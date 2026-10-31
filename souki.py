#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
想起 (Souki) -- ComfyUI 出力台帳  v2
====================================

ComfyUI が出力した PNG に埋め込まれている API 形式ワークフロー (tEXt: prompt)
を読み取って SQLite に台帳化し、ブラウザから絞り込み・再投入できるようにする。
キャラの判定は chara.py / vision.py が受け持つ。

ふだんは 起動.bat から起動する（専用の Python 環境で動く）。
    python souki.py serve                          # 画面を開く
    python souki.py index  F:/ComfyUI/output      # 台帳だけ作る / 更新する

台帳の部分は Pillow だけで動く。キャラの部分は numpy と ONNX Runtime が要る。
"""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import random
import re
import socket
import sqlite3
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    sys.stderr.write("Pillow が必要です:  pip install Pillow\n")
    raise

VERSION = "2.3.0"
APP_NAME = "想起"
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "souki.db"
THUMB_DIR = DATA_DIR / "thumbs"
CONFIG_PATH = APP_DIR / "config.json"
BRIDGE_SRC = APP_DIR / "comfyui_node" / "souki_bridge"   # ComfyUI に入れる受け口
BRIDGE_VERSION = "1.0.0"

THUMB_MAX = 360
THUMB_QUALITY = 82
SCAN_EXTS = {".png", ".webp", ".jpg", ".jpeg"}

DEFAULT_CONFIG = {
    "output_dirs": [],
    "comfy_url": "http://127.0.0.1:8188",
    "server_port": 8765,
    "scan_workers": 6,
}


# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------

def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception as e:
            sys.stderr.write(f"config.json の読み込みに失敗しました: {e}\n")
    return cfg


def save_config(cfg: dict) -> None:
    """書きかけで止まっても壊れないよう、別名に書いてから差し替える。
    キャラの設定（"chara" 欄）は chara.py が書くので、ここでは読み直して残す。"""
    try:
        cur = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    except Exception:
        cur = {}
    merged = dict(cur)
    merged.update({k: v for k, v in cfg.items() if k != "chara"})
    tmp = CONFIG_PATH.with_name(CONFIG_PATH.name + ".tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CONFIG_PATH)


# --------------------------------------------------------------------------
# データベース
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id            INTEGER PRIMARY KEY,
    path          TEXT UNIQUE NOT NULL,
    root          TEXT,
    rel_path      TEXT,
    filename      TEXT,
    folder        TEXT,
    mtime         REAL,
    size          INTEGER,
    width         INTEGER,
    height        INTEGER,
    dhash         TEXT,
    has_api       INTEGER DEFAULT 0,
    has_ui        INTEGER DEFAULT 0,
    checkpoint    TEXT,
    positive      TEXT,
    negative      TEXT,
    sampler       TEXT,
    scheduler     TEXT,
    steps         INTEGER,
    cfg           REAL,
    seed          TEXT,
    prompt_json   TEXT,
    workflow_json TEXT,
    node_types    TEXT,
    model_files   TEXT,
    indexed_at    REAL
);
CREATE INDEX IF NOT EXISTS ix_images_mtime      ON images(mtime DESC);
CREATE INDEX IF NOT EXISTS ix_images_checkpoint ON images(checkpoint);
CREATE INDEX IF NOT EXISTS ix_images_dhash      ON images(dhash);
CREATE INDEX IF NOT EXISTS ix_images_folder     ON images(folder);

CREATE TABLE IF NOT EXISTS loras (
    image_id  INTEGER NOT NULL,
    name      TEXT NOT NULL,
    strength  REAL
);
CREATE INDEX IF NOT EXISTS ix_loras_image ON loras(image_id);
CREATE INDEX IF NOT EXISTS ix_loras_name  ON loras(name);

CREATE TABLE IF NOT EXISTS tags (
    image_id  INTEGER NOT NULL,
    tag       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_tags_image ON tags(image_id);
CREATE INDEX IF NOT EXISTS ix_tags_tag   ON tags(tag);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    val TEXT
);

CREATE TABLE IF NOT EXISTS outfits (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    words TEXT NOT NULL,
    character TEXT,
    created_at REAL
);
"""


def connect(readonly: bool = False) -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con


# --------------------------------------------------------------------------
# ワークフロー解析
# --------------------------------------------------------------------------

SAMPLER_HINTS = ("ksampler", "samplercustom", "sampler_custom")
SEED_KEYS = ("seed", "noise_seed")
MODEL_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".sft", ".gguf")

CKPT_CLASSES = {
    "CheckpointLoaderSimple": "ckpt_name",
    "CheckpointLoader": "ckpt_name",
    "UNETLoader": "unet_name",
    "UnetLoaderGGUF": "unet_name",
    "ImageOnlyCheckpointLoader": "ckpt_name",
}
LORA_CLASSES = {
    "LoraLoader": ("lora_name", "strength_model"),
    "LoraLoaderModelOnly": ("lora_name", "strength_model"),
    "LoraLoader|pysssss": ("lora_name", "strength_model"),
}
TEXT_KEYS = ("text", "text_g", "text_l", "prompt", "string", "positive", "wildcard_text")


def is_link(v) -> bool:
    """ComfyUI の API 形式では、他ノードからの入力は [node_id, slot] になる。"""
    return (
        isinstance(v, list)
        and len(v) == 2
        and isinstance(v[0], (str, int))
        and isinstance(v[1], int)
    )


def parse_api_prompt(raw: str):
    try:
        d = json.loads(raw)
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    # {"prompt": {...}} で包まれている場合をほどく
    if "prompt" in d and isinstance(d.get("prompt"), dict) and "class_type" not in d:
        d = d["prompt"]
    ok = any(isinstance(v, dict) and "class_type" in v for v in d.values())
    return d if ok else None


def find_sampler_nodes(prompt: dict) -> list[str]:
    out = []
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        ct = str(node.get("class_type", ""))
        if any(h in ct.lower() for h in SAMPLER_HINTS):
            out.append(nid)
    return out


def _collect_text(prompt: dict, nid, seen: set, depth: int = 0) -> list[str]:
    """ノードから遡って、実際に書かれたテキストを集める。"""
    if depth > 12:
        return []
    key = str(nid)
    if key in seen:
        return []
    seen.add(key)
    node = prompt.get(key)
    if not isinstance(node, dict):
        return []
    inputs = node.get("inputs") or {}
    found = []
    for k in TEXT_KEYS:
        v = inputs.get(k)
        if isinstance(v, str) and v.strip():
            found.append(v.strip())
    if found:
        return found
    # テキストが直接なければ、リンクをたどる
    for v in inputs.values():
        if is_link(v):
            found += _collect_text(prompt, v[0], seen, depth + 1)
    return found


def extract_workflow_info(prompt: dict) -> dict:
    """API形式ワークフローから台帳に載せる情報を抜き出す。"""
    info = {
        "checkpoint": None,
        "positive": None,
        "negative": None,
        "sampler": None,
        "scheduler": None,
        "steps": None,
        "cfg": None,
        "seed": None,
        "loras": [],
        "node_types": [],
        "model_files": [],
    }

    node_types, model_files = set(), set()
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        ct = str(node.get("class_type", ""))
        if ct:
            node_types.add(ct)
        inputs = node.get("inputs") or {}

        if ct in CKPT_CLASSES and not info["checkpoint"]:
            v = inputs.get(CKPT_CLASSES[ct])
            if isinstance(v, str):
                info["checkpoint"] = v

        if ct in LORA_CLASSES:
            nk, sk = LORA_CLASSES[ct]
            name = inputs.get(nk)
            st = inputs.get(sk)
            if isinstance(name, str) and name:
                info["loras"].append(
                    {"name": name, "strength": float(st) if isinstance(st, (int, float)) else None}
                )

        for v in inputs.values():
            if isinstance(v, str) and v.lower().endswith(MODEL_SUFFIXES):
                model_files.add(v)

    info["node_types"] = sorted(node_types)
    info["model_files"] = sorted(model_files)

    # サンプラ系ノードから条件を拾う。steps が入っているものを優先。
    samplers = find_sampler_nodes(prompt)
    best = None
    for nid in samplers:
        inputs = (prompt[nid].get("inputs") or {})
        score = sum(1 for k in ("steps", "cfg", "sampler_name") if k in inputs)
        if best is None or score > best[0]:
            best = (score, nid)

    if best:
        nid = best[1]
        inputs = prompt[nid].get("inputs") or {}
        for k, dst, cast in (
            ("sampler_name", "sampler", str),
            ("scheduler", "scheduler", str),
            ("steps", "steps", int),
            ("cfg", "cfg", float),
        ):
            v = inputs.get(k)
            if v is not None and not is_link(v):
                try:
                    info[dst] = cast(v)
                except Exception:
                    pass
        for k in SEED_KEYS:
            v = inputs.get(k)
            if isinstance(v, (int, float)) and info["seed"] is None:
                info["seed"] = str(int(v))

        pos = inputs.get("positive")
        neg = inputs.get("negative")
        if is_link(pos):
            t = _collect_text(prompt, pos[0], set())
            if t:
                info["positive"] = "\n".join(t)
        if is_link(neg):
            t = _collect_text(prompt, neg[0], set())
            if t:
                info["negative"] = "\n".join(t)

    # サンプラが無い / 拾えなかった場合は CLIPTextEncode を順に見る
    if not info["positive"]:
        texts = []
        for nid, node in prompt.items():
            if not isinstance(node, dict):
                continue
            if "cliptextencode" in str(node.get("class_type", "")).lower():
                v = (node.get("inputs") or {}).get("text")
                if isinstance(v, str) and v.strip():
                    texts.append((nid, v.strip()))
        if texts:
            texts.sort(key=lambda x: str(x[0]))
            info["positive"] = texts[0][1]
            if len(texts) > 1 and not info["negative"]:
                info["negative"] = texts[1][1]

    if info["seed"] is None:
        for nid, node in prompt.items():
            if not isinstance(node, dict):
                continue
            for k in SEED_KEYS:
                v = (node.get("inputs") or {}).get(k)
                if isinstance(v, (int, float)):
                    info["seed"] = str(int(v))
                    break
            if info["seed"]:
                break

    return info


# --------------------------------------------------------------------------
# タグ正規化
# --------------------------------------------------------------------------

WEIGHT_RE = re.compile(r":\s*-?\d+(?:\.\d+)?\s*$")
LORA_TAG_RE = re.compile(r"<[^>]*>")


def normalize_tags(positive: str | None) -> list[str]:
    if not positive:
        return []
    txt = LORA_TAG_RE.sub(" ", positive)
    txt = txt.replace("\n", ",").replace("|", ",")
    out, seen = [], set()
    for chunk in txt.split(","):
        t = chunk.strip()
        if not t:
            continue
        # (tag:1.2) → tag / [tag] → tag / {tag} → tag
        for _ in range(4):
            t2 = t.strip()
            if len(t2) >= 2 and t2[0] in "([{" and t2[-1] in ")]}":
                t2 = t2[1:-1]
            t2 = WEIGHT_RE.sub("", t2).strip()
            if t2 == t:
                break
            t = t2
        t = t.replace("\\(", "(").replace("\\)", ")").replace("\\", "")
        t = re.sub(r"\s+", " ", t).strip().lower()
        if not t or t in ("break", "and"):
            continue
        if len(t) > 80:      # 文章っぽいものはタグ扱いしない
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# --------------------------------------------------------------------------
# 画像処理
# --------------------------------------------------------------------------

def dhash(im: Image.Image, size: int = 8) -> str:
    g = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = g.tobytes()
    bits = 0
    n = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (1 if px[base + col] > px[base + col + 1] else 0)
            n += 1
    return f"{bits:0{n // 4}x}"


DEGENERATE_HASHES = {"0" * 16, "f" * 16}


def useful_hash(h: str | None) -> str | None:
    """平坦な画像 (単色・べた塗り) は dHash が全0/全fに潰れて
    無関係な画像同士が同一判定されるので、重複判定から外す。"""
    return None if (not h or h.lower() in DEGENERATE_HASHES) else h


def thumb_path(image_id: int) -> Path:
    sub = f"{image_id % 256:02x}"
    return THUMB_DIR / sub / f"{image_id}.jpg"


def thumb_bytes(im: Image.Image) -> bytes:
    """デコード済み画像を JPEG バイト列に落とす。
    PIL の Image をそのまま持ち回すとメモリが持たないので、
    走査ワーカーの中でここまで済ませてしまう。"""
    t = im.copy()
    t.thumbnail((THUMB_MAX, THUMB_MAX), Image.Resampling.LANCZOS)
    if t.mode not in ("RGB", "L"):
        t = t.convert("RGB")
    buf = io.BytesIO()
    t.save(buf, "JPEG", quality=THUMB_QUALITY, optimize=True)
    t.close()
    return buf.getvalue()


def read_embedded(im: Image.Image) -> tuple[str | None, str | None]:
    """PNG tEXt / WebP・JPEG の EXIF から prompt / workflow を取り出す。"""
    info = im.info or {}
    api = info.get("prompt")
    ui = info.get("workflow")
    if isinstance(api, bytes):
        api = api.decode("utf-8", "replace")
    if isinstance(ui, bytes):
        ui = ui.decode("utf-8", "replace")

    if not api:
        # ComfyUI は WebP/JPEG では EXIF UserComment に入れることがある
        try:
            exif = im.getexif()
            for tag in (0x9286, 0x010E):  # UserComment, ImageDescription
                v = exif.get(tag)
                if not v:
                    continue
                if isinstance(v, bytes):
                    v = v.decode("utf-8", "replace")
                v = v.lstrip("\x00").replace("UNICODE\x00", "").strip()
                if "Prompt:" in v:
                    v = v.split("Prompt:", 1)[1]
                if v.strip().startswith("{"):
                    api = v.strip()
                    break
        except Exception:
            pass
    return api, ui


def scan_file(path: Path, root: Path, want_thumb: bool) -> dict | None:
    try:
        st = path.stat()
    except OSError:
        return None

    rec: dict = {
        "path": str(path),
        "root": str(root),
        "rel_path": str(path.relative_to(root)).replace("\\", "/"),
        "filename": path.name,
        "folder": str(path.parent.relative_to(root)).replace("\\", "/"),
        "mtime": st.st_mtime,
        "size": st.st_size,
        "width": None, "height": None, "dhash": None,
        "has_api": 0, "has_ui": 0,
        "checkpoint": None, "positive": None, "negative": None,
        "sampler": None, "scheduler": None, "steps": None,
        "cfg": None, "seed": None,
        "prompt_json": None, "workflow_json": None,
        "node_types": None, "model_files": None,
        "loras": [], "tags": [], "_thumb": None,
        "indexed_at": time.time(),
    }
    if rec["folder"] == ".":
        rec["folder"] = ""

    api_raw = ui_raw = None
    try:
        with Image.open(path) as im:
            rec["width"], rec["height"] = im.size
            api_raw, ui_raw = read_embedded(im)
            im.load()
            rec["dhash"] = useful_hash(dhash(im))
            if want_thumb:
                rec["_thumb"] = thumb_bytes(im)
    except Exception:
        return rec

    if ui_raw:
        rec["has_ui"] = 1
        rec["workflow_json"] = ui_raw
    if api_raw:
        prompt = parse_api_prompt(api_raw)
        if prompt:
            rec["has_api"] = 1
            rec["prompt_json"] = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
            info = extract_workflow_info(prompt)
            for k in ("checkpoint", "positive", "negative", "sampler",
                      "scheduler", "steps", "cfg", "seed"):
                rec[k] = info[k]
            rec["loras"] = info["loras"]
            rec["node_types"] = json.dumps(info["node_types"], ensure_ascii=False)
            rec["model_files"] = json.dumps(info["model_files"], ensure_ascii=False)
            rec["tags"] = normalize_tags(info["positive"])
    return rec


# --------------------------------------------------------------------------
# インデクサ
# --------------------------------------------------------------------------

INSERT_COLS = [
    "path", "root", "rel_path", "filename", "folder", "mtime", "size",
    "width", "height", "dhash", "has_api", "has_ui", "checkpoint",
    "positive", "negative", "sampler", "scheduler", "steps", "cfg",
    "seed", "prompt_json", "workflow_json", "node_types", "model_files",
    "indexed_at",
]


def run_index(dirs: list[str], full: bool = False, workers: int = 6,
              progress=None) -> dict:
    con = connect()
    cur = con.cursor()

    known = {}
    for r in cur.execute("SELECT id, path, mtime, size FROM images"):
        known[r["path"]] = (r["id"], r["mtime"], r["size"])

    stats = {"total_found": 0, "scanned": 0, "with_api": 0, "no_meta": 0,
             "errors": 0, "removed": 0, "moved": 0, "skipped_roots": [],
             "elapsed": 0.0}
    t0 = time.time()

    # --- 走査できたフォルダだけを削除判定の対象にする ------------------
    # 外付け・別ドライブが繋がっていないときに、そこの記録を
    # 「消えたファイル」と誤判定して台帳ごと消してしまわないための防波堤。
    reachable: list[Path] = []
    for d in dirs:
        root = Path(d).expanduser()
        if root.is_dir():
            reachable.append(root)
        else:
            stats["skipped_roots"].append(str(root))
            sys.stderr.write(f"  !! フォルダに到達できません（この分は台帳を保持します）: {root}\n")

    if not reachable:
        con.close()
        stats["elapsed"] = time.time() - t0
        return stats

    def under_scan(p: str) -> bool:
        try:
            rp = Path(p)
        except Exception:
            return False
        return any(rp.is_relative_to(r) for r in reachable)

    # --- 全ファイルを列挙 ---------------------------------------------
    found: list[tuple[Path, Path, os.stat_result]] = []
    seen_paths = set()
    for root in reachable:
        for p in root.rglob("*"):
            if p.suffix.lower() not in SCAN_EXTS or not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            seen_paths.add(str(p))
            found.append((p, root, st))
    stats["total_found"] = len(seen_paths)

    # --- 消えたファイル（到達できたフォルダ配下のみ）------------------
    gone = {p: known[p] for p in known if p not in seen_paths and under_scan(p)}

    # --- 移動の検出 ----------------------------------------------------
    # C: から K: へ退避したときに、同じ中身を読み直すのは無駄なので、
    # ファイル名とサイズが一致する「消えた記録」があればパスだけ書き換える。
    # 一意に決まる場合だけ移動とみなす（曖昧なら普通に読み直す）。
    move_pool: dict[tuple, list[str]] = {}
    for p, (iid, mt, sz) in gone.items():
        move_pool.setdefault((Path(p).name, sz), []).append(p)

    targets: list[tuple[Path, Path]] = []
    moves: list[tuple[str, Path, Path]] = []
    for p, root, st in found:
        sp = str(p)
        if sp in known:
            _id, mt, sz = known[sp]
            if not full and abs(st.st_mtime - (mt or 0)) < 1 and st.st_size == sz:
                continue
            targets.append((p, root))
            continue
        cand = move_pool.get((p.name, st.st_size)) if not full else None
        if cand and len(cand) == 1:
            moves.append((cand[0], p, root))
            move_pool.pop((p.name, st.st_size))
        else:
            targets.append((p, root))

    for old_path, p, root in moves:
        iid = gone.pop(old_path)[0]
        rel = str(p.relative_to(root)).replace("\\", "/")
        folder = str(p.parent.relative_to(root)).replace("\\", "/")
        mt = p.stat().st_mtime
        cur.execute(
            "UPDATE images SET path=?, root=?, rel_path=?, folder=?, mtime=? WHERE id=?",
            (str(p), str(root), rel, "" if folder == "." else folder, mt, iid),
        )
        # 同じファイルが場所を変えただけなので、キャラの解析結果もそのまま使う
        try:
            cur.execute("UPDATE c_analysis SET mtime=? WHERE image_id=?", (mt, iid))
        except sqlite3.Error:
            pass
    stats["moved"] = len(moves)
    if moves:
        con.commit()

    def worker(item):
        p, root = item
        try:
            return scan_file(p, root, True)
        except Exception:
            return {"__error__": str(p), "trace": traceback.format_exc()}

    # 読み直す前の見た目（dHash と大きさ）を控えておく。中身が同じなら
    # キャラの判定結果を残せる（メタデータだけ書き換わった場合など）。
    before = {}
    for i in range(0, len(targets), 800):
        paths = [str(p) for p, _ in targets[i:i + 800] if str(p) in known]
        if paths:
            qm = ",".join("?" * len(paths))
            for r in cur.execute(f"SELECT id, dhash, width, height FROM images "
                                 f"WHERE path IN ({qm})", paths):
                before[r["id"]] = (r["dhash"], r["width"], r["height"])
    new_ids: list[int] = []

    # ThreadPoolExecutor.map は渡した分を一度に全部投入してしまうので、
    # チャンクに切って処理する。こうしないと数万枚でメモリが尽きる。
    CHUNK = 240
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i in range(0, len(targets), CHUNK):
            batch = []
            for rec in ex.map(worker, targets[i:i + CHUNK]):
                if rec is None:
                    continue
                if "__error__" in rec:
                    stats["errors"] += 1
                    continue
                batch.append(rec)
                stats["scanned"] += 1
                if rec["has_api"]:
                    stats["with_api"] += 1
                else:
                    stats["no_meta"] += 1
            if batch:
                new_ids += _flush(con, cur, batch, known)
            if progress:
                progress(stats["scanned"], len(targets))

    # --- キャラの記録の後始末 -------------------------------------------
    # 読み直した絵のうち、見た目が変わっていないものは判定を残す。
    # 消えた絵と同じ見た目の絵が新しく現れたら（リネームやメタデータの
    # 削除で保存し直されたもの）、顔と確定をそちらへ引き継ぐ。
    chara_mod = load_chara()
    if chara_mod is not None:
        chara_mod.connect().close()
        if before:
            ids = list(before)
            same = []
            for i in range(0, len(ids), 800):
                ch = ids[i:i + 800]
                qm = ",".join("?" * len(ch))
                for r in cur.execute(f"SELECT id, dhash, width, height, mtime, size "
                                     f"FROM images WHERE id IN ({qm})", ch):
                    if useful_hash(r["dhash"]) and before[r["id"]] == (r["dhash"], r["width"],
                                                                         r["height"]):
                        same.append(r)
            for r in same:
                tags = [t[0] for t in cur.execute("SELECT tag FROM tags WHERE image_id=?",
                                                  (r["id"],))]
                words = chara_mod.identity_slots(tags)
                words["multi"] = chara_mod.multi_hint(tags)
                cur.execute("UPDATE c_analysis SET mtime=?, size=?, words=? WHERE image_id=?",
                            (r["mtime"], r["size"], json.dumps(words, ensure_ascii=False),
                             r["id"]))
            stats["kept_analysis"] = len(same)
        if gone and new_ids:
            look = {}
            for i in range(0, len(new_ids), 800):
                ch = new_ids[i:i + 800]
                qm = ",".join("?" * len(ch))
                for r in cur.execute(f"SELECT id, dhash, width, height FROM images "
                                     f"WHERE id IN ({qm})", ch):
                    if useful_hash(r["dhash"]):
                        look.setdefault((r["dhash"], r["width"], r["height"]), []).append(r["id"])
            moved = 0
            gone_ids = [v[0] for v in gone.values()]
            for i in range(0, len(gone_ids), 800):
                ch = gone_ids[i:i + 800]
                qm = ",".join("?" * len(ch))
                for r in list(cur.execute(f"SELECT id, dhash, width, height FROM images "
                                          f"WHERE id IN ({qm})", ch)):
                    cands = look.get((r["dhash"], r["width"], r["height"])) or []
                    if len(cands) == 1 and chara_mod.transfer_image(con, r["id"], cands[0]):
                        moved += 1
            stats["transferred"] = moved
        con.commit()

    # --- 本当に消えたものを台帳から落とす ------------------------------
    if gone:
        gone_list = list(gone)
        for i in range(0, len(gone_list), 500):
            chunk = gone_list[i:i + 500]
            ids = [gone[p][0] for p in chunk]
            qm = ",".join("?" * len(ids))
            cur.execute(f"DELETE FROM images WHERE id IN ({qm})", ids)
            cur.execute(f"DELETE FROM loras  WHERE image_id IN ({qm})", ids)
            cur.execute(f"DELETE FROM tags   WHERE image_id IN ({qm})", ids)
            if chara_mod is not None:
                chara_mod.purge_images(con, ids)
            for iid in ids:
                try:
                    thumb_path(iid).unlink(missing_ok=True)
                except OSError:
                    pass
        stats["removed"] = len(gone_list)
        con.commit()

    con.commit()
    cur.execute("INSERT OR REPLACE INTO meta(key,val) VALUES('last_index',?)",
                (str(time.time()),))
    cur.execute("INSERT OR REPLACE INTO meta(key,val) VALUES('roots',?)",
                (json.dumps(dirs, ensure_ascii=False),))
    con.commit()
    try:
        con.execute("PRAGMA optimize")
    except Exception:
        pass
    con.close()
    stats["elapsed"] = time.time() - t0
    return stats


def _flush(con, cur, batch: list[dict], known: dict) -> list[int]:
    """書き込んで、新しく台帳に載った画像の id を返す。"""
    fresh = []
    for rec in batch:
        vals = [rec[c] for c in INSERT_COLS]
        cur.execute(
            f"INSERT INTO images ({','.join(INSERT_COLS)}) "
            f"VALUES ({','.join('?' * len(INSERT_COLS))}) "
            f"ON CONFLICT(path) DO UPDATE SET "
            + ",".join(f"{c}=excluded.{c}" for c in INSERT_COLS if c != "path"),
            vals,
        )
        row = cur.execute("SELECT id FROM images WHERE path=?", (rec["path"],)).fetchone()
        iid = row["id"]
        if rec["path"] not in known:
            fresh.append(iid)
        cur.execute("DELETE FROM loras WHERE image_id=?", (iid,))
        cur.execute("DELETE FROM tags  WHERE image_id=?", (iid,))
        if rec["loras"]:
            cur.executemany(
                "INSERT INTO loras(image_id,name,strength) VALUES(?,?,?)",
                [(iid, l["name"], l["strength"]) for l in rec["loras"]],
            )
        if rec["tags"]:
            cur.executemany(
                "INSERT INTO tags(image_id,tag) VALUES(?,?)",
                [(iid, t) for t in rec["tags"]],
            )
        tb = rec.pop("_thumb", None)
        if tb:
            try:
                tp = thumb_path(iid)
                tp.parent.mkdir(parents=True, exist_ok=True)
                tp.write_bytes(tb)
            except Exception:
                pass
    con.commit()
    return fresh


# --------------------------------------------------------------------------
# ComfyUI クライアント
# --------------------------------------------------------------------------

class ComfyClient:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self._obj = None
        self._obj_at = 0.0
        self._obj_ms = 0.0
        self._obj_err = None
        self._fetching = False
        self._lock = threading.Lock()

    def _get(self, path: str, timeout: float = 20):
        req = urllib.request.Request(self.base + path, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def object_info(self, force: bool = False):
        """今持っている object_info を返す。無ければ None。

        ComfyUI はこの要求のたびにモデルフォルダを走査するので、
        構成によっては応答に数秒かかる。要求の中で待つと、その裏で
        画面の表示まで巻き込んで止まる。だから取得は裏で行い、
        呼び出し側は「今あるもの」だけを受け取る。"""
        with self._lock:
            fresh = self._obj is not None and (time.time() - self._obj_at) < 900
            if fresh and not force:
                return self._obj
            if self._fetching:
                return self._obj          # 取得中。今あるもので済ませる
            self._fetching = True
        threading.Thread(target=self._fetch_object_info, daemon=True).start()
        return self._obj

    def _fetch_object_info(self) -> None:
        """取得と解釈は重い。画面の初回表示と重ならないよう、
        少し待ってから始める。ランプは後から灯ればよい。"""
        time.sleep(2.0)
        t0 = time.time()
        try:
            data = self._get("/object_info", timeout=120)
            with self._lock:
                self._obj = data
                self._obj_at = time.time()
                self._obj_ms = (time.time() - t0) * 1000
        except Exception as e:
            with self._lock:
                self._obj_err = str(e)
                self._obj_at = time.time() - 840   # 1分後に再挑戦
        finally:
            with self._lock:
                self._fetching = False

    def ping_now(self, timeout: float = 5) -> dict:
        """その場で 1 回だけ問い合わせる（起動時の表示用）。"""
        try:
            q = self._get("/queue", timeout=timeout)
            return {"online": True, "state": "ok", "running": len(q.get("queue_running", [])),
                    "pending": len(q.get("queue_pending", []))}
        except Exception as e:
            return {"online": False, "state": "down", "error": str(e)}

    def ping(self) -> dict:
        """ComfyUI の様子（裏で 3 秒ごとに問い合わせた結果を返す。ここでは待たない）。

        生成中、とくにモデルを読み込んでいる間は、ComfyUI の返事が一時的に遅れることがある。
        1 回返事が遅れただけで「未接続」にすると投入できなくなるので、
          ok   … 返事があった
          slow … 返事が遅れているが、少し前（2 分以内）までは返事があった（生成中かも）
          down … つながらない（拒否された、または 2 分以上返事がない）
        の 3 段で返す。投入は down でも試せる（画面は知らせるだけ）。"""
        with self._lock:
            if not getattr(self, "_pinger", None):
                self._pinger = threading.Thread(target=self._ping_loop, daemon=True,
                                                name="souki-comfy-ping")
                self._ping_state = {"online": False, "state": "down", "running": 0,
                                    "pending": 0, "checking": True}
                self._pinger.start()
            return dict(self._ping_state)

    def stop(self) -> None:
        self._stopped = True

    def _ping_loop(self) -> None:
        last_ok = 0.0
        last = {"running": 0, "pending": 0}
        while not getattr(self, "_stopped", False):
            t0 = time.time()
            try:
                q = self._get("/queue", timeout=4)
                last = {"running": len(q.get("queue_running", [])),
                        "pending": len(q.get("queue_pending", []))}
                last_ok = time.time()
                st = {"online": True, "state": "ok", **last,
                      "ms": round((time.time() - t0) * 1000)}
            except Exception as e:
                refused = isinstance(getattr(e, "reason", None), ConnectionRefusedError) or \
                    isinstance(e, ConnectionRefusedError)
                if not refused and last_ok and time.time() - last_ok < 120:
                    st = {"online": True, "state": "slow", **last,
                          "since": round(time.time() - last_ok), "error": str(e)}
                else:
                    st = {"online": False, "state": "down", "running": 0, "pending": 0,
                          "error": str(e)}
            with self._lock:
                self._ping_state = st
            time.sleep(max(0.5, 3.0 - (time.time() - t0)))

    def bridge(self) -> dict:
        """ComfyUI に想起の受け口（souki_bridge）が入っているか。"""
        try:
            d = self._get("/souki/ping", timeout=5)
            return {"online": True, "installed": True, "version": d.get("version"),
                    "clients": int(d.get("clients") or 0)}
        except urllib.error.HTTPError as e:
            return {"online": True, "installed": False, "status": e.code}
        except Exception as e:
            return {"online": False, "installed": False, "error": str(e)}

    def open_in_ui(self, data: dict) -> dict:
        """開いている ComfyUI の画面にワークフローを読み込ませる（受け口が要る）。"""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base + "/souki/open", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read().decode("utf-8"))
                return {"ok": bool(d.get("ok")), "how": "bridge",
                        "clients": int(d.get("clients") or 0), "error": d.get("error")}
        except urllib.error.HTTPError as e:
            if e.code in (404, 405):
                return {"ok": False, "no_bridge": True}
            return {"ok": False, "error": f"ComfyUI が受け付けませんでした（{e.code}）"}
        except Exception as e:
            return {"ok": False, "offline": True, "error": f"ComfyUI に届きません: {e}"}

    def submit(self, prompt: dict, client_id: str = "souki", extra_pnginfo: dict | None = None,
               front: bool = False) -> dict:
        """投入する。extra_pnginfo に画面用のワークフローを入れると、できた PNG にも残る
        （ComfyUI の画面から回したときと同じ。入れないと API 用の形しか残らない）。"""
        payload = {"prompt": prompt, "client_id": client_id}
        if extra_pnginfo:
            payload["extra_data"] = {"extra_pnginfo": extra_pnginfo}
        if front:
            payload["front"] = True          # キューの先頭へ（今の生成の次）
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/prompt", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            # 生成中は返事が遅れることがあるので長めに待つ
            with urllib.request.urlopen(req, timeout=120) as r:
                return {"ok": True, "result": json.loads(r.read().decode("utf-8"))}
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                detail = json.loads(raw)
            except Exception:
                detail = {"raw": raw}
            return {"ok": False, "status": e.code, "detail": detail}
        except (TimeoutError, socket.timeout) as e:
            return {"ok": False, "status": 0, "timeout": True,
                    "detail": {"raw": "ComfyUI から 120 秒返事がありませんでした（生成中で忙しいのかもしれません）。"
                                      "キューに入ったかどうかは ComfyUI で確かめてください。"}}
        except Exception as e:
            if isinstance(getattr(e, "reason", None), (TimeoutError, socket.timeout)):
                return {"ok": False, "status": 0, "timeout": True,
                        "detail": {"raw": "ComfyUI から返事がありませんでした（生成中で忙しいのかもしれません）。"
                                          "キューに入ったかどうかは ComfyUI で確かめてください。"}}
            return {"ok": False, "status": 0, "detail": {"raw": str(e)}}


def enum_values(schema: dict, key: str):
    """object_info のノード定義から、その入力が取りうる選択肢リストを返す。"""
    for section in ("required", "optional"):
        d = (schema.get("input") or {}).get(section) or {}
        if key in d:
            spec = d[key]
            if isinstance(spec, list) and spec and isinstance(spec[0], list):
                return spec[0]
    return None


_ENUM_CACHE: dict = {}
_ENUM_CACHE_AT = [0.0]


def enum_set(obj: dict, ct: str, key: str):
    """選択肢を集合にして覚えておく。

    毎回リストから集合を作り直していた。モデル一覧が数百〜数千あると、
    画像1枚ごと・入力ごとにそれを繰り返すことになって効かない。"""
    if _ENUM_CACHE_AT[0] != id(obj):
        _ENUM_CACHE.clear()
        _ENUM_CACHE_AT[0] = id(obj)
    k = (ct, key)
    if k not in _ENUM_CACHE:
        vals = enum_values(obj.get(ct) or {}, key)
        _ENUM_CACHE[k] = ({str(c).replace("\\", "/") for c in vals}
                          if vals is not None else None)
    return _ENUM_CACHE[k]


def validate_prompt(prompt: dict, obj: dict) -> dict:
    """今の ComfyUI 環境でこのワークフローが再現できるか調べる。"""
    missing_nodes, missing_models = [], []
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        ct = str(node.get("class_type", ""))
        if not ct:
            continue
        schema = obj.get(ct)
        if schema is None:
            if ct not in missing_nodes:
                missing_nodes.append(ct)
            continue
        for k, v in (node.get("inputs") or {}).items():
            if not isinstance(v, str) or not v.lower().endswith(MODEL_SUFFIXES):
                continue
            norm = enum_set(obj, ct, k)
            if norm is None:
                continue
            if v.replace("\\", "/") not in norm:
                entry = {"node": ct, "input": k, "value": v}
                if entry not in missing_models:
                    missing_models.append(entry)

    if missing_nodes:
        status = "fault"
    elif missing_models:
        status = "partial"
    else:
        status = "ok"
    return {"status": status, "missing_nodes": missing_nodes,
            "missing_models": missing_models}


def seed_targets(prompt: dict) -> list[dict]:
    """seed を差し替える候補ノードを列挙する。サンプラ系を優先。"""
    prio, other = [], []
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        ct = str(node.get("class_type", ""))
        inputs = node.get("inputs") or {}
        for k in SEED_KEYS:
            v = inputs.get(k)
            if isinstance(v, (int, float)) and not is_link(v):
                item = {"node_id": str(nid), "class_type": ct,
                        "key": k, "current": int(v)}
                lowered = ct.lower()
                if any(h in lowered for h in SAMPLER_HINTS) or "noise" in lowered:
                    prio.append(item)
                else:
                    other.append(item)
    return prio if prio else other


def apply_seed(prompt: dict, mode: str, value: int | None = None) -> tuple[dict, list[dict]]:
    """mode: 'same' | 'random' | 'fixed'"""
    changed = []
    if mode == "same":
        return prompt, changed
    for t in seed_targets(prompt):
        new = value if (mode == "fixed" and value is not None) else random.randint(0, 2**63 - 1)
        prompt[t["node_id"]]["inputs"][t["key"]] = new
        changed.append({**t, "new": new})
    return prompt, changed


# --------------------------------------------------------------------------
# HTTP サーバ
# --------------------------------------------------------------------------

class State:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._local = threading.local()
        self.con = connect()          # 書き込み用（直列でよい）
        self.comfy = ComfyClient(cfg["comfy_url"])
        self.index_lock = threading.Lock()
        self.index_status = {"running": False, "done": 0, "total": 0, "last": None}
        # 書き込みは 1 本の接続を複数の要求で使い回すので、同時に書かないよう鍵をかける
        self.wlock = threading.RLock()
        if load_chara() is not None:
            CHARA.connect().close()          # キャラ用の表を用意しておく

    @property
    def rcon(self) -> sqlite3.Connection:
        """読み取り用の接続をスレッドごとに持つ。

        接続を1本で共有すると SQLite がそこで直列化するので、
        時間のかかる処理の後ろに軽い検索が並んで待たされる。
        実測で、単独 55ms の検索が同時実行時に 3000ms になっていた。"""
        c = getattr(self._local, "con", None)
        if c is None:
            c = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA query_only=ON")
            c.execute("PRAGMA cache_size=-40000")
            self._local.con = c
        return c


STATE: State | None = None


def q(sql: str, args=()) -> list[dict]:
    cur = STATE.rcon.execute(sql, args)
    return [dict(r) for r in cur.fetchall()]


def api_facets() -> dict:
    ck = q("SELECT checkpoint AS v, COUNT(*) AS n FROM images "
           "WHERE checkpoint IS NOT NULL AND checkpoint<>'' "
           "GROUP BY checkpoint ORDER BY n DESC LIMIT 100")
    lo = q("SELECT name AS v, COUNT(*) AS n FROM loras "
           "GROUP BY name ORDER BY n DESC LIMIT 150")
    tg = q("SELECT tag AS v, COUNT(*) AS n FROM tags "
           "GROUP BY tag ORDER BY n DESC LIMIT 300")
    fo = q("SELECT folder AS v, COUNT(*) AS n FROM images "
           "GROUP BY folder ORDER BY n DESC LIMIT 100")
    tot = q("SELECT COUNT(*) AS n, SUM(has_api) AS api FROM images")[0]
    last = q("SELECT val FROM meta WHERE key='last_index'")
    try:
        # 確定と自動の絵を数える。確認待ちはまだ本人か分からないので入れない。
        ch = q("SELECT c.name AS v, c.name AS label, COUNT(DISTINCT i.image_id) AS n "
               "FROM c_characters c JOIN c_assign a ON a.character_id=c.id "
               "AND a.state IN ('confirmed','auto') "
               "JOIN c_instances i ON i.id=a.instance_id "
               "GROUP BY c.id ORDER BY n DESC LIMIT 60")
    except Exception:
        ch = []
    return {
        "characters": ch,
        "checkpoints": ck, "loras": lo, "tags": tg, "folders": fo,
        "total": tot["n"] or 0, "with_api": tot["api"] or 0,
        "last_index": float(last[0]["val"]) if last else None,
        "comfy_url": STATE.cfg["comfy_url"],
        "roots": json.loads((q("SELECT val FROM meta WHERE key='roots'") or [{"val": "[]"}])[0]["val"]),
    }


def build_where(p: dict) -> tuple:
    """検索条件を組み立てる。一覧と時系列グラフで同じものを使う。
    片方だけ条件が抜けると、グラフと件数が食い違う。"""
    where, args = [], []

    def multi(key):
        v = p.get(key)
        if not v:
            return []
        return [x for x in (v if isinstance(v, list) else [v]) if x]

    for ck in multi("checkpoint"):
        where.append("i.checkpoint = ?"); args.append(ck)
    for fo in multi("folder"):
        where.append("i.folder = ?"); args.append(fo)
    for name in multi("lora"):
        where.append("EXISTS (SELECT 1 FROM loras l WHERE l.image_id=i.id AND l.name=?)")
        args.append(name)
    for tag in multi("tag"):
        where.append("EXISTS (SELECT 1 FROM tags t WHERE t.image_id=i.id AND t.tag=?)")
        args.append(tag)
    for tag in multi("nottag"):
        where.append("NOT EXISTS (SELECT 1 FROM tags t WHERE t.image_id=i.id AND t.tag=?)")
        args.append(tag)
    for nm in multi("chara"):
        where.append("EXISTS (SELECT 1 FROM c_instances ci "
                     "JOIN c_assign a ON a.instance_id=ci.id "
                     "AND a.state IN ('confirmed','auto') "
                     "JOIN c_characters c ON c.id=a.character_id "
                     "WHERE ci.image_id=i.id AND c.name=?)")
        args.append(str(nm))

    # 期間。mtime は epoch 秒なので、境界を秒に直して比べる。
    df = _as_epoch(p.get("date_from"))
    dt = _as_epoch(p.get("date_to"), end=True)
    if df is not None:
        where.append("i.mtime >= ?"); args.append(df)
    if dt is not None:
        where.append("i.mtime <= ?"); args.append(dt)

    text = (p.get("text") or "").strip()
    if text:
        for term in [t for t in text.split() if t]:
            where.append("(i.positive LIKE ? OR i.filename LIKE ?)")
            args += [f"%{term}%", f"%{term}%"]

    if p.get("api_only"):
        where.append("i.has_api = 1")
    if p.get("seed"):
        where.append("i.seed = ?"); args.append(str(p["seed"]).strip())
    return where, args


def _as_epoch(v, end: bool = False):
    """'2026-08' や '2026-08-16' を epoch 秒に直す。
    end=True ならその日/その月の終わりを返す。"""
    import calendar
    v = str(v or "").strip()
    if not v:
        return None
    try:
        if v.isdigit() and len(v) >= 9:
            return float(v)
        parts = [int(x) for x in re.split(r"[-/]", v) if x != ""]
        if not parts:
            return None
        y = parts[0]
        m = parts[1] if len(parts) > 1 else (12 if end else 1)
        if len(parts) > 2:
            d = parts[2]
        else:
            d = calendar.monthrange(y, m)[1] if end else 1
        d = min(d, calendar.monthrange(y, m)[1])
        return time.mktime((y, m, d, 23 if end else 0, 59 if end else 0,
                            59 if end else 0, 0, 0, -1))
    except Exception:
        return None


def api_search(p: dict) -> dict:
    where, args = build_where(p)

    sql_where = ("WHERE " + " AND ".join(where)) if where else ""

    order = {"new": "i.mtime DESC", "old": "i.mtime ASC",
             "name": "i.filename ASC", "random": "RANDOM()"}.get(p.get("sort", "new"), "i.mtime DESC")

    limit = min(int(p.get("limit", 120) or 120), 500)
    offset = max(int(p.get("offset", 0) or 0), 0)

    cols = ("i.id,i.filename,i.folder,i.rel_path,i.mtime,i.width,i.height,i.dhash,"
            "i.has_api,i.checkpoint,i.seed,i.steps,i.cfg,i.sampler,"
            "substr(i.positive,1,400) AS positive_head")

    if p.get("dedup"):
        # 同一 dhash は 1 枚に畳む（差分ワークフローのノイズ対策）
        sql = (f"SELECT {cols}, grp.n AS dup_count FROM images i "
               f"JOIN (SELECT COALESCE(NULLIF(dhash,''),'#'||id) AS k, MAX(id) AS keep, COUNT(*) AS n "
               f"      FROM images GROUP BY k) grp ON grp.keep = i.id "
               f"{sql_where} ORDER BY {order} LIMIT ? OFFSET ?")
        rows = q(sql, args + [limit, offset])
        cnt = q(f"SELECT COUNT(*) AS n FROM images i "
                f"JOIN (SELECT COALESCE(NULLIF(dhash,''),'#'||id) AS k, MAX(id) AS keep "
                f"      FROM images GROUP BY k) grp ON grp.keep=i.id {sql_where}", args)
    else:
        sql = f"SELECT {cols}, 1 AS dup_count FROM images i {sql_where} ORDER BY {order} LIMIT ? OFFSET ?"
        rows = q(sql, args + [limit, offset])
        cnt = q(f"SELECT COUNT(*) AS n FROM images i {sql_where}", args)

    return {"items": rows, "total": cnt[0]["n"], "offset": offset, "limit": limit}


def api_timeline(p: dict) -> dict:
    """月ごとの枚数。1万枚を「新しい順」で追うのは現実的でないので、
    どの時期に固まっているかを見て、そこへ飛べるようにする。
    自分自身の期間条件は外して集計する（外すと全期間の形が見えるため）。"""
    p2 = dict(p); p2.pop("date_from", None); p2.pop("date_to", None)
    where, args = build_where(p2)
    sql_where = ("WHERE " + " AND ".join(where)) if where else ""
    rows = q(f"SELECT strftime('%Y-%m', i.mtime, 'unixepoch', 'localtime') AS ym, "
             f"COUNT(*) AS n FROM images i {sql_where} "
             f"GROUP BY ym HAVING ym IS NOT NULL ORDER BY ym", args)
    return {"months": rows, "total": sum(r["n"] for r in rows)}


SEARCH_COLS = ("i.id,i.filename,i.folder,i.rel_path,i.mtime,i.width,i.height,i.dhash,"
               "i.has_api,i.checkpoint,i.seed,i.steps,i.cfg,i.sampler,"
               "substr(i.positive,1,400) AS positive_head")


def api_similar(body: dict) -> dict:
    """指定した画像に似た絵を探す。

    顔のベクトルが取れていればそれで、無ければ dHash で。
    「前にこんな感じのを描いたな」を辿るための道。"""
    iid = int(body.get("image_id"))
    limit = max(1, min(int(body.get("limit", 60)), 200))

    if chara_ok() and chara_ready_tables() and CHARA.ENGINE.ready():
        import numpy as np
        eng = CHARA.ENGINE
        try:
            eng.ensure(STATE.rcon)
        except Exception:
            eng = None
        store = eng.store if eng else None
        seeds = [r["id"] for r in q(
            "SELECT id FROM c_instances WHERE image_id=? ORDER BY (x2-x1)*(y2-y1) DESC",
            (iid,))] if store is not None else []
        seeds = [s for s in seeds if s in store.pos] if store is not None else []
        if seeds:
            x = store.rows([store.pos[seeds[0]]])
            d = np.empty(len(store.ids), dtype=np.float32)
            for a in range(0, len(store.ids), 50000):
                d[a:a + 50000] = eng.metric.diff(x, store.rows(np.arange(a, min(
                    a + 50000, len(store.ids)))))[0]
            best = {}
            for k in np.argsort(d)[:limit * 6]:
                img = int(store.image[k])
                if img == iid:
                    continue
                if img not in best:
                    best[img] = float(d[k])
                if len(best) >= limit:
                    break
            ids = list(best)
            if ids:
                qm = ",".join("?" * len(ids))
                got = q(f"SELECT {SEARCH_COLS}, 1 AS dup_count FROM images i "
                        f"WHERE i.id IN ({qm})", ids)
                order = {k: n for n, k in enumerate(ids)}
                for g in got:
                    g["similarity"] = round(1.0 - best.get(g["id"], 1.0), 3)
                got.sort(key=lambda g: order.get(g["id"], 9999))
                return {"ok": True, "by": "face", "items": got, "total": len(got)}

    row = q("SELECT dhash FROM images WHERE id=?", (iid,))
    if row and row[0]["dhash"]:
        got = q(f"SELECT {SEARCH_COLS}, 1 AS dup_count FROM images i "
                f"WHERE i.dhash=? AND i.id<>? LIMIT ?",
                (row[0]["dhash"], iid, limit))
        return {"ok": True, "by": "hash", "items": got, "total": len(got)}
    return {"ok": False, "error": "似た絵を探す手がかりがありません"}


def api_detail(iid: int) -> dict:
    rows = q("SELECT * FROM images WHERE id=?", (iid,))
    if not rows:
        return {"error": "not found"}
    r = rows[0]
    r["loras"] = q("SELECT name, strength FROM loras WHERE image_id=?", (iid,))
    r["tags"] = [x["tag"] for x in q("SELECT tag FROM tags WHERE image_id=? ORDER BY rowid", (iid,))]
    for k in ("node_types", "model_files"):
        try:
            r[k] = json.loads(r[k]) if r[k] else []
        except Exception:
            r[k] = []
    if useful_hash(r.get("dhash")):
        r["siblings"] = q("SELECT id, filename FROM images WHERE dhash=? AND id<>? LIMIT 40",
                          (r["dhash"], iid))
    else:
        r["siblings"] = []
    if r.get("prompt_json"):
        try:
            r["seed_targets"] = seed_targets(json.loads(r["prompt_json"]))
        except Exception:
            r["seed_targets"] = []
    else:
        r["seed_targets"] = []
    r.pop("workflow_json", None)   # 大きいので詳細では返さない
    return r


VALIDATE_CACHE: dict = {}
VALIDATE_EPOCH = [0]


def api_validate(ids: list[int]) -> dict:
    obj = STATE.comfy.object_info()
    if not obj:
        # まだ取得できていない。待たずに返す。少し後に画面から再度尋ねてもらう。
        return {"online": True, "pending": True, "results": {}}

    # object_info が入れ替わったら（ノードやモデルを足した等）覚え直す
    if VALIDATE_EPOCH[0] != id(obj):
        VALIDATE_CACHE.clear()
        VALIDATE_EPOCH[0] = id(obj)

    ids = ids[:200]
    out, need = {}, []
    for iid in ids:
        c = VALIDATE_CACHE.get(iid)
        if c is not None:
            out[str(iid)] = c
        else:
            need.append(iid)

    if need:
        # 1件ずつ問い合わせず、まとめて取る
        for i in range(0, len(need), 200):
            chunk = need[i:i + 200]
            qm = ",".join("?" * len(chunk))
            rows = q(f"SELECT id, prompt_json FROM images WHERE id IN ({qm})", chunk)
            got = {r["id"]: r["prompt_json"] for r in rows}
            for iid in chunk:
                pj = got.get(iid)
                if not pj:
                    r = {"status": "nometa"}
                else:
                    try:
                        r = validate_prompt(json.loads(pj), obj)
                    except Exception as e:
                        r = {"status": "fault", "missing_nodes": [], "error": str(e)}
                VALIDATE_CACHE[iid] = r
                out[str(iid)] = r
        if len(VALIDATE_CACHE) > 40000:
            VALIDATE_CACHE.clear()
    return {"online": True, "results": out}


def comfy_roots(cfg: dict) -> list[Path]:
    """出力フォルダから遡って ComfyUI のルートを推定する。"""
    roots = []
    for d in cfg.get("output_dirs", []):
        qd = Path(d)
        for _ in range(3):
            qd = qd.parent
            if (qd / "models").is_dir() or (qd / "user").is_dir():
                if qd not in roots:
                    roots.append(qd)
                break
    return roots


def save_workflow_to_comfy(name: str, ui_workflow: str) -> dict:
    """画面用のワークフローを ComfyUI 側に置く。

    /prompt に投げるだけだと生成は走るが画面には何も出ない。
    投げているのが API 形式で、画面が読むのは UI 形式だから。
    UI 形式を ComfyUI のワークフロー置き場に書けば、
    左のワークフロー一覧から開いて手直しできる。

    まず ComfyUI の userdata API を試し、駄目ならファイルを直接置く。"""
    cfg = STATE.cfg
    fname = re.sub(r'[<>:"/\\|?*]', "_", name)[:80] or "souki"
    fname = f"想起_{fname}.json"

    # 1) userdata API
    try:
        import urllib.parse
        enc = urllib.parse.quote("workflows/" + fname, safe="")
        for base in (f"{STATE.comfy.base}/api/userdata/{enc}",
                     f"{STATE.comfy.base}/userdata/{enc}"):
            req = urllib.request.Request(
                base + "?overwrite=true", data=ui_workflow.encode("utf-8"),
                method="POST", headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=15) as r:
                    if r.status < 300:
                        return {"ok": True, "how": "api", "name": fname}
            except Exception:
                continue
    except Exception:
        pass

    # 2) ファイルを直接置く
    for root in comfy_roots(cfg):
        for rel in ("user/default/workflows", "user/workflows"):
            d = root / rel
            try:
                if not d.parent.exists():
                    continue
                d.mkdir(parents=True, exist_ok=True)
                (d / fname).write_text(ui_workflow, encoding="utf-8")
                return {"ok": True, "how": "file", "name": fname, "dir": str(d)}
            except Exception:
                continue
    return {"ok": False, "error": "ComfyUI のワークフロー置き場が見つかりませんでした"}


# --------------------------------------------------------------------------
# ComfyUI の画面で開く（受け口 souki_bridge 経由）
# --------------------------------------------------------------------------

def comfy_custom_nodes_dirs() -> list[Path]:
    """ComfyUI の custom_nodes フォルダの候補。ComfyUI に聞き、だめなら出力フォルダから推定する。"""
    out: list[Path] = []

    def add(p):
        try:
            pp = Path(str(p))
            if pp.is_dir() and pp not in out:
                out.append(pp)
        except Exception:
            pass

    if STATE.cfg.get("comfy_custom_nodes"):
        add(STATE.cfg["comfy_custom_nodes"])
    for path in ("/internal/folder_paths", "/api/internal/folder_paths"):
        try:
            d = STATE.comfy._get(path, timeout=8)
            v = d.get("custom_nodes") if isinstance(d, dict) else None
            if isinstance(v, list):
                for x in v:
                    for y in (x if isinstance(x, list) else [x]):
                        if isinstance(y, str):
                            add(y)
            if out:
                break
        except Exception:
            continue
    for root in comfy_roots(STATE.cfg):
        add(root / "custom_nodes")
    return out


def api_bridge_status() -> dict:
    b = STATE.comfy.bridge()
    b["latest"] = BRIDGE_VERSION
    b["url"] = STATE.comfy.base
    b["outdated"] = bool(b.get("installed") and b.get("version") != BRIDGE_VERSION)
    b["can_install"] = BRIDGE_SRC.is_dir()
    return b


def api_bridge_install(body: dict) -> dict:
    """受け口を ComfyUI の custom_nodes に写す。効くのは ComfyUI を再起動してから。"""
    import shutil
    if not BRIDGE_SRC.is_dir():
        return {"ok": False, "error": "想起のフォルダに comfyui_node\\souki_bridge がありません（zip の上書き漏れ）"}
    want = str(body.get("dir") or "").strip()
    dirs = [Path(want)] if want else comfy_custom_nodes_dirs()
    dirs = [d for d in dirs if d.is_dir()]
    if not dirs:
        return {"ok": False, "need_dir": True,
                "error": "ComfyUI の custom_nodes フォルダが見つかりませんでした。"
                         "ComfyUI を起動してからもう一度押すか、フォルダの場所を入れてください。"}
    dst = dirs[0] / "souki_bridge"
    try:
        shutil.copytree(BRIDGE_SRC, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    except Exception as e:
        return {"ok": False, "error": f"写せませんでした: {e}"}
    if want:
        cfg = load_config()
        cfg["comfy_custom_nodes"] = str(dirs[0])
        save_config(cfg)
        STATE.cfg["comfy_custom_nodes"] = str(dirs[0])
    return {"ok": True, "dir": str(dst), "restart": True}


def api_comfy_open(body: dict) -> dict:
    """ComfyUI の画面に開く（差し替え・服・保存名も反映。名前が決まらなければ聞く）。"""
    P = prepare(int(body.get("id") or body.get("image_id")), body)
    if not P.get("ok"):
        return P
    return open_prepared(P)


def api_requeue(body: dict) -> dict:
    return run_submit(body)


# --------------------------------------------------------------------------
# キャラ — chara.py の判定を画面へつなぐ
# --------------------------------------------------------------------------

CHARA = None
CHARA_ERROR = None


def load_chara():
    """chara.py を読み込む。numpy や ONNX Runtime が無い環境でも台帳は使えるように、
    失敗しても落とさず、理由だけ覚えておく。"""
    global CHARA, CHARA_ERROR
    if CHARA is not None or CHARA_ERROR is not None:
        return CHARA
    try:
        import chara as _c
        if _c.VERSION != VERSION:
            raise ImportError(f"chara.py の版（{_c.VERSION}）が souki.py（{VERSION}）と違います。"
                              "ファイルの差し替え漏れがないか確認してください。")
        CHARA = _c
    except Exception as e:
        CHARA_ERROR = f"{type(e).__name__}: {e}"
    return CHARA


def chara_ok() -> bool:
    return load_chara() is not None


def chara_ready_tables() -> bool:
    try:
        STATE.rcon.execute("SELECT 1 FROM c_characters LIMIT 1")
        return True
    except Exception:
        return False


def _need_chara():
    if not chara_ok():
        return {"ok": False, "error": "キャラ判定の部品を読み込めませんでした: " + str(CHARA_ERROR),
                "hint": "起動.bat から起動してください（専用の Python 環境で動かす必要があります）"}
    return None


def api_chara_status() -> dict:
    err = _need_chara()
    if err:
        return {"available": False, **err, "version": VERSION}
    st = CHARA.status(STATE.rcon)
    st["available"] = True
    st["old_data"] = old_data_info()          # 覚えた値（数え直しは裏で 1 回だけ）
    return st


def api_chara_action(body: dict) -> dict:
    err = _need_chara()
    if err:
        return err
    kind = str(body.get("kind") or "")
    try:
        with STATE.wlock:
            r = CHARA.act(STATE.con, kind, body)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    if r.get("ok"):
        _CHAR_CACHE_BUST()
        # 顔が未分類に戻る操作のときだけ候補を作り直す（名前を付けただけなら不要）
        CHARA.request_engine(proposals=kind in ("undismiss", "unassign", "delete"))
    return r


def api_chara_undo() -> dict:
    err = _need_chara()
    if err:
        return err
    with STATE.wlock:
        r = CHARA.undo_last(STATE.con)
    if r.get("ok"):
        _CHAR_CACHE_BUST()
        CHARA.request_engine(proposals=True)
    return r


def api_chara_job(body: dict) -> dict:
    err = _need_chara()
    if err:
        return err
    return CHARA.start_job(str(body.get("kind") or ""), **{k: v for k, v in body.items()
                                                         if k != "kind"})


def api_chara_settings(body: dict) -> dict:
    err = _need_chara()
    if err:
        return err
    patch = {k: v for k, v in body.items() if k in CHARA.DEFAULTS}
    before = CHARA.feature_variant()
    try:
        s = CHARA.save_settings(patch)
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    out = {"ok": True, "settings": s}
    if "feature_variant" in patch and CHARA.feature_variant() != before:
        # 見方を変えたら、足りない特徴を取り足してから判定し直す
        CHARA.ENGINE.invalidate()
        out["job"] = CHARA.start_job("variant", variant=CHARA.feature_variant())
    elif "device" in patch:
        CHARA.ENGINE.invalidate()
    return out


def api_chara_export(body: dict) -> dict:
    err = _need_chara()
    if err:
        return err
    out_dir = Path(body.get("out") or (APP_DIR / "export"))
    mode = body.get("mode", "sheet")
    res = {"ok": True, "dir": str(out_dir)}
    con = STATE.rcon
    if mode in ("sheet", "both"):
        res["sheet"] = CHARA.export_sheets(con, out_dir / "キャラ設定シート.html")
    if mode in ("lora", "both"):
        ids = body.get("ids")
        chars = CHARA.list_characters(con)
        if ids:
            chars = [c for c in chars if c["id"] in {int(x) for x in ids}]
        res["lora"] = [CHARA.export_lora(con, c["id"], out_dir / "lora",
                                         repeats=int(body.get("repeats", 10)),
                                         cap=int(body.get("max", 200)),
                                         include_auto=bool(body.get("include_auto")))
                       for c in chars]
    if sys.platform == "win32" and body.get("open"):
        try:
            os.startfile(str(out_dir))          # noqa: E1101 (Windows のみ)
        except Exception:
            pass
    return res


def _profile_words(cid: int) -> list:
    try:
        return CHARA.character_profile(STATE.rcon, cid)
    except Exception:
        return []


_OLD_INFO = {"value": None, "running": False}
_OLD_LOCK = threading.Lock()


def old_data_info(refresh: bool = False) -> dict:
    """前の版（顔クラスタ方式）が残したものの大きさ。

    前の版の顔の切り抜きは何万ファイルもあり、数えるだけで数秒〜数十秒かかる。
    画面を開くたびに数えると全部の操作が遅くなるので、裏で 1 回だけ数えて覚えておく。"""
    with _OLD_LOCK:
        if _OLD_INFO["value"] is not None and not refresh:
            return _OLD_INFO["value"]
        tables = []
        try:
            names = {r[0] for r in STATE.rcon.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            tables = sorted(names & {"faces", "clusters", "face_scan"})
        except Exception:
            pass
        has_files = (DATA_DIR / "faces").exists() or (DATA_DIR / "models").exists()
        if not has_files:
            _OLD_INFO["value"] = {"tables": tables, "faces_dir_mb": 0.0, "files": []}
            return _OLD_INFO["value"]
        base = {"tables": tables, "faces_dir_mb": None, "files": [], "counting": True}
        if _OLD_INFO["running"]:
            return base
        _OLD_INFO["running"] = True

    def count():
        tot = 0
        for d, pat in ((DATA_DIR / "faces", "*.jpg"), (DATA_DIR / "models", "*.pt")):
            if d.exists():
                for p in d.rglob(pat):
                    try:
                        tot += p.stat().st_size
                    except OSError:
                        pass
        with _OLD_LOCK:
            _OLD_INFO["value"] = {"tables": tables, "faces_dir_mb": round(tot / 1e6, 1), "files": []}
            _OLD_INFO["running"] = False

    threading.Thread(target=count, daemon=True, name="souki-oldinfo").start()
    return base


def api_old_data_delete() -> dict:
    import shutil
    with STATE.wlock:
        for t in ("faces", "clusters", "face_scan"):
            STATE.con.execute(f"DROP TABLE IF EXISTS {t}")
        STATE.con.commit()
    fd = DATA_DIR / "faces"
    if fd.exists():
        shutil.rmtree(fd, ignore_errors=True)
    md = DATA_DIR / "models"          # 前の版の顔検出モデル（*.pt）。新しい版は models\ を使う
    if md.exists():
        for p in md.rglob("*.pt"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            md.rmdir()
        except OSError:
            pass
    try:
        STATE.con.execute("VACUUM")
    except Exception:
        pass
    with _OLD_LOCK:
        _OLD_INFO["value"] = None
    return {"ok": True, "left": old_data_info()}


_CHAR_CACHE_BUST = lambda: (_TRAITS_CACHE.clear(),  # noqa: E731
                            _CHARTAGS_CACHE.update({"tags": None, "at": 0.0}))


def api_settings(body: dict | None = None) -> dict:
    """出力フォルダなどの基本設定。書くときは読んでから一部だけ書き換える。"""
    cfg = load_config()
    if body:
        if "output_dirs" in body:
            dirs = [str(Path(str(d).strip().strip('"')).expanduser())
                    for d in body["output_dirs"] if str(d).strip()]
            cfg["output_dirs"] = dirs
        if "comfy_url" in body and str(body["comfy_url"]).strip():
            cfg["comfy_url"] = str(body["comfy_url"]).strip()
        if "save_template" in body:
            t = str(body.get("save_template") or "").strip()
            if t and t != DEFAULT_SAVE_TEMPLATE:
                cfg["save_template"] = t
            else:
                cfg.pop("save_template", None)
            STATE.cfg["save_template"] = cfg.get("save_template")
        save_config(cfg)
        STATE.cfg.update({k: cfg[k] for k in ("output_dirs", "comfy_url") if k in cfg})
        if "comfy_url" in body:
            old = STATE.comfy
            STATE.comfy = ComfyClient(STATE.cfg["comfy_url"])
            old.stop()
    return {"ok": True, "output_dirs": cfg.get("output_dirs", []),
            "comfy_url": cfg.get("comfy_url"), "save_template": save_template(),
            "reachable": {d: Path(d).is_dir() for d in cfg.get("output_dirs", [])}}


# --------------------------------------------------------------------------
# キャラの差し替え — 状況はそのまま、キャラだけ入れ替えて投入する
# --------------------------------------------------------------------------

# 体の造形を指す語。着替えても変わらないもの。
BODY_PARTS = (
    "hair", "eyes", "eye", "horn", "horns", "ears", "ear", "tail", "tails",
    "wings", "wing", "halo", "fang", "fangs", "freckles", "mole", "skin",
    "髪", "目", "瞳", "角", "耳", "尻尾", "翼",
)

# 着替えや場面で変わる語。特徴として拾わない。
SITUATION_WORDS = (
    "uniform", "shirt", "skirt", "dress", "jacket", "coat", "hoodie", "sweater",
    "kimono", "yukata", "swimsuit", "bikini", "apron", "glove", "sock", "shoe",
    "boot", "hat", "cap", "scarf", "armor", "cape", "camisole",
    "pants", "jeans", "shorts", "smile", "blush", "open mouth",
    "looking", "sitting", "standing", "lying", "holding", "background",
    "indoors", "outdoors", "night", "day", "sky", "light", "shadow",
    "upper body", "full body", "cowboy shot", "portrait", "close-up",
    "from side", "from behind", "from above", "from below", "masterpiece",
    "best quality", "highres", "absurdres", "solo", "1girl", "1boy", "2girls",
)


# 部位名を伴わずに単独で使う造形語。
# blonde のように「hair」を書かない書き方が多いので、語そのもので持つ。
STANDALONE_IDENTITY = {
    # 髪の色・質
    "blonde", "blond", "brunette", "redhead", "platinum blonde", "silver-haired",
    "white-haired", "black-haired", "blonde girl", "albino",
    # 髪型
    "twintails", "twintail", "ponytail", "side ponytail", "ahoge", "bangs",
    "blunt bangs", "braid", "braids", "twin braids", "hime cut", "bob cut",
    "bun", "double bun", "sidelocks", "drill hair", "updo", "afro", "mohawk",
    "messy hair", "wavy hair", "straight hair", "curly hair",
    # 年齢・体格（キャラを分ける記述になりやすい）
    "loli", "shota", "child", "female child", "male child", "toddler",
    "teenage", "teenager", "teenage girl", "teenage boy", "young girl",
    "young boy", "adult", "young adult", "mature", "mature female",
    "mature male", "milf", "middle-aged", "elderly", "old woman", "old man",
    "aged up", "aged down", "petite", "tall", "short", "tall girl",
    "flat chest", "small breasts", "medium breasts", "large breasts",
    "huge breasts", "muscular", "muscular female", "toned", "plump", "chubby",
    "slender", "slim", "skinny", "thin", "curvy", "wide hips", "thick thighs",
    "slender body", "slim body", "thin body",
    # 種族・属性
    "elf", "dark elf", "half-elf", "demon", "demon girl", "demon boy",
    "oni", "angel", "fallen angel", "vampire", "dragon girl", "dragon boy",
    "catgirl", "foxgirl", "wolf girl", "bunny girl", "mermaid", "android",
    "robot", "cyborg", "ghost", "monster girl", "kemonomimi", "youkai",
}

# 語尾がこれなら造形（部位名を伴わない書き方）
IDENTITY_SUFFIX = ("-haired", "-eyed", "-skinned", "haired", "eyed")

# 語そのものが飾り・表情・道具であるもの（部位名を含んでいても造形ではない）
NOT_IDENTITY_EXACT = (
    "hair ribbon", "hair ornament", "hair clip", "hair bow", "hairband",
    "hair band", "hair flower", "hairpin", "hair pin", "hair over one eye",
    "hair between eyes", "closed eyes", "half-closed eyes", "eyes visible",
    "eyes visible through hair", "glowing eyes", "eyepatch", "eyewear",
    "glasses", "sunglasses", "earrings", "ear piercing", "eyelashes",
    "eyeshadow", "eyeliner", "headphones", "cat ear headphones",
    "wind lift", "hair up", "hair down",
)


def _extra_words(key: str) -> set:
    """config.json で語を足せるようにする。
    実際の書き方は人によって違うので、こちらで抱え込まない。"""
    try:
        v = (STATE.cfg or {}).get(key) or []
        return {str(x).strip().lower() for x in v if str(x).strip()}
    except Exception:
        return set()


def is_body_tag(tag: str) -> bool:
    """その語が「その子である」ことを表しているか。

    完全一致だけで消そうとすると、silver hair / long silver hair /
    silver_hair のような表記ゆれで消し残る。残ると新旧の特徴が混ざるので、
    語の中身を見て判定する。

    判定は3通り。
      ・部位名で終わる     … red horns, blue eyes, long silver hair
      ・単独で使う造形語   … blonde, loli, elf, flat chest
      ・語尾が -haired 等  … silver-haired
    飾り（hair ribbon）や表情（closed eyes）は先に除外する。"""
    t = re.sub(r"[_\-]+(?=[a-z])", " ", tag.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    if not t:
        return False
    if t in _extra_words("not_identity_words"):
        return False
    if t in NOT_IDENTITY_EXACT or any(t.endswith(" " + w) for w in NOT_IDENTITY_EXACT):
        return False
    if t in STANDALONE_IDENTITY or t in _extra_words("identity_words"):
        return True
    if any(t.endswith(sfx) for sfx in IDENTITY_SUFFIX):
        return True
    words = t.split(" ")
    if words[-1] in BODY_PARTS:
        return True
    # 「… の loli」「… hair」のように、末尾が単独語のときも拾う
    if words[-1] in STANDALONE_IDENTITY:
        return True
    return any(w in t for w in ("髪", "瞳", "角", "尻尾", "翼"))


def _is_identity(tag: str) -> bool:
    if any(w in tag.lower() for w in SITUATION_WORDS):
        return False
    return is_body_tag(tag)


_TRAITS_CACHE: dict = {}
_TRAITS_AT = [0.0]


def identity_tags(name: str) -> dict:
    """そのキャラを「その子」たらしめている語を選ぶ。

    キャラシートは偏っている語をすべて出すが、その中には
    「いつも同じ服を着ている」ような語も混ざる。それを持ち込むと
    差し替えたときに服まで付いてくるので、ここで分ける。

    判定は 2 段階。
      ・髪/目/角 のような語 … 出現率が中くらいでも採る
      ・それ以外            … ほぼ毎回出ている語だけ採る（服は落ちる）
    手で直した内容があれば、そちらを優先する。
    集計に使うのは確定と自動の 1 人の絵だけ（確認待ちはまだ本人か分からない）。"""
    if not chara_ok() or not chara_ready_tables():
        return {"tags": [], "source": "none"}
    rows = q("SELECT words FROM c_characters WHERE name=? AND words IS NOT NULL "
             "AND words<>'' LIMIT 1", (name,))
    if rows:
        return {"tags": [t.strip() for t in rows[0]["words"].split(",") if t.strip()],
                "source": "manual"}
    # 同じキャラを何度も計算し直さない。集計が重いので効く。
    if time.time() - _TRAITS_AT[0] > 600:
        _TRAITS_CACHE.clear(); _TRAITS_AT[0] = time.time()
    hit = _TRAITS_CACHE.get(name)
    if hit is not None:
        return hit
    try:
        sheet = CHARA.character_sheet(STATE.rcon, name=name)
    except Exception:
        return {"tags": [], "source": "none"}

    picked, dropped = [], []
    for t in sheet.get("tags", []):
        tag, sup, lift = t["tag"], t["support"], t["lift"]
        if lift < 1.5:
            continue
        if _is_identity(tag) and sup >= 0.35:
            picked.append(tag)
        elif sup >= 0.85 and not any(w in tag.lower() for w in SITUATION_WORDS):
            picked.append(tag)
        else:
            dropped.append(tag)
    out = {"tags": picked[:16], "dropped": dropped[:16],
           "source": "auto", "n_images": sheet.get("n_images", 0)}
    _TRAITS_CACHE[name] = out
    return out


_CHARTAGS_CACHE = {"tags": None, "at": 0.0}


def character_distinguishing_tags() -> dict:
    """命名済みキャラのうち、特定の子に強く偏っている語を集める。

    語リストだけに頼ると必ず漏れる（blonde のような書き方は無数にある）。
    そこで台帳そのものから学ぶ。ある語が特定のキャラにだけ集中して
    出ているなら、それはそのキャラを他と分けている記述であり、
    差し替えのときに残してはいけない語である。

    逆に、複数のキャラにまたがって出る語（school uniform など）は
    状況の語なので残す。"""
    if (_CHARTAGS_CACHE["tags"] is not None
            and time.time() - _CHARTAGS_CACHE["at"] < 600):
        return _CHARTAGS_CACHE["tags"]
    out = {}
    try:
        if not chara_ok():
            raise RuntimeError("chara unavailable")
        chars = q("SELECT id, name FROM c_characters")
        if len(chars) >= 2:
            # キャラごとの画像数と、タグの出現数（確定と自動の 1 人の絵）
            per = {}
            for ch in chars:
                nm = ch["name"]
                ids = CHARA.character_image_ids(STATE.rcon, ch["id"])
                if not ids:
                    continue
                cnt = {}
                for i in range(0, len(ids), 900):
                    part = ids[i:i + 900]
                    qm = ",".join("?" * len(part))
                    for r in q(f"SELECT tag, COUNT(DISTINCT image_id) c FROM tags "
                               f"WHERE image_id IN ({qm}) GROUP BY tag", part):
                        cnt[r["tag"]] = cnt.get(r["tag"], 0) + r["c"]
                per[nm] = (cnt, len(ids))

            # 各タグについて「どのキャラでどれだけ出るか」を見る
            tags = set()
            for cnt, _ in per.values():
                tags |= set(cnt)
            for t in tags:
                rates = [(cnt.get(t, 0) / n, nm) for nm, (cnt, n) in per.items()]
                rates.sort(reverse=True)
                top, owner = rates[0]
                others = [r for r, _ in rates[1:]]
                second = max(others) if others else 0.0
                # その子ではほぼ毎回出て、他の子ではめったに出ない語
                if top >= 0.6 and second <= 0.15:
                    out[t] = owner
    except Exception:
        pass
    _CHARTAGS_CACHE.update({"tags": out, "at": time.time()})
    return out


def find_positive_nodes(prompt: dict) -> list:
    """書き換えるべきプロンプト欄を探す。

    サンプラの positive 入力から遡って、文字列を持つノードに行き着いた
    ところが本体。ネガティブ側は触らない。"""
    out, seen = [], set()

    def walk(nid, depth=0):
        key = str(nid)
        if depth > 12 or key in seen:
            return
        seen.add(key)
        node = prompt.get(key)
        if not isinstance(node, dict):
            return
        inputs = node.get("inputs") or {}
        hit = False
        for k in ("text", "text_g", "text_l", "prompt", "string", "wildcard_text"):
            v = inputs.get(k)
            if isinstance(v, str) and v.strip():
                out.append((key, k)); hit = True
        if hit:
            return
        for v in inputs.values():
            if is_link(v):
                walk(v[0], depth + 1)

    for nid in find_sampler_nodes(prompt):
        pos = (prompt[nid].get("inputs") or {}).get("positive")
        if is_link(pos):
            walk(pos[0])
    if not out:
        for nid, node in prompt.items():
            if not isinstance(node, dict):
                continue
            if "cliptextencode" in str(node.get("class_type", "")).lower():
                v = (node.get("inputs") or {}).get("text")
                if isinstance(v, str) and v.strip():
                    out.append((str(nid), "text"))
                    break
    return out


def body_tags_in(text: str) -> list:
    """その本文に書かれている「体の造形」の語を全部拾う。"""
    out = []
    for chunk in text.split(","):
        norm = normalize_tags(chunk)
        core = norm[0] if norm else chunk.strip().lower()
        if core and is_body_tag(core):
            out.append(core)
    return out


def swap_in_text(text: str, remove: list, add: list) -> dict:
    """プロンプト本文の中で、キャラの語だけを入れ替える。

    書式は極力そのまま残す。重み付き `(tag:1.2)` も、中身を見て
    一致すれば丸ごと外す。入れる場所は、最初に外した語のところ。
    外すものが無ければ先頭に足す。"""
    rm = {r.strip().lower() for r in remove if r.strip()}
    parts = [c for c in text.split(",")]
    kept, removed, insert_at = [], [], None
    for chunk in parts:
        norm = normalize_tags(chunk)
        core = norm[0] if norm else chunk.strip().lower()
        if not core:
            kept.append(chunk)
            continue
        if core in rm:
            removed.append(chunk.strip())
            if insert_at is None:
                insert_at = len(kept)
            continue
        kept.append(chunk)
    if insert_at is None:
        insert_at = 0
    add_clean = [a for a in add if a.strip()
                 and a.strip().lower() not in
                 {x for c in kept for x in normalize_tags(c)}]
    out = kept[:insert_at] + [" " + a for a in add_clean] + kept[insert_at:]
    new = ",".join(out)
    new = re.sub(r",\s*,", ", ", new).strip().strip(",")
    return {"text": new, "removed": removed, "added": add_clean}


DATE_IN_PATH = re.compile(r"(?<!\d)(\d{4})([-/_.])(\d{1,2})\2(\d{1,2})(?!\d)")
SAVE_KEYS = ("filename_prefix", "filename", "output_path", "path", "save_path",
             "subdirectory", "folder", "directory")


def retarget_save_date(prompt: dict, when=None) -> dict:
    """保存先に焼き込まれた日付を、今日のものに差し替える。

    ComfyUI に投げる時点では %date:yyyy% のような書式ではなく、
    生成した当時の日付が文字列として入っている。そのまま投げ直すと
    何年も前のフォルダに保存されて見つからない。
    桁数は元の書き方（2025-3-14 か 2025-03-14 か）に合わせる。"""
    t = time.localtime(when if when is not None else time.time())
    changed = []

    def sub(m):
        sep, mo, d = m.group(2), m.group(3), m.group(4)
        nm = f"{t.tm_mon:02d}" if len(mo) == 2 else str(t.tm_mon)
        nd = f"{t.tm_mday:02d}" if len(d) == 2 else str(t.tm_mday)
        return f"{t.tm_year}{sep}{nm}{sep}{nd}"

    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for k, v in list(inputs.items()):
            if k not in SAVE_KEYS or not isinstance(v, str) or not v:
                continue
            nv = DATE_IN_PATH.sub(sub, v)
            if nv != v:
                inputs[k] = nv
                changed.append({"node": str(nid),
                                "class_type": node.get("class_type"),
                                "key": k, "before": v, "after": nv})
    return {"prompt": prompt, "changed": changed}


def apply_edits_to_ui(ui_json: str, edits=None, save_changes=None) -> str:
    """画面用ワークフローに、差し替えた文字列を反映する。
    画面側のノードは widgets_values に値を持つので、そこを置き換える。"""
    swaps = {e["before"]: e["after"] for e in (edits or [])}
    for c in (save_changes or []):
        swaps[c["before"]] = c["after"]
    if not swaps:
        return ui_json
    try:
        g = json.loads(ui_json)
        for nd in g.get("nodes", []):
            wv = nd.get("widgets_values")
            if isinstance(wv, list):
                for i, v in enumerate(wv):
                    if isinstance(v, str) and v in swaps:
                        wv[i] = swaps[v]
            elif isinstance(wv, dict):
                for k, v in list(wv.items()):
                    if isinstance(v, str) and v in swaps:
                        wv[k] = swaps[v]
        return json.dumps(g, ensure_ascii=False)
    except Exception:
        return ui_json


def source_character(image_id: int) -> str | None:
    """その絵に写っているキャラ（いちばん大きく写っている子）。"""
    if not chara_ok() or not chara_ready_tables():
        return None
    rows = q("SELECT c.name FROM c_instances i JOIN c_assign a ON a.instance_id=i.id "
             "AND a.state IN ('confirmed','auto') JOIN c_characters c ON c.id=a.character_id "
             "WHERE i.image_id=? ORDER BY (i.x2-i.x1)*(i.y2-i.y1) DESC LIMIT 1", (image_id,))
    return rows[0]["name"] if rows else None


def build_swap(image_id: int, to_name: str, extra_remove=None, extra_add=None,
               outfit=None, strength: str = "full", keep_char: bool = False) -> dict:
    """キャラ（と服）を入れ替えたプロンプトを作る。

    outfit: None なら服はそのまま。語の並びなら、元の服の語（髪飾りも含む）を外してそれを入れる。
    strength: "full" は体格・年齢・種族まで、"light" は髪と目だけ入れ替える。
    keep_char: キャラはそのまま（服だけ替える）。"""
    rows = q("SELECT prompt_json, positive, filename FROM images WHERE id=?",
             (image_id,))
    if not rows or not rows[0]["prompt_json"]:
        return {"ok": False, "error": "この画像にはワークフローがありません"}
    prompt = json.loads(rows[0]["prompt_json"])
    nodes = find_positive_nodes(prompt)
    if not nodes:
        return {"ok": False, "error": "書き換えるプロンプト欄が見つかりませんでした"}

    src = source_character(image_id)
    if not keep_char and src and src == to_name:
        if outfit is None:
            return {"ok": False,
                    "error": f"この画像はすでに「{to_name}」です。別のキャラを選ぶか、服装を選んでください"}
        keep_char = True
    if keep_char and outfit is None:
        return {"ok": False, "error": "服装を選んでください（キャラはそのまま）"}
    light = strength == "light"
    rm, add, tgt = [], [], {"tags": []}
    if not keep_char:
        rm = list(identity_tags(src)["tags"]) if src else []
        rm += [t.strip() for t in (extra_remove or []) if t.strip()]
        tgt = identity_tags(to_name)
        add = list(tgt["tags"]) + [t.strip() for t in (extra_add or []) if t.strip()]
        if light:
            rm = [t for t in rm if is_hair_eye_tag(t)]
            add = [t for t in add if is_hair_eye_tag(t)]
        if not add:
            return {"ok": False,
                    "error": f"「{to_name}」の特徴が取れませんでした。"
                             f"キャラ画面で特徴を設定してください"}
    new_outfit = [w.strip() for w in (outfit or []) if w and w.strip()] if outfit is not None else None

    edits = []
    learned = character_distinguishing_tags() if not keep_char else {}
    for nid, key in nodes:
        before = prompt[nid]["inputs"][key]
        text, removed, added, auto = before, [], [], []
        if not keep_char:
            # 元キャラの特徴だけでなく、その本文に書かれている造形の語を
            # すべて外す。表記ゆれ（silver hair / long silver hair /
            # silver_hair）で消し残ると、新旧の特徴が混ざるため。
            # 除外は3段構え。
            #   1. 元キャラのキャラシート（完全一致）
            #   2. 語の中身から造形と判る語（表記ゆれに強い）
            #   3. 台帳から学んだ「特定の子に偏っている語」（語リストの漏れを補う）
            auto = [t for t in normalize_tags(before)
                    if t in learned and learned[t] != to_name and not is_outfit_tag(t)]
            body = body_tags_in(before)
            if light:
                auto = [t for t in auto if is_hair_eye_tag(t)]
                body = [t for t in body if is_hair_eye_tag(t)]
            rm_here = list(dict.fromkeys(rm + body + auto))
            r = swap_in_text(text, rm_here, add)
            text, removed, added = r["text"], r["removed"], r["added"]
        if new_outfit is not None:
            r2 = swap_in_text(text, outfit_tags_in(text), new_outfit)
            text = r2["text"]
            removed += r2["removed"]
            added += r2["added"]
        prompt[nid]["inputs"][key] = text
        edits.append({"node": nid, "class_type": prompt[nid].get("class_type"),
                      "before": before, "after": text,
                      "removed": removed, "added": added,
                      "learned": [t for t in auto if t not in
                                  [x.lower() for x in rm]]})
    to = src if keep_char else to_name
    return {"ok": True, "prompt": prompt, "edits": edits,
            "from": src, "to": to, "keep_char": keep_char,
            "to_tags": tgt["tags"], "from_tags": rm,
            "filename": rows[0]["filename"]}


def api_swap_preview(body: dict) -> dict:
    P = prepare(int(body.get("image_id")), body, preview=True)
    return _preview_out(P)


def api_swap_submit(body: dict) -> dict:
    return run_submit(body)


# --------------------------------------------------------------------------
# 服装 — 服の語の見分けと、いつもの服・登録した服
# --------------------------------------------------------------------------

# 語の最後がこれなら服（色や状態の語が前に付いていても：red ribbon / open jacket）
OUTFIT_NOUNS = {
    "shirt", "shirts", "t-shirt", "blouse", "skirt", "skirts", "miniskirt", "dress", "dresses",
    "gown", "uniform", "uniforms", "serafuku", "blazer", "jacket", "coat", "overcoat", "raincoat",
    "hoodie", "sweater", "cardigan", "vest", "kimono", "yukata", "hakama", "haori", "obi",
    "swimsuit", "swimsuits", "bikini", "leotard", "bodysuit", "apron", "gloves", "glove", "mittens",
    "socks", "sock", "kneehighs", "thighhighs", "pantyhose", "stockings", "legwear", "shoes", "shoe",
    "boots", "boot", "sneakers", "sandals", "loafers", "heels", "footwear", "slippers", "hat", "beret",
    "cap", "helmet", "hood", "scarf", "necktie", "tie", "bowtie", "bow", "ribbon", "ribbons",
    "neckerchief", "collar", "choker", "cape", "cloak", "armor", "armour", "belt", "pants", "jeans",
    "shorts", "trousers", "leggings", "bloomers", "sleeves", "camisole", "bra", "panties", "lingerie",
    "underwear", "pajamas", "robe", "suit", "tuxedo", "jumpsuit", "overalls", "costume", "outfit",
    "clothes", "clothing", "attire", "garter", "garters", "armband", "wristband", "bracelet",
    "necklace", "earrings", "jewelry", "pendant", "brooch", "headphones", "headband", "headdress",
    "headwear", "tiara", "crown", "veil", "hairband", "hairclip", "hairpin", "scrunchie", "maid",
    "sarashi", "bandeau", "poncho", "parka", "windbreaker", "tabard", "corset", "petticoat",
}
# 2 語で 1 つの服（最後の語だけでは決まらないもの）
OUTFIT_EXACT = {"tank top", "crop top", "tube top", "hair ornament", "hair ornaments",
                "hair ribbon", "hair bow", "hair flower", "hair clip", "hair band",
                "hair pin", "hair tie", "hair bobbles", "hair bun cover"}
# 髪飾りは服の一部として扱う（服を替えると入れ替わる）
OUTFIT_SUFFIX = ("hair ornament", "hair ribbon", "hair bow", "hair flower", "hair clip",
                 "hairclip", "hairpin", "hair pin", "hairband", "hair band")


def is_outfit_tag(tag: str) -> bool:
    t = re.sub(r"[_]+", " ", tag.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    if not t:
        return False
    if t in _extra_words("not_outfit_words"):
        return False
    if t in _extra_words("outfit_words") or t in OUTFIT_EXACT:
        return True
    if any(t.endswith(s) for s in OUTFIT_SUFFIX):
        return True
    words = t.split(" ")
    last = words[-1]
    if last in OUTFIT_NOUNS:
        # 体の語（red eyes など）や、髪型の語（hair bun）は服ではない
        return not (len(words) >= 2 and words[-2] in ("hair",) and last in ("bun", "buns"))
    return False


def outfit_tags_in(text: str) -> list:
    """本文に書かれている服の語（中身の語）を、書かれている順に。"""
    out = []
    for chunk in (text or "").split(","):
        norm = normalize_tags(chunk)
        core = norm[0] if norm else chunk.strip().lower()
        if core and is_outfit_tag(core) and core not in out:
            out.append(core)
    return out


# 「髪と目だけ」で入れ替える語
HAIR_EYE_LAST = {"hair", "eyes", "eye", "bangs", "ahoge", "sidelocks", "ponytail", "twintails",
                 "twintail", "braid", "braids", "pupils", "hairstyle", "bun", "buns"}
HAIR_EYE_STANDALONE = {"blonde", "blond", "brunette", "redhead", "platinum blonde",
                       "silver-haired", "white-haired", "black-haired", "hime cut", "bob cut",
                       "bun", "double bun", "drill hair", "updo", "afro", "mohawk",
                       "side ponytail", "twin braids", "heterochromia", "albino"}


def is_hair_eye_tag(tag: str) -> bool:
    t = re.sub(r"[_\-]+(?=[a-z])", " ", tag.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    if not t or is_outfit_tag(t):
        return False
    if t in HAIR_EYE_STANDALONE:
        return True
    if t.endswith(("haired", "eyed")):
        return True
    return t.split(" ")[-1] in HAIR_EYE_LAST or any(w in t for w in ("髪", "瞳"))


_USUAL_CACHE: dict = {}


def usual_outfits(name: str, limit: int = 5) -> list:
    """その子がよく着ている服（確定・自動の絵のプロンプトから、よく一緒に書かれる服の語の組）。"""
    if not name or not chara_ok() or not chara_ready_tables():
        return []
    sig = q("SELECT COUNT(*) AS n, COALESCE(MAX(id),0) AS m FROM images")[0]
    key = (name, sig["n"], sig["m"])
    hit = _USUAL_CACHE.get(name)
    if hit and hit[0] == key:
        return hit[1][:limit]
    rows = q("SELECT DISTINCT im.id, im.positive FROM c_characters c "
             "JOIN c_assign a ON a.character_id=c.id AND a.state IN ('confirmed','auto') "
             "JOIN c_instances i ON i.id=a.instance_id JOIN images im ON im.id=i.image_id "
             "WHERE c.name=? AND i.solo=1 ORDER BY im.mtime DESC LIMIT 4000", (name,))
    sets: dict = {}
    for r in rows:
        tags = outfit_tags_in(r["positive"] or "")
        if not tags:
            continue
        fs = frozenset(tags)
        e = sets.setdefault(fs, {"n": 0, "words": tags, "sample": r["id"]})
        e["n"] += 1
    groups = []
    for fs, e in sorted(sets.items(), key=lambda kv: -kv[1]["n"]):
        for g in groups:
            inter = len(fs & g["set"])
            if inter / max(1, len(fs | g["set"])) >= 0.6:
                g["n"] += e["n"]
                break
        else:
            groups.append({"set": fs, "words": e["words"], "n": e["n"], "sample": e["sample"]})
    groups.sort(key=lambda g: -g["n"])
    out = [{"words": g["words"], "n": g["n"], "sample": g["sample"]} for g in groups[:12]]
    _USUAL_CACHE[name] = (key, out)
    return out[:limit]


def api_outfits(qs: dict) -> dict:
    name = str(qs.get("chara") or "").strip()
    if qs.get("all"):
        saved = q("SELECT id, name, words, character FROM outfits ORDER BY character, name")
    else:
        saved = q("SELECT id, name, words, character FROM outfits WHERE character IS NULL "
                  "OR character='' OR character=? ORDER BY (character IS NULL OR character=''), name",
                  (name,))
    for s in saved:
        s["words"] = [w for w in (s["words"] or "").split("\n") if w.strip()]
    return {"usual": usual_outfits(name) if name else [], "saved": saved}


def api_outfit_of(qs: dict) -> dict:
    iid = int(qs.get("id") or 0)
    rows = q("SELECT positive FROM images WHERE id=?", (iid,))
    if not rows:
        return {"words": [], "character": None}
    chars = image_characters(iid)
    return {"words": outfit_tags_in(rows[0]["positive"] or ""),
            "character": chars[0] if len(chars) == 1 else None, "characters": chars}


def api_outfit_save(body: dict) -> dict:
    name = str(body.get("name") or "").strip()
    words = body.get("words")
    if isinstance(words, str):
        words = [w.strip() for w in words.replace("\n", ",").split(",")]
    words = [str(w).strip() for w in (words or []) if str(w).strip()]
    if not name or not words:
        return {"ok": False, "error": "名前と服の語を入れてください"}
    ch = str(body.get("character") or "").strip() or None
    with STATE.wlock:
        STATE.con.execute("INSERT INTO outfits(name, words, character, created_at) VALUES(?,?,?,?)",
                          (name, "\n".join(words), ch, time.time()))
        STATE.con.commit()
    return {"ok": True}


def api_outfit_delete(body: dict) -> dict:
    with STATE.wlock:
        STATE.con.execute("DELETE FROM outfits WHERE id=?", (int(body.get("id") or 0),))
        STATE.con.commit()
    return {"ok": True}


# --------------------------------------------------------------------------
# 保存名 — %date:yyyy%-%date:M%-%date:d%/…illu{キャラ}
# --------------------------------------------------------------------------

DEFAULT_SAVE_TEMPLATE = "%date:yyyy%-%date:M%-%date:d%/%date:yyyy%-%date:M%-%date:d%illu{キャラ}"
_DATE_TOKEN = re.compile(r"%date:([^%]+)%")
_DATE_FMT = re.compile(r"yyyy|yy|MM|M|dd|d|hh|h|mm|m|ss|s")


def save_template() -> str:
    t = str((STATE.cfg or {}).get("save_template") or "").strip()
    return t if t else DEFAULT_SAVE_TEMPLATE


def expand_dates(s: str, when=None) -> str:
    """ComfyUI の画面と同じ書き方の %date:…% を、今日の日付にする。
    （API から送ると ComfyUI はこれを日付にしないので、想起が入れてから送る）"""
    t = time.localtime(when if when is not None else time.time())
    vals = {"yyyy": f"{t.tm_year}", "yy": f"{t.tm_year % 100:02d}", "MM": f"{t.tm_mon:02d}",
            "M": f"{t.tm_mon}", "dd": f"{t.tm_mday:02d}", "d": f"{t.tm_mday}",
            "hh": f"{t.tm_hour:02d}", "h": f"{t.tm_hour}", "mm": f"{t.tm_min:02d}",
            "m": f"{t.tm_min}", "ss": f"{t.tm_sec:02d}", "s": f"{t.tm_sec}"}
    return _DATE_TOKEN.sub(lambda m: _DATE_FMT.sub(lambda k: vals[k.group(0)], m.group(1)), s)


def safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", str(name or "").strip())


def image_characters(image_id: int) -> list:
    """その絵に写っているキャラ（大きく写っている順、確定と自動）。"""
    if not chara_ok() or not chara_ready_tables():
        return []
    rows = q("SELECT c.name FROM c_instances i JOIN c_assign a ON a.instance_id=i.id "
             "AND a.state IN ('confirmed','auto') JOIN c_characters c ON c.id=a.character_id "
             "WHERE i.image_id=? ORDER BY (i.x2-i.x1)*(i.y2-i.y1) DESC", (image_id,))
    out = []
    for r in rows:
        if r["name"] not in out:
            out.append(r["name"])
    return out


def has_save_prefix(prompt: dict) -> bool:
    return any(isinstance(n, dict) and isinstance((n.get("inputs") or {}).get("filename_prefix"), str)
               for n in prompt.values())


def apply_save_template(prompt: dict, name: str, old_names=()) -> list:
    """保存名をひな形で書き直す。戻り値は変えた所（API 用の値と、画面用の値）。

    元の名前が「…illu名前_up」のような形なら、後ろの付け足し（_up）は残す
    （拡大前と拡大後で 2 つ保存しているワークフローで、名前がぶつからないように）。"""
    tpl = save_template()
    head, tail = (tpl.split("{キャラ}", 1) + [""])[:2] if "{キャラ}" in tpl else (tpl, "")
    marker = re.split(r"%[^%]*%", head)[-1]
    if "/" in marker:
        marker = marker.rsplit("/", 1)[-1]
    nm = safe_name(name)
    olds = sorted({safe_name(o) for o in old_names if o}, key=len, reverse=True)
    changes = []
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        v = inputs.get("filename_prefix")
        if not isinstance(v, str):
            continue
        suffix = ""
        if marker and marker in v:
            rest = v[v.rfind(marker) + len(marker):]
            for o in olds:
                if rest.startswith(o):
                    suffix = rest[len(o):]
                    break
            else:
                suffix = re.match(r"[^_\-]*(.*)$", rest).group(1)
        ui_after = head + nm + suffix + tail
        after = expand_dates(ui_after)
        inputs["filename_prefix"] = after
        changes.append({"node": str(nid), "class_type": node.get("class_type"),
                        "key": "filename_prefix", "before": v, "after": after,
                        "ui_after": ui_after, "kind": "save", "marker": marker})
    # ほかの保存先の欄（フォルダ名など）は、日付だけ今日にする
    rr = retarget_save_date({k: {"class_type": n.get("class_type"),
                                 "inputs": {kk: vv for kk, vv in (n.get("inputs") or {}).items()
                                            if kk != "filename_prefix"}}
                             for k, n in prompt.items() if isinstance(n, dict)})
    for c in rr["changed"]:
        prompt[c["node"]]["inputs"][c["key"]] = c["after"]
        changes.append(dict(c, ui_after=None, kind="date"))
    return changes


# --------------------------------------------------------------------------
# 画面用ワークフロー — 差し替え・保存名・seed を反映する
# --------------------------------------------------------------------------

def ui_build(ui_json: str, text_edits=None, node_changes=None) -> str:
    """画面用ワークフローに、API 側で変えたことを写す。

    文字列（プロンプト）は同じ文字列を探して置き換える。保存名と seed は、
    同じ番号のノードの中で置き換える（画面側の保存名は %date:…% のままなので、
    API 側の値とは一致しないため）。"""
    if not ui_json:
        return ui_json
    try:
        g = json.loads(ui_json)
    except Exception:
        return ui_json
    swaps = {e["before"]: e["after"] for e in (text_edits or []) if e.get("before") != e.get("after")}
    nodes = {str(nd.get("id")): nd for nd in g.get("nodes", []) if isinstance(nd, dict)}

    def items(wv):
        return list(enumerate(wv)) if isinstance(wv, list) else list(wv.items()) if isinstance(wv, dict) else []

    if swaps:
        for nd in nodes.values():
            wv = nd.get("widgets_values")
            for k, v in items(wv):
                if isinstance(v, str) and v in swaps:
                    wv[k] = swaps[v]
    for c in node_changes or []:
        nd = nodes.get(str(c.get("node")))
        if not nd:
            continue
        wv = nd.get("widgets_values")
        kind = c.get("kind")
        new = c.get("ui_after") if kind == "save" else c.get("after")
        if new is None:
            new = c.get("after")
        hit = None
        for k, v in items(wv):
            if v == c.get("before") and type(v) is type(c.get("before")):
                hit = k
                break
        if hit is None and kind == "save":
            for k, v in items(wv):
                if isinstance(v, str) and ("%date:" in v or (c.get("marker") and c["marker"] in v)):
                    hit = k
                    break
        if hit is not None:
            wv[hit] = new
    return json.dumps(g, ensure_ascii=False)


# --------------------------------------------------------------------------
# 投入の準備 — 普通の投入・差し替え・服だけ・ComfyUI で開く、すべてここを通る
# --------------------------------------------------------------------------

def prepare(image_id: int, body: dict, preview: bool = False) -> dict:
    rows = q("SELECT prompt_json, workflow_json, filename FROM images WHERE id=?", (image_id,))
    if not rows:
        return {"ok": False, "error": "画像が見つかりません"}
    row = rows[0]
    if not row["prompt_json"]:
        return {"ok": False, "error": "この画像には API 形式ワークフローがありません"}
    to = str(body.get("to") or "").strip() or None
    keep = bool(body.get("keep_char"))
    outfit = body.get("outfit")
    if isinstance(outfit, str):
        outfit = [w.strip() for w in outfit.replace("\n", ",").split(",") if w.strip()]
    if outfit is not None and not isinstance(outfit, list):
        outfit = None
    strength = "light" if body.get("strength") == "light" else "full"
    chars = image_characters(image_id)
    src = chars[0] if chars else None
    out = {"ok": True, "edits": [], "from": src, "to": to or src, "filename": row["filename"]}

    if to or keep or outfit is not None:
        r = build_swap(image_id, to or src or "", body.get("remove"), body.get("add"),
                       outfit=outfit, strength=strength, keep_char=keep or not to)
        if not r.get("ok"):
            return r
        prompt = r["prompt"]
        out.update({"edits": r["edits"], "from": r.get("from"), "to": r.get("to"),
                    "to_tags": r.get("to_tags"), "from_tags": r.get("from_tags")})
        swapped = bool(to) and not keep
    else:
        prompt = json.loads(row["prompt_json"])
        swapped = False

    # 保存名
    tpl_on = body.get("template", body.get("today", True))
    save_changes = []
    if tpl_on and has_save_prefix(prompt):
        name = (to if swapped else None) or str(body.get("save_name") or "").strip() or None
        if not name and len(chars) == 1:
            name = chars[0]
        if not name:
            out["need_name"] = True
            out["candidates"] = chars
            if not preview:
                out["ok"] = False
                out["error"] = "保存名に入れるキャラの名前が決まっていません"
                return out
            name = "（名前）"
        save_changes = apply_save_template(prompt, name, chars + ([to] if to else []))
        out["save_name"] = name
    out["prompt"] = prompt
    out["save_changes"] = save_changes
    out["ui"] = row["workflow_json"]
    stem = Path(row["filename"]).stem
    out["stem"] = f"{stem}_{out['to']}" if swapped and out.get("to") else stem
    return out


def _ui_for(P: dict, seed_changes=()) -> str | None:
    if not P.get("ui"):
        return None
    nc = list(P.get("save_changes") or [])
    nc += [{"node": c["node_id"], "before": c["current"], "after": c["new"], "kind": "seed"}
           for c in seed_changes]
    return ui_build(P["ui"], P.get("edits"), nc)


def open_prepared(P: dict) -> dict:
    """準備したワークフローを ComfyUI の画面に開く。受け口が無ければ一覧に保存する。"""
    ui = _ui_for(P)
    data = {"workflow": json.loads(ui), "name": P["stem"]} if ui else \
        {"api": P["prompt"], "name": P["stem"]}
    r = STATE.comfy.open_in_ui(data)
    if r.get("ok"):
        return {"ok": True, "how": "bridge", "clients": r["clients"], "name": P["stem"]}
    if r.get("no_bridge"):
        out = {"ok": False, "no_bridge": True, "error": "ComfyUI に想起の受け口が入っていません"}
        if ui:
            out["saved"] = save_workflow_to_comfy(P["stem"], ui)
        return out
    return {"ok": False, "error": r.get("error") or "ComfyUI の画面に開けませんでした"}


def run_submit(body: dict) -> dict:
    """投入する（普通の投入も差し替えも）。画面用のワークフローも一緒に送る。"""
    import copy
    iid = int(body.get("image_id") or body.get("id"))
    P = prepare(iid, body)
    if not P.get("ok"):
        return P
    mode = body.get("seed_mode", "random")
    fixed = body.get("seed_value")
    fixed = int(fixed) if str(fixed or "").strip().isdigit() else None
    batch = 1 if mode == "same" else max(1, min(int(body.get("batch", 1) or 1), 20))
    front = bool(body.get("front"))
    results, all_changes = [], []
    for n in range(batch):
        pr = copy.deepcopy(P["prompt"])
        pr, changed = apply_seed(pr, mode, fixed)
        all_changes += changed
        ui = _ui_for(P, changed)
        extra = {"workflow": json.loads(ui)} if ui else None
        res = STATE.comfy.submit(pr, extra_pnginfo=extra, front=front)
        results.append(res)
        if not res["ok"]:
            break
    out = {"ok": all(r["ok"] for r in results), "submitted": sum(1 for r in results if r["ok"]),
           "seed_changes": all_changes, "results": results,
           "save_changes": P["save_changes"], "save_name": P.get("save_name"),
           "edits": P["edits"], "from": P.get("from"), "to": P.get("to"), "source": P["filename"]}
    if any(r.get("timeout") for r in results):
        out["timeout"] = True
    if body.get("open_workflow"):       # 受け付けられなかったときも、画面で直せるように開く
        out["workflow"] = open_prepared(P)
    return out


def _preview_out(P: dict) -> dict:
    P = dict(P)
    for k in ("prompt", "ui"):
        P.pop(k, None)
    return P


def api_characters() -> dict:
    # キャラがまだいない台帳でも、画面を壊さないよう空で返す。
    # 名前と枚数だけ返す。特徴の算出は重いので、差し替えの下見のときだけ行う。
    if not chara_ok() or not chara_ready_tables():
        return {"characters": []}
    rows = q("SELECT c.name, COUNT(DISTINCT i.image_id) AS n "
             "FROM c_characters c JOIN c_assign a ON a.character_id=c.id "
             "AND a.state IN ('confirmed','auto') JOIN c_instances i ON i.id=a.instance_id "
             "GROUP BY c.id ORDER BY n DESC")
    return {"characters": rows}


def api_set_traits(body: dict) -> dict:
    err = _need_chara()
    if err:
        return err
    name = str(body.get("name") or "").strip()
    row = q("SELECT id FROM c_characters WHERE name=?", (name,))
    if not row:
        return {"ok": False, "error": "そのキャラは見つかりません"}
    tags = body.get("tags")
    r = api_chara_action({"kind": "words", "id": row[0]["id"], "words": tags})
    _TRAITS_CACHE.pop(name, None)
    return {"ok": r.get("ok", False), "name": name, "tags": identity_tags(name)["tags"],
            "error": r.get("error")}


def start_index_bg(dirs: list[str], full: bool) -> None:
    def run():
        STATE.index_status.update({"running": True, "done": 0, "total": 0})

        def prog(done, total):
            STATE.index_status["done"] = done
            STATE.index_status["total"] = total

        try:
            st = run_index(dirs, full=full,
                           workers=STATE.cfg.get("scan_workers", 6), progress=prog)
            STATE.index_status["last"] = st
            # 新しい絵が増えていれば、続けてキャラの解析も回す
            if chara_ok() and (st.get("scanned") or st.get("removed")):
                st["auto_analyze"] = CHARA.maybe_auto_analyze()
                if st.get("removed") and not st["auto_analyze"]:
                    CHARA.ENGINE.invalidate()
                    CHARA.request_engine()
        except Exception as e:
            STATE.index_status["last"] = {"error": str(e)}
        finally:
            STATE.index_status["running"] = False

    if STATE.index_lock.acquire(blocking=False):
        try:
            threading.Thread(target=run, daemon=True).start()
        finally:
            threading.Timer(1.0, STATE.index_lock.release).start()


PERF: dict = {}
PERF_LOCK = threading.Lock()


def note_perf(path: str, ms: float) -> None:
    """どの要求に何ミリ秒かかったかを控える。遅いと感じたとき、
    推測ではなく数字で原因を絞り込めるようにするため。"""
    key = path.split("?")[0]
    for pre in ("/thumb/", "/head/", "/full/", "/api/detail/", "/api/chara/character/",
                "/api/chara/list", "/api/chara/groups",
                "/api/chara/proposal/", "/api/chara/instance/", "/api/chara/sheet/",
                "/api/chara/similar/", "/swapwf/", "/workflow/", "/api/reveal/"):
        if key.startswith(pre):
            key = pre + "*"
            break
    with PERF_LOCK:
        e = PERF.setdefault(key, {"n": 0, "total": 0.0, "max": 0.0, "last": 0.0})
        e["n"] += 1; e["total"] += ms; e["last"] = ms
        e["max"] = max(e["max"], ms)


class Server(ThreadingHTTPServer):
    """Windows では SO_REUSEADDR を付けると、使用中のポートにも重ねて待ち受けできてしまう。
    それだと二重起動（前の版が動いたまま等）に気付けないので、Windows では付けずに
    SO_EXCLUSIVEADDRUSE で独占する。"""
    daemon_threads = True
    allow_reuse_address = sys.platform != "win32"

    def server_bind(self):
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            except OSError:
                pass
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    server_version = "Souki/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8",
              extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # 画像だけキャッシュする。HTML や JSON をキャッシュすると、
        # ファイルを差し替えてもブラウザが古い画面を出し続ける。
        cacheable = ctype.startswith("image/")
        self.send_header("Cache-Control",
                         "max-age=86400" if cacheable else "no-store, must-revalidate")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"))

    def do_GET(self):
        _t0 = time.time()
        try:
            return self._do_GET()
        finally:
            note_perf(unquote(urlparse(self.path).path), (time.time() - _t0) * 1000)

    def _do_GET(self):
        u = urlparse(self.path)
        path = unquote(u.path)
        qs = {k: (v if len(v) > 1 else v[0]) for k, v in parse_qs(u.query).items()}
        try:
            if path in ("/", "/index.html"):
                f = APP_DIR / "ui.html"
                if not f.exists():
                    return self._send(404, "ui.html が見つかりません".encode("utf-8"),
                                      "text/plain; charset=utf-8")
                return self._send(200, f.read_bytes(), "text/html; charset=utf-8")

            if path in ("/chara", "/chara.html"):
                f = APP_DIR / "chara.html"
                if not f.exists():
                    return self._send(404, "chara.html が見つかりません".encode("utf-8"),
                                      "text/plain; charset=utf-8")
                return self._send(200, f.read_bytes(), "text/html; charset=utf-8")

            if path in ("/favicon.ico", "/icon/souki.png", "/icon/chara.png"):
                name = {"/favicon.ico": "souki.ico", "/icon/souki.png": "souki_256.png",
                        "/icon/chara.png": "chara_256.png"}[path]
                f = APP_DIR / name
                if not f.exists():
                    return self._send(404, b"no icon", "text/plain")
                return self._send(200, f.read_bytes(),
                                  "image/x-icon" if name.endswith(".ico") else "image/png")

            if path in ("/kao", "/kao.html"):
                # 前の版のキャラ画面。新しい画面へ案内する。
                return self._send(302, b"", "text/plain", {"Location": "/chara"})

            if path == "/api/version":
                load_chara()
                return self._json({"version": VERSION,
                                   "chara": CHARA.VERSION if CHARA else None,
                                   "chara_error": CHARA_ERROR})

            if path == "/api/settings":
                return self._json(api_settings())

            if path.startswith("/head/"):
                if not chara_ok():
                    return self._send(404, b"no head", "text/plain")
                iid = int(path.rsplit("/", 1)[1].split(".")[0])
                fp = CHARA.head_path(iid)
                if not fp.exists():
                    return self._send(404, b"no head", "text/plain")
                return self._send(200, fp.read_bytes(), "image/jpeg")

            if path.startswith("/api/chara/"):
                if path == "/api/chara/status":
                    return self._json(api_chara_status())
                if path == "/api/chara/review_summary":
                    if not chara_ok():
                        return self._json({"total": 0, "groups": []})
                    return self._json(CHARA.review_summary(STATE.rcon))
                if path == "/api/chara/counts":
                    if not chara_ok():
                        return self._json({"available": False})
                    return self._json(CHARA.counts(STATE.rcon))
                if path in ("/api/chara/list", "/api/chara/groups"):
                    if not chara_ok():
                        return self._json({"total": 0, "faces": [], "groups": []})
                    try:
                        src = json.loads(str(qs.get("src") or "{}"))
                        if path == "/api/chara/list":
                            return self._json(CHARA.list_page(
                                STATE.rcon, src, int(qs.get("offset", 0) or 0),
                                int(qs.get("limit", 300) or 300)))
                        return self._json(CHARA.list_groups(STATE.rcon, src,
                                                            int(qs.get("per", 12) or 12)))
                    except ValueError as e:
                        return self._json({"error": str(e), "total": 0, "faces": [], "groups": []})
                if path == "/api/chara/jobs":
                    # 画面が数秒おきに尋ねる軽い問い合わせ（重い集計はしない）
                    if not chara_ok():
                        return self._json({"available": False})
                    return self._json({
                        "available": True,
                        "analysis": CHARA.JOB_ANALYSIS.snapshot(),
                        "engine": CHARA.JOB_ENGINE.snapshot(),
                        "last_action": CHARA.last_action(STATE.rcon),
                        "gen": CHARA.ENGINE.gen,
                        "index": STATE.index_status})
                err = _need_chara()
                if err:
                    return self._json(err)
                con = STATE.rcon
                lim = int(qs.get("limit", 120) or 120)
                off = int(qs.get("offset", 0) or 0)
                if path == "/api/chara/characters":
                    return self._json({"characters": CHARA.list_characters(con)})
                if path.startswith("/api/chara/character/"):
                    return self._json(CHARA.faces_of(
                        con, int(path.rsplit("/", 1)[1]), str(qs.get("state", "confirmed")),
                        order=str(qs.get("order", "score")), limit=lim, offset=off))
                if path == "/api/chara/proposals":
                    return self._json({"proposals": CHARA.list_proposals(con)})
                if path.startswith("/api/chara/proposal/"):
                    return self._json(CHARA.proposal_faces(
                        con, int(path.rsplit("/", 1)[1]), limit=lim, offset=off))
                if path == "/api/chara/review":
                    cid = qs.get("character")
                    return self._json(CHARA.review_queue(
                        con, int(cid) if cid else None, limit=lim, offset=off))
                if path == "/api/chara/unassigned":
                    return self._json(CHARA.unassigned(
                        con, limit=lim, offset=off, order=str(qs.get("order", "new")),
                        dismissed=qs.get("dismissed") in ("1", "true")))
                if path.startswith("/api/chara/similar/"):
                    if not CHARA.ENGINE.ready():
                        return self._json({"total": 0, "faces": [],
                                           "error": "モデルがまだありません"})
                    return self._json(CHARA.similar_faces(
                        con, int(path.rsplit("/", 1)[1]), scope=str(qs.get("scope", "unassigned")),
                        limit=lim))
                if path.startswith("/api/chara/instance/"):
                    return self._json(CHARA.instance_detail(con, int(path.rsplit("/", 1)[1])))
                if path.startswith("/api/chara/sheet/"):
                    cid = int(path.rsplit("/", 1)[1])
                    sh = CHARA.character_sheet(con, cid)
                    nm = q("SELECT name FROM c_characters WHERE id=?", (cid,))
                    if nm:
                        sh["identity"] = identity_tags(nm[0]["name"])
                        sh["profile"] = _profile_words(cid)
                    return self._json(sh)
                return self._send(404, b"not found", "text/plain")

            if path == "/api/characters":
                return self._json(api_characters())

            if path == "/api/timeline":
                for k in ("checkpoint", "lora", "tag", "nottag", "folder", "chara"):
                    if k in qs and isinstance(qs[k], str):
                        qs[k] = [x for x in qs[k].split("\u0001") if x]
                return self._json(api_timeline(qs))

            if path == "/api/facets":
                return self._json(api_facets())

            if path == "/api/search":
                for k in ("checkpoint", "lora", "tag", "nottag", "folder", "chara"):
                    if k in qs and isinstance(qs[k], str):
                        qs[k] = [x for x in qs[k].split("\u0001") if x]
                for k in ("dedup", "api_only"):
                    qs[k] = qs.get(k) in ("1", "true", True)
                return self._json(api_search(qs))

            if path.startswith("/api/detail/"):
                return self._json(api_detail(int(path.rsplit("/", 1)[1])))

            if path == "/api/outfits":
                return self._json(api_outfits(qs))
            if path == "/api/outfit_of":
                return self._json(api_outfit_of(qs))
            if path == "/api/save_template":
                return self._json({"template": save_template(), "default": DEFAULT_SAVE_TEMPLATE})
            if path == "/api/comfy/bridge":
                return self._json(api_bridge_status())
            if path == "/api/status":
                c = STATE.comfy
                out = {
                    "version": VERSION,
                    "comfy": c.ping(), "index": STATE.index_status,
                    "objinfo": {"ready": c._obj is not None,
                                "fetching": c._fetching,
                                "ms": round(c._obj_ms),
                                "error": c._obj_err}}
                if chara_ok():
                    out["chara"] = {"analysis": CHARA.JOB_ANALYSIS.snapshot(),
                                    "engine": CHARA.JOB_ENGINE.snapshot()}
                    try:
                        out["chara"]["review"] = STATE.rcon.execute(
                            "SELECT COUNT(*) FROM c_assign WHERE state='review'").fetchone()[0]
                    except Exception:
                        out["chara"]["review"] = 0
                return self._json(out)

            if path == "/api/perf":
                if qs.get("reset"):
                    with PERF_LOCK:
                        PERF.clear()
                    return self._json({"ok": True})
                with PERF_LOCK:
                    rows = [{"path": k, "n": v["n"],
                             "avg": round(v["total"] / max(v["n"], 1)),
                             "max": round(v["max"]), "last": round(v["last"])}
                            for k, v in PERF.items()]
                rows.sort(key=lambda r: -r["avg"])
                c = STATE.comfy
                return self._json({
                    "requests": rows,
                    "objinfo_ms": round(c._obj_ms),
                    "objinfo_ready": c._obj is not None,
                    "objinfo_error": c._obj_err})

            if path.startswith("/thumb/"):
                iid = int(path.rsplit("/", 1)[1])
                tp = thumb_path(iid)
                if not tp.exists():
                    return self._send(404, b"no thumb", "text/plain")
                return self._send(200, tp.read_bytes(), "image/jpeg")

            if path.startswith("/swapwf/"):
                iid = int(path.rsplit("/", 1)[1].split(".")[0])
                P = prepare(iid, {"to": str(qs.get("to") or ""), "save_name": str(qs.get("name") or "")},
                            preview=True)
                if not P.get("ok"):
                    return self._send(400, (P.get("error") or "").encode("utf-8"),
                                      "text/plain; charset=utf-8")
                ui = _ui_for(P)
                if not ui:
                    return self._send(
                        404, "この画像には画面用のワークフローがありません".encode("utf-8"),
                        "text/plain; charset=utf-8")
                fn = urllib.parse.quote("想起_" + safe_name(P["stem"])[:70] + ".json")
                return self._send(
                    200, ui.encode("utf-8"), "application/json; charset=utf-8",
                    {"Content-Disposition":
                     f"attachment; filename=workflow.json; filename*=UTF-8''{fn}"})
            if path.startswith("/workflow/"):
                iid = int(path.rsplit("/", 1)[1].split(".")[0])
                rows = q("SELECT workflow_json, prompt_json, filename "
                         "FROM images WHERE id=?", (iid,))
                if not rows:
                    return self._send(404, b"not found", "text/plain")
                wf = rows[0]["workflow_json"]
                if not wf:
                    return self._send(
                        404, "この画像には画面用のワークフローが入っていません".encode("utf-8"),
                        "text/plain; charset=utf-8")
                stem = re.sub(r'[<>:"/\\|?*]', "_", Path(rows[0]["filename"]).stem)[:60]
                fn = urllib.parse.quote(f"想起_{stem}.json")
                return self._send(
                    200, wf.encode("utf-8"), "application/json; charset=utf-8",
                    {"Content-Disposition":
                     f"attachment; filename=workflow.json; filename*=UTF-8''{fn}"})

            if path.startswith("/full/"):
                iid = int(path.rsplit("/", 1)[1])
                rows = q("SELECT path FROM images WHERE id=?", (iid,))
                if not rows:
                    return self._send(404, b"not found", "text/plain")
                fp = Path(rows[0]["path"])
                if not fp.exists():
                    return self._send(404, b"file missing", "text/plain")
                ctype = mimetypes.guess_type(fp.name)[0] or "application/octet-stream"
                return self._send(200, fp.read_bytes(), ctype)

            if path.startswith("/api/reveal/"):
                iid = int(path.rsplit("/", 1)[1])
                rows = q("SELECT path FROM images WHERE id=?", (iid,))
                if rows and sys.platform == "win32":
                    os.system(f'explorer /select,"{rows[0]["path"]}"')
                    return self._json({"ok": True})
                return self._json({"ok": False, "error": "Windows でのみ動作します"})

            return self._send(404, b"not found", "text/plain")
        except Exception as e:
            return self._json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def do_POST(self):
        _t0 = time.time()
        try:
            return self._do_POST()
        finally:
            note_perf(unquote(urlparse(self.path).path), (time.time() - _t0) * 1000)

    def _do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            body = {}
        try:
            if u.path == "/api/outfits":
                return self._json(api_outfit_save(body))
            if u.path == "/api/outfits/delete":
                return self._json(api_outfit_delete(body))
            if u.path == "/api/comfy/open":
                return self._json(api_comfy_open(body))
            if u.path == "/api/comfy/bridge/install":
                return self._json(api_bridge_install(body))
            if u.path == "/api/requeue":
                return self._json(api_requeue(body))
            if u.path == "/api/validate":
                return self._json(api_validate([int(x) for x in body.get("ids", [])]))
            if u.path == "/api/swap/preview":
                return self._json(api_swap_preview(body))
            if u.path == "/api/swap/submit":
                return self._json(api_swap_submit(body))
            if u.path == "/api/traits":
                return self._json(api_set_traits(body))
            if u.path == "/api/similar":
                return self._json(api_similar(body))
            if u.path == "/api/chara/action":
                return self._json(api_chara_action(body))
            if u.path == "/api/chara/undo":
                return self._json(api_chara_undo())
            if u.path == "/api/chara/job":
                return self._json(api_chara_job(body))
            if u.path == "/api/chara/settings":
                return self._json(api_chara_settings(body))
            if u.path == "/api/chara/export":
                return self._json(api_chara_export(body))
            if u.path == "/api/chara/olddata/delete":
                return self._json(api_old_data_delete())
            if u.path == "/api/settings":
                return self._json(api_settings(body))
            if u.path == "/api/reindex":
                dirs = body.get("dirs") or STATE.cfg.get("output_dirs") or []
                start_index_bg(dirs, bool(body.get("full")))
                return self._json({"ok": True})
            return self._send(404, b"not found", "text/plain")
        except Exception as e:
            return self._json({"error": str(e), "trace": traceback.format_exc()}, 500)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_index(args) -> None:
    cfg = load_config()
    dirs = args.dirs or cfg.get("output_dirs") or []
    if not dirs:
        sys.stderr.write("出力フォルダを指定してください:\n"
                         "  python souki.py index F:/ComfyUI/output\n")
        sys.exit(1)
    cfg["output_dirs"] = [str(Path(d).expanduser()) for d in dirs]
    save_config(cfg)

    print(f"[{APP_NAME}] 版 {VERSION}")
    print(f"  走査中: {', '.join(cfg['output_dirs'])}")
    last = [0]

    def prog(done, total):
        if done - last[0] >= 200 or done == total:
            last[0] = done
            pct = (done / total * 100) if total else 0
            print(f"  {done}/{total}  ({pct:.0f}%)", end="\r", flush=True)

    st = run_index(cfg["output_dirs"], full=args.full,
                   workers=cfg.get("scan_workers", 6), progress=prog)
    print(" " * 60, end="\r")
    print(f"  対象ファイル   {st['total_found']:,}")
    print(f"  新規/更新      {st['scanned']:,}")
    print(f"    ワークフロー有 {st['with_api']:,}")
    print(f"    メタデータ無し {st['no_meta']:,}")
    if st["moved"]:
        print(f"  移動を検出     {st['moved']:,}  (読み直さずパスだけ更新)")
    if st["removed"]:
        print(f"  台帳から削除   {st['removed']:,}")
    for r in st.get("skipped_roots", []):
        print(f"  !! 未到達      {r}  (この分の記録は残しています)")
    if st["errors"]:
        print(f"  エラー         {st['errors']:,}")
    print(f"  所要時間       {st['elapsed']:.1f} 秒")
    print(f"  台帳: {DB_PATH}")


# 前の版の部品。新しい版では使わないので、起動時に _旧版 へ移す（消しはしない）。
OLD_FILES = ("kao.py", "kao.html", "キャラ解析.bat", "クラスタ健康診断.bat", "診断.bat",
             "書き出し.bat", "台帳を更新.bat", "python_path.txt", "kao.ico", "kao_256.png")


def tidy_old_files() -> list[str]:
    moved = []
    dest = APP_DIR / "_旧版"
    for name in OLD_FILES:
        src = APP_DIR / name
        if not src.exists():
            continue
        if name == "kao.py":
            try:
                head = src.read_text(encoding="utf-8", errors="replace")[:6000]
            except Exception:
                continue
            if "run_cluster" not in head and "HDBSCAN" not in head and "kao" not in head:
                continue
        try:
            dest.mkdir(exist_ok=True)
            target = dest / name
            if target.exists():
                target = dest / f"{src.stem}_{int(time.time())}{src.suffix}"
            src.replace(target)
            moved.append(name)
        except Exception:
            pass
    return moved


def cmd_serve(args) -> None:
    global STATE
    cfg = load_config()
    if args.comfy:
        cfg["comfy_url"] = args.comfy
    port = args.port or cfg.get("server_port", 8765)
    url = f"http://127.0.0.1:{port}/"

    print(f"[{APP_NAME}] 版 {VERSION}")
    print(f"  Python   {sys.version.split()[0]}  ({sys.executable})")

    try:
        srv = Server(("127.0.0.1", port), Handler)
    except OSError:
        # 二重に起動したとき。同じ版が動いているなら、そちらを開けば済む。
        # 前の版が動いたままだと、古い画面が開いてしまうので知らせる。
        other = None
        try:
            with urllib.request.urlopen(url + "api/version", timeout=3) as r:
                other = json.loads(r.read().decode("utf-8")).get("version")
        except Exception:
            other = None
        if other == VERSION:
            print(f"  すでに起動しています。ブラウザで開きます: {url}")
            if not args.no_browser:
                webbrowser.open(url)
        else:
            print(f"  !! ポート {port} を別のプログラムが使っています"
                  + ("（前の版の想起が動いたままのようです）" if other is None else f"（版 {other}）") + "。")
            print("     その窓を閉じてから、もう一度 起動.bat を開いてください。")
        return

    # 前の版の部品を片付ける（前の版が動いたままのときは触らない＝ここまで来ない）
    moved = tidy_old_files()
    if moved:
        print(f"  前の版の部品を _旧版 フォルダへ移しました: {', '.join(moved)}")

    STATE = State(cfg)
    if load_chara() is not None:
        old_data_info()          # 前の版のデータの大きさを裏で数え始める
        try:
            # 顔の特徴を先に読み込んでおく（最初の操作で何秒も待たせないため）
            if chara_ready_tables() and CHARA.ENGINE.ready():
                CHARA.request_engine(delay=1.5)
        except Exception:
            pass
    n = q("SELECT COUNT(*) AS n FROM images")[0]["n"]
    ping = STATE.comfy.ping_now()
    print(f"  台帳     {n:,} 件  ({DB_PATH})")
    if load_chara() is not None:
        import vision
        mr = vision.models_ready()
        env = vision.load_env()
        print(f"  キャラ   {'準備済み' if mr['ready'] else 'モデル未取得（画面の「準備する」から）'}"
              + (f" / 解析: {env.get('device')}" if mr["ready"] and env.get("device") else ""))
    else:
        print(f"  キャラ   使えません: {CHARA_ERROR}")
        print("           起動.bat から起動すると使えるようになります。")
    print(f"  ComfyUI  {cfg['comfy_url']}  ... " +
          ("接続OK" if ping.get("online") else "未接続（再投入は使えません）"))
    print(f"  画面     {url}")
    print("  終了: この窓を閉じるか Ctrl+C")

    # 起動したら出力フォルダを読み直す（変わったものだけ読むので速い）
    dirs = STATE.cfg.get("output_dirs") or []
    if dirs and cfg.get("index_on_start", True) and not args.no_index:
        start_index_bg(dirs, False)

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n終了しました。")


def main() -> None:
    ap = argparse.ArgumentParser(prog="souki", description=f"{APP_NAME} — ComfyUI 出力台帳")
    ap.add_argument("--version", action="version", version=f"souki.py 版 {VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="出力フォルダを走査して台帳を作る")
    pi.add_argument("dirs", nargs="*", help="ComfyUI の output フォルダ（複数可）")
    pi.add_argument("--full", action="store_true", help="更新判定を無視して全件やり直す")
    pi.set_defaults(func=cmd_index)

    ps = sub.add_parser("serve", help="ブラウザUIを起動する")
    ps.add_argument("--port", type=int)
    ps.add_argument("--comfy", help="ComfyUI の URL (既定 http://127.0.0.1:8188)")
    ps.add_argument("--no-browser", action="store_true")
    ps.add_argument("--no-index", action="store_true", help="起動時に出力フォルダを読み直さない")
    ps.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
