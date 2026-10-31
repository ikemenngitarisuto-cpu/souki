#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
想起 — キャラの判定 (chara.py)
==============================

考え方
  ・キャラは「要平さんが確定した顔」だけを見本にする。機械が入れた顔
    （自動・確認待ち）は見本にならない。以前の混入は、機械が入れた顔が
    見本を動かし、動いた見本がさらに別の顔を吸い込む悪循環だった。
    見本に機械の判断が混ざらないので、構造上その悪循環は起きない。
  ・見本は確定した顔 1 枚 1 枚。平均は取らない。カラー絵とマンガ絵を
    同じキャラにしても、見本が「どちらでもない中間」に落ちることがない。
  ・固定の数字は置かない。線はモデルに付いてくる公開値から始め、
    要平さんが外した顔（否）や他キャラとの近さから、厳しくする方向にだけ動く。
  ・CCIP は髪色の違いに鈍い。プロンプトの特徴語（髪色・目の色・ツノの色・
    年齢・肌）が食い違うキャラへは自動で入れない。キャラ LoRA も手がかりにする。
  ・機械は「確定」と「否」に触らない。

状態
  confirmed 確定 … 人が入れた / OK した
  auto      自動 … 機械が十分な自信で入れた
  review    確認待ち … 機械が迷った
  （c_reject） 否 … 「このキャラではない」。そのキャラへは二度と自動で入らない
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
import traceback
import zlib
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def _fresh_vision_bytecode() -> None:
    """vision.py の控え（__pycache__）を中身で確かめる形にしておく（souki.py の fresh_bytecode と同じ。
    chara.py を直接動かす setup のときのため）。"""
    import importlib.util
    import py_compile
    src = Path(__file__).resolve().parent / "vision.py"
    try:
        pyc = Path(importlib.util.cache_from_source(str(src)))
        with open(pyc, "rb") as f:
            head = f.read(16)
    except (OSError, NotImplementedError, ValueError):
        head = b""
    if len(head) == 16 and head[:4] == importlib.util.MAGIC_NUMBER and \
            int.from_bytes(head[4:8], "little") == 0b11:
        return
    try:
        py_compile.compile(str(src), cfile=str(pyc), doraise=True,
                           invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)
    except Exception:
        try:
            pyc.unlink()
        except Exception:
            pass


_fresh_vision_bytecode()
import vision as V  # noqa: E402

VERSION = "2.5.0"
if V.VERSION != VERSION:
    raise ImportError(
        f"vision.py の版（{V.VERSION}）が chara.py（{VERSION}）と違います。"
        "ファイルの差し替え漏れがないか確認してください。")

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "souki.db"
HEAD_DIR = DATA_DIR / "heads"
CONFIG_PATH = APP_DIR / "config.json"
EXPORT_DIR = APP_DIR / "export"

STATE_LABEL = {"confirmed": "確定", "auto": "自動", "review": "確認待ち"}


# --------------------------------------------------------------------------
# 設定（config.json の "chara" 欄）
# --------------------------------------------------------------------------

DEFAULTS = {
    "device": None,            # None = 自己試験で選んだ装置
    "feature_variant": None,   # None = 試運転の結果を待つ（決まるまでは full）
    "auto_analyze": True,      # 台帳の更新のあと、新しい絵を自動で解析する
    "min_head_px": 32,         # これより小さい頭は判定に使えないので拾わない
    "det_conf": 0.4,           # 頭部検出の確からしさの下限（imgutils の既定）
    "decode_workers": None,    # 画像を読むスレッド数（None = 自動）
    "exemplar_cap": 400,       # 1キャラあたり判定に使う見本の上限
}
_CFG_LOCK = threading.Lock()


def _read_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def settings() -> dict:
    out = dict(DEFAULTS)
    try:
        out.update(_read_config().get("chara") or {})
    except Exception:
        pass
    return out


def save_settings(patch: dict) -> dict:
    """config.json の "chara" 欄だけを書き換える。ほかの欄（出力フォルダなど）は触らない。"""
    with _CFG_LOCK:
        try:
            cfg = _read_config()
        except Exception as e:
            raise RuntimeError(f"config.json が読めないので保存を止めました: {e}")
        cur = dict(cfg.get("chara") or {})
        for k, v in patch.items():
            if k in DEFAULTS:
                cur[k] = v
        cfg["chara"] = cur
        tmp = CONFIG_PATH.with_name(CONFIG_PATH.name + ".tmp")
        tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(CONFIG_PATH)
    return settings()


def feature_variant() -> str:
    v = settings().get("feature_variant")
    return v if v in ("full", "crop") else "full"


# --------------------------------------------------------------------------
# データベース
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS c_analysis (
    image_id    INTEGER PRIMARY KEY,
    mtime       REAL,
    size        INTEGER,
    n_heads     INTEGER,
    status      TEXT,
    error       TEXT,
    words       TEXT,
    detector    TEXT,
    analyzed_at REAL
);
CREATE TABLE IF NOT EXISTS c_instances (
    id         INTEGER PRIMARY KEY,
    image_id   INTEGER NOT NULL,
    idx        INTEGER NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    score      REAL,
    solo       INTEGER DEFAULT 0,
    created_at REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_ci_img ON c_instances(image_id, idx);
CREATE TABLE IF NOT EXISTS c_features (
    instance_id INTEGER NOT NULL,
    model       TEXT NOT NULL,
    variant     TEXT NOT NULL,
    vec         BLOB NOT NULL,
    PRIMARY KEY (instance_id, model, variant)
);
CREATE TABLE IF NOT EXISTS c_characters (
    id         INTEGER PRIMARY KEY,
    name       TEXT UNIQUE NOT NULL,
    note       TEXT,
    words      TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS c_assign (
    instance_id  INTEGER PRIMARY KEY,
    character_id INTEGER NOT NULL,
    state        TEXT NOT NULL,
    score        REAL,
    margin       REAL,
    reason       TEXT,
    source       TEXT,
    updated_at   REAL
);
CREATE INDEX IF NOT EXISTS ix_ca_char ON c_assign(character_id, state);
CREATE INDEX IF NOT EXISTS ix_ca_state ON c_assign(state);
CREATE TABLE IF NOT EXISTS c_reject (
    instance_id  INTEGER NOT NULL,
    character_id INTEGER NOT NULL,
    created_at   REAL,
    PRIMARY KEY (instance_id, character_id)
);
CREATE INDEX IF NOT EXISTS ix_cr_char ON c_reject(character_id);
CREATE TABLE IF NOT EXISTS c_dismissed (
    instance_id INTEGER PRIMARY KEY,
    created_at  REAL
);
CREATE TABLE IF NOT EXISTS c_proposals (
    id         INTEGER PRIMARY KEY,
    n          INTEGER,
    hints      TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS c_proposal_members (
    proposal_id INTEGER NOT NULL,
    instance_id INTEGER NOT NULL,
    rank        INTEGER,
    grp         INTEGER,
    grank       INTEGER,
    PRIMARY KEY (proposal_id, instance_id)
);
CREATE INDEX IF NOT EXISTS ix_cpm_inst ON c_proposal_members(instance_id);
CREATE TABLE IF NOT EXISTS c_actions (
    id    INTEGER PRIMARY KEY,
    at    REAL,
    kind  TEXT,
    label TEXT,
    undo  TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    val TEXT
);
"""


def connect(readonly: bool = False) -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=60, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    if readonly:
        con.execute("PRAGMA query_only=ON")
    else:
        con.executescript(SCHEMA)
        _migrate(con)
    return con


def _migrate(con) -> None:
    """前の版（2.0.0）の表に足りない列を足す。何度呼んでもよい。"""
    cols = {r[1] for r in con.execute("PRAGMA table_info(c_proposal_members)")}
    for c in ("grp", "grank"):
        if c not in cols:
            con.execute(f"ALTER TABLE c_proposal_members ADD COLUMN {c} INTEGER")
    con.execute("CREATE INDEX IF NOT EXISTS ix_cpm_rank ON c_proposal_members(proposal_id, rank)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_cpm_grank ON c_proposal_members(proposal_id, grank)")
    con.commit()


def meta_get(con, key: str, default=None):
    try:
        r = con.execute("SELECT val FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default
    except Exception:
        return default


def meta_set(con, key: str, val) -> None:
    con.execute("INSERT INTO meta(key,val) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET val=excluded.val",
                (key, json.dumps(val, ensure_ascii=False)))


def head_path(instance_id: int) -> Path:
    return HEAD_DIR / f"{instance_id % 256:02x}" / f"{instance_id}.jpg"


def _qm(n: int) -> str:
    return ",".join("?" * n)


def _chunks(seq, n: int = 800):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --------------------------------------------------------------------------
# プロンプトの特徴語（キャラを書き分けている語）
# --------------------------------------------------------------------------

# 色は「系統」でまとめる。同じキャラに silver hair と white hair を
# 両方使うことはよくあるので、細かく分けると食い違いを誤検出する。
# 食い違いとして扱うのは、系統そのものが違うときだけ。
_FAMILY = {
    "black": "black", "ebony": "black", "raven": "black", "jet": "black",
    "white": "pale", "silver": "pale", "grey": "pale", "gray": "pale",
    "platinum": "pale", "ash": "pale", "ashen": "pale", "snow": "pale",
    "blonde": "blonde", "blond": "blonde", "yellow": "blonde", "gold": "blonde",
    "golden": "blonde", "honey": "blonde",
    "orange": "orange", "ginger": "orange", "copper": "orange", "amber": "orange",
    "red": "red", "crimson": "red", "scarlet": "red", "ruby": "red", "wine": "red",
    "maroon": "red", "burgundy": "red", "vermilion": "red",
    "pink": "pink", "rose": "pink", "salmon": "pink", "peach": "pink", "magenta": "pink",
    "purple": "purple", "violet": "purple", "lavender": "purple", "lilac": "purple",
    "indigo": "purple", "plum": "purple",
    "blue": "blue", "navy": "blue", "azure": "blue", "cobalt": "blue",
    "sapphire": "blue", "sky": "blue",
    "aqua": "aqua", "cyan": "aqua", "teal": "aqua", "turquoise": "aqua",
    "green": "green", "emerald": "green", "lime": "green", "mint": "green",
    "jade": "green", "olive": "green",
    "brown": "brown", "chestnut": "brown", "auburn": "brown", "hazel": "brown",
    "chocolate": "brown",
}
_KANJI_COLOR = {
    "銀": "pale", "白": "pale", "灰": "pale", "金": "blonde", "黄": "blonde",
    "黒": "black", "赤": "red", "紅": "red", "緋": "red", "桃": "pink",
    "ピンク": "pink", "紫": "purple", "青": "blue", "蒼": "blue", "碧": "blue",
    "水色": "aqua", "緑": "green", "茶": "brown", "橙": "orange",
}
COLOR_LABEL = {"black": "黒", "pale": "白銀", "blonde": "金", "orange": "橙",
               "red": "赤", "pink": "桃", "purple": "紫", "blue": "青",
               "aqua": "水色", "green": "緑", "brown": "茶"}
SLOT_LABEL = {"hair": "髪", "eyes": "目", "horns": "ツノ", "age": "年齢", "skin": "肌"}
AGE_LABEL = {"child": "子ども", "teen": "十代", "adult": "大人", "elderly": "高齢"}
SKIN_LABEL = {"dark": "褐色", "pale": "色白"}

STANDALONE_HAIR = {"blonde": "blonde", "blond": "blonde", "platinum blonde": "blonde",
                   "brunette": "brown", "redhead": "red", "blonde girl": "blonde"}
HAIR_MULTI = {"multicolored hair", "two-tone hair", "gradient hair", "streaked hair",
              "colored inner hair", "split-color hair", "colored tips", "rainbow hair",
              "two-tone", "multicolored"}
_AGE = {
    "loli": "child", "shota": "child", "child": "child", "female child": "child",
    "male child": "child", "toddler": "child", "kid": "child", "little girl": "child",
    "little boy": "child", "幼女": "child", "女児": "child", "男児": "child",
    "teenage": "teen", "teenager": "teen", "teenage girl": "teen", "teenage boy": "teen",
    "teen": "teen", "少女": "teen", "少年": "teen",
    "adult": "adult", "mature": "adult", "mature female": "adult", "mature male": "adult",
    "milf": "adult", "middle-aged": "adult", "adult woman": "adult", "adult female": "adult",
    "adult male": "adult", "大人": "adult", "成人": "adult", "熟女": "adult",
    "elderly": "elderly", "old woman": "elderly", "old man": "elderly", "老婆": "elderly",
}
_SKIN = {"dark skin": "dark", "dark-skinned female": "dark", "dark-skinned male": "dark",
         "dark skinned female": "dark", "dark skinned male": "dark", "tan": "dark",
         "tanned": "dark", "tan skin": "dark", "brown skin": "dark", "褐色": "dark",
         "褐色肌": "dark", "pale skin": "pale", "white skin": "pale", "fair skin": "pale",
         "色白": "pale"}
SLOTS = ("hair", "eyes", "horns", "age", "skin")

_RE_HAIR = re.compile(r"^(?:(?:very long|long|medium|short|shoulder-length)\s+)?"
                      r"(?P<c>[a-z][a-z ]*?)\s+hair$")
_RE_HAIRED = re.compile(r"^(?P<c>[a-z][a-z ]*?)[- ]haired$")
_RE_EYES = re.compile(r"^(?P<c>[a-z][a-z ]*?)\s+(?:eyes?|pupils)$")
_RE_EYED = re.compile(r"^(?P<c>[a-z][a-z ]*?)[- ]eyed$")
_RE_HORNS = re.compile(r"^(?P<c>[a-z][a-z ]*?)\s+horns?$")


def canon_color(s: str) -> str | None:
    s = s.strip().lower()
    if not s:
        return None
    if s in _FAMILY:
        return _FAMILY[s]
    words = s.replace("-", " ").split()
    if words and words[-1] in _FAMILY:
        return _FAMILY[words[-1]]
    fam = {_FAMILY[w] for w in words if w in _FAMILY}
    return fam.pop() if len(fam) == 1 else None


def _kanji_color(t: str) -> str | None:
    fam = {v for k, v in _KANJI_COLOR.items() if k in t}
    return fam.pop() if len(fam) == 1 else None


def _norm_tag(t: str) -> str:
    t = re.sub(r"[_]+", " ", str(t).lower()).strip()
    return re.sub(r"\s+", " ", t)


def identity_slots(tags) -> dict:
    """プロンプトの語から、髪・目・ツノの色と、年齢・肌を拾う。"""
    out = {s: set() for s in SLOTS}
    flags = set()
    for raw in tags:
        t = _norm_tag(raw)
        if not t:
            continue
        if t in HAIR_MULTI:
            flags.add("hair_multi"); continue
        if t == "heterochromia":
            flags.add("hetero"); continue
        if t in STANDALONE_HAIR:
            out["hair"].add(STANDALONE_HAIR[t]); continue
        m = _RE_HAIR.match(t) or _RE_HAIRED.match(t)
        if m:
            c = canon_color(m.group("c"))
            if c:
                out["hair"].add(c)
            continue
        m = _RE_EYES.match(t) or _RE_EYED.match(t)
        if m:
            c = canon_color(m.group("c"))
            if c:
                out["eyes"].add(c)
            continue
        m = _RE_HORNS.match(t)
        if m:
            c = canon_color(m.group("c"))
            if c:
                out["horns"].add(c)
            continue
        if t in _AGE:
            out["age"].add(_AGE[t]); continue
        if t in _SKIN:
            out["skin"].add(_SKIN[t]); continue
        if "髪" in t:
            c = _kanji_color(t)
            if c:
                out["hair"].add(c)
        elif "角" in t or "ツノ" in t:
            c = _kanji_color(t)
            if c:
                out["horns"].add(c)
        elif any(k in t for k in ("目", "瞳", "眼")):
            c = _kanji_color(t)
            if c:
                out["eyes"].add(c)
    res = {s: sorted(v) for s, v in out.items() if v}
    if flags:
        res["flags"] = sorted(flags)
    return res


_COUNT_RE = re.compile(r"^(\d+)\+?\s*(girls?|boys?|others?)$")
_MULTI_WORDS = {"multiple girls", "multiple boys", "multiple others", "group", "crowd",
                "everyone", "harem", "couple", "yuri", "yaoi", "hetero",
                "複数", "二人", "ふたり", "三人"}


def multi_hint(tags) -> bool:
    """プロンプトから「複数人の絵」と読めるか。"""
    single = 0
    for raw in tags:
        t = _norm_tag(raw)
        if t in _MULTI_WORDS:
            return True
        m = _COUNT_RE.match(t)
        if m:
            if int(m.group(1)) >= 2:
                return True
            single += 1
    return single >= 2          # 1girl と 1boy が両方ある、など


def words_label(slots: dict) -> list[str]:
    """画面に出す短い言葉（例: 髪:白銀 / ツノ:黒）。"""
    out = []
    for s in SLOTS:
        for v in slots.get(s, []):
            lab = (COLOR_LABEL.get(v) if s in ("hair", "eyes", "horns")
                   else AGE_LABEL.get(v) if s == "age" else SKIN_LABEL.get(v))
            out.append(f"{SLOT_LABEL[s]}:{lab or v}")
    return out


def _sig(slots: dict) -> tuple:
    return tuple((s, tuple(slots.get(s, ()))) for s in SLOTS + ("flags",) if slots.get(s))


def build_profile(slot_list: list[dict]) -> dict:
    """確定した1キャラぶんの画像から、各項目の傾向を出す。"""
    prof = {}
    for s in SLOTS:
        stated = [set(d.get(s, ())) for d in slot_list if d.get(s)]
        if s == "hair":
            stated = [set(d.get(s, ())) for d in slot_list
                      if d.get(s) and "hair_multi" not in d.get("flags", ())]
        if s == "eyes":
            stated = [set(d.get(s, ())) for d in slot_list
                      if d.get(s) and "hetero" not in d.get("flags", ())]
        n = len(stated)
        if not n:
            continue
        cnt = Counter(v for st in stated for v in st)
        top, tc = cnt.most_common(1)[0]
        prof[s] = {"n": n, "top": top, "share": tc / n,
                   "shares": {v: c / n for v, c in cnt.items()}}
    return prof


def conflicts(prof: dict, slots: dict) -> list[str]:
    """その顔の特徴語が、キャラの傾向とはっきり食い違う項目を返す。

    はっきり、の中身:
      ・キャラ側がその項目を 8 枚以上で書いていて、8 割以上が同じ値
      ・顔の側がその値を書いておらず、書いてある値はキャラ側で 5% 未満"""
    out = []
    flags = set(slots.get("flags", ()))
    for s, p in prof.items():
        if p["n"] < 8 or p["share"] < 0.8:
            continue
        vals = set(slots.get(s, ()))
        if not vals or p["top"] in vals:
            continue
        if s == "hair" and "hair_multi" in flags:
            continue
        if s == "eyes" and "hetero" in flags:
            continue
        if all(p["shares"].get(v, 0.0) < 0.05 for v in vals):
            out.append(s)
    return out


# --------------------------------------------------------------------------
# 特徴の置き場（メモリ）
# --------------------------------------------------------------------------

def vec_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float16).tobytes()


def blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float16).astype(np.float32)


class Feats:
    """いくつかの顔の特徴の束。

    c    : 上半身で見た特徴（すべての顔にある）
    f    : 絵全体で見た特徴（1人の絵だけ。無ければ None）
    mask : c の各行について f があるかどうか

    比べるときは同じ見方どうしで比べる。1人の絵どうしなら全体、
    それ以外（2人の絵が片方でも混ざる）なら上半身。"""
    __slots__ = ("c", "f", "mask")

    def __init__(self, c, f=None, mask=None):
        self.c, self.f, self.mask = c, f, mask

    def __len__(self):
        return len(self.c)

    def take(self, idx) -> "Feats":
        idx = np.asarray(idx, dtype=np.int64)
        if self.f is None or self.mask is None:
            return Feats(self.c[idx])
        m = self.mask[idx]
        if not m.any():
            return Feats(self.c[idx], None, m)
        # f は mask が立った行だけを詰めて持っているので、その並びを引き直す
        where = np.cumsum(self.mask) - 1
        return Feats(self.c[idx], self.f[where[idx[m]]], m)

    @staticmethod
    def concat(parts: list) -> "Feats":
        parts = [p for p in parts if len(p)]
        if not parts:
            return Feats(np.zeros((0, 0), np.float32))
        c = np.concatenate([p.c for p in parts])
        if all(p.f is None and p.mask is None for p in parts):
            return Feats(c)
        masks = [p.mask if p.mask is not None else np.zeros(len(p), bool) for p in parts]
        fs = [p.f for p in parts if p.f is not None and len(p.f)]
        f = np.concatenate(fs) if fs else None
        return Feats(c, f, np.concatenate(masks))


def mdiff(metric: V.Metric, A: Feats, B: Feats) -> np.ndarray:
    """同じ見方どうしで比べた差（len(A) × len(B)）。

    両方に全体の特徴がある組は全体どうしで、それ以外は上半身どうしで比べる。
    使わない組の計算はしない（1 人の絵が多いと、半分近くが無駄になるため）。"""
    if A.f is None or B.f is None or A.mask is None or B.mask is None:
        return metric.diff(A.c, B.c)
    ia, ib = np.where(A.mask)[0], np.where(B.mask)[0]
    if not len(ia) or not len(ib):
        return metric.diff(A.c, B.c)
    na, nb = np.where(~A.mask)[0], np.where(~B.mask)[0]
    D = np.empty((len(A), len(B)), dtype=np.float32)
    full = metric.diff(A.f, B.f)
    if len(na) == 0 and len(nb) == 0:
        return full
    D[np.ix_(ia, ib)] = full
    if len(na):
        D[na] = metric.diff(A.c[na], B.c)
    if len(nb):
        D[np.ix_(ia, nb)] = metric.diff(A.c[ia], B.c[nb])
    return D


class Store:
    """全ての顔の特徴を行列に並べて持つ。行 = 顔。"""

    def __init__(self):
        self.ids = np.zeros(0, dtype=np.int64)
        self.Xc = np.zeros((0, 0), dtype=np.float16)
        self.Xf = np.zeros((0, 0), dtype=np.float16)
        self.fpos = np.zeros(0, dtype=np.int64)
        self.solo = np.zeros(0, dtype=bool)
        self.image = np.zeros(0, dtype=np.int64)
        self.pos: dict[int, int] = {}
        self.use_full = False
        self.Xcol = np.zeros((0, V.COLOR_DIM), dtype=np.float32)
        self.has_col = np.zeros(0, dtype=bool)
        self.sig = None

    @staticmethod
    def signature(con, model: str) -> tuple:
        r = con.execute("SELECT COUNT(*), COALESCE(MAX(instance_id),0) FROM c_features "
                        "WHERE model=?", (model,)).fetchone()
        n = con.execute("SELECT COUNT(*) FROM c_instances").fetchone()[0]
        return (int(r[0]), int(r[1]), int(n), feature_variant())

    @classmethod
    def load(cls, con, metric: V.Metric, model: str) -> "Store":
        s = cls()
        s.use_full = feature_variant() == "full"
        info = {r[0]: (int(r[1] or 0), int(r[2])) for r in con.execute(
            "SELECT id, solo, image_id FROM c_instances")}
        crop, full = {}, {}
        for iid, var, vec in con.execute(
                "SELECT instance_id, variant, vec FROM c_features WHERE model=?", (model,)):
            if iid in info:
                (crop if var == "crop" else full)[iid] = vec
        ids = sorted(set(crop) | set(full))
        if not ids:
            s.sig = cls.signature(con, model)
            return s
        dim = len(blob_to_vec((crop.get(ids[0]) or full.get(ids[0]))))
        n = len(ids)
        Xc = np.empty((n, dim), dtype=np.float16)
        fpos = np.full(n, -1, dtype=np.int64)
        frows = []
        for k, iid in enumerate(ids):
            b = crop.get(iid) or full.get(iid)       # 上半身が無い古い記録は全体で代用
            Xc[k] = blob_to_vec(b)
            if s.use_full and info[iid][0] == 1 and iid in full:
                fpos[k] = len(frows)
                frows.append(full[iid])
        for a in range(0, n, 8192):
            Xc[a:a + 8192] = metric.prep(Xc[a:a + 8192].astype(np.float32)).astype(np.float16)
        if frows:
            Xf = np.empty((len(frows), dim), dtype=np.float16)
            for k, b in enumerate(frows):
                Xf[k] = blob_to_vec(b)
            for a in range(0, len(frows), 8192):
                Xf[a:a + 8192] = metric.prep(Xf[a:a + 8192].astype(np.float32)).astype(np.float16)
        else:
            Xf = np.zeros((0, dim), dtype=np.float16)
        s.ids = np.asarray(ids, dtype=np.int64)
        s.Xc, s.Xf, s.fpos = Xc, Xf, fpos
        s.solo = np.array([info[i][0] == 1 for i in ids], dtype=bool)
        s.image = np.array([info[i][1] for i in ids], dtype=np.int64)
        s.pos = {int(i): k for k, i in enumerate(ids)}
        s.Xcol = np.zeros((len(ids), V.COLOR_DIM), dtype=np.float32)
        s.has_col = np.zeros(len(ids), dtype=bool)
        for iid, vec in con.execute("SELECT instance_id, vec FROM c_features WHERE model=?",
                                    (V.COLOR_MODEL,)):
            k = s.pos.get(iid)
            if k is not None:
                v = blob_to_vec(vec)
                if len(v) == V.COLOR_DIM:
                    s.Xcol[k] = v
                    s.has_col[k] = True
        s.sig = cls.signature(con, model)
        return s

    def rows(self, positions) -> np.ndarray:
        """上半身の特徴（似た絵探しなど、見方をそろえたいときに使う）。"""
        return self.Xc[np.asarray(positions, dtype=np.int64)].astype(np.float32)

    def feats(self, positions) -> Feats:
        P = np.asarray(positions, dtype=np.int64)
        c = self.Xc[P].astype(np.float32)
        if not self.use_full or not len(self.Xf):
            return Feats(c)
        fi = self.fpos[P]
        m = fi >= 0
        f = self.Xf[fi[m]].astype(np.float32) if m.any() else None
        return Feats(c, f, m)


def fps_subset(metric: V.Metric, X: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    """多様さを保ったまま見本を間引く（最も遠いものから順に選ぶ）。
    確定が数千枚あるキャラでも、見た目の違うまとまりを取りこぼさない。"""
    n = len(X)
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(n))]
    dmin = metric.diff(X, X[chosen])[:, 0]
    for _ in range(cap - 1):
        j = int(np.argmax(dmin))
        chosen.append(j)
        dmin = np.minimum(dmin, metric.diff(X, X[j:j + 1])[:, 0])
    return np.array(sorted(chosen))


def _robust(D: np.ndarray, k: int) -> np.ndarray:
    """各行の小さいほうから k 個の平均。1 枚だけ近い、という偶然に引っ張られない。"""
    if D.shape[1] == 0:
        return np.full(D.shape[0], np.inf, dtype=np.float32)
    k = max(1, min(k, D.shape[1]))
    if D.shape[1] > k:
        part = np.partition(D, k - 1, axis=1)[:, :k]
    else:
        part = D
    return part.mean(axis=1)


def _poskey(P) -> str:
    a = np.ascontiguousarray(np.asarray(P, dtype=np.int64))
    return f"{len(a)}:" + hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()


class DistCache:
    """キャラごとの「見本までの近さ」を、全ての顔について覚えておく。

    判定は人が操作するたびに走る。15 万の顔 × 数万の見本を毎回総当たりすると
    1 分を超えるので、見本の顔ぶれが変わったキャラの分だけ計算し直す。
    覚えた値は、特徴の表（Store）が読み直されたら捨てる。"""

    def __init__(self):
        self.owner = None
        self.near: dict = {}     # cid -> (key, (n, 3))  見本までの差の小さい順 3 つ
        self.neg: dict = {}      # cid -> (key, (n, 1))  否にした顔までのいちばん小さい差
        self.sub: dict = {}      # (種類, cid) -> (key, 間引いた位置)
        self.veto: dict = {}     # cid -> (書き分け, 書き方ごとの食い違い)
        self.outl: dict = {}     # cid -> (key, 浮いている顔)
        self.aux: dict = {}      # 顔ごとの特徴語・LoRA の番号
        self.computed = 0

    def _check(self, store) -> None:
        if self.owner is not store:
            self.near.clear(); self.neg.clear(); self.sub.clear()
            self.veto.clear(); self.outl.clear()
            self.owner = store

    def subset(self, metric, store, tag, ps: np.ndarray, cap: int, seed: int) -> np.ndarray:
        """多すぎる見本を間引く（結果を覚えておく）。"""
        self._check(store)
        if len(ps) <= cap:
            return ps
        key = _poskey(ps) + f":{cap}"
        hit = self.sub.get(tag)
        if hit and hit[0] == key:
            return hit[1]
        sel = ps[fps_subset(metric, store.rows(ps), cap, seed=seed)]
        self.sub[tag] = (key, sel)
        return sel

    def ensure(self, metric, store, table: str, items: list, k: int, progress=None,
               stage: str = "") -> list:
        """items = [(cid, 位置の配列)]。足りない分をまとめて 1 回の走査で計算し、
        各 cid の (n, k) を返す（位置が空なら None）。"""
        self._check(store)
        tab = self.near if table == "near" else self.neg
        todo = []
        for cid, P in items:
            if not len(P):
                tab.pop(cid, None)
                continue
            hit = tab.get(cid)
            if not (hit and hit[0] == _poskey(P)):
                todo.append((cid, np.asarray(P, dtype=np.int64)))
        if todo:
            n = len(store.ids)
            dt = np.float16 if metric.normalized else np.float32
            allP = np.concatenate([P for _, P in todo])
            segs, off = [], 0
            for cid, P in todo:
                segs.append((cid, off, off + len(P)))
                off += len(P)
            outs = {cid: np.full((n, k), np.inf, dtype=dt) for cid, _ in todo}
            FE = store.feats(allP)
            step = int(max(256, min(4096, 48_000_000 // max(len(allP), 1))))
            for a in range(0, n, step):
                rows = np.arange(a, min(a + step, n))
                D = mdiff(metric, store.feats(rows), FE)
                for cid, s0, s1 in segs:
                    sub = D[:, s0:s1]
                    kk = min(k, s1 - s0)
                    part = (np.partition(sub, kk - 1, axis=1)[:, :kk] if sub.shape[1] > kk
                            else sub.copy())
                    part.sort(axis=1)
                    outs[cid][a:a + len(rows), :kk] = part
                if progress:
                    progress(stage, min(a + step, n), n)
            for cid, P in todo:
                tab[cid] = (_poskey(P), outs[cid])
            self.computed += len(todo)
        return [tab[cid][1] if len(P) else None for cid, P in items]


# --------------------------------------------------------------------------
# 判定エンジン
# --------------------------------------------------------------------------

class Engine:
    """モデルと特徴をまとめて持つ。サーバ内に 1 つだけ置く。"""

    def __init__(self):
        self.lock = threading.RLock()
        self.metric: V.Metric | None = None
        self.base: dict | None = None
        self.model_name = V.MODELS["ccip"]["dir"]
        self.store: Store | None = None
        self.calib: dict = {}
        self.gen = 0             # 人の操作があるたびに増える
        self.cache = DistCache()

    def ready(self) -> bool:
        return V.models_ready()["ready"]

    def ensure(self, con, log=None) -> None:
        with self.lock:
            if self.metric is None:
                self.metric = V.load_metric(log=log or (lambda *a: None))
                self.base = V.ccip_thresholds()
            sig = Store.signature(con, self.model_name)
            if self.store is None or self.store.sig != sig:
                self.store = Store.load(con, self.metric, self.model_name)

    def invalidate(self) -> None:
        with self.lock:
            self.store = None


ENGINE = Engine()


def _load_labels(con):
    chars = {r["id"]: r["name"] for r in con.execute("SELECT id, name FROM c_characters")}
    confirmed = {r[0]: r[1] for r in con.execute(
        "SELECT instance_id, character_id FROM c_assign WHERE state='confirmed'")}
    rejects: dict[int, set] = {}
    for iid, cid in con.execute("SELECT instance_id, character_id FROM c_reject"):
        rejects.setdefault(iid, set()).add(cid)
    return chars, confirmed, rejects


def _image_words(con) -> dict[int, dict]:
    out = {}
    for iid, w in con.execute("SELECT image_id, words FROM c_analysis WHERE words IS NOT NULL"):
        try:
            out[iid] = json.loads(w)
        except Exception:
            pass
    return out


def _image_loras(con) -> dict[int, set]:
    out: dict[int, set] = {}
    for iid, name, st in con.execute("SELECT image_id, name, strength FROM loras"):
        if st is not None and abs(float(st)) < 0.05:
            continue
        out.setdefault(iid, set()).add(name)
    return out


def character_loras(con, store: Store, confirmed: dict, img_loras: dict) -> dict:
    """ほぼ特定のキャラでしか使っていない LoRA を「キャラ LoRA」とみなす。"""
    per: dict[str, Counter] = {}
    for iid, cid in confirmed.items():
        p = store.pos.get(iid)
        if p is None or not store.solo[p]:
            continue
        for name in img_loras.get(int(store.image[p]), ()):
            per.setdefault(name, Counter())[cid] += 1
    out = {}
    for name, cnt in per.items():
        n = sum(cnt.values())
        cid, c = cnt.most_common(1)[0]
        if n >= 10 and c / n >= 0.9:
            out[name] = cid
    return out


# 「否」にした顔のうち、確定どうしのふだんの近さ（下から 95%）より内側にあるものは、
# 線を決める材料に入れない。まとめて × を付けたときに本人の顔が少し混ざっただけで、
# そのキャラの自動が丸ごと止まってしまうのを防ぐ（その近くの顔は「否に近い」で
# 確認待ちへ回るので、そっくりさんが自動に入ることはない）。
REJ_GATE = 95


def calibrate(metric, base, store: Store, ex_parts: list, neg_parts: list, near: list) -> dict:
    """キャラごとの線を決める。モデルの公開値から始め、厳しくする方向にだけ動かす。

    impostor（別人）の近さ: 他キャラの見本と、そのキャラで「否」にした顔。
    別人が 1% 以上入ってしまう近さより内側だけを自動にする。
    材料が 100 件に満たないうちは公開値のまま。
    near[ci] は全ての顔からキャラ ci の見本までの近い順 3 つ（DistCache）。"""
    C = len(ex_parts)
    per = {}
    gen_all, imp_all = [], []
    gen_per: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(0)
    for ci in range(C):
        P = ex_parts[ci]
        if len(P) >= 6:
            sel = P if len(P) <= 300 else np.sort(rng.choice(P, 300, replace=False))
            D = mdiff(metric, store.feats(sel), store.feats(P))
            where = {int(c): j for j, c in enumerate(P)}
            for r, c in enumerate(sel):
                D[r, where[int(c)]] = np.inf
            g = _robust(D, 3)
            g = g[np.isfinite(g)]
            gen_per[ci] = g
            gen_all += g.tolist()
    for ci in range(C):
        imp = []
        k = min(3, len(ex_parts[ci]))
        if k:
            for cj in range(C):
                if cj == ci or len(ex_parts[cj]) < 3:
                    continue
                imp.append(near[ci][ex_parts[cj], :k].astype(np.float32).mean(axis=1))
            if len(neg_parts[ci]):
                dn = near[ci][neg_parts[ci], :k].astype(np.float32).mean(axis=1)
                g0 = gen_per.get(ci)
                if REJ_GATE is not None and g0 is not None and len(g0) >= 10:
                    dn = dn[dn > float(np.percentile(g0, REJ_GATE))]
                imp.append(dn)
        imp = np.concatenate(imp).astype(np.float64) if imp else np.zeros(0)
        imp = imp[np.isfinite(imp)]
        t_auto, t_rev = base["auto"], base["review"]
        why = "公開値"
        if len(imp) >= 100:
            q1 = float(np.percentile(imp, 1))
            if q1 < t_auto:
                t_auto, why = q1, "別キャラが近い"
            t_rev = max(t_auto, min(t_rev, float(np.percentile(imp, 5))))
        # そのキャラらしさ: 確定した顔どうしが普段どれくらい近いか。
        # その 2 倍より遠い顔は、別人のそっくりさんかもしれないので自動にしない。
        # （確定は候補の芯から選ばれるので、確定どうしの近さは本人のふだんの
        #   揺れより小さく出る。そのままの値を線にすると、本人の顔まで大量に
        #   確認待ちへ回ってしまう。2 倍はその偏りの分の余裕）
        g = gen_per.get(ci)
        if g is not None and len(g) >= 10:
            q = 2.0 * float(np.percentile(g, 99))
            if q < t_auto:
                t_auto, why = q, "そのキャラの普段の近さ"
        t_rev = max(t_rev, t_auto)
        per[ci] = {"auto": float(t_auto), "review": float(t_rev), "n_imp": int(len(imp)),
                   "why": why, "n_gen": int(len(g)) if g is not None else 0}
        imp_all.append(imp)
    g = np.asarray(gen_all) if gen_all else np.zeros(0)
    im = np.concatenate(imp_all) if imp_all else np.zeros(0)
    return {
        "per": per,
        "base": base,
        "gen_n": int(len(g)),
        "gen_median": float(np.median(g)) if len(g) else None,
        "gen_p95": float(np.percentile(g, 95)) if len(g) else None,
        "imp_n": int(len(im)),
        "imp_p1": float(np.percentile(im, 1)) if len(im) else None,
        "imp_p5": float(np.percentile(im, 5)) if len(im) else None,
    }


def classify_all(con, eng: Engine = None, progress=None, log=None) -> dict:
    """確定していない全ての顔について、自動 / 確認待ち / 未分類 を決め直す。"""
    eng = eng or ENGINE
    t0 = time.time()
    eng.ensure(con, log=log)
    store, metric = eng.store, eng.metric
    gen_at_start = eng.gen
    chars, confirmed, rejects = _load_labels(con)
    cids = sorted(chars)
    cindex = {c: i for i, c in enumerate(cids)}
    C = len(cids)
    stats = {"characters": C, "confirmed": len(confirmed), "auto": 0, "review": 0,
             "outliers": 0, "instances": int(len(store.ids))}
    if not len(store.ids):
        _write_machine_rows(con, [], [])
        stats["elapsed"] = time.time() - t0
        return stats

    cap = int(settings().get("exemplar_cap") or 400)
    cache = eng.cache
    # --- 見本（確定）と、否にした顔 -------------------------------------
    per_char_conf: dict[int, list] = {}
    for iid, cid in confirmed.items():
        p = store.pos.get(iid)
        if p is not None and cid in cindex:
            per_char_conf.setdefault(cindex[cid], []).append(p)
    E_parts = []
    for ci, cid in enumerate(cids):
        ps = np.array(sorted(per_char_conf.get(ci, [])), dtype=np.int64)
        E_parts.append(cache.subset(metric, store, ("ex", cid), ps, cap, seed=cid))
    n_ex = np.array([len(p) for p in E_parts], dtype=np.int64)

    per_char_neg: dict[int, list] = {}
    for iid, cs in rejects.items():
        p = store.pos.get(iid)
        if p is None:
            continue
        for cid in cs:
            if cid in cindex:
                per_char_neg.setdefault(cindex[cid], []).append(p)
    neg_parts = []
    for ci, cid in enumerate(cids):
        ps = np.array(sorted(per_char_neg.get(ci, [])), dtype=np.int64)
        neg_parts.append(cache.subset(metric, store, ("neg", cid), ps, cap, seed=100000 + cid))

    # 全ての顔から、各キャラの見本・否までの近さ（変わったキャラの分だけ計算する）
    c0 = cache.computed
    near = cache.ensure(metric, store, "near", list(zip(cids, E_parts)), 3,
                        progress=progress, stage="見本までの近さ")
    negmin = [None if x is None else x[:, 0]
              for x in cache.ensure(metric, store, "neg", list(zip(cids, neg_parts)), 1,
                                    progress=progress, stage="否にした顔までの近さ")]
    stats["recomputed"] = cache.computed - c0
    empty = np.full((len(store.ids), 3), np.inf, dtype=np.float32)
    near = [x if x is not None else empty for x in near]

    calib = calibrate(metric, eng.base, store, E_parts, neg_parts, near)
    t_auto = np.array([calib["per"][ci]["auto"] for ci in range(C)], dtype=np.float32)
    t_rev = np.array([calib["per"][ci]["review"] for ci in range(C)], dtype=np.float32)
    # 色: 確定した顔どうしの色の近さ（普段の範囲）。これより色の遠い顔は自動にしない。
    col_ex: dict[int, np.ndarray] = {}
    col_tau: dict[int, float] = {}
    for ci in range(C):
        ps = [int(p) for p in E_parts[ci] if store.has_col[p]]
        if len(ps) < 10:
            continue
        Xc_ = store.Xcol[ps]
        Dc = np.sqrt(np.maximum(((Xc_[:, None, :] - Xc_[None, :, :]) ** 2).sum(-1), 0))
        np.fill_diagonal(Dc, np.inf)
        col_ex[ci] = Xc_
        col_tau[ci] = float(np.percentile(_robust(Dc, 3), 95))
        calib["per"][ci]["color"] = round(col_tau[ci], 4)
    eng.calib = calib

    # --- 特徴語と LoRA -------------------------------------------------
    aux = _row_aux(con, eng)
    sig_id, sigs, lset_id, lsets = aux["sig_id"], aux["sigs"], aux["lset_id"], aux["lsets"]
    profiles = {}
    for ci in range(C):
        lst = [sigs[sig_id[p]] for p in per_char_conf.get(ci, [])
               if store.solo[p] and sig_id[p] >= 0]
        profiles[ci] = build_profile(lst)
    # キャラ LoRA: ほぼ特定のキャラでしか使っていない LoRA
    per_lora: dict[str, Counter] = {}
    for iid, cid in confirmed.items():
        p = store.pos.get(iid)
        if p is None or not store.solo[p] or lset_id[p] < 0:
            continue
        for name in lsets[lset_id[p]]:
            per_lora.setdefault(name, Counter())[cid] += 1
    char_lora = {}
    for name, cnt in per_lora.items():
        tot = sum(cnt.values())
        cid, c = cnt.most_common(1)[0]
        if tot >= 10 and c / tot >= 0.9:
            char_lora[name] = cid
    # 書き方ごとの食い違い（書き方の種類 × キャラ）。書き分けが変わったキャラの分だけ作り直す
    vcache = eng.cache.veto
    Vsig = np.zeros((len(sigs) + 1, max(C, 1)), dtype=bool)
    for ci, cid in enumerate(cids):
        key = (json.dumps(profiles[ci], sort_keys=True, default=str), len(sigs))
        hit = vcache.get(cid)
        if not (hit and hit[0] == key):
            col = np.array([bool(conflicts(profiles[ci], w)) for w in sigs] + [False], dtype=bool)
            vcache[cid] = (key, col)
            hit = vcache[cid]
        Vsig[:, ci] = hit[1]
    lset_char = np.full(len(lsets) + 1, -1, dtype=np.int64)
    for j, names in enumerate(lsets):
        cs = {char_lora[n] for n in names if n in char_lora}
        if len(cs) == 1:
            c = cs.pop()
            if c in cindex:
                lset_char[j] = cindex[c]
    rej_pairs = [(store.pos[iid], cindex[cid]) for iid, cs in rejects.items() if iid in store.pos
                 for cid in cs if cid in cindex]

    rows_out, outliers = [], []
    conf_mask = np.zeros(len(store.ids), dtype=bool)
    for iid in confirmed:
        p = store.pos.get(iid)
        if p is not None:
            conf_mask[p] = True
    targets = np.where(~conf_mask)[0]
    tpos = np.full(len(store.ids), -1, dtype=np.int64)
    tpos[targets] = np.arange(len(targets))
    rej_r = np.array([tpos[p] for p, c in rej_pairs if tpos[p] >= 0], dtype=np.int64)
    rej_c = np.array([c for p, c in rej_pairs if tpos[p] >= 0], dtype=np.int64)
    B = 16384
    done = 0
    kk = np.minimum(3, n_ex)
    need = np.minimum(2, np.maximum(n_ex, 1))
    INF = np.float32(np.inf)
    for a in range(0, len(targets), B):
        blk = targets[a:a + B]
        b = len(blk)
        if C == 0 or not n_ex.any():
            break
        R = np.full((b, C), INF, dtype=np.float32)
        # 見本が 2 枚以上（見本が 1 枚なら 1 枚）自動の線の内側にあるか
        S_ok = np.zeros((b, C), dtype=bool)
        NR = np.full((b, C), INF, dtype=np.float32)
        for ci in range(C):
            if n_ex[ci]:
                T = near[ci][blk].astype(np.float32)
                R[:, ci] = T[:, :kk[ci]].mean(axis=1)
                S_ok[:, ci] = T[:, need[ci] - 1] <= t_auto[ci]
            if negmin[ci] is not None:
                NR[:, ci] = negmin[ci][blk]
        m = (rej_r >= a) & (rej_r < a + b)
        R[rej_r[m] - a, rej_c[m]] = INF
        sid = sig_id[blk].copy()
        sid[(sid < 0) | ~store.solo[blk]] = len(sigs)
        Vt = Vsig[sid, :C]
        lid = lset_id[blk].copy()
        lid[(lid < 0) | ~store.solo[blk]] = len(lsets)
        L = lset_char[lid]

        rr = np.arange(b)
        Rnv = np.where(Vt, INF, R)
        close = Rnv <= t_auto[None, :]
        n_close = close.sum(axis=1)
        best_all = np.argmin(R, axis=1)
        best_nv = np.argmin(Rnv, axis=1)
        best_close = np.argmin(np.where(close, Rnv, INF), axis=1)
        # 否にした顔のほうが近いなら、そのキャラへは自動で入れない
        near_neg = NR <= R
        c_out = np.full(b, -1, dtype=np.int64)
        state = np.zeros(b, dtype=np.int8)          # 0 なし / 1 自動 / 2 確認待ち
        reason = np.zeros(b, dtype=np.int8)         # REASONS の番号
        one = n_close == 1
        c1 = best_close
        nn1 = near_neg[rr, c1]
        so1 = S_ok[rr, c1]
        lo1 = (L >= 0) & (L != c1)
        sel = one & nn1
        c_out[sel], state[sel], reason[sel] = c1[sel], 2, 1
        sel = one & ~nn1 & ~so1
        c_out[sel], state[sel], reason[sel] = c1[sel], 2, 2
        sel = one & ~nn1 & so1 & lo1
        c_out[sel], state[sel], reason[sel] = c1[sel], 2, 3
        sel = one & ~nn1 & so1 & ~lo1
        c_out[sel], state[sel], reason[sel] = c1[sel], 1, 0
        sel = n_close >= 2
        c_out[sel], state[sel], reason[sel] = c1[sel], 2, 4
        zero = n_close == 0
        l = np.where(L >= 0, L, 0)
        lora_ok = (zero & (L >= 0) & ~Vt[rr, l] & np.isfinite(R[rr, l])
                   & (R[rr, l] <= t_rev[l]) & ~near_neg[rr, l])
        c_out[lora_ok], state[lora_ok], reason[lora_ok] = l[lora_ok], 1, 5
        rv = Rnv[rr, best_nv]
        sel = zero & ~lora_ok & np.isfinite(rv) & (rv <= t_rev[best_nv])
        c_out[sel], state[sel], reason[sel] = best_nv[sel], 2, 6
        ra = R[rr, best_all]
        sel = zero & ~lora_ok & (state == 0) & np.isfinite(ra) & (ra <= t_auto[best_all])
        c_out[sel], state[sel], reason[sel] = best_all[sel], 2, 7
        # 色: 自動にする顔だけ、そのキャラの見本の色と比べる
        for ci in np.unique(c_out[state == 1]):
            if ci not in col_ex:
                continue
            idx = np.where((state == 1) & (c_out == ci) & store.has_col[blk])[0]
            if not len(idx):
                continue
            Xq = store.Xcol[blk[idx]]
            E_ = col_ex[ci]
            d2 = (Xq * Xq).sum(1)[:, None] + (E_ * E_).sum(1)[None, :] - 2.0 * (Xq @ E_.T)
            dc = np.sqrt(np.maximum(d2, 0))
            k3 = np.partition(dc, min(2, dc.shape[1] - 1), axis=1)[:, :3].mean(axis=1)
            bad = idx[k3 > col_tau[ci]]
            state[bad], reason[bad] = 2, 8
        hit = np.where(state > 0)[0]
        if len(hit):
            cs = c_out[hit]
            sc = R[hit, cs]
            other = Rnv[hit].copy()
            other[np.arange(len(hit)), cs] = INF
            om = other.min(axis=1) if C > 1 else np.full(len(hit), INF)
            ids_h = store.ids[blk[hit]]
            for j in range(len(hit)):
                mg = float(om[j] - sc[j]) if np.isfinite(om[j]) else None
                rows_out.append((int(ids_h[j]), cids[int(cs[j])],
                                 "auto" if state[hit[j]] == 1 else "review",
                                 float(sc[j]), mg, REASONS[int(reason[hit[j]])]))
        done += b
        if progress:
            progress("判定", done, len(targets))

    # --- 確定の中で浮いている顔（うっかり確定の保険） ----------------------
    # 「浮いている」は、モデル自身の基準（公開値）でも別人に見えるほど、
    # ほかの確定から離れている顔。キャラごとに厳しくした線は、そっくりな
    # 別キャラがいると本人のふだんの揺れまで下回るので、ここでは使わない。
    ocache = eng.cache.outl
    line_out = float(eng.base["review"])
    for ci, cid in enumerate(cids):
        ps = np.array(sorted(per_char_conf.get(ci, [])), dtype=np.int64)
        if len(ps) < 10 or n_ex[ci] < 6:
            continue
        key = (_poskey(ps), _poskey(E_parts[ci]), line_out)
        hit = ocache.get(cid)
        if hit and hit[0] == key:
            outliers += hit[1]
            continue
        found = []
        ex_set = {int(x): j for j, x in enumerate(E_parts[ci])}
        Ec = store.feats(E_parts[ci])
        for a in range(0, len(ps), 4096):
            chunk = ps[a:a + 4096]
            D = mdiff(metric, store.feats(chunk), Ec)
            for r, p in enumerate(chunk):
                j = ex_set.get(int(p))
                if j is not None:
                    D[r, j] = np.inf
            rr_ = _robust(D, 3)
            found += [int(store.ids[chunk[r]]) for r in np.where(rr_ > line_out)[0]]
        ocache[cid] = (key, found)
        outliers += found

    w = _write_machine_rows(con, rows_out, outliers)
    stats["written"] = w
    stats["auto"] = sum(1 for x in rows_out if x[2] == "auto")
    stats["review"] = sum(1 for x in rows_out if x[2] == "review")
    stats["outliers"] = len(outliers)
    stats["elapsed"] = round(time.time() - t0, 2)
    stats["stale"] = eng.gen != gen_at_start
    meta_set(con, "chara_calib", {k: v for k, v in calib.items() if k != "per"}
             | {"per": {cids[ci]: v for ci, v in calib["per"].items()}})
    meta_set(con, "chara_last_classify", stats)
    con.commit()
    return stats


REASONS = (None, "否に近い", "見本が少ない", "LoRAと食い違う", "複数のキャラに近い", "LoRA",
           "やや遠い", "特徴語が食い違う", "色が違う")


def _row_aux(con, eng: Engine) -> dict:
    """顔ごとの「特徴語の書き方」と「使った LoRA の組」を番号にしておく（覚えて使い回す）。"""
    store = eng.store
    sig = tuple(con.execute("SELECT COUNT(*), COALESCE(MAX(analyzed_at),0), "
                            "COALESCE(SUM(LENGTH(words)),0) FROM c_analysis").fetchone())
    sig += tuple(con.execute("SELECT COUNT(*), COALESCE(MAX(rowid),0) FROM loras").fetchone())
    c = eng.cache
    if c.aux and c.aux.get("owner") is store and c.aux.get("sig") == sig:
        return c.aux
    img_words = _image_words(con)
    img_loras = _image_loras(con)
    sig_index: dict = {}
    sigs: list = []
    lset_index: dict = {}
    lsets: list = []
    n = len(store.ids)
    sig_id = np.full(n, -1, dtype=np.int64)
    lset_id = np.full(n, -1, dtype=np.int64)
    img_sig: dict = {}
    img_ls: dict = {}
    for p in range(n):
        im = int(store.image[p])
        if im not in img_sig:
            w = img_words.get(im)
            if w:
                k = _sig(w)
                j = sig_index.get(k)
                if j is None:
                    j = sig_index[k] = len(sigs)
                    sigs.append(w)
                img_sig[im] = j
            else:
                img_sig[im] = -1
            ls = img_loras.get(im)
            if ls:
                k = frozenset(ls)
                j = lset_index.get(k)
                if j is None:
                    j = lset_index[k] = len(lsets)
                    lsets.append(k)
                img_ls[im] = j
            else:
                img_ls[im] = -1
        sig_id[p] = img_sig[im]
        lset_id[p] = img_ls[im]
    c.aux = {"owner": store, "sig": sig, "sig_id": sig_id, "sigs": sigs,
             "lset_id": lset_id, "lsets": lsets}
    c.veto.clear()
    return c.aux


def _write_machine_rows(con, rows: list, outliers: list) -> int:
    """機械の判定を書き込む。確定と否には触らない。変わった行だけ書く。

    判定の途中で人が操作していても壊さないよう、書き込む瞬間に
    「確定が入っていないか」「否になっていないか」を確かめ直す。"""
    now = time.time()
    old = {r[0]: (r[1], r[2], r[3], r[4]) for r in con.execute(
        "SELECT instance_id, character_id, state, reason, score FROM c_assign "
        "WHERE state IN ('auto','review')")}
    new = {}
    for (i, c, s, sc, mg, rs) in rows:
        new[i] = (c, s, sc, mg, rs)
    gone = [i for i in old if i not in new]
    changed = []
    for i, (c, s, sc, mg, rs) in new.items():
        o = old.get(i)
        if (o is None or o[0] != c or o[1] != s or o[2] != rs
                or o[3] is None or abs(float(o[3]) - sc) > 1e-5):
            changed.append((i, c, s, sc, mg, rs, now, i, c))
    old_out = {r[0] for r in con.execute(
        "SELECT instance_id FROM c_assign WHERE state='confirmed' AND reason='浮いている'")}
    new_out = {int(x) for x in outliers}
    out_off = list(old_out - new_out)
    out_on = new_out - old_out
    if out_on:
        plain = {r[0] for r in con.execute(
            "SELECT instance_id FROM c_assign WHERE state='confirmed' AND reason IS NULL")}
        out_on &= plain
    out_on = list(out_on)
    n_rej = con.execute("SELECT COUNT(*) FROM c_assign a JOIN c_reject r ON r.instance_id=a.instance_id "
                        "AND r.character_id=a.character_id WHERE a.state IN ('auto','review')").fetchone()[0]
    total = len(gone) + len(changed) + len(out_off) + len(out_on) + n_rej
    if not total:
        return 0
    GENS["machine"] += 1
    with con:
        for ch in _chunks(gone):
            con.execute(f"DELETE FROM c_assign WHERE state IN ('auto','review') "
                        f"AND instance_id IN ({_qm(len(ch))})", ch)
        con.executemany(
            "INSERT INTO c_assign(instance_id,character_id,state,score,margin,reason,source,"
            "updated_at) SELECT ?,?,?,?,?,?,'model',? WHERE NOT EXISTS "
            "(SELECT 1 FROM c_reject r WHERE r.instance_id=? AND r.character_id=?) "
            "ON CONFLICT(instance_id) DO UPDATE SET character_id=excluded.character_id, "
            "state=excluded.state, score=excluded.score, margin=excluded.margin, "
            "reason=excluded.reason, source='model', updated_at=excluded.updated_at "
            "WHERE c_assign.state IN ('auto','review')", changed)
        # 否になった組は、機械の行を残さない
        con.execute("DELETE FROM c_assign WHERE state IN ('auto','review') AND EXISTS "
                    "(SELECT 1 FROM c_reject r WHERE r.instance_id=c_assign.instance_id "
                    "AND r.character_id=c_assign.character_id)")
        for ch in _chunks(out_off):
            con.execute(f"UPDATE c_assign SET reason=NULL WHERE state='confirmed' "
                        f"AND reason='浮いている' AND instance_id IN ({_qm(len(ch))})", ch)
        for ch in _chunks(out_on):
            con.execute(f"UPDATE c_assign SET reason='浮いている' WHERE state='confirmed' "
                        f"AND reason IS NULL AND instance_id IN ({_qm(len(ch))})", ch)
    return total


# --------------------------------------------------------------------------
# 候補（まだ名前のない顔のまとまり）
# --------------------------------------------------------------------------

FIT_MAX = 12000


def _group_sums(X: np.ndarray, lab: np.ndarray, k: int) -> np.ndarray:
    """かたまりごとの和（np.add.at は遅いので、0/1 の表との掛け算で足す）。"""
    out = np.zeros((k, X.shape[1]), dtype=np.float32)
    B = 8192
    for a in range(0, len(lab), B):
        lb = lab[a:a + B]
        M = np.zeros((k, len(lb)), dtype=np.float32)
        M[lb, np.arange(len(lb))] = 1.0
        out += M @ X[a:a + B]
    return out


def similarity_groups(store: Store, P: np.ndarray, suspicion: np.ndarray | None = None,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """顔を「似た顔のかたまり」に分けて並べ直す。

    何万枚もの候補を 1 枚ずつ目で追うのは無理なので、似た顔を隣り合わせに並べ、
    かたまりごとに見分けられるようにする（別の子はたいてい別のかたまりになる）。
    かたまりは怪しい順（候補の芯から遠いものが先）、かたまりの中は典型的な顔から。

    suspicion: 各顔の「芯からの遠さ」（大きいほど怪しい）。無ければ全体の中心からの遠さ。
    戻り値: (並べ直した位置, 各位置のかたまり番号 0..)"""
    P = np.asarray(P, dtype=np.int64)
    n = len(P)
    if n == 0:
        return P, np.zeros(0, dtype=np.int64)
    X = store.rows(P)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    if n <= 40:
        k = 1
    else:
        k = int(min(300, max(2, round(1.5 * math.sqrt(n)))))
    rng = np.random.default_rng(seed)
    if k == 1:
        lab = np.zeros(n, dtype=np.int64)
        C = X.mean(axis=0, keepdims=True)
    else:
        # かたまりが 1〜2 枚ばかりにならないよう、小さすぎるかたまりは途中で畳んで、
        # その顔をほかのかたまりへ配り直す（見本の列として意味のある大きさにする）
        # 何万枚もあるときは、中心を決めるのは抜き出した 1 万 2 千枚で行い、最後に全部を振り分ける
        Xf = X[np.sort(rng.choice(n, FIT_MAX, replace=False))] if n > FIT_MAX else X
        nf = len(Xf)
        min_size = max(3, int(nf / (k * 3)))
        C = Xf[rng.choice(nf, k, replace=False)].copy()

        def assign(Y, C):
            out = np.empty(len(Y), dtype=np.int64)
            for a in range(0, len(Y), 16384):
                out[a:a + 16384] = np.argmax(Y[a:a + 16384] @ C.T, axis=1)
            return out

        for it in range(10):
            lab = assign(Xf, C)
            cnt = np.bincount(lab, minlength=len(C))
            keep = cnt >= (min_size if it >= 1 else 1)
            if not keep.any():
                keep[np.argmax(cnt)] = True
            if not keep.all():
                C = C[keep]
                lab = assign(Xf, C)
            C = _group_sums(Xf, lab, len(C))
            C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
        lab = assign(X, C)
        used = np.unique(lab)
        remap = np.full(len(C), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        lab = remap[lab]
        C = C[used]
        k = len(used)
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
    sim_c = (X * C[lab]).sum(axis=1)              # かたまりの中心との近さ
    if suspicion is None:
        g = X.mean(axis=0)
        g /= max(float(np.linalg.norm(g)), 1e-9)
        suspicion = 1.0 - X @ g
    susp = np.asarray(suspicion, dtype=np.float64)
    o = np.lexsort((susp, lab))                    # かたまりごとに怪しさ順
    cnt = np.bincount(lab, minlength=k)
    start = np.concatenate(([0], np.cumsum(cnt)[:-1]))
    sl = susp[o]
    gscore = 0.5 * (sl[start + (cnt - 1) // 2] + sl[start + cnt // 2])   # 中央値
    gorder = np.argsort(-gscore, kind="stable")    # 怪しいかたまりから
    gpos = np.empty(k, dtype=np.int64)
    gpos[gorder] = np.arange(k)
    order = np.lexsort((-sim_c, gpos[lab]))
    return P[order], gpos[lab][order]


def _knn_within(metric, FS: Feats, t: float, K: int, progress=None) -> np.ndarray:
    """各顔の近い順 K 件（t より遠いものは -1）。"""
    n = len(FS)
    k = min(K, n - 1)
    nbr = np.full((n, K), -1, dtype=np.int64)
    for a in range(0, n, 1024):
        rows_ = np.arange(a, min(a + 1024, n))
        D = mdiff(metric, FS.take(rows_), FS)
        D[np.arange(len(rows_)), rows_] = np.inf
        if k > 0:
            idx = np.argpartition(D, k - 1, axis=1)[:, :k]
            val = np.take_along_axis(D, idx, 1)
            order = np.argsort(val, axis=1, kind="stable")
            idx = np.take_along_axis(idx, order, 1)
            val = np.take_along_axis(val, order, 1)
            idx[val > t] = -1
            nbr[rows_, :k] = idx
        if progress:
            progress("候補づくり", int(rows_[-1]) + 1, n)
    return nbr


def _membership(nbr: np.ndarray, label: np.ndarray, K: int) -> np.ndarray:
    """近い顔（まとまりの大きさ - 1 件まで）のうち、同じまとまりにいる割合。
    まとまりの芯にいる顔は 1 に近く、別々のまとまりの間に立つ「橋」の顔は低い。"""
    n = len(label)
    out = np.zeros(n, dtype=np.float32)
    has = label >= 0
    if not has.any():
        return out
    sizes = Counter(label[has].tolist())
    kk = np.array([min(K, sizes[int(l)] - 1) if l >= 0 else 0 for l in label], dtype=np.int64)
    valid = (nbr >= 0) & (np.arange(nbr.shape[1])[None, :] < kk[:, None])
    lab_nb = np.where(nbr >= 0, label[np.clip(nbr, 0, None)], -1)
    same = valid & (lab_nb == label[:, None])
    cnt = valid.sum(axis=1)
    out[cnt > 0] = same.sum(axis=1)[cnt > 0] / cnt[cnt > 0]
    out[~has] = 0.0
    return out


def _group_by_links(n: int, nbr: np.ndarray, K: int, min_s: int, share: int,
                    keep: float = 0.5, rounds: int = 5) -> np.ndarray:
    """互いに近く、近所の顔ぶれも重なる 2 枚だけをつなぎ、まとまりを作る。

    つないだあと、近所の半分以上が同じまとまりにいない顔（橋）を外して
    つなぎ直す。崩れた絵やぼやけた絵は誰にでも少しずつ近いので、
    外さないと別々のキャラを 1 つのまとまりにつないでしまう。"""
    nset = [set(int(x) for x in row if x >= 0) for row in nbr]
    core = np.array([len(s) >= min_s for s in nset], dtype=bool)
    edges = []
    for i in np.where(core)[0]:
        for j in nset[i]:
            if j > i and core[j] and int(i) in nset[j] and len(nset[i] & nset[j]) >= share - 1:
                edges.append((int(i), j))
    alive = core.copy()
    banned = np.zeros(n, dtype=bool)

    def labels() -> np.ndarray:
        parent = np.arange(n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in edges:
            if alive[i] and alive[j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
        lab = np.full(n, -1, dtype=np.int64)
        for i in np.where(alive)[0]:
            lab[i] = find(int(i))
        # 芯でない顔は、いちばん近い芯のまとまりに付ける
        for i in np.where(~core & ~banned)[0]:
            for j in nbr[i]:
                if j >= 0 and alive[j]:
                    lab[i] = lab[j]
                    break
        # 1 枚だけのまとまりは、まとまりとみなさない
        sizes = Counter(lab[lab >= 0].tolist())
        for i in np.where(lab >= 0)[0]:
            if sizes[int(lab[i])] < 2:
                lab[i] = -1
        return lab

    lab = labels()
    for _ in range(rounds):
        m = _membership(nbr, lab, K)
        drop = np.where((lab >= 0) & (m < keep))[0]
        if not len(drop):
            break
        banned[drop] = True
        alive[drop] = False
        lab = labels()
    lab[banned] = -1
    return lab


def build_proposals(con, eng: Engine = None, sample_cap: int = 20000,
                    min_size: int = 8, progress=None, log=None) -> dict:
    """名前の付いていない顔を、同じキャラらしいまとまりに分けて候補にする。

    まとまりの純度を何より優先する（候補から外れた顔は、名前が付いたあと
    判定で拾われるので失われない。混ざった顔は、人が 1 枚ずつ外す手間になる）。
      1. つなぐのは「互いに相手を近いと見ていて、近所の顔ぶれも重なる」2 枚だけ
      2. 近所の半分以上がよそのまとまりにいる顔（橋）は外す
      3. 髪色やツノの色がはっきり二手に分かれていれば分ける
      4. まとまりの大半と違う特徴語を書いている絵、色が大きく外れる顔は外す
    同じキャラが絵柄違いで 2 つの候補に分かれることはあるが、
    その場合は「既存のキャラに追加」で 1 人にまとめられる。"""
    eng = eng or ENGINE
    t0 = time.time()
    eng.ensure(con, log=log)
    store, metric = eng.store, eng.metric
    t_link = float(eng.base["auto"])
    assigned = {r[0] for r in con.execute("SELECT instance_id FROM c_assign")}
    dismissed = {r[0] for r in con.execute("SELECT instance_id FROM c_dismissed")}
    pool = np.array([p for p, iid in enumerate(store.ids)
                     if int(iid) not in assigned and int(iid) not in dismissed], dtype=np.int64)
    result = {"pool": int(len(pool)), "proposals": 0}

    def save(props):
        # 作り直しても、同じまとまりには同じ番号を付ける（見ている途中で番号が
        # 入れ替わると、どの候補を見ていたのか分からなくなる）。
        old: dict[int, set] = {}
        for pid_, iid_ in con.execute("SELECT proposal_id, instance_id FROM c_proposal_members"):
            old.setdefault(int(pid_), set()).add(int(iid_))
        nxt = max([int(meta_get(con, "chara_proposal_next") or 1)] + [k + 1 for k in old])
        members_ids = [{int(store.ids[p]) for p in pr["members"]} for pr in props]
        pairs = []
        for k, m in enumerate(members_ids):
            for pid_, om in old.items():
                inter = len(m & om)
                if inter and inter >= 0.5 * min(len(m), len(om)):
                    pairs.append((inter / len(m | om), k, pid_))
        pairs.sort(reverse=True)
        given, used = {}, set()
        for _, k, pid_ in pairs:
            if k not in given and pid_ not in used:
                given[k] = pid_
                used.add(pid_)
        for k in range(len(props)):
            if k not in given:
                given[k] = nxt
                nxt += 1
        now = time.time()
        with con:
            con.execute("DELETE FROM c_proposal_members")
            con.execute("DELETE FROM c_proposals")
            for k, pr in enumerate(props):
                pid = given[k]
                con.execute("INSERT INTO c_proposals(id,n,hints,created_at) VALUES(?,?,?,?)",
                            (pid, len(pr["members"]), json.dumps(pr["hints"], ensure_ascii=False),
                             now))
                gi = {int(p): (int(g), r) for r, (p, g) in enumerate(zip(pr["gpos"], pr["ggrp"]))}
                con.executemany(
                    "INSERT INTO c_proposal_members(proposal_id,instance_id,rank,grp,grank) "
                    "VALUES(?,?,?,?,?)",
                    [(pid, int(store.ids[p]), r, gi[int(p)][0], gi[int(p)][1])
                     for r, p in enumerate(pr["members"])])
            meta_set(con, "chara_proposal_next", nxt)
        meta_set(con, "chara_proposals_at", {"at": time.time(), "pool": int(len(pool))})
        GENS["proposals"] += 1
        con.commit()

    if len(pool) < min_size:
        save([])
        result["elapsed"] = round(time.time() - t0, 2)
        return result

    # 大きいときは標本で骨組みを作り、残りを後から振り分ける
    if len(pool) > sample_cap:
        keyed = sorted(pool.tolist(), key=lambda p: (int(store.ids[p]) * 2654435761) % 4294967296)
        S = np.array(sorted(keyed[:sample_cap]), dtype=np.int64)
    else:
        S = pool
    FS = store.feats(S)
    K, MIN_S, SHARE = 15, 4, 4
    nbr = _knn_within(metric, FS, t_link, K, progress=progress)
    label = _group_by_links(len(S), nbr, K, MIN_S, SHARE)
    groups: dict[int, list] = {}
    for i, lb in enumerate(label):
        if lb >= 0:
            groups.setdefault(int(lb), []).append(i)
    groups = {g: v for g, v in groups.items() if len(v) >= max(MIN_S, min_size // 2)}
    members = {g: [int(S[i]) for i in v] for g, v in groups.items()}

    # 標本に入らなかった顔: 近い順 K 件の半分以上が同じまとまりなら、そこへ入れる
    if len(pool) > len(S) and groups:
        grouped = np.array([i for v in groups.values() for i in v], dtype=np.int64)
        glab = label[grouped]
        FG = FS.take(grouped)
        rest = np.setdiff1d(pool, S)
        for a in range(0, len(rest), 1024):
            blk = rest[a:a + 1024]
            D = mdiff(metric, store.feats(blk), FG)
            k = min(K, D.shape[1])
            idx = np.argpartition(D, k - 1, axis=1)[:, :k]
            val = np.take_along_axis(D, idx, 1)
            # 近い順 k 件のうち、いちばん多いまとまりとその数（1 行ずつ回さずにまとめて数える）
            L = np.where(val <= t_link, glab[idx], -1)
            cnt = (L[:, :, None] == L[:, None, :]).sum(axis=2)
            cnt[L < 0] = 0
            j = np.argmax(cnt, axis=1)
            best = L[np.arange(len(blk)), j]
            bc = cnt[np.arange(len(blk)), j]
            for r in np.where((best >= 0) & (bc >= 2) & (bc >= 0.5 * k))[0]:
                members[int(best[r])].append(int(blk[r]))
            if progress:
                progress("候補づくり（残り）", min(a + 1024, len(rest)), len(rest))

    img_words = _image_words(con)
    img_loras = _image_loras(con)
    props = []
    for g, ps in members.items():
        for part in _split_by_words(metric, store, ps, img_words, t_link):
            # 外すと残りの顔ぶれの比率が変わるので、変わらなくなるまで繰り返す
            for _ in range(4):
                n0 = len(part)
                part = _clean_by_color(store, _clean_by_words(store, part, img_words))
                if len(part) == n0 or len(part) < min_size:
                    break
            if len(part) < min_size:
                continue
            order = _order_by_density(metric, store, part)
            # 似た顔のかたまり（芯から遠い顔が多いかたまりほど先）
            gpos, ggrp = similarity_groups(store, np.asarray(order, dtype=np.int64),
                                           suspicion=np.arange(len(order), dtype=np.float64),
                                           seed=len(props))
            props.append({"members": order, "gpos": gpos, "ggrp": ggrp,
                          "hints": _hints(store, order, img_words, img_loras)})
    props.sort(key=lambda p: -len(p["members"]))
    save(props)
    result["proposals"] = len(props)
    result["elapsed"] = round(time.time() - t0, 2)
    return result


def _order_by_density(metric, store: Store, part: list, cap: int = 3000) -> list:
    """まとまりの中で、仲間に囲まれている顔ほど前に。最初のページが一番確かな顔になる。"""
    P = np.asarray(part, dtype=np.int64)
    if len(P) <= 2:
        return [int(x) for x in P]
    ref = P
    if len(P) > cap:
        ref = np.sort(np.random.default_rng(0).choice(P, cap, replace=False))
    FR = store.feats(ref)
    where = {int(x): j for j, x in enumerate(ref)}
    key = np.empty(len(P), dtype=np.float64)
    for a in range(0, len(P), 2048):
        blk = P[a:a + 2048]
        D = mdiff(metric, store.feats(blk), FR)
        for r, x in enumerate(blk):
            j = where.get(int(x))
            if j is not None:
                D[r, j] = np.inf
        key[a:a + len(blk)] = _robust(D, 5)
    return [int(P[i]) for i in np.argsort(key, kind="stable")]


def _clean_by_words(store: Store, ps: list, img_words: dict, min_stated: int = 8,
                    top_share: float = 0.6, minor_share: float = 0.2) -> list:
    """まとまりの大半（6 割以上）が同じ髪色・目の色・ツノを書いているのに、
    それと違う少数派の語だけを書いている絵は、候補から外す。
    （外した顔は失われない。名前が付いたあと、判定で拾われる）"""
    if len(ps) < min_stated:
        return ps
    drop = set()
    for slot in ("hair", "eyes", "horns"):
        vals = {}
        for p in ps:
            if not store.solo[p]:
                continue
            w = img_words.get(int(store.image[p]), {})
            if slot == "hair" and "hair_multi" in w.get("flags", ()):
                continue
            v = w.get(slot)
            if v:
                vals[p] = set(v)
        if len(vals) < min_stated:
            continue
        cnt = Counter(x for s in vals.values() for x in s)
        top, c = cnt.most_common(1)[0]
        if c / len(vals) < top_share:
            continue
        for p, s in vals.items():
            if top not in s and all(cnt[x] / len(vals) <= minor_share for x in s):
                drop.add(p)
    return [p for p in ps if p not in drop] if drop else ps


def _clean_by_color(store: Store, ps: list, min_n: int = 12) -> list:
    """まとまりの色の広がりから大きく外れた顔を外す（CCIP は色に鈍い）。
    「普段の広がり」は、まとまりの中で色の近い半分から測る。"""
    cp = [p for p in ps if store.has_col[p]]
    if len(cp) < min_n:
        return ps
    X = store.Xcol[cp].astype(np.float32)
    n = len(cp)
    rng = np.random.default_rng(0)
    sel = np.arange(n) if n <= 1500 else np.sort(rng.choice(n, 1500, replace=False))
    R = X[sel]
    rn = (R * R).sum(1)
    r = np.empty(n, dtype=np.float32)
    pos = np.full(n, -1, dtype=np.int64)
    pos[sel] = np.arange(len(sel))
    for a in range(0, n, 8192):
        B = X[a:a + 8192]
        d2 = (B * B).sum(1)[:, None] + rn[None, :] - 2.0 * (B @ R.T)
        D = np.sqrt(np.maximum(d2, 0))
        own = pos[a:a + 8192]
        rows = np.where(own >= 0)[0]
        D[rows, own[rows]] = np.inf
        r[a:a + 8192] = _robust(D, 3)
    core = r <= np.median(r)
    tau = 2.5 * float(np.percentile(r[core], 90)) if core.any() else np.inf
    tau = max(tau, 1e-6)
    drop = {cp[i] for i in np.where(r > tau)[0]}
    return [p for p in ps if p not in drop] if drop else ps


def _split_by_words(metric, store, ps: list, img_words: dict, link: float) -> list:
    """CCIP は色に鈍いので、髪色やツノの色がはっきり二手に分かれる候補は分ける。"""
    if len(ps) < 16:
        return [ps]
    for slot in ("hair", "horns", "eyes"):
        vals = {}
        for p in ps:
            if not store.solo[p]:
                continue
            w = img_words.get(int(store.image[p]), {})
            if slot == "hair" and "hair_multi" in w.get("flags", ()):
                continue
            v = w.get(slot)
            if v and len(v) == 1:
                vals[p] = v[0]
        if len(vals) < 12:
            continue
        cnt = Counter(vals.values())
        major = [v for v, c in cnt.items() if c >= 6 and c / len(vals) >= 0.25]
        if len(major) < 2:
            continue
        parts = {v: [p for p, x in vals.items() if x == v] for v in major}
        cents = {v: metric.prep(store.rows(pp).mean(axis=0, keepdims=True))
                 for v, pp in parts.items()}
        for p in ps:
            if p in vals and vals[p] in parts:
                continue
            # 色を書いていない絵。どちらにも近いなら（CCIP では見分けられない組）、
            # 当て推量で入れずに候補から外しておく。名前が付いたあと、
            # 判定で「複数のキャラに近い」として確認待ちに回る。
            x = store.rows([p])
            d = {v: float(metric.diff(x, c)[0, 0]) for v, c in cents.items()}
            near = [v for v, dv in d.items() if dv <= link]
            if len(near) >= 2:
                continue
            parts[min(d, key=d.get)].append(p)
        out = []
        for v, pp in parts.items():
            out += (_split_by_words(metric, store, pp, img_words, link)
                    if len(pp) < len(ps) else [pp])
        return out
    return [ps]


def _hints(store: Store, members: list, img_words: dict, img_loras: dict) -> dict:
    solo = [p for p in members if store.solo[p]]
    words = Counter()
    stated = Counter()
    for p in solo:
        w = img_words.get(int(store.image[p]), {})
        for s in SLOTS:
            if w.get(s):
                stated[s] += 1
                for v in w[s]:
                    words[(s, v)] += 1
    wl = []
    for (s, v), c in words.most_common():
        if stated[s] and c / stated[s] >= 0.3 and c >= 3:
            wl.append({"slot": s, "value": v, "share": round(c / stated[s], 2),
                       "label": words_label({s: [v]})[0]})
    # 髪色などが 2 通り以上に割れている = 別人の寄せ集めかもしれない
    mixed = sorted({w["slot"] for w in wl if sum(1 for x in wl if x["slot"] == w["slot"]) >= 2
                    and w["slot"] in ("hair", "eyes", "horns")})
    lo = Counter()
    imgs = {int(store.image[p]) for p in members}
    for i in imgs:
        for name in img_loras.get(i, ()):
            lo[name] += 1
    ll = [{"name": k, "share": round(c / max(len(imgs), 1), 2)}
          for k, c in lo.most_common(3) if c / max(len(imgs), 1) >= 0.25]
    return {"n": len(members), "solo": len(solo), "words": wl[:6], "loras": ll,
            "mixed": [SLOT_LABEL[x] for x in mixed]}


# --------------------------------------------------------------------------
# 人の操作（すべて取り消せる）
# --------------------------------------------------------------------------

def _snapshot(con, instance_ids=(), char_ids=(), whole_chars=()) -> dict:
    """操作の前の状態を写しておく。取り消しはこれを書き戻すだけ。"""
    iids = sorted({int(x) for x in instance_ids})
    cids = sorted({int(x) for x in char_ids} | {int(x) for x in whole_chars})
    snap = {"iids": iids, "cids": cids, "whole": sorted({int(x) for x in whole_chars}),
            "assign": [], "reject": [], "dismissed": [], "chars": []}
    cols = "instance_id,character_id,state,score,margin,reason,source,updated_at"
    for ch in _chunks(iids):
        snap["assign"] += [list(r) for r in con.execute(
            f"SELECT {cols} FROM c_assign WHERE instance_id IN ({_qm(len(ch))})", ch)]
        snap["dismissed"] += [list(r) for r in con.execute(
            f"SELECT instance_id,created_at FROM c_dismissed WHERE instance_id IN ({_qm(len(ch))})", ch)]
        if cids:
            snap["reject"] += [list(r) for r in con.execute(
                f"SELECT instance_id,character_id,created_at FROM c_reject "
                f"WHERE instance_id IN ({_qm(len(ch))}) AND character_id IN ({_qm(len(cids))})",
                ch + cids)]
    for cid in snap["whole"]:
        snap["assign"] += [list(r) for r in con.execute(
            f"SELECT {cols} FROM c_assign WHERE character_id=?", (cid,))]
        snap["reject"] += [list(r) for r in con.execute(
            "SELECT instance_id,character_id,created_at FROM c_reject WHERE character_id=?", (cid,))]
    for cid in cids:
        r = con.execute("SELECT id,name,note,words,created_at FROM c_characters WHERE id=?",
                        (cid,)).fetchone()
        snap["chars"].append([cid, list(r) if r else None])
    return snap


def _record(con, kind: str, label: str, snap: dict) -> int:
    # 何万枚もの操作の控えは大きいので縮めて置く（取り消しのときに戻す）
    blob = zlib.compress(json.dumps(snap, ensure_ascii=False).encode("utf-8"), 6)
    cur = con.execute("INSERT INTO c_actions(at,kind,label,undo) VALUES(?,?,?,?)",
                      (time.time(), kind, label, blob))
    con.execute("DELETE FROM c_actions WHERE id NOT IN "
                "(SELECT id FROM c_actions ORDER BY id DESC LIMIT 100)")
    ENGINE.gen += 1
    return cur.lastrowid


def undo_last(con) -> dict:
    r = con.execute("SELECT id, kind, label, undo FROM c_actions ORDER BY id DESC LIMIT 1").fetchone()
    if not r:
        return {"ok": False, "error": "取り消せる操作がありません"}
    raw = r["undo"]
    snap = json.loads(zlib.decompress(raw).decode("utf-8") if isinstance(raw, bytes) else raw)
    with con:
        iids, cids, whole = snap["iids"], snap["cids"], snap["whole"]
        for ch in _chunks(iids):
            con.execute(f"DELETE FROM c_assign WHERE instance_id IN ({_qm(len(ch))})", ch)
            con.execute(f"DELETE FROM c_dismissed WHERE instance_id IN ({_qm(len(ch))})", ch)
            if cids:
                con.execute(f"DELETE FROM c_reject WHERE instance_id IN ({_qm(len(ch))}) "
                            f"AND character_id IN ({_qm(len(cids))})", ch + cids)
        for cid in whole:
            con.execute("DELETE FROM c_assign WHERE character_id=?", (cid,))
            con.execute("DELETE FROM c_reject WHERE character_id=?", (cid,))
        for cid, row in snap["chars"]:
            con.execute("DELETE FROM c_characters WHERE id=?", (cid,))
            if row:
                con.execute("INSERT INTO c_characters(id,name,note,words,created_at) "
                            "VALUES(?,?,?,?,?)", row)
            else:
                con.execute("DELETE FROM c_assign WHERE character_id=?", (cid,))
                con.execute("DELETE FROM c_reject WHERE character_id=?", (cid,))
        con.executemany("INSERT OR REPLACE INTO c_assign(instance_id,character_id,state,score,"
                        "margin,reason,source,updated_at) VALUES(?,?,?,?,?,?,?,?)", snap["assign"])
        con.executemany("INSERT OR REPLACE INTO c_reject(instance_id,character_id,created_at) "
                        "VALUES(?,?,?)", snap["reject"])
        con.executemany("INSERT OR REPLACE INTO c_dismissed(instance_id,created_at) VALUES(?,?)",
                        snap["dismissed"])
        con.execute("DELETE FROM c_assign WHERE character_id NOT IN (SELECT id FROM c_characters)")
        con.execute("DELETE FROM c_reject WHERE character_id NOT IN (SELECT id FROM c_characters)")
        con.execute("DELETE FROM c_actions WHERE id=?", (r["id"],))
    ENGINE.gen += 1
    return {"ok": True, "undone": r["label"]}


def last_action(con) -> dict | None:
    r = con.execute("SELECT id, at, kind, label FROM c_actions ORDER BY id DESC LIMIT 1").fetchone()
    return dict(r) if r else None


def _character_id(con, target, create: bool = True) -> tuple[int, bool]:
    """target は {"id": n} か {"name": "..."}。名前が新しければ作る。"""
    if isinstance(target, dict) and target.get("id") is not None:
        cid = int(target["id"])
        if not con.execute("SELECT 1 FROM c_characters WHERE id=?", (cid,)).fetchone():
            raise ValueError("そのキャラは見つかりません")
        return cid, False
    name = str((target or {}).get("name") if isinstance(target, dict) else target or "").strip()
    if not name:
        raise ValueError("名前を入れてください")
    r = con.execute("SELECT id FROM c_characters WHERE name=?", (name,)).fetchone()
    if r:
        return int(r[0]), False
    if not create:
        raise ValueError("そのキャラは見つかりません")
    cid = (con.execute("SELECT COALESCE(MAX(id),0)+1 FROM c_characters").fetchone()[0])
    return int(cid), True


def act(con, kind: str, body: dict) -> dict:
    """画面からの操作。どれも 1 回の取り消しで元に戻る。"""
    now = time.time()
    ids = [int(x) for x in (body.get("ids") or [])]
    if kind == "assign":
        # 名前を付けて確定（新しいキャラでも既存でもよい）。外した顔は否にする。
        confirm = [int(x) for x in (body.get("confirm") or ids)]
        reject = [int(x) for x in (body.get("reject") or [])]
        if not confirm:
            return {"ok": False, "error": "確定する顔がありません"}
        cid, new = _character_id(con, body.get("to"))
        name = (body.get("to") or {}).get("name") if new else None
        snap = _snapshot(con, confirm + reject, [cid])
        with con:
            if new:
                con.execute("INSERT INTO c_characters(id,name,created_at) VALUES(?,?,?)",
                            (cid, name, now))
            _confirm(con, confirm, cid, now)
            _reject(con, reject, cid, now)
            label = (f"「{_cname(con, cid)}」に {len(confirm)} 枚を確定"
                     + (f"・{len(reject)} 枚を否" if reject else ""))
            _record(con, kind, label, snap)
        return {"ok": True, "character_id": cid, "created": new, "label": label}

    if kind in ("confirm", "reject", "move"):
        if not ids:
            return {"ok": False, "error": "顔が選ばれていません"}
        cid, new = _character_id(con, body.get("to"), create=(kind == "move"))
        cur_chars = {r[0] for ch in _chunks(ids) for r in con.execute(
            f"SELECT character_id FROM c_assign WHERE instance_id IN ({_qm(len(ch))})", ch)}
        snap = _snapshot(con, ids, {cid} | cur_chars)
        with con:
            if new:
                con.execute("INSERT INTO c_characters(id,name,created_at) VALUES(?,?,?)",
                            (cid, str(body["to"]["name"]).strip(), now))
            if kind == "confirm":
                _confirm(con, ids, cid, now)
                label = f"「{_cname(con, cid)}」に {len(ids)} 枚を確定"
            elif kind == "reject":
                _reject(con, ids, cid, now)
                label = f"{len(ids)} 枚を「{_cname(con, cid)}」ではないとした"
            else:
                src = {}
                for ch in _chunks(ids):
                    for iid, c in con.execute(
                            f"SELECT instance_id, character_id FROM c_assign "
                            f"WHERE instance_id IN ({_qm(len(ch))})", ch):
                        src[iid] = c
                _confirm(con, ids, cid, now)
                for iid, c in src.items():
                    if c != cid:
                        con.execute("INSERT OR IGNORE INTO c_reject(instance_id,character_id,"
                                    "created_at) VALUES(?,?,?)", (iid, c, now))
                label = f"{len(ids)} 枚を「{_cname(con, cid)}」へ移した"
            _record(con, kind, label, snap)
        return {"ok": True, "character_id": cid, "label": label}

    if kind == "unassign":
        if not ids:
            return {"ok": False, "error": "顔が選ばれていません"}
        snap = _snapshot(con, ids)
        with con:
            for ch in _chunks(ids):
                con.execute(f"DELETE FROM c_assign WHERE instance_id IN ({_qm(len(ch))})", ch)
            label = f"{len(ids)} 枚を未分類に戻した"
            _record(con, kind, label, snap)
        return {"ok": True, "label": label}

    if kind == "scope":
        # 一覧の範囲をまとめて決める（何万枚でも 1 回の操作・1 回の取り消し）
        verb = str(body.get("verb") or "confirm")
        target, rej, src = scope_ids(con, body)
        if verb == "dismiss":
            ids2 = target + [i for i in rej if i not in set(target)]
            if not ids2:
                return {"ok": False, "error": "顔がありません"}
            snap = _snapshot(con, ids2)
            with con:
                con.executemany("INSERT OR IGNORE INTO c_dismissed(instance_id,created_at) "
                                "VALUES(?,?)", [(i, now) for i in ids2])
                label = f"{len(ids2)} 枚を候補から外した"
                _record(con, kind, label, snap)
            return {"ok": True, "label": label, "n": len(ids2)}
        if not target and not rej:
            return {"ok": False, "error": "対象の顔がありません"}
        if verb == "keep":
            cid = int(src.get("id"))
            snap = _snapshot(con, target + rej, [cid])
            with con:
                for ch in _chunks(target):
                    con.execute(f"UPDATE c_assign SET reason='見直し済み' WHERE state='confirmed' "
                                f"AND instance_id IN ({_qm(len(ch))})", ch)
                _reject(con, rej, cid, now)
                label = (f"{len(target)} 枚を見直し済みにした"
                         + (f"・{len(rej)} 枚を否" if rej else ""))
                _record(con, kind, label, snap)
            return {"ok": True, "label": label}
        if verb == "reject":
            cid, _ = _character_id(con, body.get("to"), create=False)
            ids2 = target + rej
            snap = _snapshot(con, ids2, [cid])
            with con:
                _reject(con, ids2, cid, now)
                label = f"{len(ids2)} 枚を「{_cname(con, cid)}」ではないとした"
                _record(con, kind, label, snap)
            return {"ok": True, "label": label}
        # confirm（新しい名前なら作る）
        if not target:
            # 全部に × ＝「この中にこの子は 1 枚もいない」。すでにいるキャラなら、全部を否にする
            try:
                cid0, _ = _character_id(con, body.get("to"), create=False)
            except Exception:
                cid0 = None
            if not cid0:
                return {"ok": False, "error": "全部に × が付いています。まだいない名前なので、確定する顔が要ります"}
            snap = _snapshot(con, rej, [cid0])
            with con:
                _reject(con, rej, cid0, now)
                label = f"{len(rej):,} 枚を「{_cname(con, cid0)}」ではないとした"
                _record(con, kind, label, snap)
            return {"ok": True, "character_id": cid0, "created": False, "label": label,
                    "n": 0, "rejected": len(rej)}
        cid, new = _character_id(con, body.get("to"))
        cur_chars = set()
        for ch in _chunks(target + rej):
            cur_chars |= {r[0] for r in con.execute(
                f"SELECT character_id FROM c_assign WHERE instance_id IN ({_qm(len(ch))})", ch)}
        snap = _snapshot(con, target + rej, {cid} | cur_chars)
        with con:
            if new:
                con.execute("INSERT INTO c_characters(id,name,created_at) VALUES(?,?,?)",
                            (cid, str((body.get("to") or {}).get("name")).strip(), now))
            _confirm(con, target, cid, now)
            _reject(con, rej, cid, now)
            label = (f"「{_cname(con, cid)}」に {len(target):,} 枚を確定"
                     + (f"・{len(rej):,} 枚を否" if rej else ""))
            _record(con, kind, label, snap)
        return {"ok": True, "character_id": cid, "created": new, "label": label,
                "n": len(target), "rejected": len(rej)}

    if kind == "unreject":
        # 「このキャラではない」を取り消す（確定はしない。判定に任せる）
        cid = int((body.get("to") or {}).get("id") or body.get("id") or 0)
        if not ids or not cid:
            return {"ok": False, "error": "顔かキャラが選ばれていません"}
        snap = _snapshot(con, ids, [cid])
        with con:
            for ch in _chunks(ids):
                con.execute(f"DELETE FROM c_reject WHERE character_id=? AND instance_id IN "
                            f"({_qm(len(ch))})", [cid] + ch)
            label = f"{len(ids)} 枚の「{_cname(con, cid)}ではない」を取り消した"
            _record(con, kind, label, snap)
        return {"ok": True, "label": label}

    if kind == "keep":
        # 「浮いている」と出た確定の顔を、見直したうえで本人だとする
        if not ids:
            return {"ok": False, "error": "顔が選ばれていません"}
        snap = _snapshot(con, ids)
        with con:
            for ch in _chunks(ids):
                con.execute(f"UPDATE c_assign SET reason='見直し済み' WHERE state='confirmed' "
                            f"AND instance_id IN ({_qm(len(ch))})", ch)
            label = f"{len(ids)} 枚を見直し済みにした"
            _record(con, kind, label, snap)
        return {"ok": True, "label": label}

    if kind in ("dismiss", "undismiss"):
        if not ids:
            return {"ok": False, "error": "顔が選ばれていません"}
        snap = _snapshot(con, ids)
        with con:
            if kind == "dismiss":
                con.executemany("INSERT OR IGNORE INTO c_dismissed(instance_id,created_at) "
                                "VALUES(?,?)", [(i, now) for i in ids])
                label = f"{len(ids)} 枚を候補から外した"
            else:
                for ch in _chunks(ids):
                    con.execute(f"DELETE FROM c_dismissed WHERE instance_id IN ({_qm(len(ch))})", ch)
                label = f"{len(ids)} 枚を候補に戻した"
            _record(con, kind, label, snap)
        return {"ok": True, "label": label}

    if kind in ("rename", "note", "words"):
        cid = int(body.get("id"))
        snap = _snapshot(con, (), [cid])
        with con:
            if kind == "rename":
                name = str(body.get("name") or "").strip()
                if not name:
                    return {"ok": False, "error": "名前を入れてください"}
                other = con.execute("SELECT id FROM c_characters WHERE name=? AND id<>?",
                                    (name, cid)).fetchone()
                if other:
                    return {"ok": False, "error": f"「{name}」はすでにあります。まとめるなら「統合」を使ってください"}
                old = _cname(con, cid)
                con.execute("UPDATE c_characters SET name=? WHERE id=?", (name, cid))
                label = f"「{old}」を「{name}」に改名"
            elif kind == "note":
                con.execute("UPDATE c_characters SET note=? WHERE id=?",
                            (str(body.get("note") or "") or None, cid))
                label = f"「{_cname(con, cid)}」のメモを変更"
            else:
                w = body.get("words")
                if isinstance(w, list):
                    w = ", ".join(str(x).strip() for x in w if str(x).strip())
                con.execute("UPDATE c_characters SET words=? WHERE id=?", ((w or None), cid))
                label = f"「{_cname(con, cid)}」の特徴語を変更"
            _record(con, kind, label, snap)
        return {"ok": True, "label": label}

    if kind == "merge":
        src, dst = int(body.get("src")), int(body.get("dst"))
        if src == dst:
            return {"ok": False, "error": "同じキャラです"}
        snap = _snapshot(con, (), [src, dst], whole_chars=[src, dst])
        with con:
            sname, dname = _cname(con, src), _cname(con, dst)
            # 確定は確定のまま移す。src で否にした顔は dst でも否にする（安全側）
            con.execute("UPDATE OR REPLACE c_assign SET character_id=? WHERE character_id=? "
                        "AND state='confirmed'", (dst, src))
            con.execute("DELETE FROM c_assign WHERE character_id=?", (src,))
            con.execute("INSERT OR IGNORE INTO c_reject(instance_id,character_id,created_at) "
                        "SELECT instance_id, ?, created_at FROM c_reject WHERE character_id=?",
                        (dst, src))
            con.execute("DELETE FROM c_reject WHERE character_id=?", (src,))
            con.execute("DELETE FROM c_reject WHERE character_id=? AND instance_id IN "
                        "(SELECT instance_id FROM c_assign WHERE character_id=? "
                        " AND state='confirmed')", (dst, dst))
            con.execute("DELETE FROM c_characters WHERE id=?", (src,))
            label = f"「{sname}」を「{dname}」に統合"
            _record(con, kind, label, snap)
        return {"ok": True, "label": label}

    if kind == "delete":
        cid = int(body.get("id"))
        snap = _snapshot(con, (), [cid], whole_chars=[cid])
        with con:
            name = _cname(con, cid)
            con.execute("DELETE FROM c_assign WHERE character_id=?", (cid,))
            con.execute("DELETE FROM c_reject WHERE character_id=?", (cid,))
            con.execute("DELETE FROM c_characters WHERE id=?", (cid,))
            label = f"「{name}」を削除"
            _record(con, kind, label, snap)
        return {"ok": True, "label": label}

    return {"ok": False, "error": f"知らない操作です: {kind}"}


def _cname(con, cid: int) -> str:
    r = con.execute("SELECT name FROM c_characters WHERE id=?", (cid,)).fetchone()
    return r[0] if r else f"#{cid}"


def _confirm(con, ids, cid, now):
    con.executemany("INSERT OR REPLACE INTO c_assign(instance_id,character_id,state,score,margin,"
                    "reason,source,updated_at) VALUES(?,?,'confirmed',NULL,NULL,NULL,'user',?)",
                    [(i, cid, now) for i in ids])
    for ch in _chunks(ids):
        con.execute(f"DELETE FROM c_reject WHERE character_id=? AND instance_id IN ({_qm(len(ch))})",
                    [cid] + ch)
        con.execute(f"DELETE FROM c_dismissed WHERE instance_id IN ({_qm(len(ch))})", ch)


def _reject(con, ids, cid, now):
    con.executemany("INSERT OR IGNORE INTO c_reject(instance_id,character_id,created_at) "
                    "VALUES(?,?,?)", [(i, cid, now) for i in ids])
    for ch in _chunks(ids):
        con.execute(f"DELETE FROM c_assign WHERE character_id=? AND instance_id IN ({_qm(len(ch))})",
                    [cid] + ch)


# --------------------------------------------------------------------------
# 画面向けの問い合わせ
# --------------------------------------------------------------------------

def list_characters(con) -> list[dict]:
    rows = {r["id"]: {"id": r["id"], "name": r["name"], "note": r["note"],
                      "words": r["words"], "confirmed": 0, "auto": 0, "review": 0,
                      "outliers": 0, "rejected": 0, "cover": []}
            for r in con.execute("SELECT id, name, note, words FROM c_characters")}
    for cid, st, n in con.execute("SELECT character_id, state, COUNT(*) FROM c_assign "
                                  "GROUP BY character_id, state"):
        if cid in rows and st in rows[cid]:
            rows[cid][st] = n
    for cid, n in con.execute("SELECT character_id, COUNT(*) FROM c_assign "
                              "WHERE state='confirmed' AND reason='浮いている' GROUP BY character_id"):
        if cid in rows:
            rows[cid]["outliers"] = n
    for cid, n in con.execute("SELECT character_id, COUNT(*) FROM c_reject GROUP BY character_id"):
        if cid in rows:
            rows[cid]["rejected"] = n
    for cid in rows:
        rows[cid]["cover"] = [r[0] for r in con.execute(
            "SELECT a.instance_id FROM c_assign a JOIN c_instances i ON i.id=a.instance_id "
            "WHERE a.character_id=? AND a.state='confirmed' "
            "ORDER BY i.solo DESC, a.updated_at DESC LIMIT 4", (cid,))]
    out = list(rows.values())
    out.sort(key=lambda r: -(r["confirmed"] + r["auto"]))
    return out


_FACE_COLS = ("i.id AS instance_id, i.image_id, i.solo, i.x1, i.y1, i.x2, i.y2, "
              "im.filename, im.mtime")


def faces_of(con, cid: int, state: str, order: str = "score", limit: int = 120,
             offset: int = 0) -> dict:
    limit = max(1, min(int(limit), 600))
    offset = max(0, int(offset))
    od = {"score": "a.score ASC, i.id DESC", "far": "a.score DESC, i.id DESC",
          "new": "im.mtime DESC", "old": "im.mtime ASC"}.get(order, "a.score ASC, i.id DESC")
    if state == "rejected":
        base = ("FROM c_reject r JOIN c_instances i ON i.id=r.instance_id "
                "JOIN images im ON im.id=i.image_id WHERE r.character_id=?")
        tot = con.execute(f"SELECT COUNT(*) {base}", (cid,)).fetchone()[0]
        od2 = {"new": "im.mtime DESC", "old": "im.mtime ASC"}.get(order, "r.created_at DESC")
        rows = con.execute(f"SELECT {_FACE_COLS}, 'rejected' AS state, NULL AS score, "
                           f"NULL AS reason {base} ORDER BY {od2} LIMIT ? OFFSET ?",
                           (cid, limit, offset)).fetchall()
    else:
        cond = "a.state=?"
        args = [cid, state]
        if state == "outlier":
            cond, args = "a.state='confirmed' AND a.reason='浮いている'", [cid]
        base = ("FROM c_assign a JOIN c_instances i ON i.id=a.instance_id "
                f"JOIN images im ON im.id=i.image_id WHERE a.character_id=? AND {cond}")
        tot = con.execute(f"SELECT COUNT(*) {base}", args).fetchone()[0]
        if state == "confirmed" and order == "score":
            od = "i.solo DESC, a.updated_at DESC, i.id DESC"
        rows = con.execute(f"SELECT {_FACE_COLS}, a.state, a.score, a.reason {base} "
                           f"ORDER BY {od} LIMIT ? OFFSET ?", args + [limit, offset]).fetchall()
    return {"total": tot, "offset": offset, "faces": [dict(r) for r in rows]}


def review_queue(con, cid: int | None = None, limit: int = 240, offset: int = 0) -> dict:
    limit = max(1, min(int(limit), 1000))
    where = "a.state='review'" + (" AND a.character_id=?" if cid else "")
    args = [cid] if cid else []
    tot = con.execute(f"SELECT COUNT(*) FROM c_assign a WHERE {where}", args).fetchone()[0]
    groups = [dict(r) for r in con.execute(
        f"SELECT a.character_id, c.name, COUNT(*) AS n FROM c_assign a "
        f"JOIN c_characters c ON c.id=a.character_id WHERE {where} "
        f"GROUP BY a.character_id ORDER BY n DESC", args)]
    rows = con.execute(
        f"SELECT {_FACE_COLS}, a.character_id, a.state, a.score, a.reason FROM c_assign a "
        f"JOIN c_instances i ON i.id=a.instance_id JOIN images im ON im.id=i.image_id "
        f"WHERE {where} ORDER BY a.character_id, a.score ASC LIMIT ? OFFSET ?",
        args + [limit, int(offset)]).fetchall()
    return {"total": tot, "groups": groups, "faces": [dict(r) for r in rows]}


def unassigned(con, limit: int = 120, offset: int = 0, order: str = "new",
               dismissed: bool = False) -> dict:
    """どのキャラにも入っていない顔。dismissed=True なら「見送った」顔。"""
    limit = max(1, min(int(limit), 600))
    od = {"new": "im.mtime DESC", "old": "im.mtime ASC", "big": "(i.x2-i.x1) DESC"}.get(
        order, "im.mtime DESC")
    base = ("FROM c_instances i JOIN images im ON im.id=i.image_id "
            "LEFT JOIN c_assign a ON a.instance_id=i.id "
            "LEFT JOIN c_dismissed d ON d.instance_id=i.id "
            "WHERE a.instance_id IS NULL AND d.instance_id IS "
            + ("NOT NULL" if dismissed else "NULL"))
    tot = con.execute(f"SELECT COUNT(*) {base}").fetchone()[0]
    rows = con.execute(f"SELECT {_FACE_COLS} {base} ORDER BY {od} LIMIT ? OFFSET ?",
                       (limit, int(offset))).fetchall()
    return {"total": tot, "faces": [dict(r) for r in rows]}


def similar_faces(con, iid: int, scope: str = "unassigned", limit: int = 120,
                  eng: Engine = None) -> dict:
    """ある顔に近い顔を近い順に。未分類の山から同じ子を拾い集めるときに使う。"""
    eng = eng or ENGINE
    eng.ensure(con)
    store, metric = eng.store, eng.metric
    if iid not in store.pos:
        return {"total": 0, "faces": [], "error": "この顔はまだ解析されていません"}
    limit = max(1, min(int(limit), 600))
    if scope == "all":
        cand = np.arange(len(store.ids))
    else:
        taken = {r[0] for r in con.execute("SELECT instance_id FROM c_assign")}
        taken |= {r[0] for r in con.execute("SELECT instance_id FROM c_dismissed")}
        cand = np.array([p for p, i in enumerate(store.ids) if int(i) not in taken],
                        dtype=np.int64)
    if not len(cand):
        return {"total": 0, "faces": []}
    x = store.feats([store.pos[iid]])
    d = np.empty(len(cand), dtype=np.float32)
    for a in range(0, len(cand), 20000):
        d[a:a + 20000] = mdiff(metric, x, store.feats(cand[a:a + 20000]))[0]
    order = np.argsort(d, kind="stable")[:limit + 1]
    ids = [int(store.ids[cand[k]]) for k in order if int(store.ids[cand[k]]) != iid][:limit]
    dist = {int(store.ids[cand[k]]): float(d[k]) for k in order}
    rows = {}
    for ch in _chunks(ids):
        for r in con.execute(f"SELECT {_FACE_COLS} FROM c_instances i JOIN images im "
                             f"ON im.id=i.image_id WHERE i.id IN ({_qm(len(ch))})", ch):
            rows[r["instance_id"]] = dict(r)
    faces = []
    for i in [iid] + ids:
        if i == iid:
            r = con.execute(f"SELECT {_FACE_COLS} FROM c_instances i JOIN images im "
                            f"ON im.id=i.image_id WHERE i.id=?", (iid,)).fetchone()
            if r:
                faces.append(dict(r) | {"score": 0.0, "seed": True})
            continue
        if i in rows:
            faces.append(rows[i] | {"score": round(dist[i], 4)})
    t_auto = float(eng.base["auto"]) if eng.base else None
    return {"total": len(faces), "faces": faces, "line": t_auto}



# --------------------------------------------------------------------------
# 一覧（候補・キャラ・未分類）: 並べ替え・ページ・かたまり
# --------------------------------------------------------------------------
# 何万枚もの一覧を、どの画面でも同じ仕組みで扱う。
#   src = {"type": "proposal", "id": 3, "order": "group"}
#         {"type": "char", "id": 5, "state": "auto", "order": "far"}
#         {"type": "unassigned", "order": "new", "dismissed": false}
# 並びは一度計算したら覚えておき、ページをめくるたびに計算し直さない。

GENS = {"machine": 0, "proposals": 0}
_LIST_CACHE: dict = {}
_LIST_LOCK = threading.Lock()

ORDERS = {
    "proposal": ("group", "sus", "rep", "new", "old", "name"),
    "char": ("group", "far", "score", "new", "old", "name"),
    "unassigned": ("new", "old", "big", "name", "group"),
}


def _src_norm(src: dict) -> dict:
    t = str(src.get("type") or "")
    if t not in ORDERS:
        raise ValueError(f"知らない一覧です: {t}")
    order = str(src.get("order") or ORDERS[t][0])
    if order not in ORDERS[t]:
        order = ORDERS[t][0]
    out = {"type": t, "order": order}
    if t in ("proposal", "char"):
        out["id"] = int(src.get("id"))
    if t == "char":
        st = str(src.get("state") or "confirmed")
        out["state"] = st if st in ("confirmed", "auto", "review", "rejected", "outlier") else "confirmed"
    if t == "unassigned":
        out["dismissed"] = bool(src.get("dismissed"))
    return out


def listing(con, src: dict, eng: Engine = None) -> dict:
    """一覧の中身（並べた顔の番号と、かたまり番号）。"""
    src = _src_norm(src)
    eng = eng or ENGINE
    inst = con.execute("SELECT COUNT(*), COALESCE(MAX(id),0) FROM c_instances").fetchone()
    token = (eng.gen, GENS["machine"], GENS["proposals"], int(inst[0]), int(inst[1]))
    key = json.dumps(src, sort_keys=True)
    with _LIST_LOCK:
        hit = _LIST_CACHE.get(key)
        if hit and hit[0] == token:
            return hit[1]
    t, order = src["type"], src["order"]
    grp = None
    extra = None
    if t == "proposal" and order == "group":
        if con.execute("SELECT 1 FROM c_proposal_members WHERE proposal_id=? AND grp IS NULL LIMIT 1",
                       (src["id"],)).fetchone():
            fill_proposal_groups(con, eng, src["id"])
            inst = con.execute("SELECT COUNT(*), COALESCE(MAX(id),0) FROM c_instances").fetchone()
            token = (eng.gen, GENS["machine"], GENS["proposals"], int(inst[0]), int(inst[1]))
    if t == "proposal":
        od = {"rep": "m.rank ASC", "sus": "m.rank DESC",
              "group": "COALESCE(m.grank, m.rank) ASC",
              "new": "im.mtime DESC, m.instance_id DESC", "old": "im.mtime ASC, m.instance_id ASC",
              "name": "im.rel_path ASC, m.instance_id ASC"}[order]
        rows = con.execute(
            "SELECT m.instance_id, COALESCE(m.grp, 0) FROM c_proposal_members m "
            "JOIN c_instances i ON i.id=m.instance_id JOIN images im ON im.id=i.image_id "
            "LEFT JOIN c_assign a ON a.instance_id=m.instance_id "
            "LEFT JOIN c_dismissed d ON d.instance_id=m.instance_id "
            f"WHERE m.proposal_id=? AND a.instance_id IS NULL AND d.instance_id IS NULL ORDER BY {od}",
            (src["id"],)).fetchall()
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        if order == "group":
            grp = np.array([r[1] for r in rows], dtype=np.int64)
    else:
        if t == "char":
            st = src["state"]
            if st == "rejected":
                base = ("FROM c_reject r JOIN c_instances i ON i.id=r.instance_id "
                        "JOIN images im ON im.id=i.image_id WHERE r.character_id=?")
                args = [src["id"]]
                score_col, upd = "NULL", "r.created_at"
            else:
                cond = ("a.state='confirmed' AND a.reason='浮いている'" if st == "outlier"
                        else "a.state=?")
                base = ("FROM c_assign a JOIN c_instances i ON i.id=a.instance_id "
                        f"JOIN images im ON im.id=i.image_id WHERE a.character_id=? AND {cond}")
                args = [src["id"]] + ([] if st == "outlier" else [st])
                score_col, upd = "a.score", "a.updated_at"
            ods = {"score": (f"i.solo DESC, {upd} DESC, i.id DESC" if st in ("confirmed", "rejected")
                             else f"{score_col} ASC, i.id DESC"),
                   "far": f"{score_col} DESC, i.id DESC", "group": f"{score_col} ASC, i.id DESC",
                   "new": "im.mtime DESC, i.id DESC", "old": "im.mtime ASC, i.id ASC",
                   "name": "im.rel_path ASC, i.id ASC"}
            rows = con.execute(f"SELECT i.id, {score_col} {base} ORDER BY {ods[order]}", args).fetchall()
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            extra = np.array([r[1] if r[1] is not None else np.nan for r in rows], dtype=np.float64)
        else:
            od = {"new": "im.mtime DESC, i.id DESC", "old": "im.mtime ASC, i.id ASC",
                  "big": "(i.x2-i.x1)*(i.y2-i.y1) DESC, i.id DESC",
                  "name": "im.rel_path ASC, i.id ASC", "group": "i.id ASC"}[order]
            rows = con.execute(
                "SELECT i.id FROM c_instances i JOIN images im ON im.id=i.image_id "
                "LEFT JOIN c_assign a ON a.instance_id=i.id "
                "LEFT JOIN c_dismissed d ON d.instance_id=i.id "
                "WHERE a.instance_id IS NULL AND d.instance_id IS "
                + ("NOT NULL" if src["dismissed"] else "NULL") + f" ORDER BY {od}").fetchall()
            ids = np.array([r[0] for r in rows], dtype=np.int64)
        if order == "group" and len(ids):
            ids, grp = _group_ids(con, eng, ids, extra)
    res = {"ids": ids, "grp": grp, "src": src,
           "n_groups": int(grp.max()) + 1 if grp is not None and len(grp) else 0}
    with _LIST_LOCK:
        _LIST_CACHE[key] = (token, res)
        while len(_LIST_CACHE) > 12:
            _LIST_CACHE.pop(next(iter(_LIST_CACHE)))
    return res


def fill_proposal_groups(con, eng: Engine = None, pid: int | None = None) -> int:
    """前の版（2.0）で作った候補には「かたまり」の列が無い。無いものだけ作って書き込む。
    （作り直すと候補の番号が変わるので、中身はそのままにして列だけ足す）"""
    eng = eng or ENGINE
    q = "SELECT DISTINCT proposal_id FROM c_proposal_members WHERE grp IS NULL"
    pids = [r[0] for r in con.execute(q + (" AND proposal_id=?" if pid is not None else ""),
                                      (() if pid is None else (int(pid),)))]
    if not pids:
        return 0
    wcon = connect()
    try:
        for p in pids:
            ids = np.array([r[0] for r in con.execute(
                "SELECT instance_id FROM c_proposal_members WHERE proposal_id=? ORDER BY rank", (p,))],
                dtype=np.int64)
            if not len(ids):
                continue
            oi, og = _group_ids(con, eng, ids, np.arange(len(ids), dtype=np.float64))
            rows = [(int(g), k, int(p), int(i)) for k, (i, g) in enumerate(zip(oi.tolist(), og.tolist()))]
            with wcon:
                wcon.executemany("UPDATE c_proposal_members SET grp=?, grank=? "
                                 "WHERE proposal_id=? AND instance_id=?", rows)
    finally:
        wcon.close()
    GENS["proposals"] += 1
    return len(pids)


def _group_ids(con, eng: Engine, ids: np.ndarray, score: np.ndarray | None):
    """特徴のある顔をかたまりに分ける。特徴の無い顔（まれ）は最後にまとめる。"""
    eng.ensure(con)
    store = eng.store
    pos = np.array([store.pos.get(int(i), -1) for i in ids], dtype=np.int64)
    have = pos >= 0
    susp = None
    if score is not None and np.isfinite(score[have]).any():
        sc = score[have].copy()
        sc[~np.isfinite(sc)] = np.nanmax(sc) if np.isfinite(sc).any() else 0
        susp = sc
    gp, gg = similarity_groups(store, pos[have], suspicion=susp)
    back = {int(p): int(i) for p, i in zip(pos[have], ids[have])}
    out_ids = [back[int(p)] for p in gp]
    out_grp = gg.tolist()
    rest = ids[~have]
    if len(rest):
        g = (int(gg.max()) + 1) if len(gg) else 0
        out_ids += rest.tolist()
        out_grp += [g] * len(rest)
    return np.array(out_ids, dtype=np.int64), np.array(out_grp, dtype=np.int64)


def _face_rows(con, ids) -> dict:
    """顔の表示に要る列をまとめて引く。"""
    out = {}
    for ch in _chunks([int(x) for x in ids]):
        for r in con.execute(
                f"SELECT {_FACE_COLS}, a.state, a.score, a.reason, a.character_id "
                f"FROM c_instances i JOIN images im ON im.id=i.image_id "
                f"LEFT JOIN c_assign a ON a.instance_id=i.id WHERE i.id IN ({_qm(len(ch))})", ch):
            out[r["instance_id"]] = dict(r)
    return out


def list_page(con, src: dict, offset: int = 0, limit: int = 300) -> dict:
    L = listing(con, src)
    ids, grp = L["ids"], L["grp"]
    limit = max(1, min(int(limit), 2000))
    offset = max(0, min(int(offset), max(len(ids) - 1, 0)))
    sl = ids[offset:offset + limit]
    rows = _face_rows(con, sl)
    faces = []
    for k, i in enumerate(sl):
        r = rows.get(int(i))
        if r is None:
            continue
        if grp is not None:
            r["grp"] = int(grp[offset + k])
        faces.append(r)
    return {"total": int(len(ids)), "offset": offset, "faces": faces, "src": L["src"],
            "n_groups": L["n_groups"]}


def list_groups(con, src: dict, per: int = 12) -> dict:
    """かたまりごとの顔の数と、見本の顔（かたまりの典型から per 枚）。"""
    src = dict(src, order="group")
    L = listing(con, src)
    ids, grp = L["ids"], L["grp"]
    if grp is None or not len(ids):
        return {"total": int(len(ids)), "groups": [], "src": L["src"]}
    starts = np.r_[0, np.where(np.diff(grp) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(ids)]
    per = max(1, min(int(per), 40))
    sample = []
    for a, b in zip(starts, ends):
        sample += ids[a:min(b, a + per)].tolist()
    rows = _face_rows(con, sample)
    groups = []
    for a, b in zip(starts, ends):
        groups.append({"grp": int(grp[a]), "n": int(b - a), "offset": int(a),
                       "faces": [rows[int(i)] for i in ids[a:min(b, a + per)] if int(i) in rows]})
    return {"total": int(len(ids)), "groups": groups, "src": L["src"]}


def scope_ids(con, body: dict) -> tuple[list, list, dict]:
    """範囲の指定（このページ / ここまで / すべて）を顔の番号に直す。
    戻り値: (対象の顔, × の顔, 一覧の src)"""
    L = listing(con, body.get("src") or {})
    ids = L["ids"].tolist()
    inset = set(ids)
    scope = str(body.get("scope") or "page")
    if scope == "page":
        sel = [int(i) for i in (body.get("page_ids") or []) if int(i) in inset]
    elif scope == "upto":
        u = int(body.get("upto_id") or -1)
        if u not in inset:
            raise ValueError("一覧が変わっています。画面を読み直してから、もう一度押してください")
        sel = ids[:ids.index(u) + 1]
    elif scope == "all":
        sel = ids
    else:
        raise ValueError(f"知らない範囲です: {scope}")
    rej = {int(i) for i in (body.get("reject") or []) if int(i) in inset}
    rg = {int(g) for g in (body.get("reject_groups") or [])}
    if rg and L["grp"] is not None:
        rej |= {int(i) for i, g in zip(L["ids"], L["grp"]) if int(g) in rg}
    excl = {int(i) for i in (body.get("exclude") or [])}
    target = [i for i in sel if i not in rej and i not in excl]
    return target, sorted(rej), L["src"]


def list_proposals(con) -> list[dict]:
    """候補の一覧。残りの枚数は 1 回の集計で数える（候補ごとに数えると遅い）。"""
    left = dict(con.execute(
        "SELECT m.proposal_id, COUNT(*) FROM c_proposal_members m "
        "LEFT JOIN c_assign a ON a.instance_id=m.instance_id "
        "LEFT JOIN c_dismissed d ON d.instance_id=m.instance_id "
        "WHERE a.instance_id IS NULL AND d.instance_id IS NULL GROUP BY m.proposal_id").fetchall())
    out = []
    for r in con.execute("SELECT id, n, hints FROM c_proposals"):
        n = int(left.get(r["id"], 0))
        if n < 3:
            continue
        cover = [x[0] for x in con.execute(
            "SELECT m.instance_id FROM c_proposal_members m LEFT JOIN c_assign a "
            "ON a.instance_id=m.instance_id LEFT JOIN c_dismissed d ON d.instance_id=m.instance_id "
            "WHERE m.proposal_id=? AND a.instance_id IS NULL AND d.instance_id IS NULL "
            "ORDER BY m.rank LIMIT 8", (r["id"],))]
        out.append({"id": r["id"], "n": n, "n0": r["n"], "hints": json.loads(r["hints"] or "{}"),
                    "cover": cover})
    out.sort(key=lambda x: (-x["n"], x["id"]))
    return out


def review_summary(con, per: int = 12) -> dict:
    """確認待ちをキャラごとにまとめる（数・迷った理由・見本の顔）。"""
    groups = []
    for cid, name, n in con.execute(
            "SELECT a.character_id, c.name, COUNT(*) FROM c_assign a "
            "JOIN c_characters c ON c.id=a.character_id WHERE a.state='review' "
            "GROUP BY a.character_id ORDER BY COUNT(*) DESC"):
        reasons = dict(con.execute(
            "SELECT COALESCE(reason,'—'), COUNT(*) FROM c_assign WHERE state='review' "
            "AND character_id=? GROUP BY reason ORDER BY COUNT(*) DESC", (cid,)).fetchall())
        faces = [dict(r) for r in con.execute(
            f"SELECT {_FACE_COLS}, a.reason, a.score FROM c_assign a "
            f"JOIN c_instances i ON i.id=a.instance_id JOIN images im ON im.id=i.image_id "
            f"WHERE a.character_id=? AND a.state='review' ORDER BY a.score ASC LIMIT ?",
            (cid, int(per)))]
        groups.append({"character_id": cid, "name": name, "n": n, "reasons": reasons,
                       "faces": faces})
    return {"total": sum(g["n"] for g in groups), "groups": groups}


def counts(con) -> dict:
    """画面の上の数字（軽い問い合わせだけ）。"""
    st = dict(con.execute("SELECT state, COUNT(*) FROM c_assign GROUP BY state").fetchall())
    inst = con.execute("SELECT COUNT(*) FROM c_instances").fetchone()[0]
    dism = con.execute("SELECT COUNT(*) FROM c_dismissed").fetchone()[0]
    return {"characters": con.execute("SELECT COUNT(*) FROM c_characters").fetchone()[0],
            "proposals": con.execute("SELECT COUNT(*) FROM c_proposals").fetchone()[0],
            "confirmed": st.get("confirmed", 0), "auto": st.get("auto", 0),
            "review": st.get("review", 0), "instances": inst, "dismissed": dism,
            "unassigned": max(0, inst - sum(st.values()) - dism)}


def proposal_faces(con, pid: int, limit: int = 60, offset: int = 0) -> dict:
    base = ("FROM c_proposal_members m JOIN c_instances i ON i.id=m.instance_id "
            "JOIN images im ON im.id=i.image_id LEFT JOIN c_assign a ON a.instance_id=m.instance_id "
            "LEFT JOIN c_dismissed d ON d.instance_id=m.instance_id "
            "WHERE m.proposal_id=? AND a.instance_id IS NULL AND d.instance_id IS NULL")
    tot = con.execute(f"SELECT COUNT(*) {base}", (pid,)).fetchone()[0]
    rows = con.execute(f"SELECT {_FACE_COLS} {base} ORDER BY m.rank LIMIT ? OFFSET ?",
                       (pid, max(1, min(int(limit), 600)), max(0, int(offset)))).fetchall()
    r = con.execute("SELECT hints FROM c_proposals WHERE id=?", (pid,)).fetchone()
    return {"total": tot, "faces": [dict(x) for x in rows],
            "hints": json.loads(r[0]) if r and r[0] else {}}


def character_profile(con, cid: int) -> list[dict]:
    """確定した 1 人の絵から見た、そのキャラの書き分け（髪・目・ツノ…）。
    自動判定で「食い違い」とみなす根拠そのものなので、画面にも出す。"""
    lst = []
    for (w,) in con.execute(
            "SELECT an.words FROM c_assign a JOIN c_instances i ON i.id=a.instance_id "
            "JOIN c_analysis an ON an.image_id=i.image_id "
            "WHERE a.character_id=? AND a.state='confirmed' AND i.solo=1 "
            "AND an.words IS NOT NULL", (cid,)):
        try:
            lst.append(json.loads(w))
        except Exception:
            pass
    prof = build_profile(lst)
    out = []
    for s in SLOTS:
        p = prof.get(s)
        if not p:
            continue
        out.append({"slot": s, "top": p["top"], "share": round(p["share"], 2), "n": p["n"],
                    "label": words_label({s: [p["top"]]})[0],
                    "strong": bool(p["n"] >= 8 and p["share"] >= 0.8)})
    return out


def instance_detail(con, iid: int, eng: Engine = None) -> dict:
    eng = eng or ENGINE
    r = con.execute(f"SELECT {_FACE_COLS}, i.score AS det FROM c_instances i "
                    f"JOIN images im ON im.id=i.image_id WHERE i.id=?", (iid,)).fetchone()
    if not r:
        return {"error": "見つかりません"}
    d = dict(r)
    a = con.execute("SELECT a.state, a.score, a.reason, c.name, c.id FROM c_assign a "
                    "JOIN c_characters c ON c.id=a.character_id WHERE a.instance_id=?",
                    (iid,)).fetchone()
    d["assign"] = dict(a) if a else None
    d["rejected"] = [x[0] for x in con.execute(
        "SELECT c.name FROM c_reject r JOIN c_characters c ON c.id=r.character_id "
        "WHERE r.instance_id=?", (iid,))]
    w = con.execute("SELECT words FROM c_analysis WHERE image_id=?", (d["image_id"],)).fetchone()
    d["words"] = words_label(json.loads(w[0])) if w and w[0] else []
    d["near"] = []
    try:
        if eng.store is not None and iid in eng.store.pos:
            store, metric = eng.store, eng.metric
            x = store.feats([store.pos[iid]])
            best = []
            for cid, name in con.execute("SELECT id, name FROM c_characters"):
                ps = [store.pos[i] for (i,) in con.execute(
                    "SELECT instance_id FROM c_assign WHERE character_id=? AND state='confirmed' "
                    "LIMIT 2000", (cid,)) if i in store.pos and i != iid]
                if ps:
                    best.append((float(_robust(mdiff(metric, x, store.feats(ps)), 3)[0]), name))
            best.sort()
            d["near"] = [{"name": n, "score": round(s, 3)} for s, n in best[:4]]
    except Exception:
        pass
    return d


def status(con) -> dict:
    q1 = lambda s, a=(): con.execute(s, a).fetchone()[0]
    total = q1("SELECT COUNT(*) FROM images")
    analyzed = q1("SELECT COUNT(*) FROM c_analysis a JOIN images i ON i.id=a.image_id "
                  "WHERE ABS(a.mtime-i.mtime)<=1 AND a.size=i.size")
    errors = q1("SELECT COUNT(*) FROM c_analysis WHERE status='error'")
    inst = q1("SELECT COUNT(*) FROM c_instances")
    st = dict(con.execute("SELECT state, COUNT(*) FROM c_assign GROUP BY state").fetchall())
    assigned = sum(st.values())
    dism = q1("SELECT COUNT(*) FROM c_dismissed")
    env = V.load_env()
    s = settings()
    try:
        manifest = json.loads((V.MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        manifest = None
    return {
        "version": VERSION,
        "vision_version": V.VERSION,
        "manifest": manifest,
        "env": {k: env.get(k) for k in ("python", "ort", "providers", "tested_at")},
        "models": V.models_ready(),
        "device": s.get("device") or env.get("device"),
        "devices": env.get("devices", []),
        "tested_at": env.get("tested_at"),
        "settings": s,
        "variant": feature_variant(),
        "images": total, "analyzed": analyzed, "errors": errors,
        "instances": inst,
        "confirmed": st.get("confirmed", 0), "auto": st.get("auto", 0),
        "review": st.get("review", 0),
        "unassigned": max(0, inst - assigned - dism), "dismissed": dism,
        "characters": q1("SELECT COUNT(*) FROM c_characters"),
        "proposals": q1("SELECT COUNT(*) FROM c_proposals"),
        "preview": meta_get(con, "chara_preview"),
        "calib": meta_get(con, "chara_calib"),
        "last_classify": meta_get(con, "chara_last_classify"),
        "full_done": bool(meta_get(con, "chara_full_done")),
        "last_action": last_action(con),
        "jobs": {"analysis": JOB_ANALYSIS.snapshot(), "engine": JOB_ENGINE.snapshot()},
    }


# --------------------------------------------------------------------------
# 解析（頭部検出 → 切り出し → CCIP）
# --------------------------------------------------------------------------

def _analysis_targets(con, image_ids=None) -> list:
    sql = ("SELECT i.id, i.path, i.mtime, i.size FROM images i "
           "LEFT JOIN c_analysis a ON a.image_id=i.id "
           "WHERE (a.image_id IS NULL OR ABS(a.mtime-i.mtime)>1 OR a.size<>i.size)")
    rows = con.execute(sql + " ORDER BY i.mtime DESC").fetchall()
    if image_ids is not None:
        want = {int(x) for x in image_ids}
        rows = [r for r in rows if r["id"] in want]
    return rows


def purge_images(con, image_ids) -> int:
    """画像が台帳から消えたとき・中身が変わったときに、その顔の記録を消す。"""
    image_ids = [int(x) for x in image_ids]
    n = 0
    for ch in _chunks(image_ids):
        iids = [r[0] for r in con.execute(
            f"SELECT id FROM c_instances WHERE image_id IN ({_qm(len(ch))})", ch)]
        for ich in _chunks(iids):
            q = _qm(len(ich))
            con.execute(f"DELETE FROM c_features WHERE instance_id IN ({q})", ich)
            con.execute(f"DELETE FROM c_assign WHERE instance_id IN ({q})", ich)
            con.execute(f"DELETE FROM c_reject WHERE instance_id IN ({q})", ich)
            con.execute(f"DELETE FROM c_dismissed WHERE instance_id IN ({q})", ich)
            con.execute(f"DELETE FROM c_proposal_members WHERE instance_id IN ({q})", ich)
            for i in ich:
                try:
                    head_path(i).unlink(missing_ok=True)
                except OSError:
                    pass
        con.execute(f"DELETE FROM c_instances WHERE image_id IN ({_qm(len(ch))})", ch)
        con.execute(f"DELETE FROM c_analysis WHERE image_id IN ({_qm(len(ch))})", ch)
        n += len(iids)
    return n


def transfer_image(con, old_id: int, new_id: int) -> bool:
    """同じ絵が別名で保存し直された（リネームやメタデータ削除）とき、
    顔と確定をそのまま新しい記録へ引き継ぐ。"""
    if con.execute("SELECT 1 FROM c_instances WHERE image_id=?", (new_id,)).fetchone():
        return False
    row = con.execute("SELECT mtime, size FROM images WHERE id=?", (new_id,)).fetchone()
    if not row:
        return False
    tags = [r[0] for r in con.execute("SELECT tag FROM tags WHERE image_id=?", (new_id,))]
    con.execute("DELETE FROM c_analysis WHERE image_id=?", (new_id,))
    con.execute("UPDATE c_instances SET image_id=? WHERE image_id=?", (new_id, old_id))
    con.execute("UPDATE c_analysis SET image_id=?, mtime=?, size=?, words=? WHERE image_id=?",
                (new_id, row[0], row[1],
                 json.dumps(identity_slots(tags) | {"multi": multi_hint(tags)},
                            ensure_ascii=False), old_id))
    return True


class _Timer:
    def __init__(self):
        self.t = Counter()

    def add(self, k, dt):
        self.t[k] += dt


def run_analysis(image_ids=None, variants_solo=None, progress=None, stop=None,
                 log=None, device: str | None = None) -> dict:
    """まだ見ていない画像（と中身の変わった画像）を解析する。途中で止めても続きから。"""
    log = log or (lambda *a: None)
    con = connect()
    st = settings()
    env = V.load_env()
    if not device and not st.get("device") and not env.get("devices"):
        # まだ一度も GPU を試験していない（モデルを手で置いた等）。黙って CPU で
        # 何時間も回すことにならないよう、先に試験して装置を決める。
        try:
            if progress:
                progress("GPU の試験", 0, 1)
            V.selftest(None, log=log, quick=True)
            env = V.load_env()
        except Exception as e:
            log(f"  GPU の試験に失敗しました（{e}）。CPU で解析します。")
    device = device or st.get("device") or env.get("device") or "cpu"
    # 上半身はすべての顔で取る（2人の絵とも比べられるように）。
    # 1人の絵は、全体で見るほうがよければ全体も取る。
    want = tuple(variants_solo or (("crop", "full") if feature_variant() == "full" else ("crop",)))
    targets = _analysis_targets(con, image_ids)
    stats = {"targets": len(targets), "images": 0, "instances": 0, "missing": 0,
             "errors": 0, "heads": Counter(), "device": device, "stopped": False}
    if not targets:
        con.close()
        stats["heads"] = dict(stats["heads"])
        return stats
    t_load = time.time()
    try:
        det = V.HeadDetector(device, conf=float(st.get("det_conf") or 0.4))
        ccip = V.CCIP(device)
    except Exception as e:
        if device == "cpu":
            raise
        log(f"  {device} で開けませんでした（{e}）。CPU で続けます。")
        device = "cpu"
        det = V.HeadDetector(device, conf=float(st.get("det_conf") or 0.4))
        ccip = V.CCIP(device)
        stats["device"] = device
    stats["load_s"] = round(time.time() - t_load, 2)
    min_px = int(st.get("min_head_px") or 32)
    workers = int(st.get("decode_workers") or max(2, min(8, (os.cpu_count() or 4) - 2)))
    tm = _Timer()
    HEAD_DIR.mkdir(parents=True, exist_ok=True)

    def decode(row):
        t0 = time.time()
        p = Path(row["path"])
        if not p.exists():
            return {"row": row, "missing": True}
        try:
            im0 = Image.open(p)
            im0.load()
            im = V.load_rgb(im0)
            arr, old, new = det.prepare(im)
            tm.add("decode", time.time() - t0)
            return {"row": row, "im": im, "arr": arr, "old": old, "new": new}
        except Exception as e:
            return {"row": row, "error": f"{type(e).__name__}: {e}"}

    def crops(im, heads, solo):
        t0 = time.time()
        W, H = im.size
        out = []
        for box, _score in heads:
            thumb, col = V.head_thumb_and_color(im, box)
            item = {"thumb": thumb, "color": col, "inputs": {}}
            for v in (want if solo else ("crop",)):
                if v == "full":
                    item["inputs"]["full"] = V.ccip_prepare(im)
                else:
                    item["inputs"]["crop"] = V.ccip_prepare(im.crop(V.upper_body_box(box, W, H)))
            out.append(item)
        tm.add("crop", time.time() - t0)
        return out

    queue_in: list = []       # (instance_id, variant, array)
    BATCH = 16
    now = time.time()

    def flush_ccip():
        if not queue_in:
            return
        t0 = time.time()
        feats = ccip.features(np.stack([a for _, _, a in queue_in]))
        tm.add("ccip", time.time() - t0)
        con.executemany("INSERT OR REPLACE INTO c_features(instance_id,model,variant,vec) "
                        "VALUES(?,?,?,?)",
                        [(iid, ccip.name, v, vec_to_blob(f))
                         for (iid, v, _), f in zip(queue_in, feats)])
        queue_in.clear()

    wall0 = time.time()
    done = 0
    last_commit = [0]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        it = iter(targets)
        pend = deque()
        for _ in range(workers * 2):
            r = next(it, None)
            if r is None:
                break
            pend.append(pool.submit(decode, r))
        crop_pend = deque()

        def finish_one(entry):
            row, heads, solo, words, fut = entry
            items = fut.result()
            iid_img = row["id"]
            old = con.execute("SELECT 1 FROM c_analysis WHERE image_id=?", (iid_img,)).fetchone()
            if old or con.execute("SELECT 1 FROM c_instances WHERE image_id=?",
                                  (iid_img,)).fetchone():
                purge_images(con, [iid_img])
            Wd, Hd = row["_size"]
            for k, ((box, score), item) in enumerate(zip(heads, items)):
                x1, y1, x2, y2 = box
                cur = con.execute(
                    "INSERT INTO c_instances(image_id,idx,x1,y1,x2,y2,score,solo,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (iid_img, k, x1 / Wd, y1 / Hd, x2 / Wd, y2 / Hd, float(score),
                     1 if solo else 0, now))
                inst = cur.lastrowid
                hp = head_path(inst)
                hp.parent.mkdir(parents=True, exist_ok=True)
                hp.write_bytes(item["thumb"])
                con.execute("INSERT OR REPLACE INTO c_features(instance_id,model,variant,vec) "
                            "VALUES(?,?,?,?)", (inst, V.COLOR_MODEL, "color",
                                                vec_to_blob(item["color"])))
                for v, arr in item["inputs"].items():
                    queue_in.append((inst, v, arr))
                stats["instances"] += 1
            con.execute("INSERT OR REPLACE INTO c_analysis(image_id,mtime,size,n_heads,status,"
                        "error,words,detector,analyzed_at) VALUES(?,?,?,?,'ok',NULL,?,?,?)",
                        (iid_img, row["mtime"], row["size"], len(heads),
                         json.dumps(words, ensure_ascii=False), det.name, time.time()))
            stats["heads"][min(len(heads), 3)] += 1
            if len(queue_in) >= BATCH:
                flush_ccip()

        while pend or crop_pend:
            if stop is not None and stop.is_set():
                stats["stopped"] = True
                break
            if pend:
                res = pend.popleft().result()
                r = next(it, None)
                if r is not None:
                    pend.append(pool.submit(decode, r))
                row = dict(res["row"])
                if res.get("missing"):
                    stats["missing"] += 1
                elif res.get("error"):
                    stats["errors"] += 1
                    purge_images(con, [row["id"]])
                    con.execute("INSERT OR REPLACE INTO c_analysis(image_id,mtime,size,n_heads,"
                                "status,error,words,detector,analyzed_at) "
                                "VALUES(?,?,?,0,'error',?,NULL,?,?)",
                                (row["id"], row["mtime"], row["size"], res["error"][:300],
                                 det.name, time.time()))
                else:
                    t0 = time.time()
                    heads = det.detect_prepared(res["arr"], res["old"], res["new"])
                    tm.add("detect", time.time() - t0)
                    heads = [(b, s) for b, s in heads
                             if min(b[2] - b[0], b[3] - b[1]) >= min_px]
                    heads.sort(key=lambda h: -((h[0][2] - h[0][0]) * (h[0][3] - h[0][1])))
                    tags = [x[0] for x in con.execute(
                        "SELECT tag FROM tags WHERE image_id=?", (row["id"],))]
                    words = identity_slots(tags)
                    words["multi"] = multi_hint(tags)
                    solo = len(heads) == 1 and not words["multi"]
                    row["_size"] = res["im"].size
                    crop_pend.append((row, heads, solo, words,
                                      pool.submit(crops, res["im"], heads, solo)))
                    stats["images"] += 1
                done += 1
            while crop_pend and (crop_pend[0][4].done() or len(crop_pend) > workers * 2
                                 or not pend):
                finish_one(crop_pend.popleft())
            # 書き込んだ顔の特徴がすべて揃ったところで確定させる。
            # 途中で落ちても、確定していない分は次回やり直されるだけで済む。
            if done - last_commit[0] >= 64:
                flush_ccip()
                con.commit()
                last_commit[0] = done
            if progress:
                progress("解析", done, len(targets))
        # 残りを片付ける（止めた場合も、読み終えた分は書いておく）
        while crop_pend:
            finish_one(crop_pend.popleft())
        for f in pend:
            f.cancel()
    flush_ccip()
    con.commit()
    wall = time.time() - wall0
    stats["wall_s"] = round(wall, 2)
    stats["rate"] = round(stats["images"] / wall, 2) if wall > 0 and stats["images"] else None
    stats["timing"] = {k: round(v, 2) for k, v in tm.t.items()}
    stats["heads"] = {str(k): v for k, v in sorted(stats["heads"].items())}
    stats["workers"] = workers
    con.close()
    return stats


def run_missing_variants(variant: str, image_ids=None, progress=None, stop=None,
                         device: str | None = None) -> dict:
    """1人の絵について、足りない見方（全身 / 上半身）の特徴だけを取り足す。
    検出はやり直さないので、確定などはそのまま残る。"""
    con = connect()
    st = settings()
    device = device or st.get("device") or V.load_env().get("device") or "cpu"
    ccip = V.CCIP(device)
    rows = con.execute(
        "SELECT ci.id, ci.image_id, ci.x1, ci.y1, ci.x2, ci.y2, im.path FROM c_instances ci "
        "JOIN images im ON im.id=ci.image_id WHERE ci.solo=1 AND NOT EXISTS "
        "(SELECT 1 FROM c_features f WHERE f.instance_id=ci.id AND f.variant=? AND f.model=?)",
        (variant, ccip.name)).fetchall()
    if image_ids is not None:
        want = {int(x) for x in image_ids}
        rows = [r for r in rows if r["image_id"] in want]
    n = 0
    buf = []

    def flush():
        if not buf:
            return
        feats = ccip.features(np.stack([a for _, a in buf]))
        con.executemany("INSERT OR REPLACE INTO c_features(instance_id,model,variant,vec) "
                        "VALUES(?,?,?,?)",
                        [(i, ccip.name, variant, vec_to_blob(f)) for (i, _), f in zip(buf, feats)])
        buf.clear()

    def prep(r):
        p = Path(r["path"])
        if not p.exists():
            return None
        try:
            im = V.load_rgb(Image.open(p))
        except Exception:
            return None
        W, H = im.size
        if variant == "full":
            return V.ccip_prepare(im)
        box = (r["x1"] * W, r["y1"] * H, r["x2"] * W, r["y2"] * H)
        return V.ccip_prepare(im.crop(V.upper_body_box(box, W, H)))

    workers = int(st.get("decode_workers") or max(2, min(8, (os.cpu_count() or 4) - 2)))
    rows = [dict(r) for r in rows]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for a in range(0, len(rows), 64):
            if stop is not None and stop.is_set():
                break
            chunk = rows[a:a + 64]
            for r, arr in zip(chunk, pool.map(prep, chunk)):
                if arr is None:
                    continue
                buf.append((r["id"], arr))
                n += 1
                if len(buf) >= 16:
                    flush()
            flush()
            con.commit()
            if progress:
                progress("特徴の取り足し", min(a + 64, len(rows)), len(rows))
    flush()
    con.commit()
    con.close()
    return {"computed": n, "variant": variant}


# --------------------------------------------------------------------------
# 試運転
# --------------------------------------------------------------------------

def _auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """差の値が「同じ組」で小さく「違う組」で大きいほど 1 に近づく。"""
    if len(pos) < 20 or len(neg) < 20:
        return None
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    rp = ranks[:len(pos)].sum()
    u = rp - len(pos) * (len(pos) + 1) / 2
    return float(1.0 - u / (len(pos) * len(neg)))


def preview_report(con, image_ids: list, run: dict | None, eng: Engine = None) -> dict:
    """試運転の結果をまとめる。数字はすべて要平さんの絵から測る。"""
    eng = eng or ENGINE
    eng.ensure(con)
    metric = eng.metric
    model = eng.model_name
    ids = [int(x) for x in image_ids]
    heads = Counter()
    for n, in con.execute(f"SELECT n_heads FROM c_analysis WHERE status='ok' AND image_id IN "
                          f"({_qm(len(ids))})", ids):
        heads[min(int(n or 0), 3)] += 1
    ok_imgs = sum(heads.values())
    inst = con.execute(f"SELECT ci.id, ci.image_id, ci.solo FROM c_instances ci "
                       f"WHERE ci.image_id IN ({_qm(len(ids))})", ids).fetchall()
    solo = [r for r in inst if r["solo"]]
    img_words = {r[0]: json.loads(r[1]) for r in con.execute(
        f"SELECT image_id, words FROM c_analysis WHERE words IS NOT NULL AND image_id IN "
        f"({_qm(len(ids))})", ids)}
    # 特徴語の組み合わせ（髪・ツノ・目）で仮の名札を作る
    sigs = {}
    for r in solo:
        w = img_words.get(r["image_id"], {})
        if "hair_multi" in w.get("flags", ()):
            continue
        key = tuple(tuple(w.get(s, ())) for s in ("hair", "horns", "eyes"))
        if sum(1 for k in key if k) >= 2 and all(len(k) <= 1 for k in key):
            sigs[r["id"]] = key
    groups = Counter(sigs.values())
    keep = {k for k, c in groups.items() if c >= 5}
    lab = {i: k for i, k in sigs.items() if k in keep}
    sep = {}
    for variant in ("full", "crop"):
        vecs = {}
        for iid in lab:
            r = con.execute("SELECT vec FROM c_features WHERE instance_id=? AND model=? AND variant=?",
                            (iid, model, variant)).fetchone()
            if r:
                vecs[iid] = blob_to_vec(r[0])
        if len(vecs) < 20:
            sep[variant] = None
            continue
        ks = sorted(vecs)
        X = metric.prep(np.stack([vecs[k] for k in ks]))
        D = metric.diff(X, X)
        pos, neg = [], []
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                la, lb = lab[ks[a]], lab[ks[b]]
                if la == lb:
                    pos.append(D[a, b])
                else:
                    # 髪かツノの系統がはっきり違う組だけを「別人」とみなす
                    if (la[0] and lb[0] and la[0] != lb[0]) or (la[1] and lb[1] and la[1] != lb[1]):
                        neg.append(D[a, b])
        pos, neg = np.asarray(pos), np.asarray(neg)
        sep[variant] = {
            "auc": _auc(pos, neg), "n_same": int(len(pos)), "n_diff": int(len(neg)),
            "same_median": float(np.median(pos)) if len(pos) else None,
            "diff_p5": float(np.percentile(neg, 5)) if len(neg) else None,
            "same_below_review": float((pos <= eng.base["review"]).mean()) if len(pos) else None,
            "diff_below_auto": float((neg <= eng.base["auto"]).mean()) if len(neg) else None,
        }
    a_full = (sep.get("full") or {}).get("auc")
    a_crop = (sep.get("crop") or {}).get("auc")
    if a_full is None and a_crop is None:
        rec = "full"
        rec_why = "特徴語のそろった絵が少なく、見分けの効き具合は測れませんでした。"
    elif a_crop is not None and (a_full is None or a_crop > a_full + 0.01):
        rec = "crop"
        rec_why = "上半身で見たほうが、特徴語の違う絵をよく見分けました。"
    else:
        rec = "full"
        rec_why = "絵全体で見たほうが、特徴語の違う絵をよく見分けました（または同等）。"
    best = max([x for x in (a_full, a_crop) if x is not None], default=None)
    verdict = None
    if best is not None:
        verdict = ("good" if best >= 0.9 else "fair" if best >= 0.8 else "weak")
    remaining = len(_analysis_targets(con))
    rate = (run or {}).get("rate")
    sample_faces = [r["id"] for r in inst]
    random.Random(1).shuffle(sample_faces)
    rep = {
        "at": time.time(),
        "images": len(ids), "analyzed_ok": ok_imgs,
        "heads": {str(k): heads.get(k, 0) for k in range(4)},
        "instances": len(inst), "solo": len(solo),
        "labeled": len(lab), "label_groups": len(keep),
        "separation": sep, "recommend": rec, "recommend_why": rec_why,
        "verdict": verdict,
        "rate": rate, "device": (run or {}).get("device"),
        "timing": (run or {}).get("timing"), "workers": (run or {}).get("workers"),
        "remaining": remaining,
        "eta_s": round(remaining / rate) if rate else None,
        "thresholds": eng.base, "metric": metric.mode,
        "sample_faces": sample_faces[:36],
        "errors": (run or {}).get("errors", 0), "missing": (run or {}).get("missing", 0),
    }
    meta_set(con, "chara_preview", rep)
    con.commit()
    return rep


def run_preview(n: int = 500, progress=None, stop=None, log=None, seed: int = 0) -> dict:
    con = connect()
    ids = [r[0] for r in con.execute("SELECT id FROM images")]
    con.close()
    if not ids:
        return {"error": "台帳に画像がありません。先に「更新」で出力フォルダを読んでください。"}
    rng = random.Random(seed)
    sample = rng.sample(ids, min(n, len(ids)))
    run = run_analysis(image_ids=sample, variants_solo=("full", "crop"),
                       progress=progress, stop=stop, log=log)
    for v in ("full", "crop"):
        run_missing_variants(v, image_ids=sample, stop=stop)
    con = connect()
    ENGINE.invalidate()
    rep = preview_report(con, sample, run)
    meta_set(con, "chara_preview_ids", sample)
    con.commit()
    con.close()
    return rep


# --------------------------------------------------------------------------
# キャラシート・書き出し
# --------------------------------------------------------------------------

_GLOB = {"at": 0.0, "total": 0, "glob": {}}
TRIVIAL_TAGS = {"masterpiece", "best quality", "high quality", "absurdres", "highres",
                "very aesthetic", "newest", "general", "amazing quality"}


def character_image_ids(con, cid: int, states=("confirmed", "auto"), solo_only=True) -> list[int]:
    q = ("SELECT DISTINCT i.image_id FROM c_assign a JOIN c_instances i ON i.id=a.instance_id "
         f"WHERE a.character_id=? AND a.state IN ({_qm(len(states))})"
         + (" AND i.solo=1" if solo_only else ""))
    return [r[0] for r in con.execute(q, [cid] + list(states))]


def character_sheet(con, cid: int | None = None, name: str | None = None,
                    min_support: float = 0.35, min_lift: float = 1.6, top: int = 40) -> dict:
    """そのキャラの絵で「よく出る」だけでなく「そこに偏っている」語を抜く。
    1人の絵だけで数える（複数人の絵は語が混ざるため）。"""
    if cid is None and name is not None:
        r = con.execute("SELECT id FROM c_characters WHERE name=?", (name,)).fetchone()
        cid = r[0] if r else None
    empty = {"n_images": 0, "tags": [], "checkpoints": [], "loras": [], "prompt_block": ""}
    if cid is None:
        return empty
    img_ids = character_image_ids(con, cid)
    n_all = len(img_ids)
    if not n_all:
        return empty
    # 何万枚もいる子は、決まった 3000 枚で数える（割合を見るだけなので十分。全部数えると遅い）
    if n_all > 3000:
        img_ids = sorted(random.Random(cid).sample(img_ids, 3000))
    n = len(img_ids)
    total = con.execute("SELECT COUNT(*) FROM images WHERE has_api=1").fetchone()[0] or 1
    if not (_GLOB["at"] and time.time() - _GLOB["at"] < 600 and _GLOB["total"] == total):
        _GLOB["glob"] = {r[0]: r[1] for r in con.execute(
            "SELECT tag, COUNT(DISTINCT image_id) FROM tags GROUP BY tag")}
        _GLOB.update({"at": time.time(), "total": total})
    glob = _GLOB["glob"]
    counts = Counter()
    ck, lo = Counter(), {}
    for ch in _chunks(img_ids, 900):
        q = _qm(len(ch))
        for t, c in con.execute(f"SELECT tag, COUNT(DISTINCT image_id) FROM tags "
                                f"WHERE image_id IN ({q}) GROUP BY tag", ch):
            counts[t] += c
        for v, c in con.execute(f"SELECT checkpoint, COUNT(*) FROM images WHERE id IN ({q}) "
                                f"AND checkpoint IS NOT NULL GROUP BY checkpoint", ch):
            ck[v] += c
        for v, s, c in con.execute(f"SELECT name, AVG(strength), COUNT(*) FROM loras "
                                   f"WHERE image_id IN ({q}) GROUP BY name", ch):
            e = lo.setdefault(v, {"v": v, "n": 0, "s": 0.0})
            e["n"] += c; e["s"] = s
    tags = []
    for t, c in counts.items():
        sup = c / n
        base = glob.get(t, 1) / total
        lift = sup / base if base > 0 else 0
        if sup >= min_support and lift >= min_lift:
            tags.append({"tag": t, "support": round(sup, 3), "lift": round(lift, 2), "count": c})
    tags.sort(key=lambda t: -(t["support"] * math.log(max(t["lift"], 1.01))))
    return {"n_images": n_all, "tags": tags[:top],
            "checkpoints": [{"v": k, "n": v} for k, v in ck.most_common(6)],
            "loras": sorted(lo.values(), key=lambda x: -x["n"])[:8],
            "prompt_block": ", ".join(t["tag"] for t in tags[:14]), "id": cid}


def _norm_tags(positive: str | None) -> list[str]:
    if not positive:
        return []
    txt = re.sub(r"<[^>]*>", " ", positive).replace("\n", ",").replace("|", ",")
    out, seen = [], set()
    for chunk in txt.split(","):
        t = chunk.strip()
        for _ in range(4):
            t2 = t.strip()
            if len(t2) >= 2 and t2[0] in "([{" and t2[-1] in ")]}":
                t2 = t2[1:-1]
            t2 = re.sub(r":\s*-?\d+(?:\.\d+)?\s*$", "", t2).strip()
            if t2 == t:
                break
            t = t2
        t = t.replace("\\(", "(").replace("\\)", ")").replace("\\", "")
        t = re.sub(r"\s+", " ", t).strip().lower()
        if t and t not in ("break", "and") and len(t) <= 80 and t not in seen:
            seen.add(t); out.append(t)
    return out


def gather_images(con, cid: int, include_auto: bool = False, solo_only: bool = True,
                  min_face: float = 0.10, cap: int = 300) -> list[dict]:
    """LoRA の学習に向く順に集める。確定を優先し、同じ絵柄（dHash 一致）は 1 枚に畳む。"""
    states = ("confirmed", "auto") if include_auto else ("confirmed",)
    rows = con.execute(
        f"SELECT a.state, a.score, i.id AS inst, i.solo, i.x1, i.y1, i.x2, i.y2, i.image_id, "
        f"im.path, im.filename, im.dhash, im.positive FROM c_assign a "
        f"JOIN c_instances i ON i.id=a.instance_id JOIN images im ON im.id=i.image_id "
        f"WHERE a.character_id=? AND a.state IN ({_qm(len(states))})",
        [cid] + list(states)).fetchall()
    best = {}
    for r in rows:
        d = dict(r)
        if solo_only and not d["solo"]:
            continue
        frac = max(abs(d["x2"] - d["x1"]), abs(d["y2"] - d["y1"]))
        if frac < min_face:
            continue
        d["face_frac"] = frac
        d["rank"] = ((0 if d["state"] == "confirmed" else 1), -min(frac / 0.4, 1.0),
                     d["score"] if d["score"] is not None else 0.0)
        key = d["dhash"] or f"#{d['image_id']}"
        if key not in best or d["rank"] < best[key]["rank"]:
            best[key] = d
    out = sorted(best.values(), key=lambda x: x["rank"])
    return out[:cap] if cap else out


def make_caption(positive: str, traits: list[str], trigger: str, keep_traits: bool = False) -> str:
    tl = {t.lower() for t in traits}
    keep = [t for t in _norm_tags(positive)
            if t not in TRIVIAL_TAGS and (keep_traits or t not in tl)]
    return ", ".join(([trigger] if trigger else []) + keep)


def export_lora(con, cid: int, out_dir: Path, repeats: int = 10, cap: int = 200,
                include_auto: bool = False, trigger: str | None = None,
                keep_traits: bool = False, copy_images: bool = True) -> dict:
    import shutil
    name = _cname(con, cid)
    sheet = character_sheet(con, cid)
    traits = [t["tag"] for t in sheet["tags"][:18]]
    trig = trigger if trigger is not None else re.sub(r"\s+", "_", name.strip())
    picks = gather_images(con, cid, include_auto=include_auto, cap=cap)
    safe = re.sub(r'[<>:"/\\|?*]', "_", name).strip() or "chara"
    dest = Path(out_dir) / f"{repeats}_{safe}"
    dest.mkdir(parents=True, exist_ok=True)
    written = missing = 0
    for i, d in enumerate(picks, 1):
        src = Path(d["path"])
        if not src.exists():
            missing += 1
            continue
        stem = f"{i:04d}"
        try:
            if copy_images:
                shutil.copy2(src, dest / (stem + src.suffix.lower()))
            (dest / (stem + ".txt")).write_text(
                make_caption(d["positive"] or "", traits, trig, keep_traits), encoding="utf-8")
            written += 1
        except Exception:
            missing += 1
    meta = {"character": name, "trigger": trig, "repeats": repeats, "images": written,
            "missing": missing, "candidates": len(picks), "include_auto": include_auto,
            "traits_removed_from_caption": [] if keep_traits else traits,
            "checkpoints": sheet["checkpoints"], "loras": sheet["loras"]}
    (dest / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["path"] = str(dest)
    return meta


SHEET_CSS = """:root{--ink:#0D141C;--panel:#141E28;--rule:#29394A;--text:#D3DCE5;--dim:#78899A;
--faint:#4E5F70;--lamp:#E9A23B;--live:#46C9B4;--sans:"BIZ UDPGothic","Yu Gothic UI",
"Hiragino Sans",system-ui,sans-serif;--mono:"Cascadia Mono",Consolas,ui-monospace,monospace}
*{box-sizing:border-box}body{background:var(--ink);color:var(--text);font-family:var(--sans);
margin:0;padding:28px;line-height:1.7;font-size:13px}header{border-bottom:1px solid var(--rule);
padding-bottom:14px;margin-bottom:22px}h1{margin:0;font-size:20px;letter-spacing:.2em}
header p{margin:6px 0 0;font-family:var(--mono);font-size:10px;letter-spacing:.12em;color:var(--faint)}
section{background:var(--panel);border:1px solid var(--rule);border-radius:5px;padding:16px 18px;
margin-bottom:16px;break-inside:avoid}h2{margin:0 0 12px;font-size:16px;display:flex;
align-items:baseline;gap:12px}h2 em{font-style:normal;font-family:var(--mono);font-size:10px;
color:var(--faint)}h3{font-family:var(--mono);font-size:10px;letter-spacing:.16em;color:var(--faint);
margin:0 0 8px;font-weight:400}.covers{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}
.covers img{width:84px;height:84px;object-fit:cover;border-radius:3px;border:1px solid var(--rule)}
.cols{display:grid;grid-template-columns:1.2fr 1fr;gap:22px}table{width:100%;border-collapse:collapse}
td{padding:1px 0;font-size:12px}td.t{max-width:200px;overflow:hidden;text-overflow:ellipsis;
white-space:nowrap}td.b{width:74px;padding:0 8px}td.b i{display:block;height:4px;background:var(--live);
border-radius:2px}td.n,td.l{font-family:var(--mono);font-size:10px;text-align:right;width:38px}
td.l{color:var(--lamp)}pre{background:#0A1119;border:1px solid var(--rule);border-radius:3px;
padding:9px;font-family:var(--mono);font-size:11px;white-space:pre-wrap;word-break:break-word;
color:#B9C6D2;margin:0 0 14px}ul{list-style:none;margin:0 0 14px;padding:0}li{font-family:var(--mono);
font-size:11px;color:var(--dim);display:flex;justify-content:space-between;gap:10px}
@media print{body{background:#fff;color:#111}section{background:#fff;border-color:#ccc}}
@media (max-width:760px){.cols{grid-template-columns:1fr}}"""


def _esc(t) -> str:
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def export_sheets(con, out_path: Path, covers: int = 8) -> dict:
    """全キャラの設定シートを 1 枚の HTML にする。画像は埋め込むので単体で持ち歩ける。"""
    import base64
    parts, total = [], 0
    for c in list_characters(con):
        sh = character_sheet(con, c["id"])
        if not sh["n_images"]:
            continue
        total += sh["n_images"]
        faces = [r[0] for r in con.execute(
            "SELECT instance_id FROM c_assign WHERE character_id=? AND state='confirmed' "
            "ORDER BY updated_at DESC LIMIT ?", (c["id"], covers))]
        imgs = []
        for f in faces:
            try:
                imgs.append(base64.b64encode(head_path(f).read_bytes()).decode("ascii"))
            except Exception:
                pass
        parts.append((c, sh, imgs))
    body = []
    for c, sh, imgs in parts:
        cov = "".join(f'<img src="data:image/jpeg;base64,{b}" alt="">' for b in imgs)
        rows = "".join(
            f'<tr><td class="t">{_esc(t["tag"])}</td><td class="b"><i style="width:'
            f'{min(100, round(t["support"] * 100))}%"></i></td><td class="n">'
            f'{round(t["support"] * 100)}%</td><td class="l">x{t["lift"]:.1f}</td></tr>'
            for t in sh["tags"][:26])
        mdl = "".join(f"<li>{_esc(str(x['v']).rsplit('.', 1)[0])} <span>{x['n']}</span></li>"
                      for x in sh["checkpoints"])
        lor = "".join(f"<li>{_esc(str(x['v']).rsplit('.', 1)[0])} <span>{x['n']}回</span></li>"
                      for x in sh["loras"])
        body.append(
            f'<section><h2>{_esc(c["name"])} <em>{sh["n_images"]:,} 枚（確定 {c["confirmed"]:,}）</em>'
            f'</h2><div class="covers">{cov}</div><div class="cols"><div><h3>偏っているタグ</h3>'
            f'<table>{rows}</table></div><div><h3>プロンプト断片</h3><pre>{_esc(sh["prompt_block"])}'
            f'</pre><h3>よく使ったモデル</h3><ul>{mdl or "<li>—</li>"}</ul><h3>よく使った LoRA</h3>'
            f'<ul>{lor or "<li>—</li>"}</ul></div></div></section>')
    head = (f"SOUKI / {len(parts)} CHARACTERS / {total:,} IMAGES / {time.strftime('%Y-%m-%d')}")
    html = ('<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>キャラクター設定シート</title><style>{SHEET_CSS}</style></head><body>'
            f'<header><h1>キャラクター設定シート</h1><p>{head}</p></header>{"".join(body)}'
            '</body></html>')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return {"characters": len(parts), "images": total, "path": str(out_path)}


# --------------------------------------------------------------------------
# 裏で動く仕事
# --------------------------------------------------------------------------

class Job:
    """時間のかかる処理を 1 本ずつ裏で回す。進み具合と見込み時間を持つ。"""

    def __init__(self, name: str):
        self.name = name
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.s = {"kind": None, "running": False, "stage": "", "done": 0, "total": 0,
                  "started": 0.0, "eta_s": None, "last": None, "error": None, "finished": 0.0}
        self._rate = None

    def snapshot(self) -> dict:
        """状態の写し。結果の大きな中身（試運転の顔の一覧など）は含めない。"""
        with self.lock:
            s = dict(self.s)
        last = s.get("last")
        if isinstance(last, dict):
            last = {k: v for k, v in last.items() if k != "trace"}
            res = last.get("result")
            if isinstance(res, dict):
                last["result"] = {k: v for k, v in res.items()
                                  if k not in ("sample_faces", "models", "separation")}
            s["last"] = last
        return s

    def progress(self, stage, done, total):
        with self.lock:
            now = time.time()
            self.s.update({"stage": stage, "done": int(done), "total": int(total)})
            el = now - (self.s.get("stage_t0") or now)
            if self.s.get("stage_name") != stage:
                self.s["stage_name"] = stage
                self.s["stage_t0"] = now
                self.s["stage_d0"] = done
                el = 0
            dd = done - (self.s.get("stage_d0") or 0)
            if el > 3 and dd > 0 and total:
                self.s["eta_s"] = round((total - done) * el / dd)
            else:
                self.s["eta_s"] = None

    def start(self, kind: str, fn, *args, **kw) -> bool:
        with self.lock:
            if self.s["running"]:
                return False
            self.cancel.clear()
            self.s.update({"kind": kind, "running": True, "stage": "準備中", "done": 0,
                           "total": 0, "started": time.time(), "eta_s": None,
                           "error": None, "stage_name": None})

        def run():
            try:
                res = fn(*args, **kw)
                with self.lock:
                    self.s["last"] = {"kind": kind, "result": res, "at": time.time()}
            except Exception as e:
                with self.lock:
                    self.s["error"] = f"{type(e).__name__}: {e}"
                    self.s["last"] = {"kind": kind, "error": self.s["error"],
                                      "trace": traceback.format_exc()[-2000:], "at": time.time()}
            finally:
                with self.lock:
                    self.s["running"] = False
                    self.s["finished"] = time.time()
                    self.s["stage"] = ""
                if self._after:
                    try:
                        self._after(kind)
                    except Exception:
                        pass

        threading.Thread(target=run, daemon=True, name=f"souki-{self.name}").start()
        return True

    _after = None


JOB_ANALYSIS = Job("analysis")
JOB_ENGINE = Job("engine")
_ENGINE_WANT = {"pending": False, "proposals": False, "timer": None}
_ENGINE_LOCK = threading.Lock()


def _engine_run(proposals: bool) -> dict:
    con = connect()
    try:
        before = con.execute("SELECT COUNT(*) FROM c_assign").fetchone()[0]
        res = {"classify": classify_all(con, progress=JOB_ENGINE.progress)}
        after = con.execute("SELECT COUNT(*) FROM c_assign").fetchone()[0]
        last = meta_get(con, "chara_proposals_at") or {}
        pool_now = res["classify"]["instances"] - after
        # 名前を付けて顔が減っただけなら、今の候補はそのまま使える（片付いた顔は
        # 一覧から消える）。作り直すのは、顔が増えたとき（解析・取り消し）だけ。
        grew = pool_now - (last.get("pool") or 0) > max(200, 0.02 * max(pool_now, 1))
        if proposals or grew or not last:
            res["proposals"] = build_proposals(con, progress=JOB_ENGINE.progress)
        elif pool_now < (last.get("pool") or 0):
            # 増えたかどうかは、いちばん減ったところから測る
            meta_set(con, "chara_proposals_at", dict(last, pool=int(pool_now)))
            con.commit()
        if "proposals" not in res:
            try:
                n = fill_proposal_groups(con)   # 前の版で作った候補に「かたまり」を足す
                if n:
                    res["groups_filled"] = n
            except Exception:
                pass
        res["assigned_delta"] = after - before
        return res
    finally:
        con.close()


def _engine_after(kind):
    with _ENGINE_LOCK:
        again = _ENGINE_WANT["pending"]
        props = _ENGINE_WANT["proposals"]
        _ENGINE_WANT["pending"] = False
        _ENGINE_WANT["proposals"] = False
    if again:
        request_engine(proposals=props, delay=0.2)


JOB_ENGINE._after = _engine_after


def request_engine(proposals: bool = False, delay: float = 1.2) -> None:
    """判定のやり直しを頼む。続けて操作したときにまとめて 1 回で済むよう少し待つ。"""
    if not ENGINE.ready():
        return
    with _ENGINE_LOCK:
        if proposals:
            _ENGINE_WANT["proposals"] = True
        t = _ENGINE_WANT.get("timer")
        if t:
            t.cancel()

        def fire():
            with _ENGINE_LOCK:
                props = _ENGINE_WANT["proposals"]
                if JOB_ENGINE.snapshot()["running"]:
                    _ENGINE_WANT["pending"] = True
                    return
                _ENGINE_WANT["proposals"] = False
            JOB_ENGINE.start("classify", _engine_run, props)

        _ENGINE_WANT["timer"] = threading.Timer(delay, fire)
        _ENGINE_WANT["timer"].daemon = True
        _ENGINE_WANT["timer"].start()


def _analysis_then_engine(kind: str, fn, *args, **kw):
    res = fn(*args, **kw)
    ENGINE.invalidate()
    if kind == "analyze" and not (res or {}).get("stopped"):
        con = connect()
        if not _analysis_targets(con):
            meta_set(con, "chara_full_done", True)
            con.commit()
        con.close()
    request_engine(proposals=True, delay=0.2)
    return res


def start_job(kind: str, **kw) -> dict:
    """画面の「試運転」「全件を解析」などから呼ぶ。"""
    if kind == "stop":
        JOB_ANALYSIS.cancel.set()
        return {"ok": True}
    if kind in ("classify", "proposals"):
        if not ENGINE.ready():
            return {"ok": False, "error": "モデルがまだありません"}
        request_engine(proposals=(kind == "proposals"), delay=0.05)
        return {"ok": True}
    if kind == "setup":
        ok = JOB_ANALYSIS.start("setup", setup_all, progress=JOB_ANALYSIS.progress)
        return {"ok": ok, "error": None if ok else "ほかの処理が動いています"}
    if not ENGINE.ready():
        return {"ok": False, "error": "モデルがまだありません。先に「準備する」を押してください"}
    if kind == "preview":
        n = int(kw.get("n") or 500)
        ok = JOB_ANALYSIS.start("preview", _analysis_then_engine, "preview", run_preview, n,
                                progress=JOB_ANALYSIS.progress, stop=JOB_ANALYSIS.cancel)
    elif kind == "analyze":
        ok = JOB_ANALYSIS.start("analyze", _analysis_then_engine, "analyze", run_analysis,
                                progress=JOB_ANALYSIS.progress, stop=JOB_ANALYSIS.cancel)
    elif kind == "variant":
        v = kw.get("variant") or feature_variant()
        ok = JOB_ANALYSIS.start("variant", _analysis_then_engine, "variant", run_missing_variants,
                                v, progress=JOB_ANALYSIS.progress, stop=JOB_ANALYSIS.cancel)
    elif kind == "selftest":
        ok = JOB_ANALYSIS.start("selftest", V.selftest, settings().get("device"),
                                log=lambda *a: None)
    else:
        return {"ok": False, "error": f"知らない処理です: {kind}"}
    return {"ok": ok, "error": None if ok else "ほかの処理が動いています"}


def maybe_auto_analyze() -> bool:
    """台帳を更新したあと、新しい絵があれば自動で解析する（全件解析を一度済ませてから）。"""
    if not ENGINE.ready() or not settings().get("auto_analyze", True):
        return False
    con = connect()
    try:
        if not meta_get(con, "chara_full_done"):
            return False
        if not _analysis_targets(con):
            return False
    finally:
        con.close()
    return start_job("analyze").get("ok", False)


# --------------------------------------------------------------------------
# 準備（モデルの取得と装置の試験）
# --------------------------------------------------------------------------

def setup_all(progress=None, log=print) -> dict:
    res = {"version": VERSION}
    if progress:
        progress("モデルの取得", 0, 1)

    def prog(name, done, total):
        if progress:
            progress(f"取得中 {name}", done, total or 1)

    fm = V.fetch_models(progress=prog, log=log)
    res["models"] = fm
    if not fm["ok"]:
        res["ok"] = False
        res["error"] = ("モデルを取得できませんでした。回線を確かめてから、もう一度「準備する」を"
                        "押してください。" + (f"（{fm['error']}）" if fm.get("error") else ""))
        return res
    if progress:
        progress("GPU の試験", 0, 1)
    res["selftest"] = V.selftest(settings().get("device"), log=log)
    V.load_metric(log=log)
    res["ok"] = bool(res["selftest"].get("ok"))
    return res


# --------------------------------------------------------------------------
# コマンド
# --------------------------------------------------------------------------

def _bar(stage, done, total):
    pct = done / total * 100 if total else 0
    sys.stdout.write(f"\r  {stage}  {done:,}/{total:,}  ({pct:4.1f}%)   ")
    sys.stdout.flush()


def cmd_setup(args) -> int:
    print(f"[想起] キャラ判定の準備  版 {VERSION}")
    print(f"  Python {sys.version.split()[0]} / {sys.executable}")
    try:
        import onnxruntime as o
        print(f"  ONNX Runtime {o.__version__}  ({', '.join(o.get_available_providers())})")
    except Exception as e:
        print(f"  !! ONNX Runtime を読み込めません: {e}")
        return 1
    print()
    print("  1/2 モデルを取得します（初回のみ、約 200MB）")
    last = [0.0]

    def prog(stage, done, total):
        if time.time() - last[0] > 0.5 or done >= total:
            last[0] = time.time()
            _bar(stage, done, total)

    res = setup_all(progress=prog, log=lambda m: print("\n" + m if m.startswith("  !!") else m))
    print()
    if not res["models"]["ok"]:
        print()
        print("  !! モデルを取得できませんでした。")
        print("     回線を確かめて、キャラ画面の「準備する」を押すか、")
        print("     setup フォルダの setup.bat をもう一度開いてください。")
        print("     （想起は起動しますが、キャラ画面は準備が終わるまで使えません）")
        return 2
    print()
    print("  2/2 GPU で正しく計算できるか確かめます")
    st = res["selftest"]
    print()
    print(f"  解析に使う装置: {st.get('device')}")
    return 0


def cmd_preview(args) -> int:
    con = connect()
    ENGINE.ensure(con)
    con.close()
    print(f"[想起] 試運転 {args.n} 枚  版 {VERSION}")
    rep = run_preview(args.n, progress=_bar)
    print()
    print(json.dumps({k: v for k, v in rep.items() if k != "sample_faces"},
                     ensure_ascii=False, indent=1))
    return 0


def cmd_analyze(args) -> int:
    print(f"[想起] 解析  版 {VERSION}")
    st = run_analysis(progress=_bar, log=print)
    print()
    print(json.dumps(st, ensure_ascii=False, indent=1, default=str))
    con = connect()
    print(json.dumps(classify_all(con), ensure_ascii=False, indent=1))
    print(json.dumps(build_proposals(con), ensure_ascii=False, indent=1))
    con.close()
    return 0


def cmd_classify(args) -> int:
    con = connect()
    print(json.dumps(classify_all(con), ensure_ascii=False, indent=1))
    print(json.dumps(build_proposals(con), ensure_ascii=False, indent=1))
    con.close()
    return 0


def cmd_selftest(args) -> int:
    r = V.selftest(settings().get("device"))
    print(json.dumps(r, ensure_ascii=False, indent=1))
    return 0 if r.get("ok") else 1


def cmd_export(args) -> int:
    con = connect()
    out = Path(args.out) if args.out else EXPORT_DIR
    chars = list_characters(con)
    if args.charas:
        want = {n.strip() for n in args.charas.split(",") if n.strip()}
        chars = [c for c in chars if c["name"] in want]
    if args.mode in ("sheet", "both"):
        r = export_sheets(con, out / "キャラ設定シート.html")
        print(f"  設定シート {r['characters']} 人 -> {r['path']}")
    if args.mode in ("lora", "both"):
        for c in chars:
            m = export_lora(con, c["id"], out / "lora", repeats=args.repeats, cap=args.max,
                            include_auto=args.include_auto)
            print(f"  {c['name']:<16} {m['images']:>4} 枚 -> {m['path']}")
    con.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="chara", description="想起 — キャラの判定")
    ap.add_argument("--version", action="version", version=f"chara.py 版 {VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="モデルを取得し、GPU を試験する").set_defaults(func=cmd_setup)
    sub.add_parser("selftest", help="GPU の試験だけやり直す").set_defaults(func=cmd_selftest)
    p = sub.add_parser("preview", help="試運転")
    p.add_argument("--n", type=int, default=500)
    p.set_defaults(func=cmd_preview)
    sub.add_parser("analyze", help="全件を解析").set_defaults(func=cmd_analyze)
    sub.add_parser("classify", help="判定と候補をやり直す").set_defaults(func=cmd_classify)
    p = sub.add_parser("export", help="LoRA 学習データ / 設定シートを書き出す")
    p.add_argument("--mode", choices=("sheet", "lora", "both"), default="both")
    p.add_argument("--out")
    p.add_argument("--charas")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--max", type=int, default=200)
    p.add_argument("--include-auto", action="store_true")
    p.set_defaults(func=cmd_export)
    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
