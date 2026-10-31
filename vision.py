#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
想起 — 画像を見る部分 (vision.py)
==================================

モデルの取得、GPU の確認、頭部検出、CCIP（アニメキャラの同一判定）を受け持つ。
キャラの判定ロジックそのものは chara.py にある。

使っているモデル（どちらも DeepGHS の公開モデル）
  頭部検出  deepghs/anime_head_detection   head_detect_v2.0_s
  CCIP      deepghs/ccip_onnx               ccip-caformer-24-randaug-pruned

前処理・後処理は dghs-imgutils（MIT License, Copyright (c) 2022 DeepGHS）の
実装を写したもの。ライブラリごと入れない理由は二つ。
  ・imgutils は numpy<2 を要求し、OpenCV など大きな依存を連れてくる
  ・計算そのものは小さく、同じ結果になることを試験で確かめられる
写した関数は tests/test_parity.py で元の実装と突き合わせている。

GPU は DirectML（Windows 標準の仕組み）で使う。CUDA や torch は要らない。
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

VERSION = "2.4.2"

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
RUNTIME_DIR = APP_DIR / "runtime"
ENV_PATH = RUNTIME_DIR / "env.json"
METRIC_PATH = RUNTIME_DIR / "metric.json"

USER_AGENT = f"souki/{VERSION} (+local tool)"


# --------------------------------------------------------------------------
# モデルの置き場所と取得
# --------------------------------------------------------------------------

# rev は取得元の版。取れなければ main を使う（中身は sha256 で確かめる）。
MODELS = {
    "head": {
        "title": "頭部検出",
        "repo": "deepghs/anime_head_detection",
        "rev": "06604feee81983792a57c21081e539c0ae229833",
        "dir": "head_detect_v2.0_s",
        # ファイル名: 必須かどうか
        "files": {"model.onnx": True, "model_type.json": False,
                  "threshold.json": False, "labels.json": False},
        "approx_mb": 45,
    },
    "ccip": {
        "title": "キャラ判定 (CCIP)",
        "repo": "deepghs/ccip_onnx",
        "rev": "main",
        "dir": "ccip-caformer-24-randaug-pruned",
        "files": {"model_feat.onnx": True, "model_metrics.onnx": True,
                  "metrics.json": True, "cluster.json": False},
        "approx_mb": 150,
    },
}


def model_path(key: str, fname: str) -> Path:
    return MODEL_DIR / key / fname


def models_ready() -> dict:
    """必須ファイルが揃っているか。揃っていなければ何が足りないかを返す。"""
    missing = []
    for key, m in MODELS.items():
        for fn, req in m["files"].items():
            p = model_path(key, fn)
            if req and (not p.exists() or p.stat().st_size == 0):
                missing.append(f"{key}/{fn}")
    return {"ready": not missing, "missing": missing}


def _hf_url(repo: str, rev: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{rev}/{path}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _probe(url: str, timeout: float = 30) -> tuple[int, dict]:
    """転送先に飛ばずに、最初の応答の見出しを読む。
    HuggingFace は大きなファイルの sha256 と大きさを、転送前の応答に載せてくる。"""
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}
    except (urllib.error.URLError, OSError, ValueError) as e:
        # つながらない（回線・プロキシ・セキュリティソフト）。0 で返して呼び元が知らせる
        return 0, {"error": str(getattr(e, "reason", e))}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download(url: str, dest: Path, size: int | None, sha: str | None,
              progress=None, tries: int = 4) -> dict:
    """途中で切れても続きから取り直す。取り終えたら大きさと sha256 を確かめる。"""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(tries):
        have = part.stat().st_size if part.exists() else 0
        if size and have > size:
            part.unlink(missing_ok=True); have = 0
        headers = {"User-Agent": USER_AGENT}
        if have and size and have < size:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    have = 0          # 続きからを受け付けなかった
                total = size or (int(r.headers.get("Content-Length") or 0) + have)
                with open(part, "ab" if have else "wb") as f:
                    done = have
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
            got = part.stat().st_size
            if size and got != size:
                last_err = f"大きさが合いません（{got} / {size}）"
                continue
            if sha:
                h = _sha256(part)
                if h != sha:
                    part.unlink(missing_ok=True)
                    last_err = "中身の検査（sha256）に失敗しました"
                    continue
            part.replace(dest)
            return {"ok": True, "size": got, "sha256": sha or _sha256(dest)}
        except Exception as e:           # 回線の切断など
            last_err = str(e)
            time.sleep(min(2 ** attempt, 8))
    return {"ok": False, "error": last_err or "取得に失敗しました"}


def fetch_models(progress=None, log=print) -> dict:
    """足りないモデルを取ってくる。すでにあるものは取り直さない。"""
    manifest_p = MODEL_DIR / "manifest.json"
    try:
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    results = []
    for key, m in MODELS.items():
        for fn, required in m["files"].items():
            dest = model_path(key, fn)
            if dest.exists() and dest.stat().st_size > 0:
                results.append({"file": f"{key}/{fn}", "ok": True, "skipped": True})
                continue
            rel = f"{m['dir']}/{fn}"
            got = None
            for rev in dict.fromkeys([m["rev"], "main"]):
                url = _hf_url(m["repo"], rev, rel)
                code, hd = _probe(url)
                if code == 0:
                    got = {"ok": False, "error": f"huggingface.co につながりません（{hd.get('error')}）"}
                    break
                if code == 404:
                    continue
                if code >= 400 and code not in (401, 403):
                    continue
                sha = (hd.get("x-linked-etag") or "").strip('"').lower() or None
                if sha and len(sha) != 64:
                    sha = None             # 小さいファイルの etag は sha256 ではない
                size = hd.get("x-linked-size") or None
                size = int(size) if size and str(size).isdigit() else None
                log(f"  取得中  {m['title']}  {fn}"
                    + (f"  ({size / 1e6:.0f} MB)" if size else ""))

                def prog(done, total, _fn=fn):
                    if progress:
                        progress(f"{key}/{_fn}", done, total)
                got = _download(url, dest, size, sha, progress=prog)
                if got["ok"]:
                    manifest.setdefault(key, {})[fn] = {
                        "sha256": got["sha256"], "size": got["size"],
                        "rev": rev, "at": time.time()}
                    break
            if got is None:
                got = {"ok": False, "error": "見つかりませんでした"}
            got["file"] = f"{key}/{fn}"
            got["required"] = required
            results.append(got)
            if not got["ok"]:
                lvl = "!!" if required else "  "
                log(f"  {lvl} {key}/{fn}: {got.get('error')}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_p.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = all(r["ok"] for r in results if r.get("required", True) and not r.get("skipped"))
    bad = [r for r in results if not r.get("ok")]
    return {"ok": ok and models_ready()["ready"], "results": results,
            "error": (f"{bad[0]['file']}: {bad[0].get('error')}" if bad else None)}


# --------------------------------------------------------------------------
# ONNX Runtime と GPU
# --------------------------------------------------------------------------

_ORT = None
_ORT_LOCK = threading.Lock()


def ort():
    global _ORT
    with _ORT_LOCK:
        if _ORT is None:
            import onnxruntime as o
            try:
                o.set_default_logger_severity(3)
            except Exception:
                pass
            _ORT = o
    return _ORT


def dxgi_adapters() -> list[dict]:
    """Windows に見えている GPU の一覧（DirectML の device_id と同じ並び）。

    名前を出したいだけなので、失敗しても空で返して先へ進む。"""
    if sys.platform != "win32":
        return []
    try:
        import ctypes
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                        ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]

        class LUID(ctypes.Structure):
            _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

        class DESC1(ctypes.Structure):
            _fields_ = [("Description", ctypes.c_wchar * 128),
                        ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                        ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint),
                        ("DedicatedVideoMemory", ctypes.c_size_t),
                        ("DedicatedSystemMemory", ctypes.c_size_t),
                        ("SharedSystemMemory", ctypes.c_size_t),
                        ("AdapterLuid", LUID), ("Flags", ctypes.c_uint)]

        def make_guid(s):
            h = s.replace("-", "")
            g = GUID()
            g.Data1 = int(h[0:8], 16); g.Data2 = int(h[8:12], 16); g.Data3 = int(h[12:16], 16)
            for i in range(8):
                g.Data4[i] = int(h[16 + i * 2:18 + i * 2], 16)
            return g

        def vfunc(obj, index, restype, *argtypes):
            vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
            return proto(vtbl[index])

        dxgi = ctypes.WinDLL("dxgi")
        create = dxgi.CreateDXGIFactory1
        create.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
        create.restype = ctypes.c_long
        iid = make_guid("770aae78-f26f-4dba-a829-253c83d1b387")   # IDXGIFactory1
        fac = ctypes.c_void_p()
        if create(ctypes.byref(iid), ctypes.byref(fac)) != 0 or not fac.value:
            return []
        out = []
        enum1 = vfunc(fac, 12, ctypes.c_long, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        i = 0
        while i < 16:
            ad = ctypes.c_void_p()
            if enum1(fac, i, ctypes.byref(ad)) != 0 or not ad.value:
                break
            d = DESC1()
            getdesc1 = vfunc(ad, 10, ctypes.c_long, ctypes.POINTER(DESC1))
            if getdesc1(ad, ctypes.byref(d)) == 0:
                out.append({"index": i, "name": d.Description.strip(),
                            "vram_mb": int(d.DedicatedVideoMemory // (1024 * 1024)),
                            "software": bool(d.Flags & 2)})
            vfunc(ad, 2, ctypes.c_ulong)(ad)          # Release
            i += 1
        vfunc(fac, 2, ctypes.c_ulong)(fac)
        return out
    except Exception:
        return []


def device_candidates() -> list[dict]:
    """試してみる候補。GPU を先に、最後に CPU。"""
    provs = ort().get_available_providers()
    out = []
    if "DmlExecutionProvider" in provs:
        ads = [a for a in dxgi_adapters() if not a["software"]]
        if ads:
            for a in ads:
                out.append({"device": f"dml:{a['index']}", "name": a["name"],
                            "vram_mb": a["vram_mb"]})
        else:
            for i in range(2):
                out.append({"device": f"dml:{i}", "name": f"GPU {i}", "vram_mb": 0})
    if "CUDAExecutionProvider" in provs:
        out.append({"device": "cuda:0", "name": "CUDA GPU 0", "vram_mb": 0})
    out.append({"device": "cpu", "name": "CPU", "vram_mb": 0})
    return out


def make_session(path: Path | str, device: str = "cpu", threads: int | None = None):
    """その装置で動くセッションを作る。黙って CPU に落ちたら失敗として扱う。"""
    o = ort()
    so = o.SessionOptions()
    so.graph_optimization_level = o.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if device.startswith("dml"):
        # DirectML ではこの二つを切る決まりになっている
        so.enable_mem_pattern = False
        so.execution_mode = o.ExecutionMode.ORT_SEQUENTIAL
        providers = [("DmlExecutionProvider", {"device_id": int(device.split(":")[1])}),
                     "CPUExecutionProvider"]
    elif device.startswith("cuda"):
        providers = [("CUDAExecutionProvider", {"device_id": int(device.split(":")[1])}),
                     "CPUExecutionProvider"]
    else:
        so.intra_op_num_threads = threads or max(1, (os.cpu_count() or 4) - 2)
        providers = ["CPUExecutionProvider"]
    sess = o.InferenceSession(str(path), so, providers=providers)
    used = sess.get_providers()
    if device != "cpu" and (not used or used[0] == "CPUExecutionProvider"):
        raise RuntimeError(f"{device} で動かせませんでした（CPU に切り替わりました）")
    return sess


def load_env() -> dict:
    try:
        return json.loads(ENV_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_env(env: dict) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    ENV_PATH.write_text(json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")


def _synthetic_batch(n: int, size: int, seed: int = 7) -> np.ndarray:
    """装置どうしの比較に使う入力。毎回同じものになるよう乱数を固定する。"""
    rng = np.random.default_rng(seed)
    ims = []
    for k in range(n):
        yy, xx = np.mgrid[0:size, 0:size] / size
        base = np.stack([np.sin(xx * (3 + k) + yy * 2),
                         np.cos(yy * (2 + k) - xx),
                         np.sin((xx + yy) * (4 + k))]) * 0.5 + 0.5
        base = base + rng.normal(0, 0.05, base.shape)
        ims.append(base)
    x = np.clip(np.stack(ims), 0, 1)
    return ((x - CLIP_MEAN[None]) / CLIP_STD[None]).astype(np.float32)


def choose_device(results: list[dict], preferred: str | None = None) -> str:
    """どの装置で解析するか決める。

    GPU が 2 枚あるなら、ビデオメモリの少ないほうを使う。大きいほうは
    ComfyUI の生成用に空けておくため（要平さんの構成では 5080 を空け、3060 で解析）。
    内蔵 GPU は遅いので、ほかに GPU があれば使わない。"""
    ok = [r for r in results if r.get("ok")]
    if preferred and any(r["device"] == preferred for r in ok):
        return preferred
    gpus = [r for r in ok if r["device"] != "cpu"]
    discrete = [r for r in gpus if (r.get("vram_mb") or 0) >= 2048]
    if len(discrete) >= 2:
        return sorted(discrete, key=lambda r: (r["vram_mb"], -r.get("speed", 0)))[0]["device"]
    if discrete:
        return discrete[0]["device"]
    cpu = next((r for r in ok if r["device"] == "cpu"), None)
    if gpus:
        best = max(gpus, key=lambda r: r.get("speed", 0))
        if not cpu or best.get("speed", 0) > cpu.get("speed", 0):
            return best["device"]
    return "cpu"


def selftest(preferred: str | None = None, log=print, quick: bool = False) -> dict:
    """各装置で実際に計算させ、CPU と同じ答えが出るかを確かめる。

    「GPU が見えている」と「GPU で正しく計算できる」は別の話で、
    前の版はその区別をしなかったせいで何度もつまずいた。"""
    if not models_ready()["ready"]:
        return {"ok": False, "error": "モデルがまだありません"}
    feat_p = model_path("ccip", "model_feat.onnx")
    det_p = model_path("head", "model.onnx")
    n = 4 if quick else 8
    x = _synthetic_batch(n, 384)
    ref_sess = make_session(feat_p, "cpu")
    in_name = ref_sess.get_inputs()[0].name
    t0 = time.time()
    ref = ref_sess.run(None, {in_name: x})[0]
    cpu_speed = n / max(time.time() - t0, 1e-6)

    det_ref_sess = make_session(det_p, "cpu")
    det_in = det_ref_sess.get_inputs()[0]
    dshape = [d if isinstance(d, int) and d > 0 else 640 for d in det_in.shape]
    dshape[0] = 1
    dx = np.random.default_rng(3).random(dshape, dtype=np.float32)
    det_ref = det_ref_sess.run(None, {det_in.name: dx})[0]

    results = [{"device": "cpu", "name": "CPU", "vram_mb": 0, "ok": True,
                "speed": round(cpu_speed, 2), "diff": 0.0}]
    for cand in device_candidates():
        if cand["device"] == "cpu":
            continue
        r = dict(cand)
        try:
            s = make_session(feat_p, cand["device"])
            s.run(None, {in_name: x[:1]})               # 準備の分は測らない
            t0 = time.time()
            out = s.run(None, {in_name: x})[0]
            r["speed"] = round(n / max(time.time() - t0, 1e-6), 2)
            a = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-9)
            b = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
            cos = float((a * b).sum(axis=1).min())
            r["diff"] = round(1.0 - cos, 6)
            ds = make_session(det_p, cand["device"])
            dout = ds.run(None, {det_in.name: dx})[0]
            scale = float(np.abs(det_ref).max()) or 1.0
            r["det_diff"] = round(float(np.abs(dout - det_ref).max()) / scale, 6)
            r["ok"] = bool(cos >= 0.999 and r["det_diff"] <= 0.02)
            if not r["ok"]:
                r["error"] = "CPU と答えが合いません"
        except Exception as e:
            r["ok"] = False
            r["error"] = str(e).splitlines()[0][:200]
        results.append(r)
        mark = "OK " if r["ok"] else "NG "
        log(f"  {mark} {r['device']:<7} {r.get('name', '')[:40]:<40}"
            + (f"  {r['speed']:.1f} 枚/秒" if r.get("speed") else "")
            + (f"  ({r['error']})" if r.get("error") else ""))
    log(f"  OK  cpu     CPU{'':<37}  {cpu_speed:.1f} 枚/秒")
    chosen = choose_device(results, preferred)
    env = load_env()
    env.update({
        "version": VERSION, "tested_at": time.time(),
        "python": sys.version.split()[0], "ort": ort().__version__,
        "providers": ort().get_available_providers(),
        "devices": results, "device": chosen,
    })
    save_env(env)
    return {"ok": True, "devices": results, "device": chosen}


# --------------------------------------------------------------------------
# 画像の読み込み
# --------------------------------------------------------------------------

def load_rgb(src) -> Image.Image:
    """RGB で読む。透明部分は白で埋める（imgutils と同じ扱い）。"""
    im = src if isinstance(src, Image.Image) else Image.open(src)
    im.load()
    alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in (im.info or {})
    if alpha:
        # imgutils と同じ手順。パレット画像は貼り付けに失敗して素通りする
        # （その場合は透明部分がパレットの色になる）ところまで同じにする。
        try:
            bg = Image.new("RGBA", im.size, "white")
            bg.paste(im, (0, 0), mask=im)
        except ValueError:
            bg = im
        im = bg
    if im.mode != "RGB":
        return im.convert("RGB")
    return im


# --------------------------------------------------------------------------
# 頭部検出（imgutils.generic.yolo の写し）
# --------------------------------------------------------------------------

def _yolo_xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def _yolo_nms(boxes, scores, iou_threshold: float = 0.7) -> list:
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def _image_preprocess(image: Image.Image, max_infer_size=1216,
                      allow_dynamic: bool = False, align: int = 32):
    if isinstance(max_infer_size, int):
        max_w, max_h = max_infer_size, max_infer_size
    else:
        max_w, max_h = max_infer_size
    old_w, old_h = image.width, image.height
    new_w, new_h = old_w, old_h
    if allow_dynamic:
        r = min(max_w / new_w, max_h / new_h)
        if r < 1:
            new_w, new_h = new_w * r, new_h * r
        new_w = int(math.ceil(new_w / align) * align)
        new_h = int(math.ceil(new_h / align) * align)
    else:
        new_w, new_h = max_w, max_h
    image = image.resize((new_w, new_h))
    return image, (old_w, old_h), (new_w, new_h)


def _xy_postprocess(x, y, old_size, new_size):
    old_w, old_h = old_size
    new_w, new_h = new_size
    x, y = x / new_w * old_w, y / new_h * old_h
    x = int(np.clip(x, a_min=0, a_max=old_w).round())
    y = int(np.clip(y, a_min=0, a_max=old_h).round())
    return x, y


def _end2end_postprocess(output, conf_threshold, iou_threshold, old_size, new_size, labels):
    assert output.shape[-1] == 6
    detections = []
    output = output[output[:, 4] > conf_threshold]
    selected = _yolo_nms(output[:, :4], output[:, 4])
    for x0, y0, x1, y1, score, cls in output[selected]:
        x0, y0 = _xy_postprocess(x0, y0, old_size, new_size)
        x1, y1 = _xy_postprocess(x1, y1, old_size, new_size)
        detections.append(((x0, y0, x1, y1), labels[int(cls.item())], float(score)))
    return detections


def _nms_postprocess(output, conf_threshold, iou_threshold, old_size, new_size, labels):
    assert output.shape[0] == 4 + len(labels)
    max_scores = output[4:, :].max(axis=0)
    output = output[:, max_scores > conf_threshold].transpose(1, 0)
    boxes = output[:, :4]
    scores = output[:, 4:]
    filtered = scores.max(axis=1)
    if not boxes.size:
        return []
    boxes = _yolo_xywh2xyxy(boxes)
    idx = _yolo_nms(boxes, filtered, iou_threshold=iou_threshold)
    boxes, scores = boxes[idx], scores[idx]
    detections = []
    for box, score in zip(boxes, scores):
        x0, y0 = _xy_postprocess(box[0], box[1], old_size, new_size)
        x1, y1 = _xy_postprocess(box[2], box[3], old_size, new_size)
        k = score.argmax()
        detections.append(((x0, y0, x1, y1), labels[k], float(score[k])))
    return detections


def _yolo_postprocess(output, conf_threshold, iou_threshold, old_size, new_size, labels):
    if output.shape[-1] == 6:
        return _end2end_postprocess(output, conf_threshold, iou_threshold,
                                    old_size, new_size, labels)
    return _nms_postprocess(output, conf_threshold, iou_threshold,
                            old_size, new_size, labels)


def _rtdetr_postprocess(output, conf_threshold, iou_threshold, old_size, new_size, labels):
    assert output.shape[-1] == 4 + len(labels)
    return _nms_postprocess(output.transpose(1, 0), conf_threshold, iou_threshold,
                            old_size, (1.0, 1.0), labels)


def _safe_eval_names(names_str: str) -> dict:
    node = ast.parse(names_str, mode="eval")
    out = {}
    for k, v in zip(node.body.keys, node.body.values):
        out[ast.literal_eval(k)] = ast.literal_eval(v)
    return out


def _rgb_encode(image: Image.Image) -> np.ndarray:
    """imgutils.data.rgb_encode と同じ（CHW, 0〜1 の float32）。"""
    arr = np.asarray(image)
    arr = np.transpose(arr, (2, 0, 1))
    return (arr / 255.0).astype(np.float32)


class HeadDetector:
    """アニメの頭部を探す。枠は髪や耳まで含む頭全体。"""

    def __init__(self, device: str = "cpu", conf: float = 0.4, iou: float = 0.7):
        p = model_path("head", "model.onnx")
        self.sess = make_session(p, device)
        self.device = device
        meta = self.sess.get_modelmeta().custom_metadata_map
        if "imgsz" in meta:
            sz = json.loads(meta["imgsz"])
            self.imgsz = (int(sz), int(sz)) if isinstance(sz, (int, float)) else tuple(sz)
        else:
            self.imgsz = (640, 640)
        names = _safe_eval_names(meta.get("names", "{0: 'head'}"))
        self.labels = [names[i] for i in range(len(names))]
        self.model_type = "yolo"
        mt = model_path("head", "model_type.json")
        if mt.exists():
            try:
                self.model_type = json.loads(mt.read_text(encoding="utf-8"))["model_type"]
            except Exception:
                pass
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.conf, self.iou = float(conf), float(iou)
        self.name = "head_detect_v2.0_s"
        self._lock = threading.Lock()

    def prepare(self, im: Image.Image):
        """別スレッドで呼んでよい。推論に渡す形まで整える。"""
        new_im, old, new = _image_preprocess(im, self.imgsz, allow_dynamic=False)
        return _rgb_encode(new_im), old, new

    def detect_prepared(self, arr, old, new) -> list[tuple]:
        with self._lock:
            out = self.sess.run([self.out_name], {self.in_name: arr[None, ...]})[0]
        if self.model_type == "rtdetr":
            dets = _rtdetr_postprocess(out[0], self.conf, self.iou, old, new, self.labels)
        else:
            dets = _yolo_postprocess(out[0], self.conf, self.iou, old, new, self.labels)
        return [(box, score) for box, _label, score in dets]

    def detect(self, im: Image.Image) -> list[tuple]:
        return self.detect_prepared(*self.prepare(im))


# --------------------------------------------------------------------------
# CCIP（imgutils.metrics.ccip の写し）
# --------------------------------------------------------------------------

CLIP_MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073))[:, None, None]
CLIP_STD = np.asarray((0.26862954, 0.26130258, 0.27577711))[:, None, None]
CCIP_SIZE = 384


def ccip_prepare(im: Image.Image, size: int = CCIP_SIZE) -> np.ndarray:
    """1キャラの画像を CCIP の入力にする。縦横比は保たず 384 四方に伸ばす
    （学習時と同じ扱い）。"""
    im = im.resize((size, size), resample=Image.BILINEAR)
    data = np.array(im).transpose(2, 0, 1).astype(np.float32) / 255.0
    data = (data - CLIP_MEAN) / CLIP_STD
    return data.astype(np.float32)


class CCIP:
    """アニメキャラの同一判定用の特徴を取る。"""

    def __init__(self, device: str = "cpu"):
        self.sess = make_session(model_path("ccip", "model_feat.onnx"), device)
        self.device = device
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.name = MODELS["ccip"]["dir"]
        meta = json.loads(model_path("ccip", "metrics.json").read_text(encoding="utf-8"))
        self.threshold = float(meta["threshold"])
        cl = model_path("ccip", "cluster.json")
        self.cluster = {}
        if cl.exists():
            try:
                self.cluster = json.loads(cl.read_text(encoding="utf-8"))
            except Exception:
                self.cluster = {}
        self._lock = threading.Lock()

    def features(self, batch: np.ndarray) -> np.ndarray:
        with self._lock:
            return self.sess.run([self.out_name], {self.in_name: batch})[0].astype(np.float32)


def ccip_thresholds() -> dict:
    """モデルに付いてくる基準線を読む。自分で決めた数字は使わない。

    review : 同じキャラかどうかの、公開ベンチマークで最もよく当たる線
    auto   : まとまりの純度を優先して選ばれた、より厳しい線
             （cluster.json の dbscan_2 / dbscan_free。無ければ review と同じ）"""
    meta = json.loads(model_path("ccip", "metrics.json").read_text(encoding="utf-8"))
    review = float(meta["threshold"])
    auto = review
    cl = model_path("ccip", "cluster.json")
    if cl.exists():
        try:
            c = json.loads(cl.read_text(encoding="utf-8"))
            cands = [float(c[k]["eps"]) for k in ("dbscan_2", "dbscan_free")
                     if isinstance(c.get(k), dict) and "eps" in c[k]]
            if cands:
                auto = min(min(cands), review)
        except Exception:
            pass
    return {"auto": auto, "review": review}


# --------------------------------------------------------------------------
# 差の値（CCIP の metric）
# --------------------------------------------------------------------------

class Metric:
    """特徴どうしの「差」を計算する。

    CCIP は特徴を取るモデルとは別に、差を出す小さなモデルを持っている。
    13万個の顔について総当たりで何度も使うので、そのモデルが簡単な式と
    同じ答えを出すことを最初に確かめ、同じなら行列計算で済ませる。
    合わなければモデルそのものを少しずつ回す（遅いが正しい）。"""

    def __init__(self, mode: str, params: dict | None = None, sess=None):
        self.mode = mode
        self.params = params or {}
        self.sess = sess
        self.normalized = mode in ("cos_linear", "cos_table")

    # --- 判定 ---------------------------------------------------------
    @staticmethod
    def detect(sess, dim: int, seed: int = 0) -> "Metric":
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        rng = np.random.default_rng(seed)

        def run(X):
            return sess.run([out_name], {in_name: X.astype(np.float32)})[0].astype(np.float64)

        # 1. 長さを変えても同じ答えか
        X = rng.normal(size=(48, dim))
        s1 = rng.uniform(0.5, 2.0, size=(48, 1))
        s2 = rng.uniform(0.5, 2.0, size=(48, 1))
        M1, M2 = run(X * s1), run(X * s2)
        norm_free = float(np.abs(M1 - M2).max()) < 1e-4
        # 2. 並びや組み合わせを変えても同じ答えか（組ごとに決まる量か）
        perm = rng.permutation(48)
        M3 = run(X[perm] * s1[perm])
        pairwise = float(np.abs(M3 - M1[np.ix_(perm, perm)]).max()) < 1e-4

        if norm_free and pairwise:
            # 余弦を -1〜1 まで揃えて、差との対応表を作る
            cs = np.linspace(-1.0, 1.0, 201)
            A, B = [], []
            for c in cs:
                a = rng.normal(size=dim); a /= np.linalg.norm(a)
                u = rng.normal(size=dim); u -= a * (u @ a); u /= np.linalg.norm(u)
                b = c * a + math.sqrt(max(0.0, 1 - c * c)) * u
                A.append(a); B.append(b)
            A, B = np.stack(A), np.stack(B)
            ds = []
            for i in range(0, len(cs), 50):
                M = run(np.vstack([A[i:i + 50], B[i:i + 50]]))
                k = len(A[i:i + 50])
                ds += [M[j, k + j] for j in range(k)]
            ds = np.asarray(ds)
            # 検算用: ふつうの乱数の組
            Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
            C = Xn @ Xn.T
            iu = np.triu_indices(48, 1)
            c_chk, d_chk = C[iu], M1[iu]
            # 直線で表せるか
            b, a = np.polyfit(cs, ds, 1)
            lin_err = max(float(np.abs(a + b * cs - ds).max()),
                          float(np.abs(a + b * c_chk - d_chk).max()))
            if lin_err < 2e-4:
                return Metric("cos_linear", {"a": float(a), "b": float(b), "err": lin_err},
                              sess)
            order = np.argsort(cs)
            tab_c, tab_d = cs[order], ds[order]
            tab_err = float(np.abs(np.interp(c_chk, tab_c, tab_d) - d_chk).max())
            if tab_err < 2e-3:
                return Metric("cos_table", {"c": tab_c.tolist(), "d": tab_d.tolist(),
                                            "err": tab_err}, sess)
        if pairwise:
            E = np.sqrt(np.maximum(((X * s1)[:, None, :] - (X * s1)[None, :, :]) ** 2, 0)
                        .sum(axis=2))
            if float(np.abs(E - M1).max()) < 1e-3:
                return Metric("euclid", {}, sess)
            if float(np.abs(E ** 2 - M1).max()) < 1e-3:
                return Metric("sqeuclid", {}, sess)
        return Metric("onnx", {}, sess)

    def to_json(self) -> dict:
        return {"mode": self.mode, "params": self.params}

    @staticmethod
    def from_json(d: dict, sess=None) -> "Metric":
        return Metric(d["mode"], d.get("params") or {}, sess)

    # --- 計算 ---------------------------------------------------------
    def prep(self, X: np.ndarray) -> np.ndarray:
        """保存用・計算用の形に揃える（余弦で済む場合は長さを 1 に）。"""
        X = np.asarray(X, dtype=np.float32)
        if self.normalized:
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        return X

    def diff(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """A の各行と B の各行の差（len(A) × len(B)）。A, B は prep 済みとする。"""
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        if len(A) == 0 or len(B) == 0:
            return np.zeros((len(A), len(B)), dtype=np.float32)
        if self.mode == "cos_linear":
            C = A @ B.T                  # 大きな行列になるので、その場で書き換える
            C *= np.float32(self.params["b"])
            C += np.float32(self.params["a"])
            return C
        if self.mode == "cos_table":
            C = A @ B.T
            return np.interp(C, self.params["c"], self.params["d"]).astype(np.float32)
        if self.mode in ("euclid", "sqeuclid"):
            aa = (A * A).sum(axis=1)[:, None]
            bb = (B * B).sum(axis=1)[None, :]
            D2 = np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)
            return (np.sqrt(D2) if self.mode == "euclid" else D2).astype(np.float32)
        # onnx: モデルに [A の一部; B の一部] を渡して、交差する区画だけ使う
        in_name = self.sess.get_inputs()[0].name
        out_name = self.sess.get_outputs()[0].name
        out = np.zeros((len(A), len(B)), dtype=np.float32)
        step = 256
        for i in range(0, len(A), step):
            a = A[i:i + step]
            for j in range(0, len(B), step):
                b = B[j:j + step]
                M = self.sess.run([out_name], {in_name: np.vstack([a, b])})[0]
                out[i:i + len(a), j:j + len(b)] = M[:len(a), len(a):]
        return out


def load_metric(log=print) -> Metric:
    """差の計算方法を決める。結果は runtime/metric.json に控えて使い回す。"""
    mp = model_path("ccip", "model_metrics.onnx")
    sess = make_session(mp, "cpu", threads=2)
    sig = f"{mp.stat().st_size}:{_sha256(mp)[:16]}"
    try:
        saved = json.loads(METRIC_PATH.read_text(encoding="utf-8"))
        if saved.get("sig") == sig:
            return Metric.from_json(saved, sess)
    except Exception:
        pass
    feat = make_session(model_path("ccip", "model_feat.onnx"), "cpu")
    dim = int(feat.get_outputs()[0].shape[-1] or 768)
    m = Metric.detect(sess, dim)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_PATH.write_text(json.dumps({"sig": sig, **m.to_json()}, ensure_ascii=False),
                           encoding="utf-8")
    log(f"  差の計算: {m.mode}")
    return m


# --------------------------------------------------------------------------
# 切り出し
# --------------------------------------------------------------------------

def upper_body_box(box, W: int, H: int) -> tuple:
    """頭の枠から、その子の上半身あたりを囲む枠を作る（複数人の絵で CCIP に渡す）。"""
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    cx = (x1 + x2) / 2
    bx1 = max(0, int(cx - 1.5 * w))
    bx2 = min(W, int(cx + 1.5 * w))
    by1 = max(0, int(y1 - 0.35 * h))
    by2 = min(H, int(y2 + 2.3 * h))
    if bx2 - bx1 < 8 or by2 - by1 < 8:
        return (0, 0, W, H)
    return (bx1, by1, bx2, by2)


COLOR_MODEL = "color-v1"
COLOR_DIM = 16


def color_descriptor(sq: Image.Image) -> np.ndarray:
    """頭のあたりの色の割合（髪・ツノの色を捉えるため）。

    CCIP は髪色の違いに鈍いので、色だけは別に測っておく。
    鮮やかな画素は色相 12 区分、くすんだ画素は明るさ 4 区分（黒〜白）に数え、
    合計 1 に揃えて平方根を取る（比べるときの距離が素直になる）。
    服や背景が入りにくいよう、頭の枠の上 7 割・左右の中央 7 割だけを見る。"""
    w, h = sq.size
    reg = sq.crop((int(w * 0.15), 0, int(w * 0.85), int(h * 0.70))).convert("HSV")
    a = np.asarray(reg, dtype=np.float32) / 255.0
    hue, sat, val = a[..., 0].ravel(), a[..., 1].ravel(), a[..., 2].ravel()
    hist = np.zeros(COLOR_DIM, dtype=np.float64)
    vivid = (sat > 0.25) & (val > 0.15)
    if vivid.any():
        hb = np.minimum((hue[vivid] * 12).astype(int), 11)
        np.add.at(hist, hb, sat[vivid] * val[vivid])
    dull = ~vivid
    if dull.any():
        vb = np.minimum((val[dull] * 4).astype(int), 3)
        np.add.at(hist, 12 + vb, 1.0 - sat[dull])
    s = hist.sum()
    if s <= 0:
        hist[:] = 1.0 / COLOR_DIM
    else:
        hist /= s
    return np.sqrt(hist).astype(np.float32)


def head_square(im: Image.Image, box) -> Image.Image:
    """頭を真ん中に置いた正方形。はみ出した所は白で埋める。"""
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    side = max(w, h) * 1.35
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2 - 0.08 * h
    sx1, sy1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    sx2, sy2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    canvas = Image.new("RGB", (max(sx2 - sx1, 1), max(sy2 - sy1, 1)), (255, 255, 255))
    cx1, cy1 = max(0, sx1), max(0, sy1)
    cx2, cy2 = min(im.width, sx2), min(im.height, sy2)
    if cx2 > cx1 and cy2 > cy1:
        canvas.paste(im.crop((cx1, cy1, cx2, cy2)), (cx1 - sx1, cy1 - sy1))
    return canvas


def head_thumb_and_color(im: Image.Image, box, px: int = 128, quality: int = 84):
    """画面用の小さな顔画像（JPEG）と、色の割合を一度に作る。"""
    import io
    small = head_square(im, box).resize((px, px), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue(), color_descriptor(small)


def head_thumb(im: Image.Image, box, px: int = 128, quality: int = 84) -> bytes:
    """画面に並べる顔の小さな画像。ツノが切れないよう上に余白を多めに取る。"""
    import io
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    side = max(w, h) * 1.35
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2 - 0.08 * h
    sx1, sy1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    sx2, sy2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    canvas = Image.new("RGB", (max(sx2 - sx1, 1), max(sy2 - sy1, 1)), (255, 255, 255))
    cx1, cy1 = max(0, sx1), max(0, sy1)
    cx2, cy2 = min(im.width, sx2), min(im.height, sy2)
    if cx2 > cx1 and cy2 > cy1:
        canvas.paste(im.crop((cx1, cy1, cx2, cy2)), (cx1 - sx1, cy1 - sy1))
    canvas = canvas.resize((px, px), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()
