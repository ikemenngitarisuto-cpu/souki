# -*- coding: utf-8 -*-
"""想起のテスト。偽の ComfyUI を立て、想起のサーバを立てて API を確かめる。

  python -m unittest discover -s tests -v        （リポの一番上で）

作業用のフォルダ（一時フォルダ）に想起の部品を写して動かすので、リポの data/ や config.json は汚さない。
画面のテスト（Playwright）は tests/ui_check.js。node と playwright があれば、ここから一緒に回す。
"""
import copy
import json
import os
import random
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
APP_FILES = ("souki.py", "chara.py", "vision.py", "ui.html", "chara.html", "souki.ico",
             "souki_256.png", "chara.ico", "chara_256.png")
MODEL_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".sft", ".gguf")

ENV: dict = {}


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def noise(seed, size=(64, 96), color=None):
    rnd = random.Random(seed)
    im = Image.new("RGB", size)
    im.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) if color is None else color
                for _ in range(size[0] * size[1])])
    return im


def save_png(path: Path, prompt=None, workflow=None, seed=0, im=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngInfo()
    if prompt is not None:
        info.add_text("prompt", json.dumps(prompt, ensure_ascii=False))
    if workflow is not None:
        info.add_text("workflow", json.dumps(workflow, ensure_ascii=False))
    (im or noise(seed)).save(path, pnginfo=info)


def plain_workflow(pos: str, neg: str, prefix: str):
    """ふつうの SDXL 形（CLIPTextEncode 2 つ・KSampler・SaveImage）。"""
    api = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "illustriousPencilXL_v320.safetensors"},
              "_meta": {"title": "Load Checkpoint"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["4", 1]},
              "_meta": {"title": "CLIP Text Encode (Prompt)"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["4", 1]},
              "_meta": {"title": "CLIP Text Encode (Prompt)"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 1},
              "_meta": {"title": "Empty Latent Image"}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 5, "steps": 28, "cfg": 5, "sampler_name": "euler",
                                                   "scheduler": "normal", "denoise": 1, "model": ["4", 0],
                                                   "positive": ["6", 0], "negative": ["7", 0],
                                                   "latent_image": ["5", 0]}, "_meta": {"title": "KSampler"}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
              "_meta": {"title": "VAE Decode"}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]},
              "_meta": {"title": "Save Image"}},
    }
    ui = {"nodes": [
        {"id": 6, "type": "CLIPTextEncode", "inputs": [{"name": "clip", "link": 1},
                                                      {"name": "text", "widget": {"name": "text"}, "link": None}],
         "widgets_values": [pos]},
        {"id": 7, "type": "CLIPTextEncode", "inputs": [{"name": "clip", "link": 2},
                                                      {"name": "text", "widget": {"name": "text"}, "link": None}],
         "widgets_values": [neg]},
        {"id": 3, "type": "KSampler", "inputs": [{"name": "seed", "widget": {"name": "seed"}, "link": None},
                                                {"name": "steps", "widget": {"name": "steps"}, "link": None}],
         "widgets_values": [5, "fixed", 28]},
        {"id": 9, "type": "SaveImage", "inputs": [{"name": "images", "link": 3},
                                                 {"name": "filename_prefix", "widget": {"name": "filename_prefix"},
                                                  "link": None}],
         "widgets_values": ["%date:yyyy%-%date:M%-%date:d%/%date:yyyy%-%date:M%-%date:d%illuだれか"]},
    ], "links": []}
    return api, ui


def model_classes() -> dict:
    out = {}
    for f in FIX.glob("*.api.json"):
        for node in json.loads(f.read_text(encoding="utf-8")).values():
            d = out.setdefault(node["class_type"], {})
            for k, v in node["inputs"].items():
                if isinstance(v, str) and v.lower().endswith(MODEL_SUFFIXES):
                    d.setdefault(k, [])
                    if v not in d[k]:
                        d[k].append(v)
    out.setdefault("CheckpointLoaderSimple", {}).setdefault("ckpt_name", []).append(
        "illustriousPencilXL_v320.safetensors")
    for ct in ("CLIPTextEncode", "KSampler", "EmptyLatentImage", "VAEDecode", "SaveImage"):
        out.setdefault(ct, {})
    return out


def http(method, url, body=None, timeout=60):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw.decode("utf-8")) if raw[:1] in (b"{", b"[") else raw


def api(path, body=None):
    return http("POST" if body is not None else "GET", ENV["souki"] + path, body)


def comfy(path, body=None):
    return http("POST" if body is not None else "GET", ENV["comfy"] + path, body)


def wait_ready(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            st = api("/api/status")
            rp = st["index"].get("reparse") or {}
            if not st["index"].get("running") and st["index"].get("last") and rp and not rp.get("running"):
                return st
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError("想起の準備が終わりません")


def start_souki():
    port = ENV["souki_port"]
    log = open(ENV["tmp"] / f"souki_{port}.log", "a", encoding="utf-8")
    ENV["souki_proc"] = subprocess.Popen(
        [sys.executable, "souki.py", "serve", "--port", str(port), "--no-browser"],
        cwd=ENV["app"], stdout=log, stderr=subprocess.STDOUT,
        env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
    for _ in range(200):
        try:
            api("/api/version")
            break
        except Exception:
            time.sleep(0.1)
    return wait_ready()


def stop_souki():
    p = ENV.pop("souki_proc", None)
    if p:
        p.terminate()
        p.wait(10)


def image_id(filename):
    r = api("/api/search?q=&limit=200")
    for it in r.get("items") or r.get("results") or []:
        if it["filename"] == filename:
            return it["id"]
    raise KeyError(filename)


def db():
    con = sqlite3.connect(ENV["app"] / "data" / "souki.db", timeout=30)
    con.row_factory = sqlite3.Row
    return con


def setUpModule():
    tmp = Path(tempfile.mkdtemp(prefix="souki_test_"))
    app, out, inp = tmp / "app", tmp / "output", tmp / "comfy" / "input"
    app.mkdir()
    for f in APP_FILES:
        shutil.copy(ROOT / f, app / f)
    ENV.update(tmp=tmp, app=app, out=out, input=inp)

    # --- 台帳に載せる絵 ---------------------------------------------------
    ks_api, ks_ui = load("kigae_special.api.json"), load("kigae_special.ui.json")
    save_png(out / "着替工房" / "02_セーラー服_00001_.png", ks_api, ks_ui, seed=1)
    save_png(out / "着替工房" / "一覧_00001_.png", ks_api, ks_ui, seed=2)
    kd_api, kd_ui = load("kigae_standard.api.json"), load("kigae_standard.ui.json")
    save_png(out / "着替工房" / "標準_00001_.png", kd_api, kd_ui, seed=3)
    q_api, q_ui = load("qwen21_edit.api.json"), load("qwen21_edit.ui.json")
    save_png(out / "Qwen_image_2.1_00001_.png", q_api, q_ui, seed=4)
    # 参照画像 image_1 は、台帳にある絵と同じ絵（元の絵としてつながるはず）
    portrait = noise(10, (80, 80))
    p_api, p_ui = plain_workflow("1girl, solo, silver hair, blue eyes, denim jacket",
                                 "worst quality", "2025-3-14/2025-3-14illuルル")
    save_png(out / "2025-3-14" / "2025-3-14illuルル_00001_.png", p_api, p_ui, im=portrait)
    inp.mkdir(parents=True)
    portrait.save(inp / "portrait_model_denim.png")
    noise(11, (80, 80)).save(inp / "clothing_light_blue_denim_shirt.png")
    # 2 人の絵
    m_api, m_ui = plain_workflow("2girls, multiple girls, pink hair, red eyes, silver hair, blue eyes, school uniform",
                                 "worst quality, lowres", "2025-3-14/2025-3-14illuだれか")
    save_png(out / "2025-3-14" / "2025-3-14illuだれか_00001_.png", m_api, m_ui, seed=5)
    # たどれないワークフロー（ポジティブが文字の無いノードから来る）
    b_api, b_ui = plain_workflow("x", "worst quality, bad hands", "bad")
    b_api["6"] = {"class_type": "MysteryEncode", "inputs": {"clip": ["4", 1], "seed": 3},
                  "_meta": {"title": "謎"}}
    save_png(out / "bad_00001_.png", b_api, None, seed=6)

    # --- 偽の ComfyUI ---------------------------------------------------------
    classes = tmp / "classes.json"
    classes.write_text(json.dumps(model_classes(), ensure_ascii=False), encoding="utf-8")
    cport = free_port()
    ENV["comfy_proc"] = subprocess.Popen(
        [sys.executable, str(ROOT / "tests" / "fake_comfy.py"), "--port", str(cport),
         "--input", str(inp), "--classes", str(classes)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ENV["comfy"] = f"http://127.0.0.1:{cport}"
    (app / "config.json").write_text(json.dumps(
        {"output_dirs": [str(out)], "comfy_url": ENV["comfy"], "scan_workers": 2,
         "comfy_input": str(inp)}, ensure_ascii=False), encoding="utf-8")
    ENV["souki_port"] = free_port()
    ENV["souki"] = f"http://127.0.0.1:{ENV['souki_port']}"
    for _ in range(100):
        try:
            comfy("/queue")
            break
        except Exception:
            time.sleep(0.1)
    start_souki()
    setup_characters()


def setup_characters():
    """キャラ帳の代わりに、キャラと顔の場所を直接書く（顔の解析はしない）。"""
    con = db()
    cur = con.cursor()
    ids = {r["filename"]: r["id"] for r in cur.execute("SELECT id, filename FROM images")}
    ENV["ids"] = ids
    chars = {"ルル": "silver hair, blue eyes, long hair", "ピナ": "pink hair, red eyes, short hair",
             "ナナ": "black hair, green eyes, twintails"}
    cid = {}
    for name, words in chars.items():
        cur.execute("INSERT INTO c_characters(name, words, created_at) VALUES(?,?,?)", (name, words, time.time()))
        cid[name] = cur.lastrowid

    def face(img, idx, x1, x2, who, solo=1):
        cur.execute("INSERT INTO c_instances(image_id, idx, x1, y1, x2, y2, score, solo, created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?)", (ids[img], idx, x1, 0, x2, x2 - x1, 0.9, solo, time.time()))
        cur.execute("INSERT INTO c_assign(instance_id, character_id, state, updated_at) VALUES(?,?,?,?)",
                    (cur.lastrowid, cid[who], "confirmed", time.time()))
    face("02_セーラー服_00001_.png", 0, 10, 60, "ルル")
    face("標準_00001_.png", 0, 10, 60, "ルル")
    face("2025-3-14illuルル_00001_.png", 0, 10, 60, "ルル")
    # 左にピナ（小さい）、右にルル（大きい）
    face("2025-3-14illuだれか_00001_.png", 0, 0, 30, "ピナ", solo=0)
    face("2025-3-14illuだれか_00001_.png", 1, 40, 100, "ルル", solo=0)
    con.commit()
    con.close()


def tearDownModule():
    stop_souki()
    p = ENV.get("comfy_proc")
    if p:
        p.terminate()
        p.wait(10)
    if not os.environ.get("SOUKI_KEEP_TMP"):
        shutil.rmtree(ENV["tmp"], ignore_errors=True)


def last_prompt():
    log = comfy("/_test/log")
    return log["prompts"][-1]


def ui_node(wf, nid, sub=None):
    nodes = wf["nodes"] if sub is None else next(s for s in wf["definitions"]["subgraphs"]
                                                   if s["id"] == sub)["nodes"]
    return next(n for n in nodes if str(n["id"]) == str(nid))


class Reading(unittest.TestCase):
    """台帳の読み取り（ポジティブ・ネガティブ・タグ・参照画像）。"""

    def test_kigae_special_positive_is_not_negative(self):
        d = api(f"/api/detail/{ENV['ids']['02_セーラー服_00001_.png']}")
        self.assertIn("serafuku", d["positive"])
        self.assertNotIn("maid", d["positive"])           # その絵の服の行だけ
        self.assertNotIn("worst quality", d["positive"])
        self.assertTrue(d["negative"].startswith("worst quality"))
        self.assertNotIn("worst quality", d["tags"])
        self.assertIn("silver hair", d["tags"])

    def test_kigae_standard_positive(self):
        d = api(f"/api/detail/{ENV['ids']['標準_00001_.png']}")
        self.assertEqual(d["positive"], "masterpiece, best quality, amazing quality, absurdres, 1girl, solo, "
                                        "long hair, silver hair, blue eyes, maid, maid headdress, white apron, "
                                        "black dress, frills, standing, smile, looking at viewer, full body, "
                                        "simple background, white background")
        self.assertNotIn("worst quality", d["tags"])

    def test_qwen_prose_and_refs(self):
        d = api(f"/api/detail/{ENV['ids']['Qwen_image_2.1_00001_.png']}")
        self.assertEqual(d["positive"], "Replace costumes for the character")
        self.assertEqual(d["tags"], [])                   # 文章のプロンプトはタグ一覧に入れない
        self.assertEqual([r["slot"] for r in d["refs"]], ["image_1", "image_2"])
        self.assertEqual(d["refs"][0]["filename"], "portrait_model_denim.png")
        self.assertTrue(all(r["kept"] for r in d["refs"]), d["refs"])
        self.assertEqual(d["refs"][0]["src_image_id"], ENV["ids"]["2025-3-14illuルル_00001_.png"])
        self.assertIsNone(d["refs"][1]["src_image_id"])
        raw = urllib.request.urlopen(ENV["souki"] + d["refs"][0]["url"]).read()
        self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")
        # 検索欄では探せる
        r = api("/api/search?text=" + urllib.request.quote("Replace costumes") + "&limit=50")
        names = [it["filename"] for it in (r.get("items") or r.get("results") or [])]
        self.assertIn("Qwen_image_2.1_00001_.png", names)

    def test_untraceable_keeps_negative_out(self):
        d = api(f"/api/detail/{ENV['ids']['bad_00001_.png']}")
        self.assertNotEqual(d["positive"], "worst quality, bad hands")
        self.assertNotIn("bad hands", d["tags"])


class Swapping(unittest.TestCase):
    def setUp(self):
        comfy("/_test/reset", {})
        api("/api/settings", {"save_template": ""})

    def test_kigae_special_character(self):
        iid = ENV["ids"]["02_セーラー服_00001_.png"]
        r = api("/api/swap/preview", {"image_id": iid, "to": "ピナ"})
        self.assertTrue(r["ok"], r)
        self.assertEqual([(e["node"], e["key"]) for e in r["edits"]], [("2", "キャラ")])
        self.assertIn("pink hair", r["edits"][0]["after"])
        self.assertNotIn("silver hair", r["edits"][0]["after"])
        self.assertEqual(r["kigae"]["no"], 2)
        s = api("/api/swap/submit", {"image_id": iid, "to": "ピナ", "seed_mode": "same"})
        self.assertTrue(s["ok"], s)
        p = last_prompt()
        pr = p["prompt"]
        self.assertTrue(pr["5"]["inputs"]["text"].startswith("worst quality"))   # ネガティブはそのまま
        self.assertNotIn("pink hair", pr["5"]["inputs"]["text"])
        self.assertEqual(pr["2"]["inputs"]["服の出し方"], "番号の行だけ")
        self.assertEqual(pr["2"]["inputs"]["番号"], 2)
        self.assertEqual(pr["2"]["inputs"]["服"], load("kigae_special.api.json")["2"]["inputs"]["服"])
        # 線でつながった保存名も、ひな形の名前に
        self.assertIsInstance(pr["10"]["inputs"]["filename_prefix"], str)
        self.assertTrue(pr["10"]["inputs"]["filename_prefix"].endswith("illuピナ"))
        self.assertTrue(pr["12"]["inputs"]["filename_prefix"].endswith("illuピナ_一覧"))
        # 画面用ワークフロー: 線を外して値を入れる・名前つきの値も直す
        wf = p["extra_data"]["extra_pnginfo"]["workflow"]
        n10 = ui_node(wf, 10)
        self.assertIsNone(next(i for i in n10["inputs"] if i["name"] == "filename_prefix")["link"])
        self.assertTrue(n10["widgets_values"][0].endswith("illuピナ"))
        self.assertIn("%date:", n10["widgets_values"][0])
        self.assertNotIn(3, [l[0] for l in wf["links"]])
        self.assertNotIn(3, ui_node(wf, 2)["outputs"][2]["links"])
        n2 = ui_node(wf, 2)
        self.assertEqual(n2["widgets_values"][5], "番号の行だけ")
        self.assertEqual(n2["widgets_values"][6], 2)
        self.assertEqual(n2["widgets_values"][7], "fixed")
        self.assertEqual(n2["widgets_values_named"]["番号"], 2)
        self.assertIn("pink hair", n2["widgets_values"][1])
        self.assertIn("pink hair", n2["widgets_values_named"]["キャラ"])
        self.assertTrue(ui_node(wf, 12)["widgets_values"][0].endswith("illuピナ_一覧"))
        self.assertTrue(ui_node(wf, 12)["widgets_values_named"]["filename_prefix"].endswith("illuピナ_一覧"))

    def test_kigae_special_outfit_line(self):
        api("/api/settings", {"save_template": "%date:yyyy%/{キャラ}_{服}"})
        iid = ENV["ids"]["02_セーラー服_00001_.png"]
        s = api("/api/swap/submit", {"image_id": iid, "keep_char": True, "outfit": ["one-piece swimsuit"],
                                     "outfit_name": "水着", "seed_mode": "same"})
        self.assertTrue(s["ok"], s)
        pr = last_prompt()["prompt"]
        lines = pr["2"]["inputs"]["服"].split("\n")
        self.assertEqual(lines[1], "【水着】one-piece swimsuit")
        self.assertEqual(lines[0], "【メイド】maid, maid headdress, white apron, black dress, frills")
        self.assertEqual(len(lines), 5)
        self.assertEqual(pr["2"]["inputs"]["キャラ"], "1girl, solo, long hair, silver hair, blue eyes")
        self.assertEqual(pr["10"]["inputs"]["filename_prefix"], time.strftime("%Y") + "/ルル_水着")

    def test_outfit_name_from_kigae_label(self):
        api("/api/settings", {"save_template": "%date:yyyy%/{キャラ}_{服}"})
        r = api("/api/swap/preview", {"image_id": ENV["ids"]["02_セーラー服_00001_.png"], "template": True})
        sv = [c for c in r["save_changes"] if c["kind"] == "save"]
        self.assertEqual(sv[0]["after"], time.strftime("%Y") + "/ルル_セーラー服")
        # {服} が空のときは区切りごと消える
        r = api("/api/swap/preview", {"image_id": ENV["ids"]["標準_00001_.png"], "template": True})
        sv = [c for c in r["save_changes"] if c["kind"] == "save"]
        self.assertEqual(sv[0]["after"], time.strftime("%Y") + "/ルル")

    def test_kigae_sheet_cannot_change_outfit(self):
        r = api("/api/swap/preview", {"image_id": ENV["ids"]["一覧_00001_.png"], "keep_char": True,
                                      "outfit": ["kimono"]})
        self.assertFalse(r["ok"])

    def test_kigae_standard_windows(self):
        iid = ENV["ids"]["標準_00001_.png"]
        s = api("/api/swap/submit", {"image_id": iid, "to": "ピナ", "outfit": ["hoodie", "denim shorts"],
                                     "seed_mode": "same"})
        self.assertTrue(s["ok"], s)
        p = last_prompt()
        pr = p["prompt"]
        self.assertEqual(pr["4"]["inputs"]["value"], "hoodie, denim shorts")        # 服の窓は中身ごと
        self.assertIn("pink hair", pr["3"]["inputs"]["value"])                       # キャラの窓へ
        self.assertNotIn("silver hair", pr["3"]["inputs"]["value"])
        for k in ("2", "5", "6"):                                                     # ほかの窓はそのまま
            self.assertEqual(pr[k]["inputs"]["value"], load("kigae_standard.api.json")[k]["inputs"]["value"])
        self.assertEqual(pr["16"]["inputs"]["text"], load("kigae_standard.api.json")["16"]["inputs"]["text"])
        self.assertEqual(sum("pink hair" in pr[k]["inputs"]["value"] for k in ("2", "3", "4", "5", "6")), 1)
        self.assertTrue(pr["20"]["inputs"]["filename_prefix"].endswith("illuピナ"))
        wf = p["extra_data"]["extra_pnginfo"]["workflow"]
        n4 = ui_node(wf, 4)
        self.assertEqual(n4["widgets_values"][0], "hoodie, denim shorts")
        if "widgets_values_named" in n4:
            self.assertEqual(n4["widgets_values_named"]["value"], "hoodie, denim shorts")

    def test_untraceable_refuses(self):
        r = api("/api/swap/preview", {"image_id": ENV["ids"]["bad_00001_.png"], "to": "ピナ"})
        self.assertFalse(r["ok"])
        self.assertIn("書き換える欄が見つかりません", r["error"])

    def test_qwen_no_word_swap(self):
        r = api("/api/swap/preview", {"image_id": ENV["ids"]["Qwen_image_2.1_00001_.png"], "to": "ピナ"})
        self.assertFalse(r["ok"])
        self.assertIn("文章", r["error"])


class SaveNames(unittest.TestCase):
    def setUp(self):
        comfy("/_test/reset", {})
        api("/api/settings", {"save_template": ""})

    def test_two_characters_left_to_right(self):
        iid = ENV["ids"]["2025-3-14illuだれか_00001_.png"]
        s = api("/api/requeue", {"id": iid, "seed_mode": "same"})
        self.assertTrue(s["ok"], s)
        v = last_prompt()["prompt"]["9"]["inputs"]["filename_prefix"]
        self.assertTrue(v.endswith("illuピナとルル"), v)

    def test_swap_replaces_only_that_name(self):
        iid = ENV["ids"]["2025-3-14illuだれか_00001_.png"]
        s = api("/api/swap/submit", {"image_id": iid, "to": "ナナ", "seed_mode": "same"})
        self.assertTrue(s["ok"], s)
        v = last_prompt()["prompt"]["9"]["inputs"]["filename_prefix"]
        self.assertTrue(v.endswith("illuピナとナナ"), v)

    def test_ask_only_when_nobody_known(self):
        r = api("/api/requeue", {"id": ENV["ids"]["Qwen_image_2.1_00001_.png"], "seed_mode": "same"})
        self.assertFalse(r["ok"])
        self.assertTrue(r.get("need_name"))


class QwenSubmit(unittest.TestCase):
    def setUp(self):
        comfy("/_test/reset", {})

    def test_restore_and_subgraph_seed(self):
        iid = ENV["ids"]["Qwen_image_2.1_00001_.png"]
        (ENV["input"] / "portrait_model_denim.png").unlink()
        # ランプ: 参照画像が input に無い
        for _ in range(100):
            v = api("/api/validate", {"ids": [iid]})
            if not v.get("pending"):
                break
            time.sleep(0.3)
        res = v["results"][str(iid)]
        self.assertEqual(res["status"], "partial", res)
        self.assertEqual(res["missing_images"][0]["value"], "portrait_model_denim.png")
        s = api("/api/requeue", {"id": iid, "seed_mode": "fixed", "seed_value": "12345",
                                 "save_name": "テスト"})
        self.assertTrue(s["ok"], s)
        self.assertEqual([x["ok"] for x in s["restored"]], [True])
        self.assertTrue((ENV["input"] / "portrait_model_denim.png").is_file())
        log = comfy("/_test/log")
        self.assertEqual(log["uploads"][0]["name"], "portrait_model_denim.png")
        p = log["prompts"][-1]
        self.assertEqual(p["prompt"]["459:458"]["inputs"]["seed"], 12345)
        self.assertEqual(p["prompt"]["461"]["inputs"]["filename_prefix"][-7:], "illuテスト")
        wf = p["extra_data"]["extra_pnginfo"]["workflow"]
        outer = ui_node(wf, 459)
        self.assertEqual(outer["widgets_values_named"]["seed"], 12345)
        self.assertEqual(outer["widgets_values"][13], 12345)
        self.assertEqual(outer["widgets_values_named"]["prompt"], "Replace costumes for the character")


class Ledger(unittest.TestCase):
    def test_reparse_fixes_old_rows(self):
        iid = ENV["ids"]["標準_00001_.png"]
        con = db()
        con.execute("UPDATE images SET positive='worst quality, bad quality' WHERE id=?", (iid,))
        con.execute("DELETE FROM tags WHERE image_id=?", (iid,))
        con.execute("INSERT INTO tags(image_id, tag) VALUES(?, 'worst quality')", (iid,))
        con.execute("INSERT OR REPLACE INTO c_analysis(image_id, status, words) VALUES(?, 'ok', '{}')", (iid,))
        con.execute("DELETE FROM meta WHERE key='parse_version'")
        con.commit()
        con.close()
        stop_souki()
        start_souki()
        d = api(f"/api/detail/{iid}")
        self.assertIn("silver hair", d["positive"])
        self.assertNotIn("worst quality", d["tags"])
        con = db()
        w = json.loads(con.execute("SELECT words FROM c_analysis WHERE image_id=?", (iid,)).fetchone()[0])
        con.close()
        self.assertTrue(w.get("hair"), w)              # 銀髪の語から読み直した
        self.assertFalse(w.get("multi"))


class Window(unittest.TestCase):
    def test_alive_and_open_window_guard(self):
        self.assertTrue(api("/api/alive", {"x": 10, "y": 20, "w": 1200, "h": 800})["ok"])
        cfg = json.loads((ENV["app"] / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["window"], {"x": 10, "y": 20, "w": 1200, "h": 800})
        self.assertFalse(api("/api/open_window", {"url": "file:///etc/passwd"})["ok"])

    def test_closing_window_stops_server(self):
        """専用の窓を閉じたら（画面から「閉じた」の知らせが来て、そのまま戻らなければ）想起も終わる。"""
        port = free_port()
        app2 = ENV["tmp"] / "app2"
        if not app2.exists():
            shutil.copytree(ENV["app"], app2, ignore=shutil.ignore_patterns("data"))
        p = subprocess.Popen([sys.executable, "souki.py", "serve", "--port", str(port), "--no-browser",
                              "--no-index", "--app"], cwd=app2, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        try:
            url = f"http://127.0.0.1:{port}"
            for _ in range(200):
                try:
                    http("GET", url + "/api/version")
                    break
                except Exception:
                    time.sleep(0.1)
            http("POST", url + "/api/alive", {})
            time.sleep(3)
            self.assertIsNone(p.poll())                      # 開いている間は終わらない
            http("POST", url + "/api/bye", {})
            p.wait(40)
            self.assertIsNotNone(p.poll())
        finally:
            if p.poll() is None:
                p.kill()

    def test_edge_window_has_own_taskbar_id(self):
        """Edge は --app の窓に「URL のホスト_パス」と「設定フォルダの名前」から名札を付け、名札が同じ窓を
        タスクバーで 1 つにまとめる。台割工房（http://127.0.0.1:<ポート>/ と設定フォルダ edge）と違う名札になること。"""
        d = ENV["tmp"] / "app_edge"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        for f in APP_FILES:
            shutil.copy(ROOT / f, d / f)
        (d / "data" / "edge").mkdir(parents=True)              # 前の版の設定フォルダ
        code = ("import json, souki; u = souki.app_window_url(8765); "
                "a = souki.edge_command('msedge.exe', u, {'window': {'w': 1200, 'h': 800}}); "
                "print(json.dumps([a, souki.chromium_app_id(u, souki.EDGE_DIR), "
                "souki.chromium_app_id('http://127.0.0.1:8765/', r'C:\\souki\\data\\edge'), "
                "souki.chromium_app_id('http://127.0.0.1:8770/', r'C:\\daiwari\\data\\edge')]))")
        r = subprocess.run([sys.executable, "-c", code], cwd=d, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONUTF8="1"))
        args, ours, old, daiwari = json.loads(r.stdout)
        self.assertIn("--app=http://127.0.0.1:8765/souki", args)
        udd = next(a for a in args if a.startswith("--user-data-dir="))
        self.assertTrue(udd.endswith("souki_edge"), udd)
        self.assertIn("--window-size=1200,800", args)
        self.assertEqual(old, daiwari)                          # 前の版は台割工房と同じ名札だった
        self.assertNotEqual(ours, daiwari)
        self.assertEqual(ours, "MSEdge.127.0.0.1_/souki.soukiedge.Default")
        self.assertTrue((d / "data" / "souki_edge").is_dir())   # 前の版の設定フォルダを引き継ぐ
        self.assertFalse((d / "data" / "edge").exists())

    def test_identity_is_reapplied_when_edge_resets_it(self):
        """名札を付けたあと Edge に戻されても、次に見たとき付け直す（Windows の呼び出しは偽物）。"""
        sys.path.insert(0, str(ENV["app"]))
        try:
            import souki
        finally:
            sys.path.pop(0)

        class FakeWin:
            def __init__(self):
                self.ids = {101: "MSEdge.127.0.0.1_/.edge.Default", 202: "MSEdge.127.0.0.1_/.edge.Default"}
                self.icons, self.sets = [], 0

            def windows(self):
                return [(101, "想起 — ComfyUI 出力台帳")]       # 202 は台割工房の窓（題が違うので触らない）

            def get_id(self, h):
                return self.ids.get(h)

            def set_identity(self, h, values):
                self.sets += 1
                self.ids[h] = values[souki.PID_ID]
                return [0, 0, 0, 0, 0]

            def set_icon(self, h, icon):
                self.icons.append(h)

        w, values = FakeWin(), souki.identity_values()
        seen = souki.identity_pass(w, values, {})
        self.assertEqual(w.ids[101], "Souki.Ledger")
        self.assertEqual(w.ids[202], "MSEdge.127.0.0.1_/.edge.Default")
        seen = souki.identity_pass(w, values, seen)
        self.assertEqual(w.sets, 1)                              # 付いていれば何もしない
        w.ids[101] = "MSEdge.127.0.0.1_/souki.soukiedge.Default"    # Edge が付け直した
        seen = souki.identity_pass(w, values, seen)
        self.assertEqual((w.ids[101], w.sets, w.icons), ("Souki.Ledger", 2, [101]))
        self.assertTrue(values[souki.PID_RELAUNCH_COMMAND].endswith('/c ""' + str(ENV["app"] / "起動.bat") + '""'))
        self.assertTrue(values[souki.PID_RELAUNCH_ICON].endswith("souki.ico,0"))

    def test_window_entry_redirects_to_ledger(self):
        with urllib.request.urlopen(ENV["souki"] + "/souki?image=5") as r:
            self.assertTrue(r.geturl().endswith("/?image=5"), r.geturl())
            self.assertIn("<title>想起", r.read().decode("utf-8"))

    def test_chara_page_has_no_new_tabs(self):
        html = (ROOT / "chara.html").read_text(encoding="utf-8")
        self.assertNotIn('target = "_blank"', html)


class Versions(unittest.TestCase):
    def test_versions_match(self):
        import re
        v = {}
        for f, pat in (("souki.py", r'^VERSION = "([^"]+)"'), ("chara.py", r'^VERSION = "([^"]+)"'),
                       ("vision.py", r'^VERSION = "([^"]+)"'), ("ui.html", r'UI_VERSION = "([^"]+)"'),
                       ("chara.html", r'PAGE_VERSION = "([^"]+)"')):
            v[f] = re.search(pat, (ROOT / f).read_text(encoding="utf-8"), re.M).group(1)
        self.assertEqual(len(set(v.values())), 1, v)

    def test_bat_files_ascii_crlf(self):
        for f in [ROOT / "起動.bat", *(ROOT / "setup").glob("*.bat")]:
            b = f.read_bytes()
            b.decode("ascii")
            self.assertNotIn(b"\n", b.replace(b"\r\n", b""), f.name)

    def test_stale_bytecode_after_overwrite(self):
        """zip で上書きして、chara.py・vision.py の日時と大きさが前の版と同じでも、
        前の版の控え（__pycache__）を使わない（v2.4.1 で「chara.py の版（2.4.0）が違います」と出た件）。"""
        d = ENV["tmp"] / "app_pyc"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        for f in APP_FILES:
            shutil.copy(ROOT / f, d / f)
        import re
        ver = re.search(r'^VERSION = "([^"]+)"', (ROOT / "souki.py").read_text(encoding="utf-8"), re.M).group(1)
        old = ver[:-1] + ("0" if ver[-1] != "0" else "1")          # 前の版（文字数は同じ）
        saved = {}
        for name in ("chara.py", "vision.py"):
            src = d / name
            cur = src.read_bytes()
            st = src.stat()
            saved[name] = (cur, st.st_mtime)
            src.write_bytes(cur.replace(f'VERSION = "{ver}"'.encode(), f'VERSION = "{old}"'.encode()))
            os.utime(src, (st.st_mtime, st.st_mtime))
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, "-c", "import chara; print(chara.VERSION)"], cwd=d,
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.stdout.strip(), old, r.stderr)        # 前の版で一度動かした（控えができた）
        for name, (cur, mt) in saved.items():                     # zip で上書き（日時も大きさも同じ）
            (d / name).write_bytes(cur)
            os.utime(d / name, (mt, mt))
        r = subprocess.run([sys.executable, "-c",
                            "import souki; souki.load_chara(); print(souki.CHARA_ERROR); "
                            "print(souki.CHARA and souki.CHARA.VERSION)"],
                           cwd=d, capture_output=True, text=True, env=env)
        self.assertEqual(r.stdout.split(), ["None", ver], r.stdout + r.stderr)

    def test_zip_has_fresh_dates(self):
        """納品の zip のファイルの日時は、作った時刻にする（版ごとに日時が変わるように）。"""
        sys.path.insert(0, str(ROOT / "tests" / "tools"))
        try:
            import make_zip
        finally:
            sys.path.pop(0)
        out = ENV["tmp"] / "test.zip"
        make_zip.build(out)
        import zipfile
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            for f in make_zip.FILES:
                self.assertIn(f, names)
            for i in z.infolist():
                t = time.mktime(i.date_time + (0, 0, -1))
                self.assertLess(abs(time.time() - t), 600, (i.filename, i.date_time))
                if any(ord(c) > 127 for c in i.filename):
                    self.assertTrue(i.flag_bits & 0x800, i.filename)


class Screen(unittest.TestCase):
    """画面（Playwright）。node と playwright が無ければ飛ばす。"""

    def test_ui(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node がありません")
        ok = subprocess.run([node, "-e", "require('playwright')"], capture_output=True).returncode == 0
        if not ok:
            self.skipTest("playwright がありません")
        r = subprocess.run([node, str(ROOT / "tests" / "ui_check.js"), ENV["souki"],
                            json.dumps(ENV["ids"], ensure_ascii=False)],
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
