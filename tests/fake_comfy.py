# -*- coding: utf-8 -*-
"""テスト用の小さな偽 ComfyUI。

想起が使う口だけに答える:
  GET  /queue  /object_info  /view  /internal/folder_paths  /souki/ping  /souki/pending
  POST /prompt  /upload/image  /souki/open
テストから中身を見るための口:
  GET  /_test/log     受け取った投入・送り込み・開く要求
  POST /_test/reset   記録を消す

  python tests/fake_comfy.py --port 18188 --input 入力フォルダ --classes 一覧.json
"""
import argparse
import json
import threading
import urllib.parse
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOG = {"prompts": [], "uploads": [], "opens": []}
LOCK = threading.Lock()
INPUT: Path = Path(".")
CLASSES: dict = {}


def object_info() -> dict:
    """ノードの一覧。LoadImage の選択肢は input フォルダの中身（ComfyUI と同じ）。"""
    files = sorted(p.relative_to(INPUT).as_posix() for p in INPUT.rglob("*") if p.is_file())
    out = {}
    for ct, inputs in CLASSES.items():
        req = {}
        for k, v in inputs.items():
            req[k] = [v, {}] if isinstance(v, list) else [v, {}]
        out[ct] = {"input": {"required": req}}
    out["LoadImage"] = {"input": {"required": {"image": [files, {"image_upload": True}]}}}
    return out


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        if u.path == "/queue":
            return self._json({"queue_running": [], "queue_pending": []})
        if u.path == "/object_info":
            return self._json(object_info())
        if u.path in ("/internal/folder_paths", "/api/internal/folder_paths"):
            return self._json({"custom_nodes": [[str(INPUT.parent / "custom_nodes")]]})
        if u.path == "/view":
            if qs.get("type", "output") != "input":
                return self._send(404)
            p = INPUT / qs.get("subfolder", "") / qs.get("filename", "")
            if p.is_file():
                return self._send(200, p.read_bytes(), "image/png")
            return self._send(404)
        if u.path == "/souki/ping":
            return self._json({"ok": True, "version": "1.0.0", "clients": 1})
        if u.path == "/souki/pending":
            return self._json({})
        if u.path == "/_test/log":
            with LOCK:
                return self._json(LOG)
        return self._send(404)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if u.path == "/prompt":
            body = json.loads(raw.decode("utf-8"))
            with LOCK:
                LOG["prompts"].append(body)
                num = len(LOG["prompts"])
            return self._json({"prompt_id": f"fake-{num}", "number": num, "node_errors": {}})
        if u.path == "/upload/image":
            msg = BytesParser(policy=email_policy).parsebytes(
                b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw)
            fields, data, fname = {}, None, None
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if part.get_filename():
                    fname, data = part.get_filename(), part.get_payload(decode=True)
                else:
                    fields[name] = part.get_content().strip()
            sub = fields.get("subfolder", "")
            d = INPUT / sub
            d.mkdir(parents=True, exist_ok=True)
            (d / fname).write_bytes(data)
            with LOCK:
                LOG["uploads"].append({"name": fname, "subfolder": sub, "size": len(data),
                                       "type": fields.get("type")})
            return self._json({"name": fname, "subfolder": sub, "type": "input"})
        if u.path == "/souki/open":
            with LOCK:
                LOG["opens"].append(json.loads(raw.decode("utf-8")))
            return self._json({"ok": True, "clients": 1})
        if u.path == "/_test/reset":
            with LOCK:
                for v in LOG.values():
                    v.clear()
            return self._json({"ok": True})
        return self._send(404)


def main():
    global INPUT, CLASSES
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--classes", required=True)
    a = ap.parse_args()
    INPUT = Path(a.input)
    INPUT.mkdir(parents=True, exist_ok=True)
    CLASSES = json.loads(Path(a.classes).read_text(encoding="utf-8"))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    srv.daemon_threads = True
    srv.serve_forever()


if __name__ == "__main__":
    main()
