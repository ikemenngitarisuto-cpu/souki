# -*- coding: utf-8 -*-
"""想起の受け口（souki_bridge）

想起（ComfyUI 出力台帳）の「ComfyUI で開く」から送られたワークフローを、
開いている ComfyUI の画面に読み込ませるための小さな拡張です。
ノードは増えません。Python の部品も入れません（ComfyUI に最初から入っているものだけを使います）。

  POST /souki/open     {"workflow": {...} または "api": {...}, "name": "..."}
                        開いている画面すべてに届けます。画面が 1 つも無いときは預かっておき、
                        次に ComfyUI の画面を開いたときに読み込みます（2 分まで）。
  GET  /souki/pending  預かっているワークフローを渡す（画面の起動時に呼ばれる）
  GET  /souki/ping     入っているかどうかと版
"""
import time

from aiohttp import web
from server import PromptServer

BRIDGE_VERSION = "1.0.0"
WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_PENDING = {"data": None, "at": 0.0}
_KEEP_S = 120

routes = PromptServer.instance.routes


@routes.get("/souki/ping")
async def souki_ping(request):
    return web.json_response({"ok": True, "version": BRIDGE_VERSION,
                              "clients": len(getattr(PromptServer.instance, "sockets", {}) or {})})


@routes.post("/souki/open")
async def souki_open(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "JSON ではありません"}, status=400)
    if not isinstance(body, dict) or not (isinstance(body.get("workflow"), dict)
                                          or isinstance(body.get("api"), dict)):
        return web.json_response({"ok": False, "error": "ワークフローがありません"}, status=400)
    data = {"workflow": body.get("workflow"), "api": body.get("api"),
            "name": str(body.get("name") or "想起")[:120]}
    n = len(getattr(PromptServer.instance, "sockets", {}) or {})
    if n:
        await PromptServer.instance.send("souki.open", data)
        _PENDING.update(data=None, at=0.0)
    else:
        _PENDING.update(data=data, at=time.time())
    return web.json_response({"ok": True, "clients": n, "version": BRIDGE_VERSION})


@routes.get("/souki/pending")
async def souki_pending(request):
    d = _PENDING["data"]
    if d is None or time.time() - _PENDING["at"] > _KEEP_S:
        _PENDING.update(data=None, at=0.0)
        return web.json_response({})
    _PENDING.update(data=None, at=0.0)
    return web.json_response(d)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
