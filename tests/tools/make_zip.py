# -*- coding: utf-8 -*-
"""納品用の zip（dist/souki_v<版>.zip）を作る。日本語のファイル名が化けないよう UTF-8 の印を立てる。

ファイルの日時は「zip を作った時刻」にする。版ごとに同じ日時にすると、版の数字だけ変わったファイル
（大きさも同じ）で、Python が前の版の控え（__pycache__）を使ってしまう（v2.4.1 で起きた）。

  python tests/tools/make_zip.py
"""
import re
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["起動.bat", "souki.py", "chara.py", "vision.py", "ui.html", "chara.html", "README.md",
         "souki.ico", "souki_256.png", "chara.ico", "chara_256.png",
         "setup/setup.bat", "setup/環境を作り直す.bat", "setup/requirements.txt", "setup/requirements.in",
         "comfyui_node/souki_bridge/__init__.py", "comfyui_node/souki_bridge/web/souki_bridge.js"]


def version() -> str:
    return re.search(r'^VERSION = "([^"]+)"', (ROOT / "souki.py").read_text(encoding="utf-8"), re.M).group(1)


def build(out: Path | None = None) -> Path:
    out = out or ROOT / "dist" / f"souki_v{version()}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    now = time.localtime()[:6]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            info = zipfile.ZipInfo(f, date_time=now)
            info.flag_bits |= 0x800
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, (ROOT / f).read_bytes())
    return out


if __name__ == "__main__":
    print(build())
