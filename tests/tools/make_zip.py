# -*- coding: utf-8 -*-
"""納品用の zip（dist/souki_v<版>.zip）を作る。日本語のファイル名が化けないよう UTF-8 の印を立てる。

  python tests/tools/make_zip.py
"""
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["起動.bat", "souki.py", "chara.py", "vision.py", "ui.html", "chara.html", "README.md",
         "souki.ico", "souki_256.png", "chara.ico", "chara_256.png",
         "setup/setup.bat", "setup/環境を作り直す.bat", "setup/requirements.txt", "setup/requirements.in",
         "comfyui_node/souki_bridge/__init__.py", "comfyui_node/souki_bridge/web/souki_bridge.js"]


def main():
    ver = re.search(r'^VERSION = "([^"]+)"', (ROOT / "souki.py").read_text(encoding="utf-8"), re.M).group(1)
    out = ROOT / "dist" / f"souki_v{ver}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            p = ROOT / f
            info = zipfile.ZipInfo(f, date_time=(2026, 10, 31, 12, 0, 0))
            info.flag_bits |= 0x800
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())
    print(out)


if __name__ == "__main__":
    main()
