# -*- coding: utf-8 -*-
"""納品用の zip（fuerutama/dist/fuerutama_v<版>.zip）を作る。日本語のファイル名が化けないよう UTF-8 の印を立てる。

  python fuerutama/tools/make_zip.py
"""
import re
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["起動.bat", "fuerutama.html", "README.md", "fuerutama.ico", "fuerutama_256.png"]


def version() -> str:
    return re.search(r'^const VERSION = "([^"]+)"', (ROOT / "fuerutama.html").read_text(encoding="utf-8"), re.M).group(1)


def build(out: Path | None = None) -> Path:
    out = out or ROOT / "dist" / f"fuerutama_v{version()}.zip"
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
