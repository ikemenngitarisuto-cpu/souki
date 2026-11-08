# -*- coding: utf-8 -*-
"""ふえる玉のアイコン（fuerutama_256.png と fuerutama.ico）を作る。

  python fuerutama/tools/make_icon.py

大きく描いてから縮めて、なめらかにする。画面（fuerutama.html）の玉と同じ色・同じ光り方にしている。
"""
import base64
import io
import math
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
S = 1024  # 描く大きさ（あとで縮める）
PAL = ["#ff3b4a", "#ff8a2b", "#ffd43b", "#c4f23a", "#38e070", "#22e8c0", "#2fd0ff", "#3d6bff", "#8c5cff", "#b14dff", "#ff45b5", "#ff6b8b"]


def rgb(h):
    n = int(h[1:], 16)
    return (n >> 16 & 255, n >> 8 & 255, n & 255)


def mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def ball(img, cx, cy, r, hexcol):
    """つやのある玉（画面と同じ: 左上が明るく、ふちが暗い）"""
    base = rgb(hexcol)
    lit, dark = mix(base, (255, 255, 255), .62), mix(base, (0, 0, 0), .5)
    hx, hy = cx - r * .38, cy - r * .42
    box = (int(cx - r - 2), int(cy - r - 2), int(cx + r + 3), int(cy + r + 3))
    px = img.load()
    for y in range(box[1], box[3]):
        for x in range(box[0], box[2]):
            d = math.hypot(x + .5 - cx, y + .5 - cy)
            if d > r + 1:
                continue
            a = max(0.0, min(1.0, r + .5 - d))
            t = min(1.0, math.hypot(x + .5 - hx, y + .5 - hy) / (r * 1.42))
            if t < .26:
                c = mix(lit, base, t / .26)
            elif t < .62:
                c = mix(base, mix(base, (0, 0, 0), .22), (t - .26) / .36)
            else:
                c = mix(mix(base, (0, 0, 0), .22), dark, (t - .62) / .38)
            # 光の点
            qx, qy = x + .5 - (cx - r * .34), y + .5 - (cy - r * .40)
            ex, ey = (qx * .8253 - qy * .5646) / (r * .27), (qx * .5646 + qy * .8253) / (r * .16)
            e = math.hypot(ex, ey)
            if e < 1:
                c = mix(c, (255, 255, 255), .5 * min(1.0, (1 - e) / .25))
            old = px[x, y]
            px[x, y] = tuple(round(o + (n - o) * a) for o, n in zip(old[:3], c)) + (255,)


def build():
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    # 角の丸い暗い四角（背景）
    bg = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(bg)
    d.rounded_rectangle((0, 0, S - 1, S - 1), radius=int(S * .2), fill=(10, 12, 26, 255))
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((S * .18, S * .18, S * .82, S * .82), fill=(40, 52, 110, 120))
    bg = Image.alpha_composite(bg, glow.filter(ImageFilter.GaussianBlur(S * .08)))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, S - 1, S - 1), radius=int(S * .2), fill=255)
    bg.putalpha(mask)
    img = Image.alpha_composite(img, bg)
    # 円（白い線）
    cx = cy = S / 2
    R = S * .4
    ring = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse((cx - R, cy - R, cx + R, cy + R), outline=(240, 244, 255, 255), width=int(S * .018))
    img = Image.alpha_composite(img, ring)
    # 玉（下半分にたまっている）
    rnd = random.Random(7)
    r = S * .056
    order = [6, 10, 2, 4, 8, 0, 5, 1, 9, 3, 7, 11]
    k = 0
    pts = []
    row = 0
    y = cy + R - r * 1.12
    while y > cy - R * .02:
        x = cx - R + (r * 1.03 if row % 2 else 0)
        while x < cx + R:
            if math.hypot(x - cx, y - cy) <= R - r * 1.12:
                pts.append((x + rnd.uniform(-r * .05, r * .05), y))
            x += r * 2.06
        y -= r * 1.78
        row += 1
    for x, y in pts:
        ball(img, x, y, r, PAL[order[k % len(order)]])
        k += 1
    # 上で分かれたばかりの 3 個と、はじけた輪
    sx, sy, sr = cx + S * .02, cy - R * .52, S * .062
    ring = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse((sx - sr * 2.3, sy - sr * 2.3, sx + sr * 2.3, sy + sr * 2.3), outline=(255, 212, 59, 150), width=int(S * .012))
    img = Image.alpha_composite(img, ring)
    for a in (-150, -30, 90):
        t = math.radians(a)
        ball(img, sx + math.cos(t) * sr * 1.25, sy + math.sin(t) * sr * 1.25, sr, PAL[2])
    return img


def main():
    big = build()
    p256 = big.resize((256, 256), Image.LANCZOS)
    p256.save(ROOT / "fuerutama_256.png", optimize=True)
    big.resize((256, 256), Image.LANCZOS).save(ROOT / "fuerutama.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    # 画面のタブ・タスクバーの絵（fuerutama.html の中に埋め込む。ファイルを開いただけでも出るように）
    buf = io.BytesIO()
    big.resize((96, 96), Image.LANCZOS).save(buf, format="PNG", optimize=True)
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    html = ROOT / "fuerutama.html"
    text = html.read_text(encoding="utf-8")
    text, n = re.subn(r'<link rel="icon" href="[^"]*">', f'<link rel="icon" href="{uri}">', text, count=1)
    assert n == 1, "fuerutama.html に <link rel=\"icon\"> がありません"
    html.write_text(text, encoding="utf-8", newline="\n")
    print("fuerutama_256.png / fuerutama.ico / favicon (", len(uri), "bytes )")


if __name__ == "__main__":
    main()
