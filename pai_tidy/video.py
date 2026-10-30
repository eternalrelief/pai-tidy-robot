"""シミュレーション映像に、AI が見たものと判断を並べて1本の動画にする。"""
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .scene import FONT

W, H = 1280, 720
LEFT_W, RIGHT_W = 840, 440
FPS = 30
INK, MUTED, ACCENT = (26, 30, 36), (72, 80, 92), (178, 44, 30)


def font(size):
    return ImageFont.truetype(FONT, size)


def wrap(text, f, width):
    lines, cur = [], ""
    for ch in text:
        if f.getlength(cur + ch) > width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    return lines + [cur] if cur else lines


class Recorder:
    def __init__(self, model, data, path):
        self.m, self.d = model, data
        self.r = mujoco.Renderer(model, H, LEFT_W)
        self.w = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
        self.view = None
        self.caption = ""
        self.instruction = ""
        self.decision = ""
        self.tick = 0
        self.frames = 0

    def _panel(self):
        p = Image.new("RGB", (RIGHT_W, H), (244, 245, 247))
        d = ImageDraw.Draw(p)
        y = 12
        d.text((16, y), "AIが見ている画面", font=font(22), fill=MUTED)
        y += 34
        if self.view is not None:
            v = self.view.resize((RIGHT_W - 32, int((RIGHT_W - 32) * 0.75)), Image.LANCZOS)
            p.paste(v, (16, y))
            y += v.height + 12
        d.text((16, y), "人の指示", font=font(22), fill=MUTED)
        y += 30
        f = font(21)
        for ln in wrap(self.instruction, f, RIGHT_W - 32)[:4]:
            d.text((16, y), ln, font=f, fill=INK)
            y += 28
        y += 8
        d.text((16, y), "AIの判断", font=font(22), fill=MUTED)
        y += 30
        for ln in wrap(self.decision, f, RIGHT_W - 32)[:4]:
            d.text((16, y), ln, font=f, fill=ACCENT)
            y += 28
        return p

    def frame(self):
        self.r.update_scene(self.d, "side")
        left = Image.fromarray(self.r.render())
        if self.caption:
            d = ImageDraw.Draw(left, "RGBA")
            d.rectangle((0, H - 66, LEFT_W, H), fill=(20, 24, 30, 215))
            d.text((24, H - 33), self.caption, font=font(30), fill=(255, 255, 255), anchor="lm")
        img = Image.new("RGB", (W, H))
        img.paste(left, (0, 0))
        img.paste(self._panel(), (LEFT_W, 0))
        self.w.append_data(np.asarray(img))
        self.frames += 1

    def on_step(self):
        self.tick += 1
        if self.tick % 3 == 0:  # 制御 100Hz → 動画 約33Hz
            self.frame()

    def hold(self, seconds):
        for _ in range(int(seconds * FPS)):
            self.frame()

    def card(self, lines, seconds):
        img = Image.new("RGB", (W, H), (24, 28, 34))
        d = ImageDraw.Draw(img)
        y = H / 2 - (len(lines) - 1) * 40
        for i, ln in enumerate(lines):
            d.text((W / 2, y), ln, font=font(50 if i == 0 else 30), fill=(255, 255, 255), anchor="mm")
            y += 80
        arr = np.asarray(img)
        for _ in range(int(seconds * FPS)):
            self.w.append_data(arr)
            self.frames += 1

    def close(self):
        self.w.close()
