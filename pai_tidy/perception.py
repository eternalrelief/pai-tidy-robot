"""俯瞰カメラの深度から物の位置と向きを測る。何の物かは AI に見せて判断させる。"""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .scene import FONT, TRAY_INNER, TRAY_JP, TRAYS

W, H = 1280, 960


@dataclass
class Blob:
    id: int
    xy: np.ndarray
    height: float
    yaw_short: float  # 短辺方向の角度（ここを指で挟む）
    px: tuple
    bbox: tuple
    where: str  # table / left / right / back


class Eye:
    def __init__(self, model):
        import mujoco
        self.mj = mujoco
        self.model = model
        self.rgb_r = mujoco.Renderer(model, H, W)
        self.dep_r = mujoco.Renderer(model, H, W)
        self.dep_r.enable_depth_rendering()
        cam = model.camera("top")
        self.cam = cam
        self.f = H / 2 / np.tan(np.deg2rad(float(model.cam_fovy[cam.id])) / 2)
        self.campos = model.cam_pos[cam.id].copy()

    def px_to_world(self, u, v, depth):
        # 画像の右 = 世界の -y、上 = 世界の +x
        y = -(u - W / 2) * depth / self.f + self.campos[1]
        x = -(v - H / 2) * depth / self.f + self.campos[0]
        return np.array([x, y])

    def world_to_px(self, x, y, z=0.0):
        depth = self.campos[2] - z
        u = W / 2 - (y - self.campos[1]) * self.f / depth
        v = H / 2 - (x - self.campos[0]) * self.f / depth
        return u, v

    def capture(self, data):
        self.rgb_r.update_scene(data, "top")
        rgb = self.rgb_r.render().copy()
        self.dep_r.update_scene(data, "top")
        depth = self.dep_r.render().copy()
        return rgb, depth

    def _wall_mask(self):
        m = np.zeros((H, W), bool)
        vv, uu = np.mgrid[0:H, 0:W]
        wx = -(vv - H / 2) * self.campos[2] / self.f + self.campos[0]
        wy = -(uu - W / 2) * self.campos[2] / self.f + self.campos[1]
        for cx, cy in TRAYS.values():
            inner = TRAY_INNER / 2 - 0.004
            outer = TRAY_INNER / 2 + 0.035
            box = (abs(wx - cx) < outer) & (abs(wy - cy) < outer)
            core = (abs(wx - cx) < inner) & (abs(wy - cy) < inner)
            m |= box & ~core
        return m

    def blobs(self, depth):
        height = self.campos[2] - depth
        mask = (height > 0.012) & (height < 0.3) & ~self._wall_mask()
        mask = ndimage.binary_opening(mask, iterations=2)
        lab, n = ndimage.label(mask)
        out = []
        for i in range(1, n + 1):
            vs, us = np.nonzero(lab == i)
            if len(us) < 150:
                continue
            h = float(np.median(height[vs, us]))
            d = float(self.campos[2] - h)
            pts = np.stack([self.px_to_world(u, v, d) for u, v in zip(us[::3], vs[::3])])
            c = pts.mean(0)
            w, v = np.linalg.eigh(np.cov((pts - c).T))
            short = v[:, 0]
            where = "table"
            for name, (cx, cy) in TRAYS.items():
                if abs(c[0] - cx) < TRAY_INNER / 2 and abs(c[1] - cy) < TRAY_INNER / 2:
                    where = name
            out.append(Blob(len(out) + 1, c, h, float(np.arctan2(short[1], short[0])),
                            (float(us.mean()), float(vs.mean())),
                            (int(us.min()), int(vs.min()), int(us.max()), int(vs.max())), where))
        return out


def _font(size):
    return ImageFont.truetype(FONT, size)


def annotate(rgb, blobs, eye, hilite=None):
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    for name, (cx, cy) in TRAYS.items():
        u, v = eye.world_to_px(cx + TRAY_INNER / 2 - 0.03, cy)
        d.text((u, v), TRAY_JP[name], font=_font(30), fill=(30, 30, 30), anchor="mm",
               stroke_width=4, stroke_fill=(255, 255, 255))
    for b in blobs:
        col = (225, 30, 30) if hilite == b.id else (20, 90, 220)
        x0, y0, x1, y1 = b.bbox
        d.rectangle((x0 - 4, y0 - 4, x1 + 4, y1 + 4), outline=col, width=3)
        cx, cy = max(x0 - 20, 22), max(y0 - 20, 22)
        d.ellipse((cx - 19, cy - 19, cx + 19, cy + 19), fill=col)
        d.text((cx, cy), str(b.id), font=_font(26), fill=(255, 255, 255), anchor="mm")
    return img


def crops(rgb, blobs, scale=4):
    """読みやすいように物ごとに切り出して拡大した一覧を作る。"""
    n = max(1, len(blobs))
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    cell = 280
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    src = Image.fromarray(rgb)
    for i, b in enumerate(blobs):
        u, v = b.px
        r = 34
        c = src.crop((int(u - r), int(v - r), int(u + r), int(v + r))).resize((cell - 16, cell - 16), Image.LANCZOS)
        x, y = (i % cols) * cell, (i // cols) * cell
        sheet.paste(c, (x + 8, y + 8))
        d.ellipse((x + 10, y + 10, x + 56, y + 56), fill=(20, 90, 220))
        d.text((x + 33, y + 33), str(b.id), font=_font(30), fill=(255, 255, 255), anchor="mm")
    return sheet
