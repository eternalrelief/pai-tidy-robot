"""机の上に物が散らかった場面（MuJoCo のモデル）を組み立てる。"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
PANDA_DIR = ROOT / "third_party/mujoco_menagerie/franka_emika_panda"
TEX_DIR = ROOT / "out/textures"
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"

# 箱の中心（x, y）。内寸 0.24m、壁の高さ 0.03m
TRAYS = {"left": (0.40, 0.38), "right": (0.40, -0.38), "back": (0.72, 0.0)}
TRAY_INNER = 0.24
TRAY_WALL_H = 0.03
TRAY_WALL_T = 0.01
TRAY_JP = {"left": "左の箱", "right": "右の箱", "back": "奥の箱"}


@dataclass(frozen=True)
class Item:
    key: str
    text: str
    color: tuple
    size: tuple  # 半分ではなく全長 (lx, ly, h)


ITEMS = [
    Item("cookie", "クッキー", (222, 170, 96), (0.075, 0.05, 0.04)),
    Item("orange", "みかん", (240, 140, 40), (0.055, 0.055, 0.045)),
    Item("choco", "チョコ", (120, 74, 48), (0.075, 0.05, 0.03)),
    Item("scissors", "ハサミ", (70, 130, 200), (0.075, 0.05, 0.03)),
    Item("eraser", "消しゴム", (235, 235, 235), (0.08, 0.05, 0.03)),
    Item("stapler", "ホチキス", (200, 60, 60), (0.08, 0.05, 0.04)),
    Item("battery", "電池", (70, 170, 110), (0.06, 0.05, 0.04)),
]
LATE_ITEM = Item("onigiri", "おにぎり", (245, 245, 235), (0.075, 0.05, 0.04))


def _atlas(item: Item) -> Path:
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    s = 256
    atlas = Image.new("RGB", (s * 4, s * 3), item.color)
    face = Image.new("RGB", (s, s), item.color)
    d = ImageDraw.Draw(face)
    fg = (255, 255, 255) if sum(item.color) < 420 else (25, 25, 25)
    n = len(item.text)
    font = ImageFont.truetype(FONT, int(min(200, 225 / n * 1.05)))
    d.text((s / 2, s / 2), item.text, font=font, fill=fg, anchor="mm")
    atlas.paste(face, (s, s))  # MuJoCo の立方体テクスチャでは F が +z（上面）
    path = TEX_DIR / f"{item.key}.png"
    atlas.save(path)
    return path


def build_xml(items) -> str:
    asset, bodies = [], []
    for i, it in enumerate(items):
        path = _atlas(it)
        asset.append(f'<texture name="t_{it.key}" type="cube" file="{path}" gridsize="3 4" gridlayout=".U..LFRB.D.."/>')
        asset.append(f'<material name="m_{it.key}" texture="t_{it.key}" specular="0.1"/>')
        lx, ly, h = it.size
        bodies.append(f'''
    <body name="item_{it.key}" pos="{-1 - 0.2 * i} 1.5 0.1">
      <freejoint name="j_{it.key}"/>
      <geom type="box" size="{lx/2} {ly/2} {h/2}" material="m_{it.key}" mass="0.07"
            friction="1.6 0.02 0.002" condim="4" solimp="0.95 0.99 0.001" solref="0.004 1"/>
    </body>''')
    walls = []
    for name, (cx, cy) in TRAYS.items():
        half = TRAY_INNER / 2 + TRAY_WALL_T / 2
        full = half + TRAY_WALL_T / 2
        z = TRAY_WALL_H / 2
        col = {"left": "0.86 0.55 0.45 1", "right": "0.45 0.62 0.82 1", "back": "0.55 0.72 0.52 1"}[name]
        for sx, sy, ex, ey in [(0, 1, full, TRAY_WALL_T / 2), (0, -1, full, TRAY_WALL_T / 2),
                               (1, 0, TRAY_WALL_T / 2, full), (-1, 0, TRAY_WALL_T / 2, full)]:
            walls.append(f'<geom type="box" pos="{cx + sx * half} {cy + sy * half} {z}" '
                         f'size="{ex} {ey} {z}" rgba="{col}"/>')
    return f'''<mujoco model="tidy">
  <include file="panda.xml"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <headlight diffuse="0.25 0.25 0.25" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global offwidth="1600" offheight="1200"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.80 0.83 0.87" rgb2="0.62 0.66 0.72" width="256" height="1536"/>
    <material name="desk" rgba="0.60 0.54 0.45 1" specular="0"/>
    {"".join(asset)}
  </asset>
  <worldbody>
    <light pos="0.4 0.2 1.8" dir="-0.1 -0.1 -1" directional="true" castshadow="true" diffuse="0.45 0.45 0.45" specular="0 0 0"/>
    <geom name="desk" type="plane" size="0 0 0.05" material="desk" friction="1 0.01 0.001"/>
    <camera name="top" pos="0.47 0 1.2" xyaxes="0 -1 0 1 0 0" fovy="36"/>
    <camera name="side" pos="1.62 0.72 0.92" xyaxes="-0.5 0.87 0 -0.4 -0.23 0.89" fovy="42"/>
    {"".join(walls)}
    {"".join(bodies)}
  </worldbody>
</mujoco>'''


def load_model(items):
    import mujoco
    path = PANDA_DIR / "_tidy_scene.xml"
    path.write_text(build_xml(items), encoding="utf-8")
    m = mujoco.MjModel.from_xml_path(str(path))
    m.light_diffuse[m.light("top").id] = 0.12  # Panda 付属の照明は机が白飛びするので弱める
    return m


def sample_layout(items, rng):
    """物どうしが重ならないように机の中央へ並べる。"""
    placed = []
    for _ in items:
        for _ in range(2000):
            p = np.array([rng.uniform(0.30, 0.50), rng.uniform(-0.18, 0.18)])
            if all(np.linalg.norm(p - q[:2]) > 0.105 for q in placed):
                placed.append(np.array([p[0], p[1], rng.uniform(-np.pi, np.pi)]))
                break
        else:
            raise RuntimeError("配置に失敗")
    return placed
