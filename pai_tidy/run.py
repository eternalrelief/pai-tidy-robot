"""机の片付けを、見る → 判断する → 動く → 確かめる の繰り返しで実行する。

  python -m pai_tidy.run --seed 3 --video out/demo.mp4          # Claude に判断させる
  python -m pai_tidy.run --seed 3 --replay logs/demo.json --video out/replay.mp4  # 記録で再現
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .agent import Brain
from .controller import Arm, LOOK
from .perception import Eye, annotate, crops
from .robot import go_look, pick, place
from .scene import ITEMS, LATE_ITEM, TRAY_JP, TRAYS, load_model, sample_layout
from .video import Recorder

INSTRUCTION = "お客さんが来るので机を片付けて。食べ物は左の箱、文房具は右の箱へ。どちらでもない物は奥の箱へ。"
EXTRA = "追加：チョコは後で食べるから、机の上に残して。"
EXPECT = {"cookie": "left", "orange": "left", "onigiri": "left", "scissors": "right", "eraser": "right",
          "stapler": "right", "battery": "back", "choco": "table"}
SLOTS = [(a, b) for a in (0.065, -0.065) for b in (0.065, -0.065)]
DEST_JP = dict(TRAY_JP, table="机の上")


def item_where(m, d, it):
    a = m.joint("j_" + it.key).qposadr[0]
    x, y = d.qpos[a:a + 2]
    for name, (cx, cy) in TRAYS.items():
        if abs(x - cx) < 0.13 and abs(y - cy) < 0.13:
            return name
    return "table" if 0.1 < x < 0.7 and abs(y) < 0.3 else "lost"


def set_item(m, d, it, x, y, yaw, z=None):
    a = m.joint("j_" + it.key).qposadr[0]
    z = it.size[2] / 2 + 0.002 if z is None else z
    d.qpos[a:a + 3] = [x, y, z]
    d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.qvel[m.joint("j_" + it.key).dofadr[0]:][:6] = 0


def free_spot(blobs, rng):
    """机の上で、他の物から一番離れた場所を選ぶ。"""
    best, score = None, -1
    table = [b.xy for b in blobs if b.where == "table"]
    for _ in range(300):
        p = np.array([rng.uniform(0.30, 0.50), rng.uniform(-0.18, 0.18)])
        s = min([np.linalg.norm(p - q) for q in table], default=1.0)
        if s > score:
            best, score = p, s
    return best


def free_slot(tray, blobs):
    c = np.array(TRAYS[tray])
    inside = [b.xy for b in blobs if b.where == tray]
    cand = [c + np.array(s) for s in SLOTS]
    return max(cand, key=lambda p: min([np.linalg.norm(p - q) for q in inside], default=1.0))


def run(seed, video=None, replay=None, disturb=True, max_steps=18, name=None):
    all_items = ITEMS + [LATE_ITEM]
    m = load_model(all_items)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    for it, (x, y, yaw) in zip(ITEMS, sample_layout(ITEMS, rng)):
        set_item(m, d, it, x, y, yaw)
    set_item(m, d, LATE_ITEM, -1.0, 1.0, 0.0)

    rec = Recorder(m, d, video) if video else None
    arm = Arm(m, d, rec.on_step if rec else None)
    arm.set_joints(LOOK)
    for _ in range(200):
        mujoco.mj_step(m, d)
    eye = Eye(m)
    brain = Brain(replay=json.loads(Path(replay).read_text(encoding="utf-8")) if replay else None)

    instruction = INSTRUCTION
    history, events = [], []
    slipped = added = extended = False
    if rec:
        rec.instruction = instruction
        rec.card(["言葉で頼める片付けロボット", "AI が画面を見て判断し、失敗も自分で立て直す", "シミュレーション映像・ロボットの動きは等倍速（AIの考える時間のみ短縮）"], 3.5)
        rec.caption = "左右はロボットから見た向き（赤が左の箱、青が右の箱）"
        rec.hold(3.0)

    def say(text, seconds=2.2):
        events.append(text)
        if rec:
            rec.caption = text
            rec.hold(seconds)

    for step in range(max_steps):
        rgb, depth = eye.capture(d)
        blobs = eye.blobs(depth)
        view = annotate(rgb, blobs, eye)
        if rec:
            rec.view = view
            rec.decision = "考え中…"
            rec.caption = "画面を見て次の一手を考える"
            rec.hold(0.8)
        r = brain.decide(view, crops(rgb, blobs), blobs, instruction, history)
        if rec:
            rec.decision = r["reason"]

        if r["decision"] == "done":
            if disturb and not extended:
                r = None
            else:
                if rec:
                    rec.caption = "AIの判断：片付け完了"
                    rec.hold(2.5)
                break
        if r is None or (disturb and not extended and step >= 4 and item_where(m, d, ITEMS[2]) == "left"):
            extended = True
            instruction = instruction + " " + EXTRA
            if rec:
                rec.instruction = instruction
            say("人から追加の指示：チョコは机に残して", 3.0)
            continue

        target = next((b for b in blobs if b.id == r["target_id"]), None)
        dest = r["destination"]
        if target is None or dest == "none":
            history.append({"step": step, "error": "存在しない番号を選んだ"})
            continue
        names = {o["id"]: o["name"] for o in r["objects"]}
        label = names.get(target.id, "物")
        rec_view = annotate(rgb, blobs, eye, hilite=target.id)
        if rec:
            rec.view = rec_view
            rec.caption = f"{label} → {DEST_JP[dest]}"
        gripped = pick(arm, target)
        goal = free_spot(blobs, rng) if dest == "table" else free_slot(dest, blobs)

        slip_now = disturb and not slipped and step == 1
        if slip_now:
            mid = (target.xy + goal) / 2
            mid[1] = np.clip(mid[1], -0.19, 0.19)
            slipped = True
            place(arm, goal, slip_at=mid)
            say(f"あっ、{label}を途中で落としてしまった", 2.0)
            history.append({"step": step, "object": label, "destination": dest, "note": "動作は実行した。結果は画像で確認すること"})
        else:
            place(arm, goal)
            history.append({"step": step, "object": label, "destination": dest, "note": "動作は実行した。結果は画像で確認すること"})
        if rec:
            rec.caption = "腕を視野の外へ戻して、結果を確かめる"
        go_look(arm)

        if disturb and not added and step == 3:
            added = True
            spot = free_spot(eye.blobs(eye.capture(d)[1]), rng)
            set_item(m, d, LATE_ITEM, spot[0], spot[1], float(rng.uniform(-1, 1)), z=0.12)
            arm.wait(0.8)
            say("途中で人がおにぎりを置いた", 2.2)

    final = {it.key: item_where(m, d, it) for it in all_items if disturb or it is not LATE_ITEM}
    expected = {k: ("left" if (k == "choco" and not extended) else v) for k, v in EXPECT.items() if k in final}
    ok = {k: final[k] == expected[k] for k in final}
    if rec:
        rec.caption = ""
        n = sum(ok.values())
        rec.card(["片付け完了", f"{len(ok)}個のうち{n}個が指示どおりの場所へ",
                  "判断：Claude（画像を見て1手ずつ決定）／ 位置の計測：深度カメラ", "ロボット：Franka Panda ／ シミュレーター：MuJoCo"], 4.5)
        rec.close()
        print("動画秒数", round(rec.frames / 30, 1))
    return {"seed": seed, "steps": len(brain.log), "final": final, "ok": ok, "success": all(ok.values()),
            "events": events, "log": brain.log}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--video")
    ap.add_argument("--replay")
    ap.add_argument("--no-disturb", action="store_true")
    ap.add_argument("--save-log")
    a = ap.parse_args()
    res = run(a.seed, a.video, a.replay, not a.no_disturb)
    print(json.dumps({k: v for k, v in res.items() if k != "log"}, ensure_ascii=False, indent=1))
    if a.save_log:
        Path(a.save_log).parent.mkdir(exist_ok=True)
        Path(a.save_log).write_text(json.dumps(res["log"], ensure_ascii=False, indent=1), encoding="utf-8")
