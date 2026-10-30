"""認識した物を掴んで指定の場所へ置く一連の動き。"""
import numpy as np

from .controller import CLOSE, HOME, LOOK, OPEN

HOVER = 0.16
CARRY = 0.20


def grasp_yaw(short_angle):
    """指の開閉方向を物の短辺に合わせる。手首の回転が小さく済む向きを選ぶ。"""
    yaw = short_angle - np.pi / 2  # Arm.down_rotation の y 軸が指の方向
    return (yaw + np.pi / 2) % np.pi - np.pi / 2


def pick(arm, blob, on_slip=None):
    x, y = blob.xy
    yaw = grasp_yaw(blob.yaw_short)
    arm.gripper(OPEN, 0.1)
    arm.move_joint(HOME, 1.3)
    arm.move_to([x, y, HOVER], yaw, 1.2)
    arm.move_to([x, y, max(0.018, blob.height - 0.015)], yaw, 0.9)
    arm.gripper(CLOSE, 0.6)
    arm.move_to([x, y, CARRY], yaw, 0.9)
    return arm.finger_gap() > 0.01


def place(arm, xy, slip_at=None):
    x, y = xy
    if slip_at is not None:
        cur = slip_at
        arm.move_to([cur[0], cur[1], CARRY], 0.0, 0.9)
        arm.gripper(OPEN, 0.8)  # 掴み損ねて落とす場面の再現
        arm.move_joint(HOME, 1.0)
        return False
    arm.move_to([x, y, CARRY], 0.0, 1.5)
    arm.move_to([x, y, 0.09], 0.0, 0.7)
    arm.gripper(OPEN, 0.5)
    arm.move_to([x, y, CARRY], 0.0, 0.7)
    return True


def go_look(arm):
    arm.move_joint(HOME, 1.0)
    arm.move_joint(LOOK, 1.4)
    arm.wait(0.3)


def pick_and_place(arm, blob, tray, slot=(0, 0)):
    ok = pick(arm, blob)
    place(arm, (tray[0] + slot[0], tray[1] + slot[1]))
    go_look(arm)
    return ok
