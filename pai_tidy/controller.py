"""Panda を逆運動学で動かす部品。カメラ映像の記録は on_frame に任せる。"""
import mujoco
import numpy as np

HOME = np.array([0, -0.5, 0, -2.3, 0, 1.8, 0.785])
LOOK = np.array([-2.0, -0.6, 0, -2.4, 0, 1.8, 0.785])  # 俯瞰カメラの視野から外れる姿勢
TCP_OFFSET = 0.1034
OPEN, CLOSE = 255.0, 0.0
CTRL_DT = 0.01


def min_jerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class Arm:
    def __init__(self, model, data, on_frame=None):
        self.m, self.d = model, data
        self.ik = mujoco.MjData(model)
        self.hand = model.body("hand").id
        self.on_frame = on_frame or (lambda: None)
        self.grip = OPEN
        self.substeps = int(round(CTRL_DT / model.opt.timestep))

    def set_joints(self, q):
        self.d.qpos[:7] = q
        self.d.qpos[7:9] = 0.04
        self.d.ctrl[:7] = q
        self.d.ctrl[7] = OPEN
        self.q = np.array(q, float)

    def _step(self, q):
        self.d.ctrl[:7] = q
        self.d.ctrl[7] = self.grip
        for _ in range(self.substeps):
            mujoco.mj_step(self.m, self.d)
        self.on_frame()

    def _tcp(self, q):
        self.ik.qpos[:7] = q
        mujoco.mj_kinematics(self.m, self.ik)
        mujoco.mj_comPos(self.m, self.ik)
        R = self.ik.xmat[self.hand].reshape(3, 3)
        return self.ik.xpos[self.hand] + R[:, 2] * TCP_OFFSET, R

    def _ik_step(self, q, p_des, R_des, iters=4):
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        q = q.copy()
        for _ in range(iters):
            p, R = self._tcp(q)
            ep = p_des - p
            eR = 0.5 * sum(np.cross(R[:, i], R_des[:, i]) for i in range(3))
            mujoco.mj_jac(self.m, self.ik, jp, jr, p, self.hand)
            J = np.vstack([jp[:, :7], jr[:, :7]])
            e = np.hstack([ep, eR])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            null = np.eye(7) - np.linalg.pinv(J) @ J
            q += dq + null @ (HOME - q) * 0.02
        return q

    @staticmethod
    def down_rotation(yaw):
        x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        z = np.array([0.0, 0.0, -1.0])
        return np.column_stack([x, np.cross(z, x), z])

    def move_joint(self, goal, T):
        start = self.q.copy()
        n = max(1, int(T / CTRL_DT))
        for i in range(1, n + 1):
            self.q = start + (goal - start) * min_jerk(i / n)
            self._step(self.q)

    def move_to(self, pos, yaw, T):
        """手先を直線で動かす。"""
        start, R0 = self._tcp(self.q)
        pos = np.asarray(pos, float)
        R1 = self.down_rotation(yaw)
        n = max(1, int(T / CTRL_DT))
        from scipy.spatial.transform import Rotation as Rot, Slerp
        sl = Slerp([0, 1], Rot.from_matrix(np.stack([R0, R1])))
        for i in range(1, n + 1):
            s = min_jerk(i / n)
            self.q = self._ik_step(self.q, start + (pos - start) * s, sl([s]).as_matrix()[0])
            self._step(self.q)

    def wait(self, T):
        for _ in range(max(1, int(T / CTRL_DT))):
            self._step(self.q)

    def gripper(self, value, T=0.5):
        self.grip = value
        self.wait(T)

    def finger_gap(self):
        return float(self.d.qpos[7] + self.d.qpos[8])
