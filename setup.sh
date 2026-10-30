#!/bin/sh
# ロボットモデル（MuJoCo Menagerie の Franka Panda）と Python 環境を用意する
set -e
mkdir -p third_party
[ -d third_party/mujoco_menagerie ] || (cd third_party && git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git && cd mujoco_menagerie && git sparse-checkout set franka_emika_panda)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
