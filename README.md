# 言葉で頼める片付けロボット

Physical AI 講座 応用編1 最終課題。ロボットアームが俯瞰カメラの画面を見て、人の日本語の指示どおりに机の上を片付ける。途中の失敗、物の追加、指示の変更があっても、見直して立て直す。

## 仕組み

1. 深度カメラで物の位置と向きを測り、番号を振る
2. 画像を Claude（`claude-opus-5-5`・推論のみ）に見せ、次の1手（どの番号を、どの箱へ）を決めさせる
3. 逆運動学で Franka Panda を動かして運ぶ
4. 腕を視野の外へ戻し、1に戻る。全部終わったと AI が判断したら止める

「何の物か」「どこへ入れるべきか」は AI、「どこにあるか」「どう動くか」は計測と制御が担当する。

| ファイル | 役割 |
|---|---|
| `pai_tidy/scene.py` | 机・箱・物の MuJoCo モデル |
| `pai_tidy/perception.py` | 深度画像からの物体検出、AI に見せる画像の作成 |
| `pai_tidy/agent.py` | Claude への問い合わせ（構造化出力） |
| `pai_tidy/controller.py` / `robot.py` | 逆運動学、掴む・運ぶ動作 |
| `pai_tidy/run.py` | 全体の繰り返しと、失敗・追加・指示変更の場面 |
| `pai_tidy/video.py` | 動画の合成 |
| `pai_tidy/evaluate.py` | 配置を変えた成功率の集計 |

## 実行

macOS（Apple Silicon）で確認。動画の文字にヒラギノを使うため、他の OS では `scene.py` の `FONT` を変更する。

```sh
sh setup.sh

# 記録した AI の応答で再現（API キー不要）
.venv/bin/python -m pai_tidy.run --seed 3 --replay logs/demo_seed3.json --video out/replay.mp4

# Claude に判断させて実行
export ANTHROPIC_API_KEY=...
.venv/bin/python -m pai_tidy.run --seed 3 --video out/demo.mp4
.venv/bin/python -m pai_tidy.evaluate
```

## 結果

配置を変えた5回で、8個すべてが指示どおりの場所に入った回が5回（物の単位で 40/40）。詳細は `logs/evaluation.json`。これはシミュレーション上の結果で、実機では確かめていない。

## 使用したもの

- ロボットモデル: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) の Franka Emika Panda（Apache-2.0。`setup.sh` で取得）
- シミュレーター: MuJoCo
- 判断: Anthropic Claude API
- 実装には、コーディング支援 AI として Claude Code を使用した
