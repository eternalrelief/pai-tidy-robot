"""Claude が俯瞰画像を見て、次に何を動かすかを1手ずつ決める。"""
import base64
import io
import json
import time

SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
                "required": ["id", "name"], "additionalProperties": False,
            },
        },
        "decision": {"type": "string", "enum": ["pick", "done"]},
        "target_id": {"type": "integer"},
        "destination": {"type": "string", "enum": ["left", "right", "back", "table", "none"]},
        "reason": {"type": "string"},
    },
    "required": ["objects", "decision", "target_id", "destination", "reason"],
    "additionalProperties": False,
}

SYSTEM = """あなたは机の片付けをするロボットアームの頭脳です。
俯瞰カメラの画像を見て、人の指示どおりに片付く「次の1手」だけを決めてください。

- 画像の番号は深度センサーが見つけた物の番号です。番号は毎回振り直されます。
- 何の物かは、上面に書かれた文字と見た目から判断します。
- location が table の物は机の上、left/right/back の物はその箱の中にあります。
- 指示の分類に合わない物が箱に入っていたら、取り出して正しい場所へ移します。
- 机に残すよう指示された物の location が table なら、触らないでください。箱に入っていたら destination を table にして戻します。
- 前の手が失敗して物が机に残っている場合があります。履歴を信用しすぎず、今の画像で判断します。
- すべて指示どおりの場所にあれば decision は done、target_id は 0、destination は none にします。
- reason には判断の理由を日本語1文、40字以内で書きます。"""


def _b64(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


class Brain:
    """model=None なら記録済みの応答を順に再生する（API キーなしで再現できる）。"""

    def __init__(self, model="claude-opus-5-5", replay=None):
        self.model, self.replay = model, replay
        self.log = []
        if replay is None:
            import anthropic
            self.client = anthropic.Anthropic()

    def decide(self, overview, sheet, blobs, instruction, history):
        listing = [{"id": b.id, "location": b.where} for b in blobs]
        text = (f"人の指示:\n{instruction}\n\n見つかった物: {json.dumps(listing, ensure_ascii=False)}\n\n"
                f"これまでの手: {json.dumps(history, ensure_ascii=False)}\n\n"
                "1枚目が全体、2枚目が物ごとの拡大です。")
        t0 = time.time()
        if self.replay is not None:
            r = self.replay[len(self.log)]["response"]
        else:
            msg = self.client.beta.messages.create(
                model=self.model, max_tokens=4000, system=SYSTEM,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(overview)}},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(sheet)}},
                    {"type": "text", "text": text},
                ]}],
            )
            if msg.stop_reason != "end_turn":
                raise RuntimeError(f"想定外の終了理由: {msg.stop_reason}")
            r = json.loads(next(b.text for b in msg.content if b.type == "text"))
        self.log.append({"instruction": instruction, "listing": listing, "response": r,
                         "seconds": round(time.time() - t0, 2)})
        return r
