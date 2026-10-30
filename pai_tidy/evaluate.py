"""配置を変えて複数回走らせ、指示どおりに片付いた割合を集計する。"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 4, 5, 6]


def one(seed):
    path = Path(f"out/eval_{seed}.json")
    r = subprocess.run([sys.executable, "-m", "pai_tidy.run", "--seed", str(seed), "--save-log", str(path)],
                       capture_output=True, text=True)
    return json.loads(r.stdout[r.stdout.index("{"):]) if r.returncode == 0 else {"seed": seed, "error": r.stderr[-400:]}


with ThreadPoolExecutor(5) as ex:
    res = list(ex.map(one, seeds))
Path("logs").mkdir(exist_ok=True)
Path("logs/evaluation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
items = sum(len(r.get("ok", {})) for r in res)
good = sum(sum(r.get("ok", {}).values()) for r in res)
print(f"完全成功 {sum(bool(r.get('success')) for r in res)}/{len(res)} 回、物の単位 {good}/{items}")
