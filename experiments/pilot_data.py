"""abandon-pilot (HA_own/experiments/2026-10-27_abandon-pilot/PLAN.md): problems.jsonl for Countdown + HumanEvalFix.
Rows carry their own `suffix` (gen_traces.py appends it instead of the math suffix) and `meta` for pilot_check.py.
Countdown: Jiayi-Pan/Countdown-Tasks-3to4 train, 4-number tasks among the first 20,000 rows, 200 drawn with Random(0).
Code: bigcode/humanevalpack python test (HumanEvalFix), all 164; prompt = "Fix bugs in <entry_point>." + buggy code + tests.
Usage: python pilot_data.py out_pilot/problems.jsonl [--n-countdown 200]"""
import argparse, json, random
from collections import Counter
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n-countdown", type=int, default=200)
a = ap.parse_args()

CD_SUFFIX = "\n\nPlease reason step by step, and put your final equation within \\boxed{}."
CODE_SUFFIX = "\n\nReturn the complete fixed function in a single ```python code block."
out = []
ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train").select(range(20000))
pool = [i for i, n in enumerate(ds["nums"]) if len(n) == 4]
for i in sorted(random.Random(0).sample(pool, a.n_countdown)):
    r = ds[i]
    nums, t = [int(x) for x in r["nums"]], int(r["target"])
    out.append({"problem_id": f"countdown_{i}", "dataset": "countdown", "answer": str(t), "suffix": CD_SUFFIX,
                "problem": f"Using the numbers {nums}, create an equation that equals {t}. You can use basic arithmetic operations "
                           f"(+, -, *, /) and each number must be used exactly once.",
                "meta": {"nums": nums, "target": t}})
for r in load_dataset("bigcode/humanevalpack", "python", split="test"):
    code = (r["declaration"] + r["buggy_solution"]).rstrip()
    out.append({"problem_id": "hefix_" + r["task_id"].split("/")[1], "dataset": "humanevalfix", "answer": "", "suffix": CODE_SUFFIX,
                "problem": f"Fix bugs in {r['entry_point']}.\n\n```python\n{code}\n```\n\nTests:\n```python\n{r['test'].strip()}\n```",
                "meta": {"entry_point": r["entry_point"], "bug_type": r["bug_type"]}})
with open(a.out, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(Counter(r["dataset"] for r in out), len(out))
