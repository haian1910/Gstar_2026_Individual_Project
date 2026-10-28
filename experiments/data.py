"""Build problems.jsonl: AIME 2025, AIME 2026, HMMT Feb 2025 (MathArena) + MATH-500 level 5."""
import json, sys
from datasets import load_dataset

def pick(row, *names):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    raise KeyError(f"none of {names} in {list(row)}")

out = []
for name, tag in [("MathArena/aime_2025", "aime25"), ("MathArena/aime_2026", "aime26"), ("MathArena/hmmt_feb_2025", "hmmt25")]:
    ds = load_dataset(name, split="train")
    for i, r in enumerate(ds):
        pid = pick(r, "problem_idx", "id", "idx") if any(k in r for k in ("problem_idx", "id", "idx")) else i + 1
        out.append({"problem_id": f"{tag}_{pid}", "dataset": tag, "problem": pick(r, "problem", "question"), "answer": str(pick(r, "answer"))})
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
for r in ds:
    if int(r["level"]) == 5:
        out.append({"problem_id": "math500_" + r["unique_id"].replace("/", "_").replace(".json", ""), "dataset": "math500_l5", "problem": r["problem"], "answer": str(r["answer"])})
with open(sys.argv[1] if len(sys.argv) > 1 else "problems.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
from collections import Counter
print(Counter(r["dataset"] for r in out), len(out))
