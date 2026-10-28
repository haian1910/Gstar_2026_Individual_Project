"""Prepared trace rows for the Fade KL sweeps on another task (prereg_fade.md, H1 task 2 rule by nti-e2): from TL Full-cell
generations (tl_gen.py rows), FINISHED rows with lo <= n_tokens <= hi of one dataset, one per problem (lowest sample_k).
Writes rows in the traces.jsonl format kv_fade.py reads (problem_id, dataset, sample_idx, output_ids, prompt_len) and the
matching problems file. prompt_len is computed with the same chat template as tl_gen.py / kv_fade.py.
Usage: python mk_task_rows.py --gen 'out_tl/gen_full.s*.jsonl' --problems P [--problems Q] --dataset countdown --out-rows R --out-problems S"""
import argparse, glob, json
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True)
ap.add_argument("--problems", action="append", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--lo", type=int, default=16384)
ap.add_argument("--hi", type=int, default=24576)
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--out-rows", required=True)
ap.add_argument("--out-problems", required=True)
a = ap.parse_args()
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."  # as tl_gen.py / kv_fade.py
probs = {r["problem_id"]: r for f in a.problems for r in map(json.loads, open(f))}
best, n_elig = {}, 0
for f in sorted(glob.glob(a.gen)):
    for r in map(json.loads, open(f)):
        if r["dataset"] != a.dataset or not r["finished"] or not a.lo <= r["n_tokens"] <= a.hi:
            continue
        n_elig += 1
        if r["problem_id"] not in best or r["sample_k"] < best[r["problem_id"]]["sample_k"]:
            best[r["problem_id"]] = r
tok = AutoTokenizer.from_pretrained(a.model)
with open(a.out_rows, "w") as fo, open(a.out_problems, "w") as fp:
    for pid in sorted(best):
        r, p = best[pid], probs[pid]
        x = tok.apply_chat_template([{"role": "user", "content": p["problem"] + p.get("suffix", SUFFIX)}],
                                    add_generation_prompt=True, enable_thinking=True, tokenize=True)
        x = x if isinstance(x, list) else x["input_ids"]
        assert len(r["gen_ids"]) == r["n_tokens"]
        fo.write(json.dumps({"problem_id": pid, "dataset": r["dataset"], "sample_idx": r["sample_k"], "output_ids": r["gen_ids"],
                             "prompt_len": len(x), "source": "tl_gen full"}) + "\n")
        fp.write(json.dumps(p) + "\n")
n = [len(best[p]["gen_ids"]) for p in best]
print(f"{a.dataset}: {n_elig} eligible rows, {len(best)} problems, mean {sum(n) / max(1, len(n)):.0f} tokens -> {a.out_rows}")
