#!/usr/bin/env python3
# POST HOC, descriptive (prereg_fade.md line 54 and the ~22:04 change-log entry): each model's full-attention NLL on the
# shared texts, the "how off-policy is the text" number. Kept out of h1_stats.py (unchanged since ~18:34) because the
# rows store full_nll per bucket ({bucket: [n_tok, nll_sum]}), which h1_stats.py skips. Per token for each model's own
# tokenizer; per character as well, the cross-model unit used for KL.
# Usage: h1_nll.py --rows q35:math:nll35_math.jsonl+fd_chars.jsonl q3:math:fdd_math.jsonl --out results/h1_nll.json
import argparse
import json
from pathlib import Path


def load(spec):
    model, task, files = spec.split(":", 2)
    path, _, side = files.partition("+")
    chars = {}
    if side:
        for line in open(side):
            if line.strip():
                r = json.loads(line)
                chars[r["problem_id"]] = r["chars"]
    rows = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            c = r.get("chars", chars.get(r["problem_id"]))
            rows[r["problem_id"]] = {"nll": r["full_nll"], "chars": c}
    return model, task, rows


def summarise(rows, ids):
    n = sum(v[0] for p in ids for v in rows[p]["nll"].values())
    s = sum(v[1] for p in ids for v in rows[p]["nll"].values())
    out = {"n_traces": len(ids), "n_tok": n, "nll_tok": s / n}
    if all(rows[p]["chars"] for p in ids):
        nc = sum(sum(rows[p]["chars"].values()) for p in ids)
        out.update({"n_chars": nc, "nll_char": s / nc})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    by_task = {}
    for spec in args.rows:
        model, task, rows = load(spec)
        by_task.setdefault(task, {})[model] = rows
    out = {}
    for task, models in by_task.items():
        ids = sorted(set.intersection(*(set(r) for r in models.values())))
        out[task] = {m: summarise(rows, ids) for m, rows in models.items()}
        for m, v in out[task].items():
            extra = f", {v['nll_char']:.4f} nats/char" if "nll_char" in v else ""
            print(f"{task} {m}: {v['n_traces']} texts, {v['nll_tok']:.4f} nats/token{extra}")
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
