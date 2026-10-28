#!/usr/bin/env python3
# Exploratory: paired difference in KL/token between two configs, per position bucket, one model and task. Bootstrap over
# traces (10,000, seed 0, the same resample indices as kv_fade_report.py); KL/token in each resample is a ratio of sums.
# This is the computation behind the step-1 auditor's per-bucket CIs (AUDIT.md item 3), written down so it can be rerun.
# Usage: h1_buckets.py --rows a.jsonl [b.jsonl ...] --pairs kv2:w8k kv2_had:w8k [--out X.json]
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="a:b, reports KL(a) - KL(b)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = sorted((json.loads(l) for f in args.rows for l in open(f) if l.strip()), key=lambda r: r["problem_id"])
    boot = np.random.default_rng(0).integers(0, len(rows), (args.n_boot, len(rows)))
    out = {"n_traces": len(rows)}
    for pair in args.pairs:
        a, b = pair.split(":")
        buckets = [k for k in rows[0]["acc"][a] if any(r["acc"][a][k][0] for r in rows)]
        res = {}
        for k in buckets + ["all"]:
            ks = buckets if k == "all" else [k]
            na, sa, nb, sb = (np.array([sum(r["acc"][c][q][i] for q in ks) for r in rows], float)
                              for c, i in ((a, 0), (a, 1), (b, 0), (b, 1)))
            db = sa[boot].sum(1) / np.maximum(1, na[boot].sum(1)) - sb[boot].sum(1) / np.maximum(1, nb[boot].sum(1))
            res[k] = {"kl_a": sa.sum() / na.sum(), "kl_b": sb.sum() / nb.sum(), "diff": sa.sum() / na.sum() - sb.sum() / nb.sum(),
                      "ci": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))], "n_tok": int(na.sum())}
            print(f"{a} - {b} {k}: {res[k]['kl_a']:.4f} - {res[k]['kl_b']:.4f} = {res[k]['diff']:+.4f} "
                  f"[{res[k]['ci'][0]:+.4f}, {res[k]['ci'][1]:+.4f}] ({res[k]['n_tok']:,} tok)")
        out[pair] = res
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
