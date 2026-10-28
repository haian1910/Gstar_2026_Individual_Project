#!/usr/bin/env python3
# H2 pulse test (prereg_fade.md, "Where does a memory error live"; exploratory, descriptive): KL(E || X) per token after a
# 1K-token 2-bit pulse, by distance d from the pulse's end, pooled over traces (ratio of sums), per problem and per trace.
# Readings by the prereg rules, applied to the pooled d >= 1,024 bucket:
#   floor (added ~18:45): if KL(Pc) and KL(Ps) are both <= 2 x KL(P0) -> "no persistent error detectable above numerical noise"
#   else (fixed ~18:26): the larger of Pc and Ps is where the error mainly persists (FA cache vs GDN state).
# No bootstrap CI: the registered sample is 10 traces of 2 problems (prereg ~22:19). Rows from kv_pulse.py:
#   acc {config: {bucket: [n_tok, kl_sum, top1_agree]}}, curve {config: [[n_tok, kl_sum] per 64-token bin]}, chars {bucket: n}.
# Usage: pulse_stats.py --hybrid pulse.jsonl [--dense pulse_dense.jsonl] --out results/pulse_stats.json
import argparse
import json
from collections import defaultdict
from pathlib import Path

LATE = "1024-4096"
FLOOR_X = 2.0


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def pooled(rows, cfg):
    """KL/token (and per 1,000 characters) per bucket, summed over rows before dividing."""
    out = {}
    for b in rows[0]["acc"][cfg]:
        n = sum(r["acc"][cfg][b][0] for r in rows)
        s = sum(r["acc"][cfg][b][1] for r in rows)
        t = sum(r["acc"][cfg][b][2] for r in rows)
        c = sum(r["chars"][b] for r in rows)
        if n == 0:  # Pc, Ps, Pb branch from P at the pulse's end: no "pulse" bucket
            continue
        out[b] = {"n_tok": n, "kl_tok": s / n, "top1": t / n, "kl_kchar": 1000 * s / c}
    return out


def curve(rows, cfg):
    k = len(rows[0]["curve"][cfg])
    return [sum(r["curve"][cfg][i][1] for r in rows) / max(1, sum(r["curve"][cfg][i][0] for r in rows)) for i in range(k)]


def reading(cells):
    pc, ps, p0 = (cells[c][LATE]["kl_tok"] for c in ("Pc", "Ps", "P0"))
    if pc <= FLOOR_X * p0 and ps <= FLOOR_X * p0:
        return "no persistent error detectable above numerical noise"
    return "mainly in the FA cache" if pc > ps else "mainly in the GDN state"


def summarise(rows, model):
    cfgs = list(rows[0]["acc"])
    by_problem = defaultdict(list)
    for r in rows:
        by_problem[r["problem_id"]].append(r)
    out = {"model": rows[0]["model"], "n_traces": len(rows), "problems": {p: len(v) for p, v in by_problem.items()},
           "gates": {k: all(r["gates"][k] if isinstance(r["gates"][k], bool) else r["gates"][k] == 0 for r in rows)
                     for k in rows[0]["gates"]},
           "pooled": {c: pooled(rows, c) for c in cfgs},
           "per_problem": {p: {c: pooled(v, c) for c in cfgs} for p, v in by_problem.items()},
           "per_trace": [{"problem_id": r["problem_id"], "sample_idx": r["sample_idx"],
                          **{c: r["acc"][c][LATE][1] / r["acc"][c][LATE][0] for c in cfgs}} for r in rows],
           "curve": {c: curve(rows, c) for c in cfgs}}
    if model == "hybrid":
        out["reading"] = reading(out["pooled"])
        out["reading_per_problem"] = {p: reading(v) for p, v in out["per_problem"].items()}
        out["late_ratio_to_P0"] = {c: out["pooled"][c][LATE]["kl_tok"] / out["pooled"]["P0"][LATE]["kl_tok"] for c in cfgs}
        out["per_trace_reading"] = []
        for r in rows:
            cells = {c: {LATE: {"kl_tok": r["acc"][c][LATE][1] / r["acc"][c][LATE][0]}} for c in ("Pc", "Ps", "P0")}
            out["per_trace_reading"].append(reading(cells))
    else:
        out["late_ratio_to_P0"] = {c: out["pooled"][c][LATE]["kl_tok"] / out["pooled"]["P0"][LATE]["kl_tok"] for c in cfgs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid", required=True)
    ap.add_argument("--dense")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {"hybrid": summarise(load(args.hybrid), "hybrid")}
    if args.dense:
        out["dense"] = summarise(load(args.dense), "dense")
    Path(args.out).write_text(json.dumps(out, indent=1))
    for m, s in out.items():
        print(f"== {m} ({s['model']}): {s['n_traces']} traces, problems {s['problems']}; gates {s['gates']}")
        buckets = list(s["pooled"]["P"])
        print("config | " + " | ".join(buckets) + f" | {LATE} / P0")
        for c, v in s["pooled"].items():
            print(f"{c} | " + " | ".join(f"{v[b]['kl_tok']:.5f}" if b in v else "-" for b in buckets) + f" | {s['late_ratio_to_P0'][c]:.2f}x")
        if "reading" in s:
            print(f"reading (pooled): {s['reading']}; per problem: {s['reading_per_problem']}")
            print(f"per trace: {s['per_trace_reading']}")


if __name__ == "__main__":
    main()
