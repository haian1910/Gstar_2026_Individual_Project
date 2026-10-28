#!/usr/bin/env python3
# Fade vs StreamingLLM on the hybrid (prereg_fade.md, entry added before any StreamingLLM row; haian's refocus on Full and
# StreamingLLM, relayed ~23:3x UTC). StreamingLLM-8K (sl-8192) keeps 4 sink tokens + the last 8,192 and drops the prompt.
# Every cell uses samples 0-1 only, so both sides of each contrast have 2 samples per problem (seeds are nominal, nothing is
# paired by sample_k). Contrasts are paired by problem, with the stratified cluster bootstrap of tl_stats (10,000, seed 0).
# Memory is analytic, prompt KV left out for every cell (as in TL and fade_stats), so SL's true memory is lower by the prompt.
# Usage: fade_sl.py --runs tl_runs_*_k4.jsonl fd_runs_*.jsonl sl_runs_*.jsonl --out results/stats_fade_sl.json
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fade_stats as F  # noqa: E402
import tl_stats as T  # noqa: E402

K_SL = 2                 # samples 0-1, the SL decode's samples
N_PROBLEMS = 120
MATCHED = 1.10           # memory rule of the Fade primary: ratio <= 1.10 counts as matched
CONTRASTS = [            # (name, a, b, labelled): labelled contrasts get the wording rule and a memory ratio
    ("SL1", "fade-kv2-had", "sl-8k", True),
    ("SL2", "fade-kv4", "sl-8k", True),
    ("SL3", "full", "sl-8k", False),
    ("SL4", "tl-w8k", "sl-8k", False),    # what keeping the prompt adds to an 8K window
    ("SL5", "fade-kv2-had", "sl-4k", True),   # only if sl-4096 ran
    ("SL6", "tl-w4k", "sl-4k", False),
]


def label(lo, hi, ref, ratio):
    mem = "at matched memory" if ratio <= MATCHED else f"at {ratio:.2f}x its memory"
    return f"better than {ref} {mem}" if lo > 0 else "worse" if hi < 0 else "inconclusive at n = 120"


def complete(idx, cell, k):
    by_p = idx.get(("math", cell), {})
    return len(by_p) == N_PROBLEMS and all(sorted(ks) == list(range(k)) for ks in by_p.values())


def analyse(runs, n_boot=10000):
    math = [r for r in runs if T.domain_of(r) == "math"]
    idx = F.index([r for r in math if r["sample_k"] < K_SL])
    need = {c for _, a, b, _ in CONTRASTS[:4] for c in (a, b)}
    bad = sorted(c for c in need if not complete(idx, c, K_SL))
    if bad:
        return {"status": "not complete", "incomplete_cells": bad}
    out = {"status": "complete", "k": K_SL,
           "cells": {c: F.summary(idx, c, n_boot) for dom, c in sorted(idx) if dom == "math"}, "contrasts": {}}
    for name, a, b, labelled in CONTRASTS:
        if not (complete(idx, a, K_SL) and complete(idx, b, K_SL)):
            out["contrasts"][name] = {"contrast": f"{a} - {b}", "status": "not run"}
            continue
        c = T.contrast(idx, "math", a, b, None, n_boot)
        res = {"contrast": f"{a} - {b}", **F.pp(c), "detail": c}
        if labelled:
            ratio = out["cells"][a]["mean_bytes"] / out["cells"][b]["mean_bytes"]
            ref = "StreamingLLM-" + b.split("-")[1].upper()
            res.update({"memory_ratio": ratio, "verdict": label(*res["ci_pp"], ref, ratio)})
        out["contrasts"][name] = res
    # Sensitivity (registered with the contrasts): Fade with every sample it has (0-3 once samples 2-3 land) vs SL 0-1.
    idx_all = F.index([r for r in math if r["sample_k"] < K_SL or r["cell"].lower().startswith(("fq", "fade"))])
    out["sensitivity_fade_all_samples"] = {}
    for name, a, b, labelled in CONTRASTS:
        if labelled and a.startswith("fade-") and complete(idx, b, K_SL):
            c = T.contrast(idx_all, "math", a, b, None, n_boot)
            ks = sorted({len(v) for v in idx_all[("math", a)].values()})
            out["sensitivity_fade_all_samples"][name] = {"contrast": f"{a} (k={ks}) - {b} (k=[2])", **F.pp(c)}
    return out


def selftest():
    import random
    rng, runs = random.Random(3), []
    for ds in ("aime25", "aime26", "hmmt25", "math500_l5"):
        for i in range(30):
            for cell, p, k in (("full", .65, 4), ("tl-w8k", .55, 4), ("tl-w4k", .42, 4), ("fq-kv2_had", .6, 4),
                               ("fq-kv4", .66, 4), ("sl-8192", .12, 2)):
                for s in range(k):
                    ok = rng.random() < p
                    runs.append({"problem_id": f"{ds}_{i}", "dataset": ds, "cell": cell, "sample_k": s, "correct": ok,
                                 "finished": ok or rng.random() < .3, "n_tokens": rng.randint(3000, 32768),
                                 "max_tokens": 32768})
    out = analyse(runs, n_boot=300)
    assert out["status"] == "complete" and out["cells"]["fade-kv2-had"]["k"] == [2] and out["cells"]["sl-8k"]["k"] == [2]
    assert out["contrasts"]["SL1"]["verdict"].startswith("better than StreamingLLM-8K")
    assert out["contrasts"]["SL5"]["status"] == "not run" and "SL5" not in out["sensitivity_fade_all_samples"]
    assert "k=[4]" in out["sensitivity_fade_all_samples"]["SL1"]["contrast"]
    assert 0.6 < out["contrasts"]["SL1"]["memory_ratio"] < 0.8, out["contrasts"]["SL1"]["memory_ratio"]
    part = analyse([r for r in runs if not (r["cell"] == "sl-8192" and r["problem_id"] == "aime25_0" and r["sample_k"] == 1)])
    assert part == {"status": "not complete", "incomplete_cells": ["sl-8k"]}
    print(f"selftest ok (SL1 {out['contrasts']['SL1']['diff_pp']:+.1f} pp, memory x{out['contrasts']['SL1']['memory_ratio']:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    runs = [json.loads(l) for f in args.runs for l in open(f) if l.strip()]
    out = analyse(runs, args.n_boot)
    Path(args.out).write_text(json.dumps(out, indent=1))
    if out["status"] != "complete":
        print(f"not complete: {out['incomplete_cells']} (no numbers reported, prereg)")
        return
    for name, c in out["contrasts"].items():
        if c.get("status") == "not run":
            print(f"{name} {c['contrast']}: not run")
            continue
        mem = f"; memory x{c['memory_ratio']:.2f}" if "memory_ratio" in c else ""
        print(f"{name} {c['contrast']}: {c['diff_pp']:+.1f} pp ({c['ci_pp'][0]:+.1f} to {c['ci_pp'][1]:+.1f}){mem}"
              + (f" -> {c['verdict']}" if "verdict" in c else ""))
    for name, c in out["sensitivity_fade_all_samples"].items():
        print(f"sensitivity {name} {c['contrast']}: {c['diff_pp']:+.1f} pp ({c['ci_pp'][0]:+.1f} to {c['ci_pp'][1]:+.1f})")
    for c in ("full", "tl-w8k", "sl-8k", "fade-kv2-had", "fade-kv4"):
        s = out["cells"][c]
        print(f"{c}: accuracy {100 * s['correct']['mean']:.1f}% ({100 * s['correct']['ci'][0]:.1f}-{100 * s['correct']['ci'][1]:.1f}),"
              f" {s['mean_bytes'] / 2 ** 20:.0f} MiB mean, cap hit {100 * s['cap_hit']:.0f}%")


if __name__ == "__main__":
    main()
