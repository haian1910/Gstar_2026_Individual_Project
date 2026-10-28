#!/usr/bin/env python3
# Pre-registered analysis for "Fade, don't forget" (prereg_fade.md): old generated tokens in the 8 FA layers stored at
# b bits, the prompt and the last ~2K generated tokens exact. Reuses tl_stats (paired-by-problem contrasts, stratified
# cluster bootstrap, wording rule); only cell names, analytic memory and the new contrasts live here.
# Usage: fade_stats.py --runs tl_runs_*.jsonl fade_runs_*.jsonl --cell-a kv2 --cell-b kv1 --out results/stats_fade.json
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tl_stats as T  # noqa: E402

R_EXACT = 2048          # recent generated tokens kept exact
BLOCK = 128             # a block is quantized once its last token is <= g - R_EXACT (KIVI-style residual)
GROUP_OVERHEAD = 0.25   # fp16 scale + zero per group of 128 -> +0.25 bits
MIX_BITS = {2: 4, 3: 4}  # mix: FA layers 2 and 3 (model 11, 15) at 4 bits, the other 6 at 2
FADE = re.compile(r"(?:(?:fade|fq)[-_]?)?(kv(\d+)(?:[-_](pre|had))?|k(\d+)only|v(\d+)only|mix)")


def fcanon(cell):
    """Fade cells -> 'fade-kv2', 'fade-kv1-had', 'fade-mix', ...; everything else goes through tl_stats.canon."""
    m = FADE.fullmatch(cell.lower())
    if m:
        return "fade-" + m.group(1).replace("_", "-")
    return T.canon(cell)


def old_bits(cell):
    """Effective bits per old-token element (K and V averaged), including the scale/zero overhead."""
    name = cell[len("fade-"):]
    if name == "mix":
        return sum(MIX_BITS.get(i, 2) + GROUP_OVERHEAD for i in range(8)) / 8
    m = re.fullmatch(r"kv(\d+)(?:-(?:pre|had))?", name)
    if m:
        b = int(m.group(1))
        return 16.0 if b >= 16 else b + GROUP_OVERHEAD
    m = re.fullmatch(r"k(\d+)only|v(\d+)only", name)
    b = int(m.group(1) or m.group(2))
    return (b + GROUP_OVERHEAD + 16) / 2


def quantized_tokens(n):
    """Old tokens held in quantized form at the last generated position g = n - 1."""
    last = n - 1 - R_EXACT
    return 0 if last < BLOCK - 1 else ((last - (BLOCK - 1)) // BLOCK + 1) * BLOCK


def sample_bytes(cell, n):
    """Analytic per-sample FA + GDN bytes at the end of a run of n generated tokens (prompt KV left out, as in TL)."""
    if not cell.startswith("fade-"):
        return T.sample_bytes(cell, n)
    q = quantized_tokens(n)
    return ((n - q) + q * old_bits(cell) / 16) * T.FA_BYTES_PER_TOKEN + T.GDN_STATE_BYTES


def index(runs):
    idx = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        key, cell = (r["dataset"], r["problem_id"]), fcanon(r["cell"])
        assert r["sample_k"] not in idx[(T.domain_of(r), cell)][key], f"duplicate run {key} {cell} k={r['sample_k']}"
        idx[(T.domain_of(r), cell)][key][r["sample_k"]] = r
    return idx


def superiority(lo, hi):
    """Primary wording rule (prereg_fade.md): two-sided 95% CI, no margin."""
    return "better than an 8K window at matched memory" if lo > 0 else "worse" if hi < 0 else "inconclusive at n = 120"


def summary(idx, cell, n_boot):
    by_p = idx.get(("math", cell), {})
    rows = [r for ks in by_p.values() for r in ks.values()]
    if not rows:
        return None
    out = {"n_problems": len(by_p), "n_runs": len(rows),
           "k": sorted({len(ks) for ks in by_p.values()}),
           "mean_bytes": sum(sample_bytes(cell, r["n_tokens"]) for r in rows) / len(rows),
           "cap_hit": sum(1 for r in rows if not r["finished"] and r["n_tokens"] >= r.get("max_tokens", 32768)) / len(rows),
           "finished_correct": [sum(1 for r in rows if r["finished"] and r["correct"]),
                                sum(1 for r in rows if r["finished"])]}
    if cell.startswith("fade-"):
        out["old_bits_effective"] = old_bits(cell)
    for m in T.METRICS:
        est, lo, hi = T.strat_boot(T.per_problem(by_p, m, None), n_boot)
        out[m] = {"mean": est, "ci": [lo, hi]}
    return out


def pp(c):
    return {"diff_pp": 100 * c["correct"]["diff"], "ci_pp": [100 * x for x in c["correct"]["ci"]]}


def analyse(runs, cell_a, cell_b, n_boot=10000, margin_pp=5.0):
    runs = [r for r in runs if T.domain_of(r) == "math" and r["sample_k"] < T.K_BASE]
    idx = index(runs)
    a, b = fcanon(cell_a), fcanon(cell_b)
    for c in (a, b, "full", "tl-w2k", "tl-w8k"):
        assert ("math", c) in idx, f"missing cell {c}"
    out = {"cells": {c: summary(idx, c, n_boot) for dom, c in sorted(idx) if dom == "math"},
           "cell_a": a, "cell_b": b}
    prim = T.contrast(idx, "math", a, "tl-w8k", None, n_boot)
    p = pp(prim)
    ratio = out["cells"][a]["mean_bytes"] / out["cells"]["tl-w8k"]["mean_bytes"]
    out["primary"] = {"contrast": f"{a} - tl-w8k", **p, "verdict": superiority(*p["ci_pp"]),
                      "memory_ratio_vs_tl_w8k": ratio, "memory_matched": ratio <= 1.10, "detail": prim}
    q1 = T.contrast(idx, "math", a, "full", None, n_boot)
    out["Q1"] = {"contrast": f"{a} - full", **pp(q1), "verdict": T.verdict(*pp(q1)["ci_pp"], margin_pp), "detail": q1}
    out["Q2"] = {f"{b} - {ref}": {**pp(c), "detail": c}
                 for ref in ("tl-w2k", "full") for c in [T.contrast(idx, "math", b, ref, None, n_boot)]}
    curve = [("tl-w2k", 0.0), (b, out["cells"][b].get("old_bits_effective")), (a, out["cells"][a].get("old_bits_effective")),
             ("full", 16.0)]
    out["Q2"]["curve"] = [{"cell": c, "old_bits": bits, "accuracy": out["cells"][c]["correct"]["mean"],
                           "ci": out["cells"][c]["correct"]["ci"]} for c, bits in curve]
    out["exploratory"] = {"by_full_length": T.by_full_length(idx, n_boot), "paired_by_full_length": by_length(idx, a, n_boot)}
    return out


def by_length(idx, a, n_boot):
    """Exploratory (prereg_fade.md): Cell A minus TL-W8K and minus Full, paired by problem, within Think-Local's bins of
    Full's per-problem mean length (samples 0-3), so the decode can be read against the KL sweep's length dependence."""
    full = idx[("math", "full")]
    flen = {k: sum(r["n_tokens"] for r in ks.values()) / len(ks) for k, ks in full.items()}
    out = {}
    for lo, hi in zip(T.LEN_BINS, T.LEN_BINS[1:]):
        keys = {k for k, v in flen.items() if v >= lo and (hi is None or v < hi)}
        name = f"{lo // 1024}k-{hi // 1024}k" if hi else f"{lo // 1024}k+"
        for ref in ("tl-w8k", "full"):
            va, vb = (T.per_problem(idx[("math", c)], "correct", None, keys) for c in (a, ref))
            est, l, h = T.strat_boot({ds: {p: va[ds][p] - vb[ds][p] for p in va[ds]} for ds in va}, n_boot)
            out[f"{a} - {ref}:{name}"] = {"n_problems": len(keys), "diff_pp": 100 * est, "ci_pp": [100 * l, 100 * h]}
    return out


def selftest():
    assert fcanon("kv2") == "fade-kv2" and fcanon("fade-kv1_had") == "fade-kv1-had" and fcanon("KV2-pre") == "fade-kv2-pre"
    assert fcanon("mix") == "fade-mix" and fcanon("k2only") == "fade-k2only" and fcanon("tl-8192") == "tl-w8k"
    assert abs(old_bits("fade-kv2") - 2.25) < 1e-9 and old_bits("fade-kv16") == 16.0
    assert abs(old_bits("fade-mix") - 2.75) < 1e-9 and abs(old_bits("fade-v2only") - 9.125) < 1e-9
    assert quantized_tokens(2048 + 127) == 0 and quantized_tokens(2048 + 128) == 128 and quantized_tokens(100) == 0
    # g = n - 1; block j is quantized iff 128 j + 127 <= g - 2048
    for n in (2175, 2176, 2177, 2303, 2304, 10000, 32768):
        g = n - 1
        assert quantized_tokens(n) == 128 * sum(1 for j in range(n // 128 + 1) if 128 * j + 127 <= g - R_EXACT), n
    assert sample_bytes("fade-kv16", 30000) == T.sample_bytes("full", 30000)
    n = 32768
    kv2 = sample_bytes("fade-kv2", n) / 2 ** 20
    w8k = T.sample_bytes("tl-w8k", n) / 2 ** 20
    assert 190 < kv2 < 260 and 300 < w8k < 310, (kv2, w8k)
    import random
    rng, runs = random.Random(1), []
    for ds in ("aime25", "hmmt25"):
        for i in range(30):
            for cell, p, k in (("full", .7, 4), ("tl-8192", .55, 4), ("tl-2048", .25, 4), ("kv2", .65, 2), ("kv1", .4, 2)):
                for s in range(k):
                    ok = rng.random() < p
                    runs.append({"problem_id": f"{ds}_{i}", "dataset": ds, "cell": cell, "sample_k": s, "correct": ok,
                                 "finished": ok or rng.random() < .3, "n_tokens": rng.randint(3000, 32768),
                                 "max_tokens": 32768})
    out = analyse(runs, "kv2", "kv1", n_boot=500)
    assert out["cells"]["fade-kv2"]["k"] == [2] and out["cells"]["full"]["k"] == [4]
    assert out["primary"]["contrast"] == "fade-kv2 - tl-w8k" and out["Q2"]["curve"][0]["old_bits"] == 0.0
    print(f"selftest ok (kv2 {kv2:.0f} MiB vs TL-W8K {w8k:.0f} MiB at 32K; primary {out['primary']['diff_pp']:+.1f} pp)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--cell-a")
    ap.add_argument("--cell-b")
    ap.add_argument("--out")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    runs = [r for f in args.runs for r in T.load_jsonl(f)]
    out = analyse(runs, args.cell_a, args.cell_b, n_boot=args.n_boot)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    p, q1 = out["primary"], out["Q1"]
    print(f"primary {p['contrast']}: {p['diff_pp']:+.1f} pp ({p['ci_pp'][0]:+.1f} to {p['ci_pp'][1]:+.1f}) -> {p['verdict']}; "
          f"memory x{p['memory_ratio_vs_tl_w8k']:.2f} of TL-W8K")
    print(f"Q1 {q1['contrast']}: {q1['diff_pp']:+.1f} pp ({q1['ci_pp'][0]:+.1f} to {q1['ci_pp'][1]:+.1f}) -> {q1['verdict']}")


if __name__ == "__main__":
    main()
