#!/usr/bin/env python3
# POST HOC, exploratory (prereg_fade.md change log, after the decode): contrasts the prereg did not name, and accuracy per
# dataset. Separate from fade_stats.py so the audited primary output (results/stats_fade.json) stays as it was.
# Usage: fade_explore.py --runs <same files as fade_stats.py> --out results/stats_fade_explore.json
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fade_stats as F  # noqa: E402
import tl_stats as T  # noqa: E402

CONTRASTS = [("fade-kv4", "tl-w8k"), ("fade-kv2-had", "fade-kv4"), ("fade-kv2-had", "tl-w4k"), ("fade-kv4", "tl-w4k")]
CELLS = ("full", "tl-w2k", "tl-w4k", "tl-w8k", "fade-kv2-had", "fade-kv4")
COMPARATOR_K = {"k0-1": {0, 1}, "k2-3": {2, 3}}  # sensitivity: which comparator samples enter (the prereg uses all 4)


def analyse(runs, n_boot=10000):
    runs = [r for r in runs if T.domain_of(r) == "math" and r["sample_k"] < T.K_BASE]
    idx = F.index(runs)
    out = {"contrasts": {}, "by_dataset": {}}
    for a, b in CONTRASTS:
        c = T.contrast(idx, "math", a, b, None, n_boot)
        ratio = F.summary(idx, a, 1)["mean_bytes"] / F.summary(idx, b, 1)["mean_bytes"]
        out["contrasts"][f"{a} - {b}"] = {**F.pp(c), "memory_ratio": ratio, "detail": c}
    for c in CELLS:
        acc = T.per_problem(idx[("math", c)], "correct", None)
        out["by_dataset"][c] = {ds: 100 * sum(v.values()) / len(v) for ds, v in sorted(acc.items())}
    out["sensitivity"] = {}
    for name, ks in COMPARATOR_K.items():
        sub = F.index([r for r in runs if F.fcanon(r["cell"]).startswith("fade-") or r["sample_k"] in ks])
        for a, b in (("fade-kv2-had", "tl-w8k"), ("fade-kv2-had", "full")):
            out["sensitivity"][f"{a} - {b}:comparator_{name}"] = F.pp(T.contrast(sub, "math", a, b, None, n_boot))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    out = analyse([r for f in args.runs for r in T.load_jsonl(f)], args.n_boot)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    for k, v in out["contrasts"].items():
        t = v["detail"]["n_tokens"]
        print(f"{k}: {v['diff_pp']:+.1f} pp ({v['ci_pp'][0]:+.1f} to {v['ci_pp'][1]:+.1f}); memory x{v['memory_ratio']:.2f}; "
              f"tokens {t['diff']:+.0f} ({t['ci'][0]:+.0f} to {t['ci'][1]:+.0f})")
    for k, v in out["sensitivity"].items():
        print(f"sensitivity {k}: {v['diff_pp']:+.1f} pp ({v['ci_pp'][0]:+.1f} to {v['ci_pp'][1]:+.1f})")
    for c, d in out["by_dataset"].items():
        print(c, " ".join(f"{ds} {x:.1f}" for ds, x in d.items()))


if __name__ == "__main__":
    main()
