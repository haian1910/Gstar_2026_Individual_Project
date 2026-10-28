"""Aggregate the post hoc joint check (layer-map PLAN Changes 16:2x UTC): per config, KL(full||cfg)/tok, top-1 agreement,
dNLL/tok, KL by bucket, and KL as a fraction of all8, with 95% CIs by bootstrap over traces (n_boot 10000, seed 0).
Determinism check: this run's all8 sums vs the map's all8 sums (out_lm), per trace.
Usage: python layer_map_joint_report.py --joint out_lm_joint/lmj.s*.jsonl --map out_lm/lm.s*.jsonl"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--joint", nargs="+", required=True)
ap.add_argument("--map", nargs="+", required=True)
a = ap.parse_args()
rows = sorted((json.loads(l) for f in a.joint for l in open(f) if l.strip()), key=lambda r: r["problem_id"])
ref = {r["problem_id"]: r for f in a.map for l in open(f) if l.strip() for r in [json.loads(l)]}
cfgs = list(rows[0]["acc"])
buckets = [b for b in rows[0]["acc"]["all8"] if any(r["acc"]["all8"][b][0] for r in rows)]
T = {c: np.array([[sum(r["acc"][c][b][i] for b in buckets) for i in range(4)] for r in rows]) for c in cfgs}
boot = np.random.default_rng(0).integers(0, len(rows), (10000, len(rows)))


def mean(a, i, idx=None):
    a = a if idx is None else a[idx]
    return a[..., i].sum(-1) / np.maximum(1, a[..., 0].sum(-1))


diff = max(abs(r["acc"]["all8"][b][i] - ref[r["problem_id"]]["acc"]["all8"][b][i])
           for r in rows for b in buckets for i in range(4))
print(f"# Joint check: {len(rows)} traces, W = {rows[0]['W']}, scored tokens {int(T['all8'][:, 0].sum()):,}")
print(f"Determinism: max |all8 sum (this run) - all8 sum (map)| over traces x buckets x fields = {diff:.3g}\n")
print("| Config (windowed FA layers) | KL/tok | 95% CI | KL / KL(all8) | 95% CI | top-1 agree | ΔNLL/tok | "
      + " | ".join(f"KL {b}" for b in buckets) + " |")
print("|---|---|---|---|---|---|---|" + "---|" * len(buckets))
k8b = mean(T["all8"], 1, boot)
for c in cfgs:
    kb = mean(T[c], 1, boot)
    fr = kb / k8b
    bk = " | ".join(f"{sum(r['acc'][c][b][1] for r in rows) / max(1, sum(r['acc'][c][b][0] for r in rows)):.4f}" for b in buckets)
    print(f"| {c} | {mean(T[c], 1):.4f} | [{np.percentile(kb, 2.5):.4f}, {np.percentile(kb, 97.5):.4f}] | "
          f"{mean(T[c], 1) / mean(T['all8'], 1):.3f} | [{np.percentile(fr, 2.5):.3f}, {np.percentile(fr, 97.5):.3f}] | "
          f"{mean(T[c], 2):.4f} | {mean(T[c], 3):+.4f} | {bk} |")
