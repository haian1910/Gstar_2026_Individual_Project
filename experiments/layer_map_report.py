"""Aggregate layer_map.py output (HA_own/experiments/2026-10-27_layer-map/PLAN.md) into the PLAN's table and decision.
Token-weighted means over all traces (generated index >= W); 95% CIs by bootstrap over traces (n_boot 10000, seed 0).
Recovery(l) = 1 - KL(except_l) / KL(all8). Decision (fixed in PLAN): sum of the top-2 recoveries >= 0.5 or one >= 0.3 -> propose a
mixed training-free cell; kill: all recoveries < 0.15 and only_l KLs within 2x of each other.
Usage: python layer_map_report.py out_lm/lm.s0.jsonl out_lm/lm.s1.jsonl > layer_map_report.md"""
import json, sys
import numpy as np

rows = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.strip()]
rows.sort(key=lambda r: r["problem_id"])
cfgs = list(rows[0]["acc"])
buckets = list(rows[0]["acc"]["all8"])
fa = rows[0]["fa_layers"]
nl = len(fa)
# per trace, per config: totals over all buckets [n, kl, agree, dnll], and per bucket
T = {c: np.array([[sum(r["acc"][c][b][i] for b in buckets) for i in range(4)] for r in rows]) for c in cfgs}
TB = {c: {b: np.array([r["acc"][c][b] for r in rows]) for b in buckets} for c in cfgs}
rng = np.random.default_rng(0)
boot = rng.integers(0, len(rows), (10000, len(rows)))


def mean(a, i, idx=None):  # token-weighted mean of field i
    a = a if idx is None else a[idx]
    return a[..., i].sum(-1) / np.maximum(1, a[..., 0].sum(-1))


def ci(x):
    return f"[{np.percentile(x, 2.5):.3f}, {np.percentile(x, 97.5):.3f}]"


print(f"# Layer-sensitivity map: {len(rows)} traces, W = {rows[0]['W']}, tokens scored (j >= W): {int(T['all8'][:, 0].sum()):,}")
print(f"FA layer l = model layer {dict(enumerate(fa))}. Problems: {', '.join(r['problem_id'] for r in rows)}\n")
print("| Config | KL(full‖cfg)/tok | top-1 agree | ΔNLL/tok | " + " | ".join(f"KL {b}" for b in buckets) + " |")
print("|---|---|---|---|" + "---|" * len(buckets))
for c in cfgs:
    bk = " | ".join(f"{mean(TB[c][b], 1):.4f} (n {int(TB[c][b][:, 0].sum())})" for b in buckets)
    print(f"| {c} | {mean(T[c], 1):.4f} | {mean(T[c], 2):.4f} | {mean(T[c], 3):+.4f} | {bk} |")

kl8 = mean(T["all8"], 1)
kl8_b = mean(T["all8"], 1, boot)
rec = np.array([1 - mean(T[f"except_{l}"], 1) / kl8 for l in range(nl)])
rec_b = np.stack([1 - mean(T[f"except_{l}"], 1, boot) / kl8_b for l in range(nl)])
only = np.array([mean(T[f"only_{l}"], 1) for l in range(nl)])
only_b = np.stack([mean(T[f"only_{l}"], 1, boot) for l in range(nl)])
print("\n| FA layer l (model layer) | Recovery(l) = 1 − KL(except_l)/KL(all8) | 95% CI | KL(only_l) | 95% CI | only_l / sum of only |")
print("|---|---|---|---|---|---|")
for l in range(nl):
    print(f"| {l} ({fa[l]}) | {rec[l]:.3f} | {ci(rec_b[l])} | {only[l]:.4f} | {ci(only_b[l])} | {only[l] / only.sum():.3f} |")
print(f"\nSum of only_l KLs = {only.sum():.4f} vs KL(all8) = {kl8:.4f} (ratio {only.sum() / kl8:.2f}; > 1 means the layers' effects overlap, < 1 means they compound).")

top = np.argsort(-rec)
top2 = rec[top[0]] + rec[top[1]]
top2_b = np.sort(rec_b, 0)[-2:].sum(0)  # per bootstrap: the best 2 in that resample
go = top2 >= 0.5 or rec.max() >= 0.3
kill = rec.max() < 0.15 and only.max() <= 2 * only.min()
print("\n## Decision (thresholds fixed in PLAN.md)")
print(f"- Top-2 recoveries: layers {top[0]} and {top[1]}, sum {top2:.3f} (bootstrap of the best-2 sum: {ci(top2_b)}); max single {rec.max():.3f}.")
print(f"- only_l KL max/min = {only.max() / max(only.min(), 1e-9):.2f}.")
if go:
    print(f"- **GO rule fires**: propose a training-free mixed cell, FA layers {sorted(top[:2].tolist())} full, the rest TL-W4K (needs haian's OK).")
elif kill:
    print("- **Kill result fires**: no layer stands out; next step is distillation/training, not layer selection.")
else:
    print("- Neither rule fires: partial concentration. Report as is; no mixed run is proposed by the pre-set rule.")
