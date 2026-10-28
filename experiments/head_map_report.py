"""Aggregate head_map.py output (nti-e2's Step 0 "few heads remember"). Per input file (one source): the 32-row table
ranked by gated_far, concentration (share of the total gated_far in the top-1/2/4/8 heads), split-half stability (odd vs even
traces in problem_id order: top-4 overlap, Spearman), per-dataset top-4, and the GO / NO-GO rule set by nti-e2 before the data:
GO if top-4 hold >= 50% of the total gated_far AND split-half top-4 overlap >= 3/4; NO-GO if spread out (top-4 near 12.5%).
If two files are given, also the top-4 overlap and Spearman between them (TL test traces vs held-out E1 traces).
Usage: python head_map_report.py out_hm/hm_tl.jsonl [out_hm/hm_e1.jsonl]"""
import json, sys
import numpy as np


def rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


def spearman(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def top(x, k):
    return set(np.argsort(-x)[:k].tolist())


def load(f):
    rows = sorted((json.loads(l) for l in open(f) if l.strip()), key=lambda r: r["problem_id"])
    A = {m: np.array([np.array(r[m]).ravel() for r in rows]) for m in ("far_mass", "gated_far", "far_share", "prompt_mass")}
    return rows, A


means = {}
for f in sys.argv[1:]:
    rows, A = load(f)
    fa = rows[0]["fa_layers"]
    kvh = len(rows[0]["far_mass"][0])
    name = lambda i: f"L{fa[i // kvh]}h{i % kvh}"
    m = {k: v.mean(0) for k, v in A.items()}
    means[f] = m
    gf = m["gated_far"]
    order = np.argsort(-gf)
    print(f"# {f}: {len(rows)} traces ({rows[0]['source']}), {rows[0]['n_q']} queries each, W = {rows[0]['W']}; "
          f"datasets {dict((d, sum(r['dataset'] == d for r in rows)) for d in sorted({r['dataset'] for r in rows}))}")
    print(f"max relative error, manual attention vs SDPA output: {max(r['max_rel_err_vs_sdpa'] for r in rows):.2e}\n")
    print("| rank | FA layer (model) | kv head | far_mass | gated_far | far_share | prompt_mass | share of total gated_far |")
    print("|---|---|---|---|---|---|---|---|")
    for j, i in enumerate(order):
        print(f"| {j + 1} | {i // kvh} ({fa[i // kvh]}) | {i % kvh} | {m['far_mass'][i]:.3f} | {gf[i]:.4f} | "
              f"{m['far_share'][i]:.3f} | {m['prompt_mass'][i]:.3f} | {gf[i] / gf.sum():.3f} |")
    conc = {k: float(np.sort(gf)[::-1][:k].sum() / gf.sum()) for k in (1, 2, 4, 8)}
    print("\nConcentration (share of total gated_far): " + ", ".join(f"top-{k} {v:.3f}" for k, v in conc.items())
          + f" (uniform: top-4 {4 / len(gf):.3f})")
    for metric in ("gated_far", "far_mass", "far_share"):
        x = A[metric]
        odd, even = x[1::2].mean(0), x[0::2].mean(0)
        print(f"Split-half ({metric}; even vs odd traces, n {len(x[0::2])}/{len(x[1::2])}): top-4 overlap "
              f"{len(top(odd, 4) & top(even, 4))}/4, top-8 overlap {len(top(odd, 8) & top(even, 8))}/8, Spearman {spearman(odd, even):.3f}")
    print("Per-dataset top-4 by gated_far: " + "; ".join(
        f"{d} (n {sum(r['dataset'] == d for r in rows)}): " + ", ".join(name(i) for i in np.argsort(-A['gated_far'][[r['dataset'] == d for r in rows]].mean(0))[:4])
        for d in sorted({r['dataset'] for r in rows})))
    x = A["gated_far"]
    ov = len(top(x[1::2].mean(0), 4) & top(x[0::2].mean(0), 4))
    go = conc[4] >= 0.5 and ov >= 3
    print(f"\n**Rule (nti-e2, fixed before data):** top-4 share {conc[4]:.3f} (GO needs >= 0.5), split-half top-4 overlap {ov}/4 "
          f"(GO needs >= 3) -> {'GO' if go else ('NO-GO (spread out)' if conc[4] < 0.2 else 'in between: report only')}\n")
if len(means) == 2:
    (fa_, a_), (fb, b_) = means.items()
    print(f"# Between sources: top-4 overlap {len(top(a_['gated_far'], 4) & top(b_['gated_far'], 4))}/4, "
          f"top-8 overlap {len(top(a_['gated_far'], 8) & top(b_['gated_far'], 8))}/8, "
          f"Spearman gated_far {spearman(a_['gated_far'], b_['gated_far']):.3f}")
