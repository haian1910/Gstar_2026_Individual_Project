# Think-Local statistics (nti-e2, 2026-10-26), as pre-registered in prereg_tl.md. Stdlib only, plus fta_stats helpers.
# Runs JSONL: {problem_id, dataset, cell, sample_k, correct, finished, n_tokens, rep_score, answer, n_reencoded, max_tokens};
# optional `repetitive` (bool, threshold fixed in the generation code). Domain is countdown if dataset starts with
# "countdown", else math. Cell names are canonicalised: full, tl-w2k, dt-c4k-m2k, sl-2k (case and separators ignored).
# The 16K view (math only): finished16 = finished and n_tokens <= 16384, correct16 = correct and finished16.
# k = 8 rule (prereg): samples 4-7 of full and tl-w2k count only if complete for both cells on every problem; then they
# enter the TL-W2K - Full contrasts only, and everything else uses samples 0-3.
# Usage: python3 tl_stats.py --runs FILE [--out FILE] [--n-boot 10000] [--margin-pp 5]
#        python3 tl_stats.py --selftest
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
from collections import defaultdict
from math import comb
from pathlib import Path

from fta_stats import answers_equal, load_jsonl

FA_BYTES_PER_TOKEN = 8 * 4 * 256 * 2 * 2                    # 8 FA layers x 4 KV heads x 256 x (K, V) x bf16 = 32 KiB
GDN_STATE_BYTES = 24 * 32 * 128 * 128 * 4 + 24 * 8192 * 3 * 2  # fp32 recurrent state + bf16 conv state, ~49 MiB
CUT_16K = 16384
VIEWS = {"math": {"cap": None, "16k": CUT_16K}, "countdown": {"cap": None}}
CD_TL = "tl-w2k"        # Countdown window, pinned from smoke-test Full lengths only (prereg S4); tl-w1k if >= 50% finish within 2,048
PRIMARY = ("math", "tl-w2k", "full")
SECONDARY = [("S1", "math", "tl-w4k", "dt-c4k-m2k"), ("S2", "math", "tl-w2k", "sl-2k"),
             ("S3", "math", "tl-w1k", "full"), ("S3", "math", "tl-w2k", "full"), ("S3", "math", "tl-w4k", "full"),
             ("S4", "countdown", CD_TL, "full")]
S5 = [("S5", "math", "tl-w2k", "full"), ("S5", "math", "tl-w4k", "dt-c4k-m2k")]    # re-scored on the 16K view


def canon(cell):
    raw = re.fullmatch(r"(tl|sl|dt)-(\d+)(?:-(\d+))?", cell.lower())   # server cell_raw in tokens: tl-8192, sl-2048, dt-4096-2048
    if raw and all(int(x) % 1024 == 0 for x in raw.groups()[1:] if x):
        kind, a, b = raw.groups()
        cell = {"tl": f"tl-w{int(a) // 1024}k", "sl": f"sl-{int(a) // 1024}k"}.get(kind) or f"dt-c{int(a) // 1024}k-m{int(b) // 1024}k"
    s = re.sub(r"[^a-z0-9]", "", cell.lower())
    for pat, fmt in ((r"full", "full"), (r"tlw(\d+)k", "tl-w{}k"), (r"dtc(\d+)km(\d+k?)", "dt-c{}k-m{}"),
                     (r"sl(\d+)k", "sl-{}k")):
        m = re.fullmatch(pat, s)
        if m:
            return fmt.format(*m.groups())
    raise ValueError(f"unknown cell name: {cell}")


def domain_of(r):
    return "countdown" if str(r["dataset"]).lower().startswith("countdown") else "math"


def view(r, cut):
    """(correct, finished, n_tokens) under a post-hoc token cut; cut=None is the run's own cap."""
    if cut is None:
        return bool(r["correct"]), bool(r["finished"]), r["n_tokens"]
    fin = bool(r["finished"]) and r["n_tokens"] <= cut
    return bool(r["correct"]) and fin, fin, min(r["n_tokens"], cut)


def fa_peak_tokens(cell, n):
    """Peak FA slots held for generated tokens; the prompt's KV is shared across samples and left out."""
    if cell == "full":
        return n
    m = re.fullmatch(r"tl-w(\d+)k", cell) or re.fullmatch(r"dt-c(\d+)k-m\d+k?", cell)
    if m:
        return min(n, int(m.group(1)) * 1024)
    m = re.fullmatch(r"sl-(\d+)k", cell)
    return min(n, int(m.group(1)) * 1024 + 4)


def sample_bytes(cell, n):
    return fa_peak_tokens(cell, n) * FA_BYTES_PER_TOKEN + GDN_STATE_BYTES


def strat_boot(strata, n_boot, seed=0, alpha=0.05):
    """strata: {stratum: {cluster: value}}. Resamples clusters within each stratum (so the dataset mix stays fixed);
    the statistic is the mean over clusters. Returns (estimate, lo, hi)."""
    groups = [list(v.values()) for v in strata.values() if v]
    n = sum(len(g) for g in groups)
    if not n:
        return (float("nan"),) * 3
    rng, boots = random.Random(seed), []
    for _ in range(n_boot):
        boots.append(sum(sum(rng.choices(g, k=len(g))) for g in groups) / n)
    boots.sort()
    return sum(sum(g) for g in groups) / n, boots[int(alpha / 2 * n_boot)], boots[int((1 - alpha / 2) * n_boot) - 1]


def verdict(lo, hi, margin):
    """Pre-registered wording rule; lo, hi and margin in the same units (pp)."""
    if lo > -margin:
        return f"non-inferior at {margin:g} pp"
    if hi < -margin:
        return "worse by more than the margin"
    if hi < 0:
        return "worse"
    return "inconclusive"


def index(runs):
    """{(domain, cell): {(dataset, problem_id): {sample_k: run}}}"""
    idx = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        key = (r["dataset"], r["problem_id"])
        cell = canon(r["cell"])
        assert r["sample_k"] not in idx[(domain_of(r), cell)][key], f"duplicate run {key} {cell} k={r['sample_k']}"
        if "correct16" in r:                      # the scorer's convenience fields must match the derivation here
            assert view(r, CUT_16K)[:2] == (bool(r["correct16"]), bool(r["finished16"])), f"16K view mismatch {key} {cell}"
        idx[(domain_of(r), cell)][key][r["sample_k"]] = r
    return idx


METRICS = {"correct": 0, "finished": 1, "n_tokens": 2}


def per_problem(by_p, metric, cut, keys=None):
    out = defaultdict(dict)
    for (ds, pid), ks in by_p.items():
        if keys is None or (ds, pid) in keys:
            vals = [view(r, cut)[METRICS[metric]] for r in ks.values()]
            out[ds][pid] = sum(vals) / len(vals)
    return out


def contrast(idx, dom, a, b, cut, n_boot, dataset=None):
    """Paired by problem: per-problem mean of cell a minus cell b, for each metric, plus mean tokens over finished runs.
    Seeds are nominal (same-seed runs diverge within a few hundred tokens), so nothing is paired by sample_k."""
    pa, pb = idx.get((dom, a), {}), idx.get((dom, b), {})
    keys = {k for k in set(pa) & set(pb) if dataset is None or k[0] == dataset}
    res = {"n_problems": len(keys), "n_only_one_cell": len((set(pa) ^ set(pb)) - keys) if dataset is None else None}
    for m in METRICS:
        va, vb = per_problem(pa, m, cut, keys), per_problem(pb, m, cut, keys)
        diff = {ds: {p: va[ds][p] - vb[ds][p] for p in va[ds]} for ds in va}
        est, lo, hi = strat_boot(diff, n_boot)
        res[m] = {"diff": est, "ci": [lo, hi]}
    fin = defaultdict(dict)
    for k in sorted(keys):  # sorted: set order follows str hashing, which changes per process
        ta = [view(r, cut)[2] for r in pa[k].values() if view(r, cut)[1]]
        tb = [view(r, cut)[2] for r in pb[k].values() if view(r, cut)[1]]
        if ta and tb:
            fin[k[0]][k[1]] = sum(ta) / len(ta) - sum(tb) / len(tb)
    est, lo, hi = strat_boot(fin, n_boot)
    res["n_tokens_finished_only"] = {"n_problems": sum(len(v) for v in fin.values()), "diff": est, "ci": [lo, hi]}
    return res


def cell_summary(idx, n_boot, rep_threshold=None):
    out = {}
    for (dom, cell), by_p in sorted(idx.items()):
        runs = [r for ks in by_p.values() for r in ks.values()]
        for vname, cut in VIEWS[dom].items():
            acc = strat_boot(per_problem(by_p, "correct", cut), n_boot)
            s = {"n_runs": len(runs), "n_problems": len(by_p), "accuracy": acc[0], "accuracy_ci": list(acc[1:]),
                 "finished": sum(view(r, cut)[1] for r in runs) / len(runs),
                 "mean_tokens": sum(view(r, cut)[2] for r in runs) / len(runs)}
            if vname == "cap":
                if all("repetitive" in r for r in runs):
                    s["repetitive"] = sum(bool(r["repetitive"]) for r in runs) / len(runs)
                elif rep_threshold is not None:
                    s["repetitive"] = sum(r["rep_score"] >= rep_threshold for r in runs) / len(runs)
                reps = [r["rep_score"] for r in runs if r.get("rep_score") is not None]
                s["mean_rep_score"] = sum(reps) / len(reps) if reps else None
                s["mean_reencoded"] = sum(r.get("n_reencoded") or 0 for r in runs) / len(runs)
                s["mean_sample_mib"] = sum(sample_bytes(cell, r["n_tokens"]) for r in runs) / len(runs) / 2 ** 20
                s["beyond_window"] = sum(fa_peak_tokens(cell, r["n_tokens"]) < r["n_tokens"] for r in runs) / len(runs)
                # post hoc (added after TL-W2K was seen): how runs end; descriptive, run-level
                fin = sorted(r["n_tokens"] for r in runs if r["finished"])
                q = lambda f: fin[min(len(fin) - 1, int(f * len(fin)))] if fin else None
                s["post_hoc"] = {"n_finished": len(fin), "n_finished_wrong": sum(r["finished"] and not r["correct"] for r in runs),
                                 "correct_if_finished": sum(r["correct"] for r in runs if r["finished"]) / len(fin) if fin else None,
                                 "closed_think": (sum(bool(r["closed_think"]) for r in runs) / len(runs)
                                                  if all("closed_think" in r for r in runs) else None),
                                 "finished_len": {"p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "max": fin[-1] if fin else None}}
            out[f"{dom}:{cell}:{vname}"] = s
    return out


def answer_classes(runs):
    """Equivalence class of each run's final answer; unfinished or answerless runs get None."""
    reps, labels = [], []
    for r in runs:
        if not r["finished"] or r.get("answer") in (None, ""):
            labels.append(None)
            continue
        for i, a in enumerate(reps):
            if answers_equal(a, r["answer"]):
                labels.append(i)
                break
        else:
            reps.append(r["answer"])
            labels.append(len(reps) - 1)
    return labels


def vote(labels, correct):
    """Expected accuracy of a majority vote under a random tie-break; runs labelled None cast no vote."""
    groups = defaultdict(list)
    for lab, c in zip(labels, correct):
        if lab is not None:
            groups[lab].append(c)
    if not groups:
        return 0.0
    top = max(len(v) for v in groups.values())
    winners = [v for v in groups.values() if len(v) == top]
    return sum(sum(v) / len(v) for v in winners) / len(winners)


def maj_at(ks, m):
    """Mean majority-vote accuracy over all m-subsets of a problem's samples."""
    runs = list(ks.values())
    labels, correct = answer_classes(runs), [bool(r["correct"]) for r in runs]
    subs = list(itertools.combinations(range(len(runs)), m))
    return sum(vote([labels[i] for i in s], [correct[i] for i in s]) for s in subs) / len(subs)


def pass_at(ks, m):
    """Unbiased pass@m (Chen et al. 2021) from n samples with c correct."""
    n, c = len(ks), sum(bool(r["correct"]) for r in ks.values())
    return 1.0 - comb(n - c, m) / comb(n, m)


def matched_memory(idx, n_boot):
    """At the mean per-sample memory of one Full sample, a cell fits m = floor(full / cell) samples (capped at k).
    Math: maj@m vs Full pass@1. Countdown: pass@m (the validator needs only the problem's inputs)."""
    out = {}
    for dom in ("math", "countdown"):
        full = idx.get((dom, "full"))
        if not full:
            continue
        full_mib = [sample_bytes("full", r["n_tokens"]) for ks in full.values() for r in ks.values()]
        full_mean = sum(full_mib) / len(full_mib)
        for (d, cell), by_p in sorted(idx.items()):
            if d != dom:
                continue
            cell_b = [sample_bytes(cell, r["n_tokens"]) for ks in by_p.values() for r in ks.values()]
            k = min(len(ks) for ks in by_p.values())
            m = max(1, min(k, math.floor(full_mean / (sum(cell_b) / len(cell_b)))))
            score = maj_at if dom == "math" else pass_at
            per = defaultdict(dict)
            for (ds, pid), ks in by_p.items():
                per[ds][pid] = score(ks, m)
            est, lo, hi = strat_boot(per, n_boot)
            out[f"{dom}:{cell}"] = {"memory_ratio": full_mean / (sum(cell_b) / len(cell_b)), "k": k, "m": m,
                                    "score": "maj@m" if dom == "math" else "pass@m", "mean": est, "ci": [lo, hi]}
    return out


def ranks(xs):
    """Average ranks (ties share their mean rank)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx, syy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return float("nan") if sxx == 0 or syy == 0 else sxy / math.sqrt(sxx * syy)


def moderator(idx, n_boot, seed=0, alpha=0.05):
    """Exploratory: Spearman between a problem's trigger density (Wait/Alternatively/... per 1K thinking tokens, from Full
    samples) and its TL - Full accuracy difference (TL-W2K for math, CD_TL for Countdown). Cross-fitted so the two never share samples: density from Full samples
    {0,1} against the difference on samples {2,3}, and the reverse; the two rhos are averaged. Bootstrap over problems
    within datasets. Also rho(density, Full accuracy), the difficulty confound."""
    halves = ((0, 1), (2, 3))
    out = {}
    for dom in ("math", "countdown"):
        full, tl = idx.get((dom, "full"), {}), idx.get((dom, "tl-w2k" if dom == "math" else CD_TL), {})
        rows = defaultdict(list)                  # dataset -> [(dens_a, diff_b, dens_b, diff_a, dens_all, full_acc)]
        for key in sorted(set(full) & set(tl)):
            f, t = full[key], tl[key]
            if not all(k in f and k in t for h in halves for k in h):
                continue
            if any(f[k].get("n_triggers_per_1k") is None for k in range(4)):
                continue
            dens = [sum(f[k]["n_triggers_per_1k"] for k in h) / 2 for h in halves]
            diff = [sum(bool(t[k]["correct"]) - bool(f[k]["correct"]) for k in h) / 2 for h in halves]
            acc = sum(bool(f[k]["correct"]) for k in range(4)) / 4
            rows[key[0]].append((dens[0], diff[1], dens[1], diff[0], sum(dens) / 2, acc))
        flat = [r for v in rows.values() for r in v]
        if len(flat) < 5:
            continue

        def stat(rs):
            cross = (spearman([r[0] for r in rs], [r[1] for r in rs]) + spearman([r[2] for r in rs], [r[3] for r in rs])) / 2
            return cross, spearman([r[4] for r in rs], [r[5] for r in rs])

        rng, boots = random.Random(seed), []
        for _ in range(n_boot):
            b = [x for v in rows.values() for x in rng.choices(v, k=len(v))]
            boots.append(stat(b))
        est = stat(flat)
        ci = lambda j: sorted(x[j] for x in boots if not math.isnan(x[j]))
        c0, c1 = ci(0), ci(1)
        q = lambda c, a: c[min(len(c) - 1, int(a * len(c)))] if c else float("nan")
        out[dom] = {"n_problems": len(flat), "rho_density_vs_diff": est[0], "ci": [q(c0, alpha / 2), q(c0, 1 - alpha / 2)],
                    "rho_density_vs_full_acc": est[1], "ci_full_acc": [q(c1, alpha / 2), q(c1, 1 - alpha / 2)]}
    return out


LEN_BINS = (0, 8192, 16384, 24576, None)   # POST HOC (2026-10-27 ~01:40 UTC, after TL-W2K was seen): Full's per-problem mean length


def by_full_length(idx, n_boot):
    """Post hoc, descriptive: each cell's accuracy and finished rate within bins of Full's per-problem mean n_tokens
    (samples 0-3). A problem's bin is set by Full only, so every cell is compared on the same problems."""
    out = {}
    for dom in ("math", "countdown"):
        full = idx.get((dom, "full"))
        if not full:
            continue
        flen = {k: sum(r["n_tokens"] for r in ks.values()) / len(ks) for k, ks in full.items()}
        for lo, hi in zip(LEN_BINS, LEN_BINS[1:]):
            keys = {k for k, v in flen.items() if v >= lo and (hi is None or v < hi)}
            if not keys:
                continue
            name = f"{lo // 1024}k-{hi // 1024}k" if hi else f"{lo // 1024}k+"
            for (d, cell), by_p in idx.items():
                if d != dom or not keys <= set(by_p):
                    continue
                acc = strat_boot(per_problem(by_p, "correct", None, keys), n_boot)
                fin = strat_boot(per_problem(by_p, "finished", None, keys), n_boot)
                out[f"{dom}:{cell}:{name}"] = {"n_problems": len(keys), "lo": lo, "hi": hi, "accuracy": acc[0],
                                               "accuracy_ci": list(acc[1:]), "finished": fin[0]}
    return out


K_BASE, K_EXT = 4, 8
K8_CELLS = {"math": ("full", "tl-w2k"), "countdown": ("full", CD_TL)}


def k8_complete(runs):
    """True if each domain's k = 8 pair has samples 0-7 on every problem that appears in any run."""
    problems = {(domain_of(r), r["dataset"], r["problem_id"]) for r in runs}
    have = defaultdict(set)
    for r in runs:
        if canon(r["cell"]) in K8_CELLS[domain_of(r)]:
            have[(canon(r["cell"]), r["dataset"], r["problem_id"])].add(r["sample_k"])
    return all(have[(c, ds, pid)] >= set(range(K_EXT)) for dom, ds, pid in problems for c in K8_CELLS[dom])


def analyse(runs, n_boot=10000, margin_pp=5.0, rep_threshold=None):
    use_k8 = k8_complete(runs)
    idx8 = index(runs) if use_k8 else None
    idx = index([r for r in runs if r["sample_k"] < K_BASE])
    pick = lambda dom, a, b: idx8 if use_k8 and {a, b} == set(K8_CELLS[dom]) else idx
    res = {"meta": {"n_runs": len(runs), "k8_used": use_k8, "countdown_tl": CD_TL, "n_boot": n_boot, "margin_pp": margin_pp, "fa_bytes_per_token": FA_BYTES_PER_TOKEN,
                    "gdn_state_bytes": GDN_STATE_BYTES, "cut_16k": CUT_16K},
           "cells": cell_summary(idx, n_boot, rep_threshold)}
    dom, a, b = PRIMARY
    if (dom, a) in idx and (dom, b) in idx:
        p = contrast(pick(dom, a, b), dom, a, b, None, n_boot)
        lo, hi = p["correct"]["ci"][0] * 100, p["correct"]["ci"][1] * 100
        p["verdict"] = verdict(lo, hi, margin_pp)
        p["text"] = (f"Math accuracy, TL-W2K - Full (32K cap): {p['correct']['diff'] * 100:+.1f} pp "
                     f"(95% CI {lo:+.1f} to {hi:+.1f}); {p['verdict']}")
        res["primary"] = p
    sec = {}
    for tag, dom, a, b in SECONDARY + S5:
        if (dom, a) not in idx or (dom, b) not in idx:
            continue
        cut = CUT_16K if tag == "S5" else None
        c = contrast(pick(dom, a, b), dom, a, b, cut, n_boot)
        if tag in ("S4", "S5") and b == "full":
            c["verdict"] = verdict(c["correct"]["ci"][0] * 100, c["correct"]["ci"][1] * 100, margin_pp)
        sec[f"{tag}:{dom}:{a}-{b}"] = c
    res["secondary"] = sec
    expl = {}
    for (dom, cell) in sorted(idx):
        if cell == "full" or (dom, "full") not in idx:
            continue
        datasets = sorted({ds for ds, _ in idx[(dom, cell)]})
        for vname, cut in VIEWS[dom].items():
            expl[f"{dom}:{cell}-full:{vname}"] = contrast(idx, dom, cell, "full", cut, n_boot)
            if vname == "cap" and len(datasets) > 1:
                for ds in datasets:
                    expl[f"{dom}:{cell}-full:{vname}:{ds}"] = contrast(idx, dom, cell, "full", cut, n_boot, dataset=ds)
    res["exploratory"] = {"contrasts": expl, "matched_memory": matched_memory(idx, n_boot), "moderator": moderator(idx, n_boot),
                          "post_hoc_by_full_length": by_full_length(idx, n_boot)}
    return res


def restart_sensitivity(runs, restarted, n_boot):
    """Pre-registered 2026-10-27 10:47 UTC (before SL-2K was scored): rows redrawn after a mid-generation stop are
    length-selected. Re-report S2 and each affected cell with those rows (a) scored unfinished and wrong, (b) left out."""
    hit = lambda r: (canon(r["cell"]), str(r["problem_id"]), int(r["sample_k"])) in restarted
    worst = lambda r: {**r, "correct": False, "finished": False, "n_tokens": r["max_tokens"], "correct16": False, "finished16": False}
    variants = {"as_generated": runs, "a_unfinished_wrong": [worst(r) if hit(r) else r for r in runs],
                "b_left_out": [r for r in runs if not hit(r)]}
    cells = sorted({c for c, _, _ in restarted})
    out = {"n_restarted": len(restarted), "by_cell": {c: sum(1 for x in restarted if x[0] == c) for c in cells}}
    for name, rs in variants.items():
        idx = index([r for r in rs if r["sample_k"] < K_BASE])
        v = {}
        for tag, dom, a, b in SECONDARY + S5:
            if tag == "S2" and (dom, a) in idx and (dom, b) in idx:
                v[f"{tag}:{dom}:{a}-{b}"] = contrast(idx, dom, a, b, None, n_boot)
        for c in cells:
            for dom in VIEWS:
                if (dom, c) in idx:
                    acc = strat_boot(per_problem(idx[(dom, c)], "correct", None), n_boot)
                    v[f"{dom}:{c}:accuracy"] = {"est": acc[0], "ci": list(acc[1:])}
        out[name] = v
    return out


def load_restarted(path):
    """TSV with a header containing problem_id and sample_k, and optionally cell (raw or canonical; default sl-2k,
    the only cell restarted mid-generation). Problem ids are unique across domains in the real data."""
    lines = [l.rstrip("\n").split("\t") for l in Path(path).read_text().splitlines() if l.strip()]
    col = {h: i for i, h in enumerate(lines[0])}
    cell = lambda r: canon(r[col["cell"]]) if "cell" in col else "sl-2k"
    return {(cell(r), r[col["problem_id"]], int(r[col["sample_k"]])) for r in lines[1:]}


def selftest():
    rng = random.Random(1)
    assert canon("TL-W2K") == canon("tl_w2k") == "tl-w2k" and canon("DT(C=4K, m=2K)") == "dt-c4k-m2k"
    assert canon("SL-2K") == "sl-2k" and canon("Full") == "full" and canon("dt-c1k-m512") == "dt-c1k-m512"
    assert canon("tl-8192") == "tl-w8k" and canon("sl-2048") == "sl-2k" and canon("dt-4096-2048") == "dt-c4k-m2k"
    assert fa_peak_tokens("dt-c1k-m512", 5000) == 1024
    assert fa_peak_tokens("tl-w2k", 10000) == 2048 and fa_peak_tokens("dt-c4k-m2k", 3000) == 3000
    assert fa_peak_tokens("sl-2k", 9000) == 2052 and fa_peak_tokens("full", 9000) == 9000
    assert abs(GDN_STATE_BYTES / 2 ** 20 - 49.125) < 1e-9
    assert view({"correct": True, "finished": True, "n_tokens": 20000}, CUT_16K) == (False, False, CUT_16K)
    assert view({"correct": True, "finished": True, "n_tokens": 9000}, CUT_16K) == (True, True, 9000)
    assert verdict(-4.0, 3.0, 5) == "non-inferior at 5 pp" and verdict(-9.0, 2.0, 5) == "inconclusive"
    assert verdict(-9.0, -1.0, 5) == "worse" and verdict(-12.0, -6.0, 5) == "worse by more than the margin"
    # vote: two say 5 (correct), one says 7, one unfinished -> 1.0; a 1-1 tie between right and wrong -> 0.5
    mk = lambda a, c, f=True: {"answer": a, "correct": c, "finished": f}
    v = lambda rs: vote(answer_classes(rs), [r["correct"] for r in rs])
    assert v([mk("5", True), mk("5", True), mk("7", False), mk("5", True, False)]) == 1.0
    assert v([mk("5", True), mk("7", False)]) == 0.5 and v([mk(None, False)]) == 0.0
    assert abs(maj_at({0: mk("5", True), 1: mk("5", True), 2: mk("7", False)}, 2) - (1 + 0.5 + 0.5) / 3) < 1e-12
    assert abs(pass_at({0: mk("5", True), 1: mk("7", False), 2: mk("7", False), 3: mk("7", False)}, 2) - 0.5) < 1e-12
    # synthetic runs: TL-W2K is 4 pp worse than Full on math and writes 1,000 more tokens
    runs, true_p = [], {}
    for ds in ("aime25", "hmmt25", "math500_l5"):
        for i in range(40):
            true_p[(ds, i)] = rng.uniform(0.1, 0.95)
    for (ds, i), p in true_p.items():
        for cell, dp, dn in (("full", 0.0, 0), ("TL-W2K", -0.04, 1000), ("DT(C=4K,m=2K)", -0.08, 1500),
                             ("SL-2K", -0.3, 2000)):
            for k in range(8):
                n = int(rng.gauss(12000 + dn, 3000))
                fin = n < 32768
                n = min(max(n, 500), 32768)
                ok = fin and rng.random() < max(0.0, p + dp)
                runs.append({"problem_id": i, "dataset": ds, "cell": cell, "sample_k": k, "correct": ok, "finished": fin,
                             "n_tokens": n, "rep_score": 0.0, "answer": "1" if ok else str(2 + k % 3),
                             "n_reencoded": 0, "max_tokens": 32768, "n_triggers_per_1k": rng.random()})
    cd = lambda i, cell, k: {"problem_id": i, "dataset": "countdown", "cell": cell, "sample_k": k,
                             "correct": rng.random() < 0.8, "finished": True, "n_tokens": 4000, "rep_score": 0.0,
                             "answer": "x", "n_reencoded": 0, "max_tokens": 16384, "n_triggers_per_1k": rng.random()}
    runs += [cd(i, cell, k) for i in range(30) for cell in ("full", "tl-w2k") for k in range(4)]
    res4 = analyse(runs, n_boot=1000)            # countdown has only k = 4, so the k = 8 rule falls back to samples 0-3
    assert not res4["meta"]["k8_used"]
    lo, hi = res4["primary"]["correct"]["ci"]
    assert lo < -0.04 < hi
    runs += [cd(i, cell, k) for i in range(30) for cell in ("full", "tl-w2k") for k in range(4, 8)]
    res = analyse(runs, n_boot=1000)
    assert res["meta"]["k8_used"]
    p = res["primary"]
    assert -0.08 < p["correct"]["diff"] < 0.0, p["correct"]
    assert p["correct"]["ci"][1] - p["correct"]["ci"][0] < hi - lo
    assert 700 < p["n_tokens"]["diff"] < 1300 and p["n_problems"] == 120
    assert res["exploratory"]["moderator"]["countdown"]["n_problems"] == 30
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0 and spearman([1, 1, 2], [3, 2, 1]) < 0
    assert "S1:math:tl-w4k-dt-c4k-m2k" not in res["secondary"] and "S2:math:tl-w2k-sl-2k" in res["secondary"]
    assert res["secondary"]["S2:math:tl-w2k-sl-2k"]["correct"]["ci"][0] > 0.15
    assert "math:tl-w2k-full:16k" in res["exploratory"]["contrasts"]
    mm = res["exploratory"]["matched_memory"]
    assert mm["math:full"]["m"] == 1 and mm["math:tl-w2k"]["m"] >= 3 and mm["countdown:tl-w2k"]["score"] == "pass@m"
    assert res["cells"]["math:sl-2k:cap"]["accuracy"] < res["cells"]["math:full:cap"]["accuracy"]
    print("selftest ok;", p["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs")
    ap.add_argument("--out")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--margin-pp", type=float, default=5.0)
    ap.add_argument("--rep-threshold", type=float, help="only if runs carry no `repetitive` field")
    ap.add_argument("--restarted", help="TSV of rows redrawn after a mid-generation stop (problem_id, sample_k, cell)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    runs = load_jsonl(a.runs)
    res = analyse(runs, a.n_boot, a.margin_pp, a.rep_threshold)
    if a.restarted:
        res["exploratory"]["restart_sensitivity"] = restart_sensitivity(runs, load_restarted(a.restarted), a.n_boot)
    s = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(s)
    print(res.get("primary", {}).get("text", "primary contrast not available yet"))
    for k, v in res["secondary"].items():
        print(f"{k}: acc {v['correct']['diff'] * 100:+.1f} pp {[round(x * 100, 1) for x in v['correct']['ci']]}, "
              f"tokens {v['n_tokens']['diff']:+.0f}" + (f"; {v['verdict']}" if "verdict" in v else ""))


if __name__ == "__main__":
    main()
