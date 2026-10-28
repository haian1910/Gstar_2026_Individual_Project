# Forget-the-Attempt prototype statistics (stdlib only; uses math_verify for answer equivalence if installed)
# Schemas: prototype_spec.md §7. Token offsets (attempt_start_tok, b_tok, c_tok) are counted from the start of the thinking text.
# Usage: python3 fta_stats.py {e1,e2,e3,kappa,posthoc,selftest} --results DIR [--json]
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

KV_BYTES_PER_TOKEN = 8 * 4 * 256 * 2 * 2     # 8 FA layers x 4 KV heads x 256 dims x K,V x bf16 = 32 KB
CELLS = ("full", "ours", "textdel", "gdn_necessity", "placebo")
CONTRASTS = (("ours", "full"), ("ours", "textdel"), ("textdel", "full"), ("gdn_necessity", "full"), ("placebo", "full"))

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
except ImportError:
    _mv_parse = _mv_verify = None


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- answer equivalence ----------

def _unwrap(s, cmd):
    while True:
        i = s.find(cmd + "{")
        if i < 0:
            return s
        depth, j = 0, i + len(cmd)
        for j in range(i + len(cmd), len(s)):
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            if depth == 0:
                break
        s = s[:i] + s[i + len(cmd) + 1:j] + s[j + 1:]


def normalize_answer(a):
    if a is None:
        return None
    s = str(a).strip().strip("$").strip()
    for cmd in ("\\boxed", "\\text", "\\mathrm", "\\textbf"):
        s = _unwrap(s, cmd)
    s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("^\\circ", "").replace("°", "").replace("\\!", "")
    if "=" in s and not s.startswith("("):
        s = s.split("=")[-1]
    s = s.strip().rstrip(".").replace(" ", "")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    return s


_LATEX_NUM = {"sqrt": "", "pi": "", "cdot": "*", "times": "*"}


def is_numeric_value(w):
    """Spec §3: a rejected value counts only if it is a number or a closed-form numeric expression
    (6^9, 3\\sqrt{2}, \\frac{3}{4}, (1,2), x=5); free text and expressions with variables do not.
    Same rule as fta_common.is_numeric_value, which writes w_is_numeric into boundaries.jsonl."""
    if w is None or str(w).strip() == "":
        return False
    s = re.sub(r"\\(sqrt|pi|cdot|times)", lambda m: _LATEX_NUM[m.group(1)], normalize_answer(w))
    return bool(re.search(r"\d", s)) and re.fullmatch(r"[0-9+\-*/^().{}\[\],]+", s) is not None


def _as_number(s):
    t = s.replace("(", "").replace(")", "")
    try:
        if re.fullmatch(r"-?\d+/-?\d+", t):
            n, d = t.split("/")
            return Fraction(int(n), int(d))
        return Fraction(t)
    except (ValueError, ZeroDivisionError):
        return None


def answers_equal(a, b):
    """True if two final answers are mathematically equal; missing answers never match."""
    if a is None or b is None or str(a).strip() == "" or str(b).strip() == "":
        return False
    if _mv_verify is not None:
        try:
            return bool(_mv_verify(_mv_parse(f"${a}$"), _mv_parse(f"${b}$")))
        except Exception:
            pass
    na, nb = normalize_answer(a), normalize_answer(b)
    if na == nb:
        return True
    xa, xb = _as_number(na), _as_number(nb)
    return xa is not None and xb is not None and xa == xb


# ---------- generic statistics ----------

def cluster_bootstrap(clusters, stat=None, n_boot=2000, seed=0, alpha=0.05):
    """clusters: {cluster_id: [values]}. Resamples clusters with replacement; stat defaults to the pooled mean.
    Returns (estimate, lo, hi)."""
    stat = stat or (lambda vals: sum(vals) / len(vals))
    keys = [k for k, v in clusters.items() if v]
    pooled = [x for k in keys for x in clusters[k]]
    if not pooled:
        return (float("nan"),) * 3
    rng, boots = random.Random(seed), []
    for _ in range(n_boot):
        sample = [x for k in rng.choices(keys, k=len(keys)) for x in clusters[k]]
        boots.append(stat(sample))
    boots.sort()
    return stat(pooled), boots[int(alpha / 2 * n_boot)], boots[int((1 - alpha / 2) * n_boot) - 1]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


def cohen_kappa(a, b):
    assert len(a) == len(b) and a, "need two equal-length, non-empty label lists"
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def noninferiority(diff, lo, hi, margin_pp, what="Ours vs Full accuracy"):
    """Wording rule: never say 'no loss'; report the difference and CI and compare the lower bound to the margin."""
    verdict = "non-inferior" if lo > -margin_pp else "not shown non-inferior"
    return f"{what}: {diff:+.1f} pp (95% CI {lo:+.1f} to {hi:+.1f}); {verdict} at a {margin_pp:.0f} pp margin"


def fmt(est, lo, hi, scale=100, unit="pp"):
    return f"{est * scale:+.1f} {unit} (95% CI {lo * scale:+.1f} to {hi * scale:+.1f})"


# ---------- E1 ----------

def _merge(spans):
    out = []
    for a, b in sorted(s for s in spans if s[1] > s[0]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


WEAK_TRIGGERS = ("wait", "alternatively", "hmm", "actually")
RESTART_TRIGGERS = ("approach", "method", "another way", "start over", "start again", "reconsider", "rethink",
                    "scrap that", "go back to")


def trigger_class(bd):
    """weak / negation / restart (prototype_spec.md §2.1); uses bd["trigger_type"] if the exporter wrote it."""
    if bd.get("trigger_type"):
        return bd["trigger_type"]
    t = str(bd.get("trigger") or "").strip().lower()
    if not t:
        return "unknown"
    if t in WEAK_TRIGGERS:
        return "weak"
    return "restart" if any(k in t for k in RESTART_TRIGGERS) else "negation"


def _attempt_span(bd):
    if bd.get("attempt_start_tok") is None or bd.get("b_tok") is None:
        return None
    return (int(bd["attempt_start_tok"]), int(bd["b_tok"]))


REASSERT_FIELDS = {"plain": ("w_reasserted_after_c", "base_reasserted_after_c"),
                   "strict": ("w_reasserted_strict", "base_reasserted_strict")}


def _reassert_summary(events, n_boot):
    excess = cluster_bootstrap({p: [r - b for r, b in v] for p, v in events.items()}, n_boot=n_boot)
    return {"n_events": sum(len(v) for v in events.values()),
            "reassert_rate": cluster_bootstrap({p: [r for r, _ in v] for p, v in events.items()}, n_boot=n_boot),
            "base_rate": cluster_bootstrap({p: [b for _, b in v] for p, v in events.items()}, n_boot=n_boot),
            "excess": excess, "excess_text": fmt(*excess)}


def e1(traces, boundaries, n_boot=2000, exclude_truncated=False):
    by_key = {(t["problem_id"], t["sample_idx"]): t for t in traces}
    by_problem = defaultdict(list)
    for t in traces:
        by_problem[t["problem_id"]].append(t)
    bds_by_trace = defaultdict(list)
    for bd in boundaries:
        bds_by_trace[(bd["problem_id"], bd["sample_idx"])].append(bd)

    labels = Counter(bd.get("judge_label") for bd in boundaries)
    labels_by_trigger = defaultdict(Counter)
    for bd in boundaries:
        labels_by_trigger[trigger_class(bd)][bd.get("judge_label")] += 1
    abandons_per_trace = [sum(bd.get("judge_label") == "abandon" for bd in bds_by_trace[k]) for k in by_key]

    # excess relapse: events are distinct (trace, rejected wrong value w)
    events = defaultdict(list)                     # problem_id -> [(relapse, base, base_norej)]
    reassert = defaultdict(lambda: defaultdict(list))   # variant[_w1|_w2plus] -> problem_id -> [(w reasserted, base)]
    n_events = n_relapse = n_missing_final = n_non_numeric = 0
    for key, bds in bds_by_trace.items():
        t = by_key.get(key)
        if t is None or (exclude_truncated and t.get("truncated")):
            continue
        seen = []
        for bd in sorted(bds, key=lambda x: x.get("b_tok") or 0):
            w = bd.get("rejected_value")
            if bd.get("judge_label") != "abandon" or w in (None, ""):
                continue
            numeric = bd.get("w_is_numeric")
            if not (is_numeric_value(w) if numeric is None else numeric):
                n_non_numeric += 1                 # free-text "values" can never relapse; counting them dilutes the excess
                continue
            w_wrong = bd.get("w_is_wrong")
            if w_wrong is None:
                w_wrong = not answers_equal(w, t.get("reference"))
            if not w_wrong or any(answers_equal(w, s) for s in seen):
                continue
            seen.append(w)
            others = [o for o in by_problem[key[0]] if o["sample_idx"] != key[1]
                      and not (exclude_truncated and o.get("truncated"))]
            if not others:
                continue
            relapse = float(answers_equal(t.get("final_answer"), w))
            base = sum(answers_equal(o.get("final_answer"), w) for o in others) / len(others)
            # sensitivity: base rate only over samples that never rejected w themselves (the main base is conservative)
            norej = [o for o in others if not any(bo.get("judge_label") == "abandon" and answers_equal(bo.get("rejected_value"), w)
                                                  for bo in bds_by_trace[(key[0], o["sample_idx"])])]
            base_norej = sum(answers_equal(o.get("final_answer"), w) for o in norej) / len(norej) if norej else None
            events[key[0]].append((relapse, base, base_norej))
            w_len = "w1" if len(normalize_answer(w)) < 2 else "w2plus"   # 1-digit values recur by chance
            for v, (fw, fb) in REASSERT_FIELDS.items():
                if bd.get(fw) is not None and bd.get(fb) is not None:
                    for name in (v, f"{v}_{w_len}"):
                        reassert[name][key[0]].append((float(bool(bd[fw])), float(bd[fb])))
            n_events += 1
            n_relapse += int(relapse)
            n_missing_final += int(t.get("final_answer") in (None, ""))

    excess = cluster_bootstrap({p: [r - b for r, b, _ in v] for p, v in events.items()}, n_boot=n_boot)
    relapse_rate = cluster_bootstrap({p: [r for r, _, _ in v] for p, v in events.items()}, n_boot=n_boot)
    base_rate = cluster_bootstrap({p: [b for _, b, _ in v] for p, v in events.items()}, n_boot=n_boot)
    excess_norej = cluster_bootstrap({p: [r - bn for r, _, bn in v if bn is not None] for p, v in events.items()}, n_boot=n_boot)

    # memory: abandoned attempt spans as a share of thinking tokens, and peak live FA cache with deletion at c_tok
    tot_think = tot_abandoned = 0
    shares, peak_full, peak_del = [], [], []
    for key, t in by_key.items():
        n = int(t.get("n_think_tokens") or 0)
        if n == 0:
            continue
        abandon = [bd for bd in bds_by_trace[key] if bd.get("judge_label") == "abandon" and _attempt_span(bd)]
        spans = _merge([_attempt_span(bd) for bd in abandon])
        freed = sum(b - a for a, b in spans)
        tot_think += n
        tot_abandoned += freed
        shares.append(freed / n)
        removed, peak = 0, 0
        for c, (a, b) in sorted((int(bd.get("c_tok") or bd["b_tok"]), _attempt_span(bd)) for bd in abandon):
            peak = max(peak, c - removed)
            removed = sum(y - x for x, y in _merge([_attempt_span(bd) for bd in abandon
                                                     if int(bd.get("c_tok") or bd["b_tok"]) <= c]))
        peak = max(peak, n - removed)
        peak_full.append(n)
        peak_del.append(peak)

    pooled_share = tot_abandoned / tot_think if tot_think else float("nan")

    # truncation vs abandoning, within problem (harder problems have both; compare samples of the same problem).
    # Per problem with >= 1 truncated and >= 1 finished sample: mean(truncated) - mean(finished), for abandons per
    # trace and abandons per 1K thinking tokens (truncated traces are longer). Association only, not causation.
    trunc_diff = defaultdict(lambda: defaultdict(list))
    for p, ts in by_problem.items():
        groups = {True: [], False: []}
        for t in ts:
            n_ab = sum(bd.get("judge_label") == "abandon" for bd in bds_by_trace[(p, t["sample_idx"])])
            n_tok = int(t.get("n_think_tokens") or 0)
            groups[bool(t.get("truncated"))].append((n_ab, 1000 * n_ab / n_tok if n_tok else 0.0))
        if groups[True] and groups[False]:
            for i, name in enumerate(("abandons_per_trace", "abandons_per_1k_tokens")):
                m = lambda g: sum(x[i] for x in g) / len(g)
                trunc_diff[name][p].append(m(groups[True]) - m(groups[False]))
    truncation = {"n_problems_mixed": len(trunc_diff["abandons_per_trace"])}
    for name, clusters in trunc_diff.items():
        est, lo, hi = cluster_bootstrap(clusters, n_boot=n_boot)
        truncation[f"{name}_truncated_minus_finished"] = {"diff": est, "ci": [lo, hi]}
    finished = [t for t in traces if not t.get("truncated")]
    truncation["accuracy_finished"] = sum(bool(t.get("correct")) for t in finished) / max(len(finished), 1)

    return {
        "n_traces": len(traces),
        "n_truncated": sum(bool(t.get("truncated")) for t in traces),
        "accuracy": sum(bool(t.get("correct")) for t in traces) / max(len(traces), 1),
        "boundary_labels": dict(labels),
        "boundary_labels_by_trigger": {k: dict(v) for k, v in sorted(labels_by_trigger.items())},
        "abandons_per_trace_mean": sum(abandons_per_trace) / max(len(abandons_per_trace), 1),
        "traces_with_abandon": sum(x > 0 for x in abandons_per_trace),
        "relapse": {
            "n_events": n_events, "n_relapse": n_relapse, "n_problems": len(events),
            "n_events_final_missing": n_missing_final,
            "n_abandons_non_numeric_w_skipped": n_non_numeric,
            "relapse_rate": relapse_rate, "base_rate": base_rate, "excess": excess,
            "excess_text": fmt(*excess),
            "excess_vs_never_rejected": excess_norej,
            "excess_vs_never_rejected_text": fmt(*excess_norej),
            "exclude_truncated": exclude_truncated,
            "in_trace": {**{v: _reassert_summary(ev, n_boot) for v, ev in sorted(reassert.items()) if ev},
                         "note": "plain: w asserted again at token >= c_tok vs other samples at the same offset (biased up by "
                                 "restating w while rejecting it); strict: from c_tok + 256, negated sentences skipped"},
        },
        "memory": {
            "abandoned_token_share_pooled": pooled_share,
            "abandoned_token_share_mean_per_trace": sum(shares) / max(len(shares), 1),
            "fa_bytes_freed_total": tot_abandoned * KV_BYTES_PER_TOKEN,
            "fa_gb_freed_per_64k_think_tokens": pooled_share * 65536 * KV_BYTES_PER_TOKEN / 1e9,
            "peak_fa_tokens_full_mean": sum(peak_full) / max(len(peak_full), 1),
            "peak_fa_tokens_with_deletion_mean": sum(peak_del) / max(len(peak_del), 1),
            "note": "think tokens only; prompt KV is identical in both and excluded",
        },
        "truncation_vs_abandon": truncation,
    }


# ---------- E2 ----------

def _problem_of(boundaries):
    return {bd["boundary_id"]: bd["problem_id"] for bd in boundaries}


E2_TARGET = "real_vs_restored"
E2_REFERENCES = ("placebo", "noise_floor")
# Paired contrasts (numerator, denominator): the H3 tests, plus which memory carries more of the attempt (FA vs GDN).
E2_PAIRS = tuple((E2_TARGET, ref) for ref in E2_REFERENCES) + (("full_vs_ours", E2_TARGET),)


def e2(rows, boundaries, n_boot=2000):
    prob = _problem_of(boundaries)
    groups = defaultdict(lambda: defaultdict(list))   # (condition, window) -> problem -> [kl]
    kl = defaultdict(dict)                            # (condition, window) -> boundary -> kl
    for r in rows:
        key = (r["condition"], str(r["window"]))
        groups[key][prob.get(r["boundary_id"], r["boundary_id"])].append(float(r["kl"]))
        kl[key][r["boundary_id"]] = float(r["kl"])
    out = {}
    for (cond, win), clusters in sorted(groups.items()):
        vals = sorted(x for v in clusters.values() for x in v)
        q = lambda f: vals[min(int(f * len(vals)), len(vals) - 1)]
        est, lo, hi = cluster_bootstrap(clusters, n_boot=n_boot)
        out[f"{cond}|{win}"] = {"n": len(vals), "mean": est, "ci": [lo, hi], "median": q(0.5), "iqr": [q(0.25), q(0.75)]}
    # real_vs_restored against each reference band, paired on boundaries that have both, clustered by problem.
    # KL is heavy-tailed, so the main number is the mean per-boundary log10 ratio (geometric-mean ratio = 10**mean).
    vs = {}
    for win in sorted({w for _, w in kl}):
        for num, ref in E2_PAIRS:
            tgt, other = kl.get((num, win), {}), kl.get((ref, win), {})
            shared = sorted(set(tgt) & set(other))
            if not shared:
                continue
            logr, frac = defaultdict(list), defaultdict(list)
            for b in shared:
                logr[prob.get(b, b)].append(math.log10(max(tgt[b], 1e-12) / max(other[b], 1e-12)))
                frac[prob.get(b, b)].append(float(tgt[b] > other[b]))
            est, lo, hi = cluster_bootstrap(logr, n_boot=n_boot)
            f_est, f_lo, f_hi = cluster_bootstrap(frac, n_boot=n_boot)
            vs[f"{num}/{ref}|{win}"] = {
                "n_boundaries": len(shared), "mean_log10_ratio": est, "ci": [lo, hi],
                "ratio": 10 ** est, "ratio_ci": [10 ** lo, 10 ** hi], "frac_above": f_est, "frac_above_ci": [f_lo, f_hi],
                "above": lo > 0,
                "text": f"{win}: {num} is {10 ** est:.2g}x {ref} (95% CI {10 ** lo:.2g}-{10 ** hi:.2g}x, "
                        f"n={len(shared)}); above in {f_est:.0%} of boundaries",
            }
    return {"conditions": out, "vs": vs,
            "note": "H3 needs real_vs_restored above placebo AND noise_floor (ci of the log ratio > 0); "
                    "above is necessary, not sufficient"}


# ---------- E3 ----------

METRICS = ("correct", "relapse", "finished", "correct_if_finished", "correct_w_correct", "repetition_same", "n_tokens",
           "backref_count")


def _metric(r, m):
    if m == "repetition_same":
        rep = r.get("repetition")
        return None if rep is None else float(str(rep).lower() == "same")
    if m == "correct_if_finished":                 # accuracy among continuations that ended with EOS within the cap
        return float(bool(r.get("correct"))) if r.get("finished") else None
    if m == "correct_w_correct":                   # filled in e3(): accuracy where the abandoned value was the right answer
        return None
    v = r.get(m)
    return None if v is None else float(v)


def e3(runs, boundaries, n_boot=2000, margin_pp=10.0):
    prob = _problem_of(boundaries)
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # metric -> cell -> boundary -> [values]
    w_wrong = {bd["boundary_id"]: bd.get("w_is_wrong") for bd in boundaries}
    for r in runs:
        ww = w_wrong.get(r["boundary_id"])
        for m in METRICS:
            v = _metric(r, m)
            if m == "relapse" and ww is False:       # returning to a correct abandoned value is not a relapse
                v = None
            if m == "correct_w_correct" and ww is False:
                v = float(bool(r.get("correct")))
            if v is not None:
                per[m][r["cell"]][r["boundary_id"]].append(v)
    mean = lambda xs: sum(xs) / len(xs)
    summary = {}
    for m in METRICS:
        summary[m] = {}
        for cell, by_b in per[m].items():
            clusters = defaultdict(list)
            for b, xs in by_b.items():
                clusters[prob.get(b, b)].append(mean(xs))
            est, lo, hi = cluster_bootstrap(clusters, n_boot=n_boot)
            summary[m][cell] = {"n_boundaries": len(by_b), "mean": est, "ci": [lo, hi]}
    contrasts = {}
    for m in METRICS:
        for x, y in CONTRASTS:
            bx, by = per[m].get(x, {}), per[m].get(y, {})
            shared = sorted(set(bx) & set(by))
            if not shared:
                continue
            clusters = defaultdict(list)
            for b in shared:                      # paired at the boundary level, clustered by problem
                clusters[prob.get(b, b)].append(mean(bx[b]) - mean(by[b]))
            est, lo, hi = cluster_bootstrap(clusters, n_boot=n_boot)
            contrasts[f"{m}:{x}-{y}"] = {"n_boundaries": len(shared), "diff": est, "ci": [lo, hi]}
    acc = contrasts.get("correct:ours-full")
    if acc:
        contrasts["noninferiority_text"] = noninferiority(acc["diff"] * 100, acc["ci"][0] * 100, acc["ci"][1] * 100, margin_pp)
    return {"cells": summary, "contrasts": contrasts}


# ---------- annotation agreement ----------

def agreement(boundaries, claude_rows, human_rows=None):
    judge = {bd["boundary_id"]: bd.get("judge_label") for bd in boundaries}
    claude = {r["boundary_id"]: r["label"] for r in claude_rows}
    shared = sorted(set(judge) & set(claude))
    out = {"n_shared": len(shared)}
    if shared:
        a, b = [judge[i] for i in shared], [claude[i] for i in shared]
        out["kappa_judge_claude"] = cohen_kappa(a, b)
        out["raw_agreement"] = sum(x == y for x, y in zip(a, b)) / len(a)
        ab = lambda xs: ["abandon" if x == "abandon" else "other" for x in xs]
        out["kappa_abandon_vs_other"] = cohen_kappa(ab(a), ab(b))
        # attempt-start agreement on boundaries both annotators call abandon (paragraph indices of the judge's input)
        jp = {bd["boundary_id"]: bd.get("attempt_start_paragraph") for bd in boundaries}
        cp = {r["boundary_id"]: r.get("attempt_start_paragraph") for r in claude_rows}
        both = [i for i in shared if judge[i] == claude[i] == "abandon" and jp.get(i) is not None and cp.get(i) is not None]
        if both:
            d = [abs(int(jp[i]) - int(cp[i])) for i in both]
            out["attempt_start"] = {"n": len(both), "exact": wilson_ci(sum(x == 0 for x in d), len(d)),
                                    "within_1": wilson_ci(sum(x <= 1 for x in d), len(d))}
    if human_rows:
        human = {r["boundary_id"]: r["label"] for r in human_rows}
        for name, ref in (("claude", claude), ("judge", judge)):
            ids = sorted(set(human) & set(ref))
            k = sum(human[i] == ref[i] for i in ids)
            out[f"human_vs_{name}"] = {"n": len(ids), "agree": k, "wilson": wilson_ci(k, len(ids))}
    out["note"] = "two LLM annotators share biases; human check is a small spot check, report its Wilson CI"
    return out


POSTHOC_KEEP = {"abandon": ("abandon",), "abandon_or_local_fix": ("abandon", "local_fix")}


def posthoc(e3_boundaries, labels, confirmer_rows=None):
    """Precision of the judge's 'abandon' on the E3 set, from blind labels that also see ~400 tokens of continuation.
    labels: {boundary_id, label: abandon|local_fix|no_change, value_dropped: yes|no|unknown, blind}; blind=false rows are dropped."""
    bd = {b["boundary_id"]: b for b in e3_boundaries}
    rows = [r for r in labels if r.get("blind", True) and r["boundary_id"] in bd]
    rates = lambda rs: {k: wilson_ci(sum(r["label"] in v for r in rs), len(rs)) for k, v in POSTHOC_KEEP.items()}
    out = {"n_labelled": len(labels), "n_blind": len(rows), "labels": dict(Counter(r["label"] for r in rows)), **rates(rows),
           "value_dropped": wilson_ci(sum(r.get("value_dropped") == "yes" for r in rows), len(rows)),
           "abandon_ids": [r["boundary_id"] for r in rows if r["label"] == "abandon"]}
    for key, of in (("trigger", trigger_class), ("w_is_wrong", lambda b: b.get("w_is_wrong"))):
        groups = defaultdict(list)
        for r in rows:
            groups[str(of(bd[r["boundary_id"]]))].append(r)
        out[f"by_{key}"] = {g: {"n": len(rs), **rates(rs)} for g, rs in sorted(groups.items())}
    if confirmer_rows:                                     # second-pass judge with continuation: agreement with these labels
        conf = {r["boundary_id"]: r["label"] for r in confirmer_rows}
        ids = [r["boundary_id"] for r in rows if r["boundary_id"] in conf]
        mine = {r["boundary_id"]: r["label"] for r in rows}
        pos = [i for i in ids if conf[i] == "abandon"]
        out["confirmer"] = {"n_shared": len(ids), "n_positive": len(pos),
                            "precision": wilson_ci(sum(mine[i] == "abandon" for i in pos), len(pos)),
                            "recall": wilson_ci(sum(conf[i] == "abandon" for i in ids if mine[i] == "abandon"),
                                                sum(mine[i] == "abandon" for i in ids)),
                            "kappa": cohen_kappa([conf[i] for i in ids], [mine[i] for i in ids]) if ids else None}
    return out


# ---------- self-test on synthetic data ----------

def selftest():
    assert answers_equal("\\boxed{\\frac{1}{2}}", "0.5") and answers_equal("1,000", "1000")
    assert answers_equal("x = 17", "17") and not answers_equal("17", "18") and not answers_equal(None, "1")
    for w in ("6^9", "3\\sqrt{2}", "\\frac{3}{4}", "(1,2)", "x=5", "2\\cdot 3"):
        assert is_numeric_value(w), w
    for w in ("local column constraints within blocks", "pos(v) = (r_v, c_v)", "x^2+1", "n=7k", "\\infty", "", None):
        assert not is_numeric_value(w), w
    p, lo, hi = wilson_ci(18, 20)
    assert abs(lo - 0.699) < 1e-3 and abs(hi - 0.972) < 1e-3
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert abs(cohen_kappa(list("aabb"), list("abab"))) < 1e-9
    ph = posthoc([{"boundary_id": i, "trigger_type": "weak", "w_is_wrong": True} for i in "abcd"],
                 [{"boundary_id": "a", "label": "abandon"}, {"boundary_id": "b", "label": "local_fix"},
                  {"boundary_id": "c", "label": "no_change"}, {"boundary_id": "d", "label": "abandon", "blind": False}],
                 [{"boundary_id": "a", "label": "abandon"}, {"boundary_id": "b", "label": "abandon"}])
    assert ph["n_blind"] == 3 and ph["abandon"][0] == 1 / 3 and ph["abandon_or_local_fix"][0] == 2 / 3, ph
    assert ph["confirmer"]["precision"][0] == 0.5 and ph["confirmer"]["recall"][0] == 1.0, ph

    rng = random.Random(1)
    traces, bds = [], []
    for p_id in range(60):                       # model relapses to rejected w=7 with prob 0.4, base rate 0.1
        ref = "3"
        for s in range(8):
            has_abandon = rng.random() < 0.5
            final = "7" if rng.random() < (0.4 if has_abandon else 0.1) else ref
            traces.append({"problem_id": p_id, "sample_idx": s, "reference": ref, "final_answer": final,
                           "correct": final == ref, "n_think_tokens": 10000, "truncated": False})
            if has_abandon:
                bds.append({"problem_id": p_id, "sample_idx": s, "boundary_id": f"{p_id}-{s}", "judge_label": "abandon",
                            "rejected_value": "7", "attempt_start_tok": 1000, "b_tok": 3000, "c_tok": 3050})
    r = e1(traces, bds, n_boot=500)
    ex, exn = r["relapse"]["excess"], r["relapse"]["excess_vs_never_rejected"]
    assert 0.08 < ex[0] < 0.22 and ex[1] > 0, ex       # truth 0.40 - (0.5*0.40 + 0.5*0.10) = 0.15
    assert 0.22 < exn[0] < 0.38 and exn[1] > 0.1, exn  # truth 0.40 - 0.10 = 0.30
    assert abs(r["memory"]["abandoned_token_share_pooled"] - 0.2 * r["traces_with_abandon"] / r["n_traces"]) < 1e-9
    text_bds = [{**bd, "boundary_id": bd["boundary_id"] + "t", "rejected_value": "pos(v) = (r_v, c_v)", "b_tok": 4000,
                 "attempt_start_tok": None} for bd in bds[:40]]
    tt = [{**t, "truncated": t["sample_idx"] < 2, "n_think_tokens": 32000 if t["sample_idx"] < 2 else 8000} for t in traces[:80]]
    tb = [{"problem_id": t["problem_id"], "sample_idx": t["sample_idx"], "boundary_id": f"x{i}", "judge_label": "abandon",
           "b_tok": 100 * i} for t in tt for i in range(4 if t["truncated"] else 1)]
    tv = e1(tt, tb, n_boot=200)["truncation_vs_abandon"]       # 10 problems: 4 vs 1 abandons, 0.125 vs 0.125 per 1K
    assert tv["n_problems_mixed"] == 10 and tv["abandons_per_trace_truncated_minus_finished"]["diff"] == 3, tv
    assert abs(tv["abandons_per_1k_tokens_truncated_minus_finished"]["diff"]) < 1e-12, tv
    rt = e1(traces, bds + text_bds, n_boot=200)["relapse"]
    assert rt["n_events"] == r["relapse"]["n_events"] and rt["n_abandons_non_numeric_w_skipped"] == 40, rt

    runs = []
    for bd in bds[:60]:
        for cell, acc in (("full", 0.5), ("ours", 0.6), ("textdel", 0.4), ("placebo", 0.45)):
            for k in range(4):
                c = rng.random() < acc
                runs.append({"boundary_id": bd["boundary_id"], "cell": cell, "sample_k": k, "correct": c,
                             "relapse": (not c) and rng.random() < 0.5, "finished": True, "n_tokens": 5000,
                             "repetition": "same" if rng.random() < 0.3 else "different", "backref_count": 0})
    r3 = e3(runs, bds, n_boot=500)
    assert "noninferiority_text" in r3["contrasts"]

    kl_rows = []                                 # real 3x placebo, placebo 30x noise floor, lognormal per-boundary noise
    for bd in bds[:60]:
        for cond, base in (("real_vs_restored", 3e-2), ("placebo", 1e-2), ("noise_floor", 3e-4)):
            if cond == "placebo" and rng.random() < 0.2:
                continue                         # placebo_e2 missing for some boundaries
            kl_rows.append({"boundary_id": bd["boundary_id"], "condition": cond, "window": "100-1k",
                            "kl": base * 10 ** rng.gauss(0, 0.3)})
    r2 = e2(kl_rows, bds, n_boot=500)
    vp, vn = r2["vs"]["real_vs_restored/placebo|100-1k"], r2["vs"]["real_vs_restored/noise_floor|100-1k"]
    assert 2 < vp["ratio"] < 4.5 and vp["above"] and vp["n_boundaries"] < 60, vp
    assert 50 < vn["ratio"] < 200 and vn["above"], vn
    print("selftest ok")
    print(" ", vp["text"])
    print(" excess relapse:", r["relapse"]["excess_text"], "| vs never-rejected:", r["relapse"]["excess_vs_never_rejected_text"],
          "| events", r["relapse"]["n_events"])
    print(" ", r3["contrasts"]["noninferiority_text"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["e1", "e2", "e3", "kappa", "posthoc", "selftest"])
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--exclude-truncated", action="store_true")
    ap.add_argument("--triggers", help="e1 sensitivity: keep only boundaries of these trigger classes, e.g. negation,restart")
    ap.add_argument("--margin-pp", type=float, default=10.0)
    ap.add_argument("--claude", default="annotations_claude.jsonl")
    ap.add_argument("--human", default="annotations_human.jsonl")
    ap.add_argument("--posthoc", default="e3_posthoc_claude.jsonl")
    ap.add_argument("--confirmer", help="posthoc: confirmer judge output {boundary_id, label} to score against the post-hoc labels")
    ap.add_argument("--out", help="write the result JSON here")
    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest()
    R = Path(args.results)
    bds = load_jsonl(R / "boundaries.jsonl") if args.cmd != "posthoc" else None   # posthoc reads the E3 set only
    if args.cmd == "e1":
        if args.triggers:                                  # judge precision differs by trigger class (annotation check)
            bds = [bd for bd in bds if trigger_class(bd) in args.triggers.split(",")]
        res = e1(load_jsonl(R / "traces.jsonl"), bds, args.n_boot, args.exclude_truncated)
        res["triggers"] = args.triggers or "all"
    elif args.cmd == "e2":
        res = e2(load_jsonl(R / "e2_kl.jsonl"), bds, args.n_boot)
    elif args.cmd == "e3":
        res = e3(load_jsonl(R / "e3_runs.jsonl"), bds, args.n_boot, args.margin_pp)
        if (R / "e3_boundaries.jsonl").exists():          # placebo coverage: placebo_e3 is skipped when no free span exists
            eb = load_jsonl(R / "e3_boundaries.jsonl")
            gaps = sorted(r["placebo_e3_gap"] for r in eb if r.get("placebo_e3") and r.get("placebo_e3_gap") is not None)
            res["placebo_coverage"] = {"n_boundaries": len(eb), "with_placebo_e3": sum(bool(r.get("placebo_e3")) for r in eb),
                                       "with_placebo_e2": sum(bool(r.get("placebo_e2")) for r in eb),
                                       "placebo_e3_gap_median": gaps[len(gaps) // 2] if gaps else None}
    elif args.cmd == "posthoc":
        conf = load_jsonl(R / args.confirmer) if args.confirmer else None
        res = posthoc(load_jsonl(R / "e3_boundaries.jsonl"), load_jsonl(R / args.posthoc), conf)
    else:
        human = R / args.human
        res = agreement(bds, load_jsonl(R / args.claude), load_jsonl(human) if human.exists() else None)
    text = json.dumps(res, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
