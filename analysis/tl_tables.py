# Think-Local write-up tables (markdown) from the JSON written by tl_stats.py --out; never recomputes statistics.
# Usage: python tl_tables.py --stats results/stats_tl.json   -> Table 1 (cells), Table 2 (contrasts), memory, moderator on stdout
from __future__ import annotations

import argparse
import json
from pathlib import Path

CELL_ORDER = ["full", "tl-w1k", "tl-w2k", "tl-w4k", "tl-w8k", "dt-c4k-m2k", "sl-2k"]
LABEL = {"full": "Full", "tl-w1k": "TL-W1K", "tl-w2k": "TL-W2K", "tl-w4k": "TL-W4K", "tl-w8k": "TL-W8K", "dt-c4k-m2k": "DT(4K/2K)", "sl-2k": "SL-2K"}
ROWS = [  # key in primary/secondary, label
    ("primary", "Primary: TL-W2K − Full"),
    ("S1:math:tl-w4k-dt-c4k-m2k", "S1: TL-W4K − DT(4K/2K)"),
    ("S2:math:tl-w2k-sl-2k", "S2: TL-W2K − SL-2K"),
    ("S3:math:tl-w1k-full", "S3: TL-W1K − Full"),
    ("S3:math:tl-w4k-full", "S3: TL-W4K − Full"),
    ("X:math:tl-w8k-full:cap", "Post hoc: TL-W8K − Full"),
    ("S4:countdown:tl-w2k-full", "S4: Countdown TL-W2K − Full"),
    ("S5:math:tl-w2k-full", "S5: TL-W2K − Full, 16K view"),
    ("S5:math:tl-w4k-dt-c4k-m2k", "S5: TL-W4K − DT, 16K view"),
]


def lookup(st, key):
    """primary, a secondary key, or X:<key> for an exploratory contrast (post hoc rows carry no verdict)."""
    if key == "primary":
        return st.get("primary")
    if key.startswith("X:"):
        return st["exploratory"]["contrasts"].get(key[2:])
    return st.get("secondary", {}).get(key)


def pct(x):
    return f"{100 * x:.1f}%"


def pp(r, scale=100, unit=" pp"):
    d, (lo, hi) = r["diff"] * scale, (v * scale for v in r["ci"])
    f = (lambda v: f"{v:+.1f}") if scale == 100 else (lambda v: f"{v:+,.0f}")
    return f"{f(d)}{unit} ({f(lo)} to {f(hi)})".replace("-", "−")


def table_cells(st, dom):
    c = st["cells"]
    cells = [x for x in CELL_ORDER if f"{dom}:{x}:cap" in c]
    if not cells:
        return ""
    has16 = any(f"{dom}:{x}:16k" in c for x in cells)
    head = "| Cell | Accuracy (95% CI) | Finished | Mean tokens | Repetitive | Re-encoded tokens | Memory per sample (MiB) | Runs beyond window |"
    lines = [head, "|---" * (head.count("|") - 1) + "|"]
    for x in cells:
        a = c[f"{dom}:{x}:cap"]
        b = c.get(f"{dom}:{x}:16k")
        acc = f"{pct(a['accuracy'])} ({100 * a['accuracy_ci'][0]:.1f}–{100 * a['accuracy_ci'][1]:.1f})"
        if b:
            acc += f" [{pct(b['accuracy'])}]"
        fin = pct(a["finished"]) + (f" [{pct(b['finished'])}]" if b else "")
        rep = pct(a["repetitive"]) if a.get("repetitive") is not None else "n/a"
        bw = "—" if x == "full" else pct(a["beyond_window"])
        lines.append(f"| {LABEL[x]} | {acc} | {fin} | {a['mean_tokens']:,.0f} | {rep} | {a['mean_reencoded']:,.0f} | {a['mean_sample_mib']:.0f} | {bw} |")
    n = c[f"{dom}:{cells[0]}:cap"]["n_problems"]
    note = f"{'Math' if dom == 'math' else 'Countdown'}, {n} problems" + (", 16K view in brackets" if has16 else "")
    return f"**Table 1{'a' if dom == 'math' else 'b'}. Cells** ({note}; k = 4 for every cell).\n\n" + "\n".join(lines)


def table_contrasts(st):
    lines = ["| Contrast | Accuracy (pp) | Mean tokens | Tokens, finished runs only | Finished (pp) | Verdict |", "|---|---|---|---|---|---|"]
    for key, label in ROWS:
        r = lookup(st, key)
        if not r:
            continue
        fo = r["n_tokens_finished_only"]
        lines.append(f"| {label} | {pp(r['correct'])} | {pp(r['n_tokens'], 1, '')} | {pp(fo, 1, '')}, n = {fo['n_problems']} | "
                     f"{pp(r['finished'])} | {r.get('verdict', '—')} |")
    k = "k = 8 for the TL-W2K − Full rows, k = 4 elsewhere" if st["meta"].get("k8_used") else "k = 4"
    return (f"**Table 2. Contrasts** (paired by problem; 95% CI from a bootstrap stratified by dataset; {k}). Verdicts follow the pre-registered "
            "wording rule for the 5 pp margin, which applies to the primary, S4 and the primary's 16K view; S1–S3 are two-sided CIs without a margin, "
            "and post hoc rows carry no verdict.\n\n" + "\n".join(lines))


def per_dataset(st, cell="tl-w2k"):
    ex = st["exploratory"]["contrasts"]
    rows = [(k.split(":")[-1], v) for k, v in sorted(ex.items()) if k.startswith(f"math:{cell}-full:cap:")]
    if not rows:
        return ""
    lines = ["| Dataset | Problems | Accuracy (pp) | Mean tokens |", "|---|---|---|---|"]
    for ds, v in rows:
        lines.append(f"| {ds} | {v['n_problems']} | {pp(v['correct'])} | {pp(v['n_tokens'], 1, '')} |")
    return f"**Per dataset, {LABEL[cell]} − Full** (exploratory, k = 4).\n\n" + "\n".join(lines)


def matched(st):
    mm = st["exploratory"].get("matched_memory", {})
    lines = ["| Cell | Domain | Memory ratio vs Full | m | Score (95% CI) |", "|---|---|---|---|---|"]
    for dom in ("math", "countdown"):
        for x in CELL_ORDER:
            r = mm.get(f"{dom}:{x}")
            if r:
                lines.append(f"| {LABEL[x]} | {dom} | {r['memory_ratio']:.1f}× | {r['m']} | {'pass@1' if r['m'] == 1 else r['score'][:-1] + str(r['m'])} "
                             f"{pct(r['mean'])} ({100 * r['ci'][0]:.1f}–{100 * r['ci'][1]:.1f}) |")
    return "**Matched memory** (exploratory; Full row = pass@1).\n\n" + "\n".join(lines) if len(lines) > 2 else ""


def moderator(st):
    md = st["exploratory"].get("moderator") or {}
    out = []
    for dom, r in md.items():
        out.append(f"- {dom}: Spearman(trigger density, TL − Full) = {r['rho_density_vs_diff']:+.2f} ({r['ci'][0]:+.2f} to {r['ci'][1]:+.2f}); "
                   f"Spearman(density, Full accuracy) = {r['rho_density_vs_full_acc']:+.2f} ({r['ci_full_acc'][0]:+.2f} to {r['ci_full_acc'][1]:+.2f}); "
                   f"n = {r['n_problems']} problems, cross-fitted.")
    return "**Moderator** (exploratory).\n" + "\n".join(out) if out else ""


def how_runs_end(st):
    c = st["cells"]
    lines = ["| Cell | Domain | Finished | Correct if finished | Closed `</think>` | Finished length p10 / p50 / p90 / max |", "|---|---|---|---|---|---|"]
    for dom in ("math", "countdown"):
        for x in CELL_ORDER:
            a = c.get(f"{dom}:{x}:cap")
            h = (a or {}).get("post_hoc")
            if not h:
                continue
            fl = h["finished_len"]
            ln = " / ".join(f"{v:,}" for v in (fl["p10"], fl["p50"], fl["p90"], fl["max"])) if fl["max"] else "—"
            cif = f"{pct(h['correct_if_finished'])} ({h['n_finished'] - h['n_finished_wrong']}/{h['n_finished']})" if h["n_finished"] else "—"
            ct = pct(h["closed_think"]) if h["closed_think"] is not None else "n/a"
            lines.append(f"| {LABEL[x]} | {dom} | {pct(a['finished'])} | {cif} | {ct} | {ln} |")
    return ("**How runs end** (post hoc, exploratory; run-level, 32K cap).\n\n" + "\n".join(lines)) if len(lines) > 2 else ""


def restart_check(st):
    rs = st["exploratory"].get("restart_sensitivity")
    if not rs:
        return ""
    names = {"as_generated": "As generated (pre-registered)", "a_unfinished_wrong": "(a) restarted rows unfinished and wrong",
             "b_left_out": "(b) restarted rows left out"}
    lines = ["| Variant | S2: TL-W2K − SL-2K (pp) | SL-2K math accuracy |", "|---|---|---|"]
    for k, lab in names.items():
        v = rs[k]
        s2 = v.get("S2:math:tl-w2k-sl-2k")
        acc = v.get("math:sl-2k:accuracy")
        lines.append(f"| {lab} | {pp(s2['correct']) if s2 else '—'} | "
                     f"{pct(acc['est']) + f' ({100 * acc['ci'][0]:.1f}–{100 * acc['ci'][1]:.1f})' if acc else '—'} |")
    return (f"**Restart check** (pre-specified 10:47 UTC; {rs['n_restarted']} rows redrawn after a mid-generation stop: "
            + ", ".join(f"{LABEL.get(c, c)} {n}" for c, n in rs["by_cell"].items()) + ").\n\n" + "\n".join(lines))


def by_length(st):
    bl = st["exploratory"].get("post_hoc_by_full_length") or {}
    out = []
    for dom in ("math", "countdown"):
        bins = sorted({(v["lo"], k.split(":")[2]) for k, v in bl.items() if k.startswith(f"{dom}:full:")})
        cells = [x for x in CELL_ORDER if any(f"{dom}:{x}:{n}" in bl for _, n in bins)]
        if not bins:
            continue
        names = [n for _, n in bins]
        head = "| Cell | " + " | ".join(f"{n.replace('k-', '–').replace('k', 'K')} (n = {bl[f'{dom}:full:{n}']['n_problems']})" for n in names) + " |"
        lines = [head, "|---" * (head.count("|") - 1) + "|"]
        for x in cells:
            lines.append(f"| {LABEL[x]} | " + " | ".join(pct(bl[f"{dom}:{x}:{n}"]["accuracy"]) if f"{dom}:{x}:{n}" in bl else "—" for n in names) + " |")
        out.append(f"*{dom}*\n\n" + "\n".join(lines))
    return ("**Accuracy by Full's per-problem mean length** (post hoc, exploratory; bins from Full's samples 0–3; k = 4).\n\n"
            + "\n\n".join(out)) if out else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    st = json.loads(Path(ap.parse_args().stats).read_text())
    m = st["meta"]
    print(f"<!-- {m['n_runs']} runs; k8_used={m.get('k8_used')}; n_boot={m['n_boot']}; margin {m['margin_pp']} pp -->\n")
    if st.get("primary"):
        print(f"**Primary.** {st['primary']['text']}\n")
    for part in (table_cells(st, "math"), table_cells(st, "countdown"), table_contrasts(st), per_dataset(st), matched(st), moderator(st), how_runs_end(st), by_length(st), restart_check(st)):
        if part:
            print(part + "\n")


if __name__ == "__main__":
    main()
