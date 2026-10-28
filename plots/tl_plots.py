# Think-Local figures (matplotlib). Reads the JSON written by tl_stats.py --out; never recomputes statistics.
# Usage: python tl_plots.py --stats results/stats_tl.json [--out results/figures]
#   fig_tl_tradeoff   math: accuracy and tokens vs Full (paired 95% CI) against per-sample memory, one point per cell
#   fig_tl_contrasts  the pre-registered contrasts (primary, S1-S5) as a forest plot with the 5 pp margin
#   fig_tl_matched    matched-memory voting: maj@m (math) and pass@m (Countdown) per cell vs Full pass@1
#   fig_tl_bylength   POST HOC: each cell's accuracy within bins of Full's per-problem mean length
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Light chart palette (dataviz reference palette, slots 1-3; Full is the reference and stays in ink).
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FAMILY = {"tl": ("Think-Local (TL-W)", "#2a78d6"), "dt": ("Delethink (DT)", "#eb6834"),
          "sl": ("StreamingLLM (SL)", "#1baf7a"), "full": ("Full", INK2)}
CELL_ORDER = ["full", "tl-w1k", "tl-w2k", "tl-w4k", "tl-w8k", "dt-c4k-m2k", "sl-2k"]
LABEL = {"full": "Full", "tl-w1k": "TL-W1K", "tl-w2k": "TL-W2K", "tl-w4k": "TL-W4K", "tl-w8k": "TL-W8K", "dt-c4k-m2k": "DT(4K/2K)", "sl-2k": "SL-2K"}
DODGE = 1.05  # multiplicative x offset on the log memory axis


def family(cell):
    return cell.split("-")[0]


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def _k_note(st):
    m = st["meta"]
    return "k = 8 for TL-W2K vs Full, k = 4 elsewhere" if m.get("k8_used") else "k = 4"


def _point(ax, x, y, lo, hi, color, vertical=True):
    if vertical:
        ax.plot([x, x], [lo, hi], color=color, lw=2, solid_capstyle="round", zorder=3)
    else:
        ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=3)
    ax.plot(x, y, "o", ms=7, color=color, mec=SURFACE, mew=2, zorder=4)


# ---------- trade-off: accuracy and tokens vs memory ----------

def _posthoc_note(st):
    return " TL-W8K was added post hoc (exploratory)." if any(":tl-w8k:" in k for k in st["cells"]) else ""


def plot_tradeoff(st, out, dom="math"):
    cells, ex = st["cells"], st["exploratory"]["contrasts"]
    present = [c for c in CELL_ORDER if f"{dom}:{c}:cap" in cells and (c == "full" or f"{dom}:{c}-full:cap" in ex)]
    if "full" not in present or len(present) < 2:
        print("tradeoff: need Full and at least one other cell; skipped")
        return
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    panels = (("correct", 100, "Accuracy − Full (pp)"), ("n_tokens", 1, "Mean tokens − Full"))
    for ax, (metric, scale, ylabel) in zip(axes, panels):
        ax.axhline(0, color=AXIS, lw=1, zorder=1)
        if metric == "correct":
            ax.axhline(-st["meta"]["margin_pp"], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.text(1.0, -st["meta"]["margin_pp"], f" −{st['meta']['margin_pp']:g} pp margin", transform=ax.get_yaxis_transform(),
                    fontsize=8, color=MUTED, va="center", ha="left")
        tl_xy, xs_all = [], []
        for c in present:
            # TL-W2K and SL-2K (and TL-W4K and DT) hold about the same memory: dodge TL left, the others right.
            x0 = cells[f"{dom}:{c}:cap"]["mean_sample_mib"]
            x = x0 / DODGE if family(c) == "tl" else x0 * DODGE if c != "full" else x0
            xs_all.append(x0)
            color = FAMILY[family(c)][1]
            if c == "full":
                _point(ax, x, 0, 0, 0, color)
                ax.annotate("Full", (x, 0), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
                continue
            r = ex[f"{dom}:{c}-full:cap"][metric]
            y, lo, hi = r["diff"] * scale, r["ci"][0] * scale, r["ci"][1] * scale
            _point(ax, x, y, lo, hi, color)
            if family(c) == "tl":
                tl_xy.append((x, y))
            # TL and SL labels under their CI, DT over its own, so no label crosses a bar (SL sits right under TL-W2K)
            below = family(c) in ("tl", "sl")
            ax.annotate(LABEL[c], (x, lo if below else hi), xytext=(0, -4 if below else 4), textcoords="offset points",
                        ha="center", va="top" if below else "bottom", fontsize=8, color=INK2)
        if len(tl_xy) > 1:
            xs, ys = zip(*sorted(tl_xy))
            ax.plot(xs, ys, color=FAMILY["tl"][1], lw=2, alpha=0.5, zorder=2, solid_capstyle="round")
        ax.set_xscale("log")
        ax.set_xticks([t for t in (50, 100, 200, 400, 800, 1600) if min(xs_all) / 1.5 <= t <= max(xs_all) * 1.5])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("FA + GDN memory per sample (MiB, analytic mean)")
        ax.set_ylabel(ylabel)
        ax.margins(x=0.2, y=0.12)
        if metric == "n_tokens":
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+,.0f}".replace("-", "−") if v else "0"))
    fams = []
    for c in present:
        if family(c) not in fams:
            fams.append(family(c))
    handles = [matplotlib.lines.Line2D([], [], marker="o", ls="", ms=7, color=FAMILY[f][1], mec=SURFACE, mew=2) for f in fams]
    fig.legend(handles, [FAMILY[f][0] for f in fams], loc="upper left", bbox_to_anchor=(0.0, 1.06), ncol=len(fams),
               fontsize=9, handletextpad=0.3, columnspacing=1.2, labelcolor=INK2)
    n = cells[f"{dom}:full:cap"]["n_problems"]
    fig.text(0.0, -0.04, f"Math, 32K cap, {n} problems, {_k_note(st)}. Paired by problem; 95% CI from a bootstrap stratified by dataset.{_posthoc_note(st)}",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, out, "fig_tl_tradeoff")


# ---------- forest plot of the pre-registered contrasts ----------

ROWS = [  # key in primary/secondary, label, compared with Full (the margin applies)
    ("primary", "Primary: TL-W2K − Full", True),
    ("S3:math:tl-w1k-full", "S3: TL-W1K − Full", True),
    ("S3:math:tl-w4k-full", "S3: TL-W4K − Full", True),
    ("X:math:tl-w8k-full:cap", "Post hoc: TL-W8K − Full", True),
    ("S1:math:tl-w4k-dt-c4k-m2k", "S1: TL-W4K − DT(4K/2K)", False),
    ("S2:math:tl-w2k-sl-2k", "S2: TL-W2K − SL-2K", False),
    ("S5:math:tl-w2k-full", "S5: TL-W2K − Full, 16K", True),
    ("S5:math:tl-w4k-dt-c4k-m2k", "S5: TL-W4K − DT, 16K", False),
    ("S4:countdown:tl-w2k-full", "S4: Countdown TL-W2K − Full", True),
]


def plot_contrasts(st, out):
    rows = []
    for key, label, vs_full in ROWS:
        r = (st.get("primary") if key == "primary" else st["exploratory"]["contrasts"].get(key[2:]) if key.startswith("X:")
             else st.get("secondary", {}).get(key))
        if r:
            rows.append((label, r, key == "primary"))
    if not rows:
        print("contrasts: nothing to plot; skipped")
        return
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.0))
    m = st["meta"]["margin_pp"]
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    ax.axvline(-m, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    for i, (label, r, primary) in enumerate(rows):
        y = len(rows) - 1 - i
        d, (lo, hi) = r["correct"]["diff"] * 100, (v * 100 for v in r["correct"]["ci"])
        _point(ax, d, y, lo, hi, FAMILY["tl"][1] if primary else INK2, vertical=False)
        ax.text(1.01, y, r.get("verdict", ""), transform=ax.get_yaxis_transform(), fontsize=8, color=MUTED, va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for lab, _, _ in reversed(rows)])
    for t, (_, _, primary) in zip(ax.get_yticklabels(), reversed(rows)):
        t.set_color(INK if primary else INK2)
        t.set_fontweight("bold" if primary else "normal")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Accuracy difference (pp), 95% CI")
    ax.text(-m, len(rows) - 0.45, f"−{m:g} pp margin\n(contrasts with Full)", fontsize=7.5, color=MUTED, ha="center", va="bottom")
    fig.text(0.0, -0.02, f"Math 32K cap unless noted; Countdown 100 tasks. {_k_note(st)}. Paired by problem, bootstrap stratified by dataset. "
             "Verdict only where pre-registered.", fontsize=7.5, color=MUTED, wrap=True)
    fig.tight_layout()
    _save(fig, out, "fig_tl_contrasts")


# ---------- matched-memory voting ----------

def plot_matched(st, out):
    mm = st["exploratory"].get("matched_memory", {})
    doms = [d for d in ("math", "countdown") if f"{d}:full" in mm]
    if not doms:
        print("matched: no matched-memory results; skipped")
        return
    _style()
    fig, axes = plt.subplots(1, len(doms), figsize=(4.3 * len(doms), 3.2), squeeze=False)
    for ax, dom in zip(axes[0], doms):
        cells = [c for c in CELL_ORDER if f"{dom}:{c}" in mm]
        base = mm[f"{dom}:full"]
        ax.axhspan(base["ci"][0] * 100, base["ci"][1] * 100, color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.axhline(base["mean"] * 100, color=INK2, lw=1, zorder=1)
        for i, c in enumerate(cells):
            r = mm[f"{dom}:{c}"]
            _point(ax, i, r["mean"] * 100, r["ci"][0] * 100, r["ci"][1] * 100, FAMILY[family(c)][1])
            ax.annotate(f"m={r['m']}", (i, r["ci"][1] * 100), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([LABEL[c] for c in cells], rotation=30, ha="right")
        ax.set_xlim(-0.6, len(cells) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        score = "majority vote over m" if base["score"] == "maj@m" else "pass@m (validator)"
        ax.set_title(f"{'Math' if dom == 'math' else 'Countdown'}: {score}", loc="left", fontsize=10, color=INK)
    fig.text(0.0, -0.06, f"m = how many samples fit in one Full sample's memory (analytic mean, capped at k). Line and band: Full pass@1, 95% CI.{_posthoc_note(st)}",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, out, "fig_tl_matched")


# ---------- post hoc: accuracy by Full's per-problem length ----------

TL_RAMP = {"tl-w1k": "#86b6ef", "tl-w2k": "#3987e5", "tl-w4k": "#1c5cab", "tl-w8k": "#0d366b"}  # ordinal blue, steps 250-700


def plot_bylength(st, out):
    bl = st["exploratory"].get("post_hoc_by_full_length") or {}
    doms = [d for d in ("math", "countdown") if any(k.startswith(f"{d}:full:") for k in bl)]
    if not doms:
        print("bylength: no post hoc length bins; skipped")
        return
    _style()
    fig, axes = plt.subplots(1, len(doms), figsize=(4.6 * len(doms), 3.4), squeeze=False, sharey=True)
    for ax, dom in zip(axes[0], doms):
        bins = sorted({(v["lo"], k.split(":")[2]) for k, v in bl.items() if k.startswith(f"{dom}:full:")})
        names = [n for _, n in bins]
        present = [c for c in CELL_ORDER if any(bl.get(f"{dom}:{c}:{n}") for n in names)]
        for j, c in enumerate(present):
            pts = [bl.get(f"{dom}:{c}:{n}") for n in names]
            color = TL_RAMP.get(c) or FAMILY[family(c)][1]
            dx = 0.06 * (j - (len(present) - 1) / 2)  # small dodge so CI whiskers don't overlap
            xs = [i + dx for i, v in enumerate(pts) if v]
            ys = [100 * v["accuracy"] for v in pts if v]
            for x, v in zip(xs, [v for v in pts if v]):
                ax.plot([x, x], [100 * v["accuracy_ci"][0], 100 * v["accuracy_ci"][1]], color=color, lw=1.2, alpha=0.6, zorder=2)
            ax.plot(xs, ys, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round", label=LABEL[c], zorder=3)
            ax.plot(xs, ys, "o", ms=7, color=color, mec=SURFACE, mew=2, zorder=4)
        ns = [bl[f"{dom}:full:{n}"]["n_problems"] for n in names]
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"{n.replace('k-', '–').replace('k', 'K')}\n(n={m})" for n, m in zip(names, ns)])
        ax.set_xlim(-0.3, len(names) - 0.7)
        ax.set_ylim(-5, 105)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_xlabel("Full's mean tokens on the problem")
        ax.set_title("Math" if dom == "math" else "Countdown", loc="left", fontsize=10, color=INK)
    axes[0][0].set_ylabel("Accuracy (32K cap)")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, labelcolor=INK2, handlelength=1.6)
    fig.text(0.0, -0.06, "POST HOC. Problems binned by Full's own mean length (samples 0-3); every cell is scored on the same problems. k = 4. Whiskers: 95% CI.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, out, "fig_tl_bylength")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, help="JSON written by tl_stats.py --out")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results" / "figures"))
    args = ap.parse_args()
    st = json.loads(Path(args.stats).read_text())
    out = Path(args.out)
    plot_tradeoff(st, out)
    plot_contrasts(st, out)
    plot_matched(st, out)
    plot_bylength(st, out)


if __name__ == "__main__":
    main()
