# Extra write-up figures for the simplified Fade write-up, in the style of fade_plots.py. Reads the JSON written by
# fade_stats.py / fade_sl.py; never recomputes statistics (outcome shares are the counts already in the JSON).
# Usage: python fade_paper_figs.py --stats results/stats_fade_k4.json --sl results/stats_fade_sl.json
#        [--out 28_10_submit/writeup/figures]
#   fig_fade_hybrid    (no data) Qwen3.5-9B's layer pattern and analytic memory per sample vs generated tokens
#   fig_fade_outcomes  (a) share of runs correct / finished wrong / hit the cap per cell;
#                      (b) exploratory: accuracy within bins of full attention's per-problem trace length
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tl_plots import GRID, INK, INK2, MUTED, SURFACE, _save, _style  # noqa: E402
from fade_plots import FADE, FULL, SL, TL  # noqa: E402

GDN_FILL, FA_FILL = "#d9d8d2", "#3d3c39"
KIB_PER_TOKEN, GDN_MIB = 32, 49.125  # 8 FA layers x 4 KV heads x 256 dims x K,V x 2 B; GDN state per sample


def plot_hybrid(out):
    _style()
    fig = plt.figure(figsize=(6.6, 2.35))
    # (a) the layer pattern
    ax = fig.add_axes([0.0, 0.02, 0.43, 0.96])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    bw, bh, x0 = 4.6, 1.55, 0.4
    ys = [6.9, 5.05, 3.2, 1.35]
    ax.add_patch(FancyBboxPatch((x0 - 0.25, 1.1), bw + 0.5, 7.6, boxstyle="round,pad=0,rounding_size=0.35",
                                fc="none", ec=MUTED, lw=0.9))
    for i, y in enumerate(ys):
        fa = i == 3
        ax.add_patch(FancyBboxPatch((x0, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.25",
                                    fc=FA_FILL if fa else GDN_FILL, ec="none"))
        ax.text(x0 + bw / 2, y + bh / 2, "Full attention" if fa else "Gated DeltaNet", ha="center", va="center",
                fontsize=8.5, color=SURFACE if fa else INK, fontweight="bold" if fa else None)
    ax.text(x0 + bw + 0.45, 4.9, "×8", ha="left", va="center", fontsize=12, color=INK2, fontweight="bold")
    ax.text(x0 + bw + 1.55, 5.95, "24 layers\nfixed-size state", ha="left", va="center", fontsize=8, color=INK2,
            linespacing=1.15)
    ax.text(x0 + bw + 1.55, 2.1, "8 layers\nKV cache grows\nwith every token", ha="left", va="center", fontsize=8,
            color=INK, linespacing=1.15)
    ax.text(0.15, 9.45, "(a) Qwen3.5-9B, a hybrid model", ha="left", va="center", fontsize=9, color=INK)
    # (b) memory per sample vs generated tokens
    bx = fig.add_axes([0.53, 0.2, 0.44, 0.7])
    n = [0, 32768]
    fa = [t * KIB_PER_TOKEN / 1024 for t in n]
    bx.fill_between(n, 0, [GDN_MIB] * 2, color=GDN_FILL, lw=0)
    bx.fill_between(n, [GDN_MIB] * 2, [GDN_MIB + f for f in fa], color=FA_FILL, alpha=0.85, lw=0)
    arrow = dict(arrowstyle="-", color=INK2, lw=0.7)
    bx.annotate("Gated DeltaNet state:\n49 MiB, fixed", (5200, 24), xytext=(700, 470), fontsize=7.5, color=INK2,
                linespacing=1.15, arrowprops=arrow)
    bx.annotate("attention KV cache:\n32 KiB per token", (19000, 660), xytext=(700, 880), fontsize=7.5, color=INK,
                linespacing=1.15, arrowprops=arrow)
    bx.text(32200, GDN_MIB + 1024 + 25, "1 GiB at 32K (95%)", ha="right", va="bottom", fontsize=7.5, color=INK,
            fontweight="bold")
    bx.set_xlim(0, 32768)
    bx.set_ylim(0, 1250)
    bx.set_xticks([0, 8192, 16384, 24576, 32768], ["0", "8K", "16K", "24K", "32K"])
    bx.set_yticks([0, 256, 512, 768, 1024])
    bx.set_xlabel("generated tokens", fontsize=8.5)
    bx.set_ylabel("MiB per sample", fontsize=8.5)
    bx.tick_params(labelsize=7.5)
    bx.grid(axis="x", visible=False)
    fig.text(0.47, 0.955, "(b) memory per sample", fontsize=9, color=INK, va="center")
    _save(fig, out, "fig_fade_hybrid")


def plot_outcomes(st, sl, out):
    _style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
    cells = [("Full attention", st["cells"]["full"]), ("Fade-4", st["cells"]["fade-kv4"]),
             ("Fade-2", st["cells"]["fade-kv2-had"]), ("8K window", st["cells"]["tl-w8k"]),
             ("2K window", st["cells"]["tl-w2k"]), ("StreamingLLM-8K*", sl["cells"]["sl-8k"])]
    cols = {"correct": "#2f8f5b", "wrong": "#c8473b", "cap": "#cfcdc4"}
    for i, (name, c) in enumerate(cells):
        right, fin = c["finished_correct"]
        n = c["n_runs"]
        parts = [(100 * right / n, "correct"), (100 * (fin - right) / n, "wrong"), (100 * (n - fin) / n, "cap")]
        left = 0
        y = len(cells) - 1 - i
        for v, k in parts:
            ax.barh(y, v, left=left, color=cols[k], height=0.62, edgecolor=SURFACE, lw=1)
            if v >= 9:
                ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color=SURFACE if k != "cap" else INK)
            left += v
    ax.set_yticks(range(len(cells)), [c[0] for c in cells][::-1], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of runs", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="y", visible=False)
    for k, lab, x in (("correct", "correct", 0), ("wrong", "finished, wrong", 30), ("cap", "hit 32K cap", 72)):
        ax.add_patch(plt.Rectangle((x, 6.05), 4, 0.32, color=cols[k], clip_on=False))
        ax.text(x + 5.5, 6.2, lab, fontsize=7.5, va="center", color=INK2)
    ax.set_ylim(-0.5, 6.5)
    ax.set_title("(a) how runs end", fontsize=9, color=INK, loc="left", pad=14)
    # (b) accuracy by full attention's trace length (exploratory)
    bins = [("0k-8k", "<8K"), ("8k-16k", "8–16K"), ("16k-24k", "16–24K"), ("24k+", "≥24K")]
    by = st["exploratory"]["by_full_length"]
    keys = {b: next(k.split(":")[2] for k in by if k.startswith("math:full:") and k.split(":")[2] == b)
            for b, _ in bins}
    series = [("full", "Full", FULL, "s"), ("fade-kv4", "Fade-4", FADE, "D"), ("fade-kv2-had", "Fade-2", FADE, "o"),
              ("tl-w8k", "8K window", TL, "o")]
    xs = range(len(bins))
    dodge = {"full": 0.0, "fade-kv4": 0.07, "fade-kv2-had": -0.07, "tl-w8k": 0.0}
    for cell, lab, color, m in series:
        ys = [100 * by[f"math:{cell}:{keys[b]}"]["accuracy"] for b, _ in bins]
        xd = [x + dodge[cell] for x in xs]
        ls = "--" if cell == "fade-kv4" else "-"
        bx.plot(xd, ys, ls, color=color, lw=1.6, marker=m, ms=5, mec=color if cell == "fade-kv4" else SURFACE, mew=1.2, label=lab,
                mfc=SURFACE if cell == "fade-kv4" else color)
    bx.set_xticks(list(xs), [f"{l}\n({by['math:full:' + keys[b]]['n_problems']})" for b, l in bins], fontsize=7.5)
    bx.set_xlabel("full attention's trace length (problems)", fontsize=8.5)
    bx.set_ylabel("accuracy (%)", fontsize=8.5)
    bx.set_ylim(0, 105)
    bx.tick_params(axis="y", labelsize=7.5)
    bx.legend(fontsize=7.5, loc="lower left", handlelength=1.8)
    bx.set_title("(b) accuracy by problem length (exploratory)", fontsize=9, color=INK, loc="left", pad=14)
    fig.tight_layout(w_pad=1.5)
    _save(fig, out, "fig_fade_outcomes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="results/stats_fade_k4.json")
    ap.add_argument("--sl", default="results/stats_fade_sl.json")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    out = Path(a.out)
    plot_hybrid(out)
    plot_outcomes(json.load(open(a.stats)), json.load(open(a.sl)), out)


if __name__ == "__main__":
    main()
