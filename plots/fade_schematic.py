# Write-up Figure 1 (no data): what each policy's attention keeps of one 32K-token trace, with analytic memory at 32K.
# Memory comes from fade_stats.sample_bytes (prompt left out, FA + GDN), the same numbers as the tables.
# Usage: python fade_schematic.py --out 28_10_submit/writeup/figures
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fade_stats as F  # noqa: E402
from tl_plots import GRID, INK, INK2, MUTED, SURFACE, _save, _style  # noqa: E402
from fade_plots import FADE  # noqa: E402

EXACT = "#3d3c39"
L, P_W = 32768, 2600  # P is drawn wider than a real prompt (median 116 tokens) so it stays visible
SINK_W = 500          # 4 sink tokens, drawn as a sliver


def _mib(cell):
    return F.sample_bytes(cell, L) / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    out = Path(ap.parse_args().out)
    _style()
    plt.rcParams["axes.grid"] = False
    fig, ax = plt.subplots(figsize=(6.4, 2.05))
    x0 = P_W
    # each row: list of (start, end, kind) in drawn units; kind in exact / low / drop
    rows = [
        ("Full attention", [(0, x0 + L, "exact")], f"{_mib('full'):,.0f} MiB"),
        ("StreamingLLM-8K", [(0, SINK_W, "exact"), (SINK_W, x0 + L - 8192, "drop"), (x0 + L - 8192, x0 + L, "exact")],
         f"{_mib('tl-w8k'):,.0f} MiB"),
        ("8K window + prompt", [(0, x0, "exact"), (x0, x0 + L - 8192, "drop"), (x0 + L - 8192, x0 + L, "exact")],
         f"{_mib('tl-w8k'):,.0f} MiB"),
        ("Fade (ours)", [(0, x0, "exact"), (x0, x0 + L - 2048, "low"), (x0 + L - 2048, x0 + L, "exact")],
         f"{_mib('fade-kv2-had'):,.0f} / {_mib('fade-kv4'):,.0f} MiB"),
    ]
    h = 0.62
    for i, (name, segs, mem) in enumerate(rows):
        y = len(rows) - 1 - i
        for a, b, kind in segs:
            if kind == "exact":
                r = Rectangle((a, y - h / 2), b - a, h, fc=EXACT, ec=SURFACE, lw=1.2)
            elif kind == "low":
                r = Rectangle((a, y - h / 2), b - a, h, fc="#f6c3ad", ec=FADE, lw=1.0, hatch="////")
            else:
                r = Rectangle((a, y - h / 2), b - a, h, fc=SURFACE, ec=MUTED, lw=0.8, ls=(0, (3, 2)))
            ax.add_patch(r)
        if name != "StreamingLLM-8K":
            ax.text(x0 / 2, y, "P", color=SURFACE, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(-600, y, name, ha="right", va="center", fontsize=9, color=INK, fontweight="bold" if "ours" in name else None)
        ax.text(x0 + L + 700, y, mem, ha="left", va="center", fontsize=8.5, color=INK2)
    ax.text(x0 + (L - 8192) / 2, 2, "dropped (prompt included)", ha="center", va="center", fontsize=8, color=MUTED)
    ax.text(x0 + (L - 8192) / 2, 1, "dropped", ha="center", va="center", fontsize=8, color=MUTED)
    ax.text(x0 + (L - 2048) / 2, 0, "older tokens at 2 or 4 bits", ha="center", va="center", fontsize=8, color=INK,
            bbox=dict(fc="#f6c3ad", ec="none", pad=1.2))
    for y, w in ((2, 8192), (1, 8192), (0, 2048)):  # size of the exact recent tail
        ax.text(x0 + L - w / 2, y, f"{w // 1024}K", color=SURFACE, ha="center", va="center", fontsize=8,
                fontweight="bold")
    # axis under the bars: generated-token positions
    for t, lab in ((0, "0"), (L - 8192, "24K"), (L, "32K")):
        ax.plot([x0 + t] * 2, [-0.52, -0.6], color=MUTED, lw=0.8)
        ax.text(x0 + t, -0.68, lab, ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.text(x0 + L + 700, len(rows) - 0.35, "memory at 32K", ha="left", va="bottom", fontsize=7.5, color=MUTED)
    ax.text(x0 + L / 2, -1.12, "generated tokens (P = prompt, not to scale)", ha="center", va="top", fontsize=7.5,
            color=MUTED)
    ax.set_xlim(-600, x0 + L + 9000)
    ax.set_ylim(-1.4, len(rows) - 0.3)
    ax.axis("off")
    fig.tight_layout()
    _save(fig, out, "fig_fade_schematic")


if __name__ == "__main__":
    main()
