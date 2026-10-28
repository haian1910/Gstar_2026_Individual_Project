# Fade figures (matplotlib), in Think-Local's style. Reads JSON written by fade_stats.py and h1_stats.py; never
# recomputes statistics.
# Usage: python fade_plots.py --stats results/stats_fade.json --kl results/h1_q35_math.json [--h1 results/h1.json]
#        [--pulse results/pulse_stats.json] [--sl results/stats_fade_sl.json] [--out results/figures] [--no-notes]
#   fig_fade_frontier  math accuracy vs memory per sample: forgetting (TL windows) vs fading (b-bit old tokens) vs Full,
#                      plus StreamingLLM-8K when --sl (fade_sl.py output) is complete
#   fig_fade_kl        teacher-forced KL vs effective bits per old token, the TL windows as references
#   fig_fade_h1_<task> hybrid vs dense (exploratory): KL per 1,000 characters vs bits, Qwen3.5-9B and Qwen3-8B
#   fig_fade_pulse     pulse test (exploratory): KL per token vs distance after a 1K-token 2-bit pulse, with the bf16 floor
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fade_stats as F  # noqa: E402
from tl_plots import AXIS, FAMILY, INK2, MUTED, SURFACE, _point, _save, _style  # noqa: E402

TL = FAMILY["tl"][1]
FADE = FAMILY["dt"][1]  # slot 2 of the reference palette; Delethink is not in these figures
FULL = FAMILY["full"][1]
SL = FAMILY["sl"][1]
TASK_NAME = {"countdown": "Countdown"}
NUDGE = {"tl-w8k": 1.03}  # display-only x nudge: TL-W8K (298 MiB) and 4-bit (286 MiB) would overlap
NOTES = True  # --no-notes drops the footnotes under each figure (the write-up puts them in LaTeX captions)


def _note(fig, *args, **kw):
    if NOTES:
        fig.text(*args, **kw)


def _log_mib_axis(ax, xs, label):
    ax.set_xscale("log")
    ax.set_xticks([t for t in (100, 200, 400, 800) if min(xs) / 1.5 <= t <= max(xs) * 1.5])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(label)


def plot_frontier(st, out, sl=None):
    cells, a, b = dict(st["cells"]), st["cell_a"], st["cell_b"]
    groups = [("tl", ["tl-w2k", "tl-w4k", "tl-w8k"], TL, {"tl-w2k": "W2K", "tl-w4k": "W4K", "tl-w8k": "W8K"}),
              ("fade", [a, b], FADE, {a: "2-bit", b: "4-bit"}), ("full", ["full"], FULL, {"full": "Full"})]
    with_sl = bool(sl) and sl.get("status") == "complete"
    if with_sl:  # SL-8K from fade_sl.py (k = 2); its CI is per cell, like every other bar
        cells["sl-8k"] = sl["cells"]["sl-8k"]
        groups.insert(1, ("sl", ["sl-8k"], SL, {"sl-8k": "SL-8K"}))
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    xs_all = []
    for name, cs, color, labels in groups:
        pts = []
        for c in cs:
            s = cells[c]
            x, y = s["mean_bytes"] / 2 ** 20 * NUDGE.get(c, 1), 100 * s["correct"]["mean"]
            lo, hi = (100 * v for v in s["correct"]["ci"])
            _point(ax, x, y, lo, hi, color)
            pts.append((x, y))
            xs_all.append(x)
            below = name in ("tl", "sl")
            ax.annotate(labels[c], (x, lo if below else hi), xytext=(0, -4 if below else 4), textcoords="offset points",
                        ha="center", va="top" if below else "bottom", fontsize=8.5, color=INK2)
        if len(pts) > 1:
            ax.plot(*zip(*sorted(pts)), color=color, lw=2, alpha=0.5, zorder=2, solid_capstyle="round")
    # the pre-registered primary: Cell A vs TL-W8K, paired by problem
    p = st["primary"]
    xa, ya = cells[a]["mean_bytes"] / 2 ** 20, 100 * cells[a]["correct"]["mean"]
    xw, yw = cells["tl-w8k"]["mean_bytes"] / 2 ** 20 * NUDGE["tl-w8k"], 100 * cells["tl-w8k"]["correct"]["mean"]
    ax.annotate("", xy=(xa * 1.02, ya), xytext=(xw * 0.98, yw),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1, shrinkA=6, shrinkB=6))
    ax.text(xw * 1.12, yw - 2,
            f"primary: {p['diff_pp']:+.1f} pp\n[{p['ci_pp'][0]:+.1f}, {p['ci_pp'][1]:+.1f}]\n{p['memory_ratio_vs_tl_w8k']:.2f}× memory",
            fontsize=8, color=INK2, ha="left", va="top")
    _log_mib_axis(ax, xs_all, "FA + GDN memory per sample (MiB, analytic mean)")
    ax.set_ylabel("Math accuracy (%)")
    ax.set_ylim(0, 100)
    ax.margins(x=0.15)
    colors = (TL, SL, FADE, FULL) if with_sl else (TL, FADE, FULL)
    names = ["Forget: prompt + last W tokens (Think-Local)"] + (["StreamingLLM: 4 sinks + last 8K"] if with_sl else []) + \
        ["Fade: keep all old tokens at b bits", "Full"]
    handles = [matplotlib.lines.Line2D([], [], marker="o", ls="", ms=7, color=c, mec=SURFACE, mew=2) for c in colors]
    fig.legend(handles, names,
               loc="upper left", bbox_to_anchor=(0.0, 1.08), ncol=2, fontsize=8.5, handletextpad=0.3, columnspacing=1.2,
               labelcolor=INK2)
    n = cells["full"]["n_problems"]
    kf = cells[a]["k"][0]
    ks = (f"k = {kf} for Fade, 4 for Full and TL" if kf != 4 else "k = 4 per cell") + (" (SL-8K: 2)" if with_sl else "")
    mb, mw = cells[b]["mean_bytes"] / 2 ** 20, cells["tl-w8k"]["mean_bytes"] / 2 ** 20
    keep = ("Prompt KV is left out of every cell's memory; Full, TL and Fade also hold it, SL-8K evicts it, so on true memory\n"
            "the other cells sit right of SL-8K by the prompt (median 116 tokens, 3.6 MiB)." if with_sl else
            "Prompt and last ~2K generated tokens exact in every cell.")
    _note(fig, 0.0, -0.05 if not with_sl else -0.09, f"Qwen3.5-9B, {n} math problems, 32K cap; {ks}. Bars: each cell's 95% CI\n"
             "(bootstrap stratified by dataset); the primary's CI is paired by problem (2-bit vs W8K). TL-W8K is nudged right by 3%\n"
             f"so it does not hide 4-bit ({mb:.0f} vs {mw:.0f} MiB). {keep}",
             fontsize=7.5, color=MUTED)
    fig.tight_layout()
    _save(fig, out, "fig_fade_frontier")


def plot_kl(kl, out):
    cells = kl["math"]["cells"]["q35"]
    plain = [(F.old_bits(f"fade-kv{b}"), cells[f"kv{b}"]["kl_tok"]) for b in (1, 2, 3, 4)]
    had = [(F.old_bits(f"fade-kv{b}-had"), cells[f"kv{b}_had"]["kl_tok"]) for b in (1, 2)]
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for name, color, W in (("w2k", TL, "TL-W2K"), ("w8k", TL, "TL-W8K")):
        y = cells[name]["kl_tok"]
        ax.axhline(y, color=color, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(1.0, y, f" {W} ({y:.4f}; {F.sample_bytes('tl-' + name, 32768) / 2 ** 20:.0f} MiB)", transform=ax.get_yaxis_transform(),
                fontsize=8, color=INK2, va="center", ha="left")
    ax.plot(*zip(*plain), color=FADE, lw=2, zorder=3, solid_capstyle="round")
    for x, y in had:
        ax.plot(x, y, "D", ms=6, mfc=SURFACE, mec=FADE, mew=1.6, zorder=4)
    for (x, y), (_, yh) in zip(plain, had + [(None, None)] * 2):
        ax.plot(x, y, "o", ms=7, color=FADE, mec=SURFACE, mew=2, zorder=5)
        txt = f"{y:.4f}" if y < 1 else f"{y:.1f}"
        if yh is not None:
            txt += f" (Hadamard {yh:.4f})" if yh < 1 else f" (Hadamard {yh:.1f})"
        dx, dy, va = {0: (10, 0, "center"), 1: (8, -10, "top")}.get(plain.index((x, y)), (6, 7, "bottom"))
        ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK2, va=va)
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 20)
    ax.set_xticks([x for x, _ in plain])
    ax.set_xticklabels([f"{x:g}\n{F.sample_bytes(f'fade-kv{b}', 32768) / 2 ** 20:.0f} MiB" for (x, _), b in zip(plain, (1, 2, 3, 4))])
    ax.set_xlabel("Effective bits per old token (memory per sample at 32K)")
    ax.set_ylabel("KL(Full ‖ policy) per token")
    ax.margins(x=0.12)
    handles = [matplotlib.lines.Line2D([], [], marker="o", color=FADE, lw=2, ms=7, mec=SURFACE, mew=2),
               matplotlib.lines.Line2D([], [], marker="D", ls="", ms=6, mfc=SURFACE, mec=FADE, mew=1.6),
               matplotlib.lines.Line2D([], [], color=TL, lw=1.2, ls=(0, (4, 3)))]
    fig.legend(handles, ["Fade, plain", "Fade, Hadamard", "Forget (window)"], loc="upper left", bbox_to_anchor=(0.0, 1.07),
               ncol=3, fontsize=8.5, handletextpad=0.6, handlelength=2.4, columnspacing=1.2, labelcolor=INK2)
    n = cells["kv2"]["n_traces"]
    _note(fig, 0.0, -0.06, f"Teacher-forced, exploratory: {n} held-out MATH-500 L5 traces, scored from generated token 2,048 to ≤ 24,576.\n"
             "Full attention has KL 0 (1,073 MiB at 32K). Memory is analytic, prompt left out.", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    _save(fig, out, "fig_fade_kl")


def plot_h1(h1, out, task="math"):
    """Hybrid vs dense: KL per 1,000 characters (tokenizers differ) against effective bits, one line per model."""
    cells = h1[task]["cells"]
    if "q3" not in cells:
        print(f"h1 {task}: no dense rows; skipped")
        return
    models = (("q35", "Qwen3.5-9B (8 attention layers quantized)", FADE, "o"), ("q3", "Qwen3-8B (all 36 layers)", INK2, "s"))
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for m, label, color, mk in models:
        pts = [(b + 0.25, cells[m][f"kv{b}"]["kl_kchar"]) for b in (1, 2, 3, 4) if f"kv{b}" in cells[m]]
        ax.plot(*zip(*pts), color=color, lw=2, marker=mk, ms=7, mec=SURFACE, mew=2, label=label, zorder=3)
    if "kv2_every4" in cells["q3"]:
        # larger and behind the lines, so a Qwen3.5 point at nearly the same value stays visible inside it
        ax.plot(2.25, cells["q3"]["kv2_every4"]["kl_kchar"], "s", ms=12, mfc="none", mec=INK2, mew=1.6, zorder=2,
                label="Qwen3-8B, every 4th layer (9 of 36)")
    ax.set_yscale("log")
    ax.set_xticks([1.25, 2.25, 3.25, 4.25])
    ax.set_xlabel("Effective bits per old token")
    ax.set_ylabel("KL(Full ‖ policy) per 1,000 characters")
    ax.legend(loc="upper right", fontsize=8.5, labelcolor=INK2)
    n = cells["q35"]["kv2"]["n_traces"]
    no3 = [name for m, name in (("q35", "Qwen3.5-9B"), ("q3", "Qwen3-8B")) if "kv3" not in cells[m]]
    no3 = {0: "", 1: f" {no3[0]} has no 3-bit run.", 2: " Neither model has a 3-bit run."}[len(no3)]
    _note(fig, 0.0, -0.06, f"Teacher-forced, exploratory: {n} {TASK_NAME.get(task, task)} texts generated by Qwen3.5-9B, re-tokenized for Qwen3-8B (off-policy for it).\n"
             f"KL per character because the vocabularies differ (248K vs 152K tokens).{no3}", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    _save(fig, out, f"fig_fade_h1_{task}")


def plot_pulse(ps, out, name="fig_fade_pulse"):
    """KL(E || X)/token against distance after the pulse, one panel per model, each with its own P0 floor."""
    SL = FAMILY["sl"][1]
    panels = (("hybrid", "Qwen3.5-9B (hybrid)", (("P", "pulse", FADE, "-"), ("Pc", "error kept in the attention cache only", TL, "-"),
                                                  ("Ps", "error kept in the recurrent state only", SL, "-"))),
              ("dense", "Qwen3-8B (dense)", (("P", "pulse, all 36 layers", FADE, "-"), ("P_every4", "pulse, every 4th layer", FADE, (0, (3, 2))))))
    panels = [p for p in panels if p[0] in ps]
    _style()
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 if len(panels) == 1 else 9.4, 3.6), sharey=True, squeeze=False)
    for ax, (m, title, series) in zip(axes[0], panels):
        s = ps[m]
        b = s["curve"]["P"]
        x = [32 + 64 * i for i in range(len(b))]  # bin centres, tokens after the pulse
        for cfg, label, color, ls in series:
            ax.plot(x, s["curve"][cfg], color=color, lw=1.6, ls=ls, label=label, zorder=3)
        floor = s["pooled"]["P0"]["1024-4096"]["kl_tok"]
        ax.plot(x, s["curve"]["P0"], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="floor: same path, no quantization (P0)", zorder=2)
        ax.hlines(2 * floor, 1024, x[-1], color=MUTED, lw=1.4, ls=(0, (1, 2)), zorder=2, label="2× floor (reading threshold)")
        ax.axvline(1024, color=AXIS, lw=0.8, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks([2e-4, 5e-4, 1e-3, 2e-3, 5e-3])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([64, 256, 1024, 4096])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Tokens after the pulse ended")
        ax.set_title(f"{title}: during the pulse {s['pooled']['P']['pulse']['kl_tok']:.4f}", fontsize=9, color=INK2, loc="left")
        ax.legend(loc="upper right", fontsize=7.5, labelcolor=INK2)
    axes[0][0].set_ylabel("KL(exact ‖ X) per token")
    n = ps["hybrid"]["n_traces"]
    probs = ", ".join(f"{p} ×{k}" for p, k in ps["hybrid"]["problems"].items())
    _note(fig, 0.0, -0.17, f"Teacher-forced, exploratory: {n} traces ({probs}). Tokens [8,192, 9,216) read old tokens at 2 bits; all other tokens read them exactly.\n"
             "64-token bins. Each model's floor is the same computation with a no-op quantizer (bf16 rounding of the pulse path). "
             "Reading rule at ≥ 1,024 tokens:\nboth single-carrier branches within 2× of the floor → no persistent error detectable. "
             "Per-token values are in each model's own tokens.\nQwen3-8B's floor sends all 36 layers through the pulse path, "
             "so it is an upper bound on the every-4th-layer floor. Restoring both carriers gives exactly 0 (gate).",
             fontsize=7.5, color=MUTED)
    fig.tight_layout()
    _save(fig, out, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--kl", required=True)
    ap.add_argument("--h1", help="h1_stats.py output with both models (optional)")
    ap.add_argument("--pulse", help="pulse_stats.py output (optional)")
    ap.add_argument("--sl", help="fade_sl.py output (optional; drawn only if complete)")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--no-notes", action="store_true", help="omit the footnote under each figure")
    args = ap.parse_args()
    global NOTES
    NOTES = not args.no_notes
    out = Path(args.out)
    plot_frontier(json.loads(Path(args.stats).read_text()), out, json.loads(Path(args.sl).read_text()) if args.sl else None)
    plot_kl(json.loads(Path(args.kl).read_text()), out)
    if args.h1:
        h1 = json.loads(Path(args.h1).read_text())
        for task in h1:
            plot_h1(h1, out, task)
    if args.pulse:
        plot_pulse(json.loads(Path(args.pulse).read_text()), out)


if __name__ == "__main__":
    main()
