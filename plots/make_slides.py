#!/usr/bin/env python3
# Builds haian's 3 pages of the GStar master deck (title + 2 content slides) from the template pages 92-94, exported
# read-only from the shared Google Slides deck (slides_build/template_vuhaian.pptx), and a PNG preview of each slide drawn
# with matplotlib and the deck's font (Lexend), because LibreOffice is not available here.
# One spec per slide feeds both outputs, so the preview shows the same boxes, text and sizes as the .pptx.
# Numbers come from results/stats_fade_k4.json (registered k = 4 rerun, audited); the StreamingLLM-8K row and figure point
# are filled from results/stats_fade_sl.json only when its status is "complete" (fade_sl.py), else shown as pending.
# Usage: make_slides.py --out 28_10_submit/fade_slides.pptx --preview slides_build/preview
import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR  # noqa: E402
from pptx.enum.dml import MSO_LINE  # noqa: E402
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE  # noqa: E402
from pptx.util import Inches, Pt, Emu  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "slides" / "build"  # course template, backgrounds and Lexend fonts; not in the repo
FONTS = {("Lexend", False): "Lexend-Regular.ttf", ("Lexend", True): "Lexend-Bold.ttf",
         ("Lexend SemiBold", False): "Lexend-SemiBold.ttf", ("Lexend SemiBold", True): "Lexend-SemiBold.ttf"}
for f in set(FONTS.values()):
    fm.fontManager.addfont(str(BUILD / "fonts" / f))
LINE = 1.25 * 1.05            # Lexend ascender - descender = 1.25 em; the template's paragraphs use 105% spacing
INSET = 0.04                  # text box inset, inches

RED, RED_T, INK, MUTED, GRID = "78141B", "731114", "1A1A1A", "5F5F5F", "D9D9D9"
PINK, PINK_INK, DROP, DROP_LINE, SL_BLUE = "EBC3C6", "78141B", "F2F2F2", "B3B3B3", "2F5D8A"
PINK_BG, LAYER, GDN_GRAY = "F8E8E9", "E3E3E3", "C9C9C9"
MIB = 2 ** 20


# ---------------------------------------------------------------- spec helpers
def run(text, size=12, bold=False, color=INK, italic=False, semi=False):
    return {"text": text, "size": size, "bold": bold, "color": color, "italic": italic,
            "font": "Lexend SemiBold" if semi else "Lexend"}


def para(*runs, align="l", after=0):
    return {"runs": list(runs), "align": align, "after": after}


def T(x, y, w, h, *paras, anchor="t", name="text"):
    return {"k": "text", "x": x, "y": y, "w": w, "h": h, "paras": list(paras), "anchor": anchor, "name": name}


def R(x, y, w, h, fill=None, line=None, lw=0.5, radius=0.0, dash=False, name="box"):
    return {"k": "rect", "x": x, "y": y, "w": w, "h": h, "fill": fill, "line": line, "lw": lw, "radius": radius,
            "dash": dash, "name": name}


def C(x, y, d, fill, name="dot"):
    return {"k": "circle", "x": x, "y": y, "d": d, "fill": fill, "name": name}


def L(x1, y1, x2, y2, color="8C8C8C", lw=0.75, name="line"):
    return {"k": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "color": color, "lw": lw, "name": name}


def I(path, x, y, w, h, name="figure", alt=""):
    return {"k": "image", "path": str(path), "x": x, "y": y, "w": w, "h": h, "name": name, "alt": alt}


# ---------------------------------------------------------------- numbers
def load_numbers():
    s = json.load(open(ROOT / "results/stats/stats_fade_k4.json"))
    cells = {k: {"acc": 100 * v["correct"]["mean"], "lo": 100 * v["correct"]["ci"][0], "hi": 100 * v["correct"]["ci"][1],
                 "mib": v["mean_bytes"] / MIB, "cap": 100 * v["cap_hit"]} for k, v in s["cells"].items()}
    sl = None
    p = ROOT / "results/stats/stats_fade_sl.json"
    if p.exists():
        j = json.load(open(p))
        if j.get("status") == "complete":
            sl = j
    return cells, sl


# ---------------------------------------------------------------- figure for slide 2
def frontier_png(cells, sl, path, w=4.7, h=3.35):
    fp = {k: fm.FontProperties(fname=str(BUILD / "fonts" / v)) for k, v in
          {"r": "Lexend-Regular.ttf", "s": "Lexend-SemiBold.ttf"}.items()}
    plt.rcParams.update({"font.family": fp["r"].get_name(), "font.size": 9.5, "axes.edgecolor": "#8C8C8C",
                         "axes.linewidth": 0.8, "xtick.color": "#4D4D4D", "ytick.color": "#4D4D4D"})
    fig, ax = plt.subplots(figsize=(w, h), dpi=300)
    ax.grid(axis="y", color=GRID_HEX, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    def pt(c, color, marker, ms=7.5, z=3):
        v = cells[c]
        ax.errorbar(v["mib"], v["acc"], yerr=[[v["acc"] - v["lo"]], [v["hi"] - v["acc"]]], fmt="none",
                    ecolor=color, elinewidth=1.0, capsize=0, alpha=0.55, zorder=z - 1)
        ax.plot(v["mib"], v["acc"], marker, color=color, ms=ms, mec="white", mew=1.2, zorder=z)
        return v

    win = [cells[c] for c in ("tl-w2k", "tl-w4k", "tl-w8k")]
    ax.plot([v["mib"] for v in win], [v["acc"] for v in win], "-", color="#9A9A9A", lw=1.6, zorder=2)
    for c, lab in (("tl-w2k", "2K"), ("tl-w4k", "4K"), ("tl-w8k", "8K")):
        v = pt(c, "#8C8C8C", "o", ms=6.5)
        ax.annotate(lab, (v["mib"], v["acc"]), xytext=(7, -3), textcoords="offset points", color="#5F5F5F",
                    fontsize=9, va="top")
    ax.text(cells["tl-w4k"]["mib"] + 18, cells["tl-w4k"]["acc"] - 9.5, "window + prompt\n(forgetting)", color="#5F5F5F",
            fontsize=8.5, ha="left", va="top", linespacing=1.05)
    full, f4, f2 = pt("full", "#1A1A1A", "s"), pt("fade-kv4", "#" + RED, "D", ms=8), pt("fade-kv2-had", "#" + RED, "D", ms=8)
    ax.annotate("Full attention", (full["mib"], full["hi"]), xytext=(0, 3), textcoords="offset points", ha="center",
                fontsize=9.5, color="#1A1A1A", fontproperties=fp["s"])
    ax.annotate("Fade 4-bit", (f4["mib"], f4["hi"]), xytext=(0, 3), textcoords="offset points", ha="center",
                fontsize=9.5, color="#" + RED, fontproperties=fp["s"])
    ax.annotate("Fade 2-bit", (f2["mib"], f2["acc"]), xytext=(-9, 0), textcoords="offset points", ha="right",
                va="center", fontsize=9.5, color="#" + RED, fontproperties=fp["s"])
    # no detectable loss at a third of the memory
    y = full["acc"] + 0.2
    ax.annotate("", xy=(f4["mib"] + 30, y), xytext=(full["mib"] - 30, y),
                arrowprops=dict(arrowstyle="-|>", color="#" + RED, lw=1.1, mutation_scale=9, ls=(0, (3, 2))))
    ax.text((f4["mib"] + full["mib"]) / 2, y - 2.2, f"no detectable loss,\n{100 * f4['mib'] / full['mib']:.0f}% of the memory",
            ha="center", va="top", fontsize=8.5, color="#" + RED, linespacing=1.05)
    if sl:
        v = {"acc": 100 * sl["cells"]["sl-8k"]["correct"]["mean"], "lo": 100 * sl["cells"]["sl-8k"]["correct"]["ci"][0],
             "hi": 100 * sl["cells"]["sl-8k"]["correct"]["ci"][1], "mib": sl["cells"]["sl-8k"]["mean_bytes"] / MIB}
        cells["sl-8k"] = v
        pt("sl-8k", "#" + SL_BLUE, "^", ms=8)
        ax.annotate("StreamingLLM-8K", (v["mib"], v["acc"]), xytext=(9, 0), textcoords="offset points", va="center",
                    fontsize=9.5, color="#" + SL_BLUE, fontproperties=fp["s"])
    ax.text(845, 1.2, "2-bit vs 8K window: the gain comes from\none of two seed pairs (post hoc)", ha="right",
            va="bottom", fontsize=7.5, color="#5F5F5F", linespacing=1.05)
    ax.set_xlim(0, 860)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Analytic memory per sample (MiB, attention + recurrent state)", fontsize=9, color="#4D4D4D")
    ax.set_ylabel("Math accuracy (%)", fontsize=9, color="#4D4D4D")
    ax.tick_params(labelsize=8.5, length=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=300)
    plt.close(fig)


GRID_HEX = "#" + GRID


# ---------------------------------------------------------------- slides
def slide_approach():
    els = [T(0.9, 0.66, 8.2, 0.52, para(run("Approach: blur old thoughts, don't delete them", 23, True, RED_T, semi=True),
                                         align="c"), name="title"),
           R(4.96, 1.3, 0.012, 3.7, fill=GRID, name="divider")]

    # left: what a hybrid is, and where its memory goes
    els += [T(0.5, 1.22, 4.35, 0.34, para(run("Problem: the attention cache grows", 13, True, RED, semi=True)),
              name="problem title"),
            T(0.5, 1.53, 4.35, 0.44, para(run("Hybrid LLMs (Qwen3.5, Kimi Linear): mostly linear layers with a fixed-size "
                                              "state, plus a few attention layers for exact recall.", 9.5, False, MUTED)),
              name="hybrid note")]
    # the 32-layer stack as a strip, its first block zoomed in
    lx, lw_ = 0.55, 4.25
    els += [T(lx - 0.04, 1.99, 2.6, 0.24, para(run("Qwen3.5-9B: 32 layers", 9, True, INK, semi=True)), name="strip label"),
            T(lx + lw_ - 2.0, 1.99, 2.04, 0.24, para(run("pattern 3 : 1, repeated 8×", 9, False, MUTED), align="r"),
              name="strip pattern")]
    sy, sp, sh = 2.25, lw_ / 32, 0.2
    for i in range(32):
        fa = i % 4 == 3
        els.append(R(lx + i * sp, sy, sp - 0.025, sh, fill=RED if fa else GDN_GRAY, name=f"strip layer {i + 1}"))
    by, bh = 2.64, 0.4
    gw_, gap = 0.85, 0.07
    fw = lw_ - 3 * gw_ - 3 * gap
    els += [R(lx - 0.03, sy - 0.03, 4 * sp + 0.035, sh + 0.06, line=INK, lw=0.75, name="zoom source"),
            L(lx - 0.03, sy + sh + 0.03, lx, by, name="zoom left"),
            L(lx + 4 * sp + 0.005, sy + sh + 0.03, lx + lw_, by, name="zoom right")]
    for i in range(4):
        fa = i == 3
        x = lx + i * (gw_ + gap)
        w = fw if fa else gw_
        els += [R(x, by, w, bh, fill=RED if fa else GDN_GRAY, radius=0.05, name=f"block layer {i + 1}"),
                T(x, by, w, bh, para(run("Full attention" if fa else "Gated DeltaNet", 9, fa, "FFFFFF" if fa else INK,
                                         semi=fa), align="c"), anchor="m", name=f"block layer {i + 1} text")]
    # what each kind of layer keeps
    cy = by + bh + 0.1
    for r_ in range(3):          # fixed-size state: a small grid
        for c_ in range(3):
            els.append(R(lx + 0.02 + c_ * 0.085, cy + 0.08 + r_ * 0.085, 0.07, 0.07, fill="9A9A9A",
                         name=f"state icon {3 * r_ + c_ + 1}"))
    els.append(T(lx + 0.32, cy, 3 * gw_ + 2 * gap - 0.32, 0.42, para(run("24 layers: fixed-size state", 9, True, INK,
                                                                         semi=True)),
                 para(run("49 MiB in total, at any length", 9, False, MUTED)), anchor="m", name="gdn note"))
    fx = lx + 3 * (gw_ + gap)
    for j in range(5):           # KV cache: grows with every token
        h_ = 0.07 + 0.045 * j
        els.append(R(fx + j * 0.055, cy + 0.33 - h_, 0.04, h_, fill=RED, name=f"cache icon {j + 1}"))
    els.append(T(fx + 0.3, cy, fw - 0.26, 0.42, para(run("KV cache grows", 9, True, RED, semi=True)),
                 para(run("32 KiB / token", 9, False, MUTED)), anchor="m", name="fa note"))
    # memory split at 32K (analytic): FA 1 GiB vs GDN 49 MiB
    my, mw = 3.64, 3.2
    gdn_w = mw * 49.125 / (1024 + 49.125)
    els += [T(0.5, my, 4.35, 0.26, para(run("Memory per sample at 32K tokens (analytic)", 9.5, False, MUTED)),
              name="memory caption"),
            R(0.55, my + 0.26, mw - gdn_w - 0.02, 0.3, fill=RED, name="memory attention"),
            T(0.55, my + 0.26, mw - gdn_w - 0.02, 0.3, para(run("attention KV: 1 GiB (95%)", 9.5, False, "FFFFFF"),
                                                            align="c"), anchor="m", name="memory attention text"),
            R(0.55 + mw - gdn_w, my + 0.26, gdn_w, 0.3, fill=GDN_GRAY, name="memory gdn"),
            T(0.55 + mw + 0.04, my + 0.26, 1.15, 0.3, para(run("GDN: 49 MiB", 9.5, False, MUTED)), anchor="m",
              name="memory gdn text")]
    els += [R(0.5, 4.42, 4.35, 0.44, fill=PINK_BG, radius=0.06, name="fix box"),
            T(0.58, 4.42, 4.19, 0.44, para(run("Fade ", 11, True, RED, semi=True),
                                          run("compresses only this cache. No training.", 11),
                                          align="c"), anchor="m", name="fix text")]

    # right: what each method keeps (generated part to scale; prompt block not to scale)
    rx, rw = 5.14, 4.42
    els += [T(rx - 0.04, 1.22, rw + 0.08, 0.34, para(run("What attention keeps of a 32K trace", 13, True, RED,
                                                         semi=True)), name="diagram title"),
            T(rx - 0.04, 1.54, rw + 0.08, 0.26, para(run("analytic memory at 32K", 9.5, False, MUTED), align="r"),
              name="memory header")]
    P = 0.34
    gw = rw - P
    k8, k2 = gw * 8192 / 32768, gw * 2048 / 32768
    rows = [
        ("Full attention", [("P", P, "x"), ("exact (16-bit)", gw, "x")], "1,073 MiB"),
        ("StreamingLLM-8K", [("", 0.07, "x"), ("dropped, including the prompt", rw - 0.07 - k8, "d"),
                             ("last 8K", k8, "x")], "305 MiB"),
        ("8K window + prompt", [("P", P, "x"), ("dropped", gw - k8, "d"), ("last 8K", k8, "x")], "305 MiB"),
        ("Fade (ours)", [("P", P, "x"), ("older tokens at 2–4 bits", gw - k2, "q"), ("2K", k2, "x")], "248–368 MiB"),
    ]
    y0, pitch, rh = 1.84, 0.66, 0.28
    for i, (lab, segs, mem) in enumerate(rows):
        yy = y0 + i * pitch
        bold = lab.startswith("Fade")
        col = RED if bold else INK
        els += [T(rx - 0.04, yy - 0.02, 2.9, 0.29, para(run(lab, 10.5, bold, col, semi=bold)), anchor="m",
                  name=f"row {i + 1} label"),
                T(rx + rw - 1.7, yy - 0.02, 1.74, 0.29, para(run(mem, 10.5, bold, col, semi=bold), align="r"), anchor="m",
                  name=f"row {i + 1} memory")]
        xx, by = rx, yy + 0.27
        for j, (txt, w, kind) in enumerate(segs):
            fill, line, tcol = {"x": (RED, None, "FFFFFF"), "q": (PINK, None, PINK_INK),
                                "d": (DROP, DROP_LINE, MUTED)}[kind]
            els.append(R(xx, by, w - 0.025, rh, fill=fill, line=line, lw=0.75, dash=kind == "d",
                         name=f"row {i + 1} segment {j + 1}"))
            if txt:
                els.append(T(xx, by, w - 0.025, rh, para(run(txt, 9 if len(txt) > 2 else 8.5, False, tcol), align="c"),
                             anchor="m", name=f"row {i + 1} segment {j + 1} text"))
            xx += w
    yl = y0 + 4 * pitch + 0.02
    els += [R(rx, yl + 0.07, 0.16, 0.13, fill=RED, name="legend exact"),
            T(rx + 0.2, yl, 0.8, 0.27, para(run("exact", 9.5, False, MUTED)), name="legend exact text"),
            R(rx + 0.95, yl + 0.07, 0.16, 0.13, fill=PINK, name="legend low-bit"),
            T(rx + 1.15, yl, 0.9, 0.27, para(run("2–4 bits", 9.5, False, MUTED)), name="legend low-bit text"),
            R(rx + 2.05, yl + 0.07, 0.16, 0.13, fill=DROP, line=DROP_LINE, lw=0.75, dash=True, name="legend dropped"),
            T(rx + 2.25, yl, 0.9, 0.27, para(run("dropped", 9.5, False, MUTED)), name="legend dropped text"),
            T(rx + 3.1, yl, rw - 3.06, 0.27, para(run("P = prompt", 9.5, False, MUTED), align="r"),
              name="legend note")]
    notes = ("Reasoning models think for tens of thousands of tokens. New models like Qwen3.5 and Kimi Linear are hybrids: "
             "most layers are linear recurrences with a fixed-size state, and a few attention layers keep exact recall. "
             "In Qwen3.5 it is three Gated DeltaNet layers, then one full-attention layer, repeated eight times. "
             "The DeltaNet layers keep a fixed state, 49 megabytes. "
             "The eight attention layers keep every token's keys and values, so at 32K tokens that cache is a gigabyte, "
             "95 percent of the memory. That cache is what Fade targets. Our earlier project tried forgetting: keep only "
             "the prompt and recent tokens. Accuracy fell 40 points, and every attention head turned out to read far "
             "back. So Fade keeps everything, but blurs it: the prompt and the last 2K tokens stay exact, and older keys "
             "and values are stored at 2 or 4 bits. No training, and the DeltaNet layers are untouched.")
    return els, notes


def slide_results(cells, sl, fig_path):
    title = "Results: Fade vs full attention and StreamingLLM" if sl else "Results: Fade vs full attention and windows"
    els = [T(0.9, 0.66, 8.2, 0.52, para(run(title, 23, True, RED_T, semi=True), align="c"), name="title"),
           I(fig_path, 0.32, 1.3, 4.7, 3.35, alt="Math accuracy vs memory per sample for full attention, Fade at 2 and "
                                                  "4 bits, and prompt-kept windows of 2K, 4K and 8K tokens"
                                                  + (", and StreamingLLM-8K" if sl else ""))]
    cards = [
        ("Setup", 1.28, [("Model", "Qwen3.5-9B (hybrid)"),
                        ("Data", "AIME 2025 · AIME 2026 · HMMT 2025"),
                        ("", "MATH-500 level 5 · 30 each, 120 total"),
                        ("Decode", "32K cap · 4 samples" + (" (StreamingLLM: 2)" if sl else " each")),
                        ("Compare", "full attention · windows · StreamingLLM"),
                        ("Analysis", "pre-registered · memory checked on A100")]),
        ("Future work", 3.45, [("Systems", "fused low-bit kernel for speed"),
                               ("Training", "QAT / distillation for 2 and 1 bit"),
                               ("Scope", "more baselines, hybrids and tasks")]),
    ]
    x, w, pitch = 5.22, 4.38, 0.265
    for i, (head, y, rows) in enumerate(cards):
        h = 0.4 + pitch * len(rows) + 0.08
        els += [R(x, y, w, h, fill="FFFFFF", line=RED, lw=0.5, radius=0.06, name=f"{head} card"),
                T(x + 0.14, y + 0.06, w - 0.28, 0.32, para(run(head, 13, True, RED, semi=True)), name=f"{head} title")]
        for j, (lab, val) in enumerate(rows):
            yy = y + 0.4 + j * pitch
            els += [T(x + 0.14, yy, 0.98, pitch, para(run(lab, 10, True, RED, semi=True)), anchor="m",
                      name=f"{head} row {j + 1} label"),
                    T(x + 1.12, yy, w - 1.22, pitch, para(run(val, 10, False, INK)), anchor="m",
                      name=f"{head} row {j + 1} value")]
    els.append(T(0.4, 4.88, 9.2, 0.34, para(run("A hybrid's attention can't forget its old thoughts, but it can blur them",
                                                 13, True, RED, semi=True),
                                             run(" (one model, math).", 10, False, MUTED), align="c"),
                 name="key finding"))
    notes = ("On 120 competition math problems, pre-registered: full attention gets 66 percent. Four-bit Fade also gets "
             "66, with no detectable loss, at 37 percent of the memory. Forgetting, the grey line, falls fast as the "
             "window shrinks. Two-bit Fade beats an 8K window that keeps the prompt by about 4 points with 30 percent "
             "less memory, a modest gain, and all of it comes from one of our two seed pairs. " +
             (f"StreamingLLM-8K, the standard way to cap the cache, drops the prompt too: it gets "
              f"{100 * sl['cells']['sl-8k']['correct']['mean']:.1f} percent on two samples per problem, and 2-bit Fade "
              f"beats it by {sl['contrasts']['SL1']['diff_pp']:.0f} points with less memory. " if sl else
              "StreamingLLM-8K, the standard way to cap the cache: result pending. ") +
             "When 2-bit Fade fails, it is not wrong answers; runs just hit the token limit. The memory on the chart is "
             "computed, and a benchmark with real packed 2- and 4-bit storage agrees within 10 percent; our prototype is "
             "slower than full attention, so no speed claim. Next: a fused low-bit kernel for speed, training to close the 2-bit gap, "
             "comparisons with more baselines such as R-KV and KVQuant, and more models and tasks. So: don't forget. Fade.")
    return els, notes


# ---------------------------------------------------------------- pptx backend
def rgb(h):
    return RGBColor.from_string(h)


def add_pptx(slide, els):
    sh = slide.shapes
    for e in els:
        if e["k"] == "rect":
            shp = sh.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if e["radius"] else MSO_SHAPE.RECTANGLE, Inches(e["x"]),
                               Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
            if e["radius"]:
                shp.adjustments[0] = e["radius"] / min(e["w"], e["h"])
            if e["fill"]:
                shp.fill.solid()
                shp.fill.fore_color.rgb = rgb(e["fill"])
            else:
                shp.fill.background()
            if e["line"]:
                shp.line.color.rgb = rgb(e["line"])
                shp.line.width = Pt(e["lw"])
                if e["dash"]:
                    shp.line.dash_style = MSO_LINE.DASH
            else:
                shp.line.fill.background()
            shp.shadow.inherit = False
            shp.name = e["name"]
        elif e["k"] == "circle":
            shp = sh.add_shape(MSO_SHAPE.OVAL, Inches(e["x"]), Inches(e["y"]), Inches(e["d"]), Inches(e["d"]))
            shp.fill.solid()
            shp.fill.fore_color.rgb = rgb(e["fill"])
            shp.line.fill.background()
            shp.shadow.inherit = False
            shp.name = e["name"]
        elif e["k"] == "line":
            ln = sh.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(e["x1"]), Inches(e["y1"]), Inches(e["x2"]), Inches(e["y2"]))
            ln.line.color.rgb = rgb(e["color"])
            ln.line.width = Pt(e["lw"])
            ln.name = e["name"]
        elif e["k"] == "image":
            pic = sh.add_picture(e["path"], Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
            pic.name = e["name"]
            pic._element.nvPicPr.cNvPr.set("descr", e["alt"])
        elif e["k"] == "text":
            tb = sh.add_textbox(Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
            tb.name = e["name"]
            tf = tb.text_frame
            tf.word_wrap = True
            tf.auto_size = MSO_AUTO_SIZE.NONE
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(INSET)
            tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE}[e["anchor"]]
            for i, p in enumerate(e["paras"]):
                par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                par.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[p["align"]]
                par.line_spacing = 1.05
                par.space_after = Pt(p["after"])
                par.space_before = Pt(0)
                for r in p["runs"]:
                    rr = par.add_run()
                    rr.text = r["text"]
                    f = rr.font
                    f.name, f.size, f.bold, f.italic = r["font"], Pt(r["size"]), r["bold"], r["italic"]
                    f.color.rgb = rgb(r["color"])


def drop_shapes(slide, keep_suffixes):
    for shp in list(slide.shapes):
        if not shp.name.endswith(keep_suffixes):
            shp._element.getparent().remove(shp._element)


def set_title_slide(slide, title, subtitle):
    shp = next(s for s in slide.shapes if s.name.endswith("2597;p146"))
    tf = shp.text_frame
    r0 = tf.paragraphs[0].runs[0]
    r0.text = title
    for extra in tf.paragraphs[0].runs[1:]:
        extra._r.getparent().remove(extra._r)
    p2 = copy.deepcopy(tf.paragraphs[0]._p)
    tf._txBody.append(p2)
    par = tf.paragraphs[1]
    par.runs[0].text = subtitle
    par.runs[0].font.size = Pt(22)
    par.runs[0].font.bold = False
    par.runs[0].font.name = "Lexend"
    par.space_before = Pt(6)
    shp.height = Inches(1.5)


def set_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ---------------------------------------------------------------- preview backend
def fprop(font, bold, size):
    return fm.FontProperties(fname=str(BUILD / "fonts" / FONTS[(font, bold)]), size=size)


def layout_lines(e, renderer, dpi):
    """Greedy word wrap of each paragraph inside the box width; returns [(para, [[(word, run, width_in)]])]."""
    width = e["w"] - 2 * INSET
    out = []
    for p in e["paras"]:
        toks = []
        for r in p["runs"]:
            parts = r["text"].split(" ")
            for j, wd in enumerate(parts):
                toks.append((wd + (" " if j < len(parts) - 1 else ""), r))
        lines, cur, cw = [], [], 0.0
        for wd, r in toks:
            if not wd:
                continue
            ww = renderer.get_text_width_height_descent(wd.replace(" ", " "), fprop(r["font"], r["bold"], r["size"]),
                                                        ismath=False)[0] / dpi
            core = renderer.get_text_width_height_descent(wd.rstrip(" "), fprop(r["font"], r["bold"], r["size"]),
                                                          ismath=False)[0] / dpi
            if cur and cw + core > width + 1e-6:
                lines.append(cur)
                cur, cw = [], 0.0
            cur.append((wd, r, ww))
            cw += ww
        if cur:
            lines.append(cur)
        out.append((p, lines))
    return out


def preview(bg, els, path, cover=None):
    dpi = 150
    fig = plt.figure(figsize=(10, 5.625), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(5.625, 0)
    ax.axis("off")
    ax.imshow(plt.imread(bg), extent=(0, 10, 5.625, 0), zorder=0)
    for x, y, w, h, col in cover or [(0.3, 0.80, 9.7, 4.50, "FFFFFF"), (1.95, 0.66, 8.05, 0.2, "FFFFFF")]:
        ax.add_patch(Rectangle((x, y), w, h, color="#" + col, zorder=1))
    renderer = fig.canvas.get_renderer()
    problems = []
    for e in els:
        if e["k"] == "rect":
            kw = dict(facecolor="#" + e["fill"] if e["fill"] else "none",
                      edgecolor="#" + e["line"] if e["line"] else "none", lw=e["lw"] * 150 / 72 / 2 if e["line"] else 0,
                      ls=(0, (4, 3)) if e["dash"] else "-", zorder=2)
            if e["radius"]:
                ax.add_patch(FancyBboxPatch((e["x"] + e["radius"], e["y"] + e["radius"]), e["w"] - 2 * e["radius"],
                                            e["h"] - 2 * e["radius"], boxstyle=f"round,pad={e['radius']}", **kw))
            else:
                ax.add_patch(Rectangle((e["x"], e["y"]), e["w"], e["h"], **kw))
        elif e["k"] == "circle":
            ax.add_patch(Circle((e["x"] + e["d"] / 2, e["y"] + e["d"] / 2), e["d"] / 2, color="#" + e["fill"], zorder=2))
        elif e["k"] == "line":
            ax.plot([e["x1"], e["x2"]], [e["y1"], e["y2"]], color="#" + e["color"], lw=e["lw"] * 150 / 72 / 2, zorder=2,
                    solid_capstyle="butt")
        elif e["k"] == "image":
            ax.imshow(plt.imread(e["path"]), extent=(e["x"], e["x"] + e["w"], e["y"] + e["h"], e["y"]), zorder=2)
        elif e["k"] == "text":
            laid = layout_lines(e, renderer, dpi)
            total = 0.0
            for p, lines in laid:
                for ln in lines:
                    total += max(r["size"] for _, r, _ in ln) * LINE / 72
                total += p["after"] / 72
            avail = e["h"] - 2 * INSET
            if total > avail + 0.01:
                problems.append(f"{e['name']}: text needs {total:.2f} in, box has {avail:.2f} in")
            y = e["y"] + INSET + (max(0.0, avail - total) / 2 if e["anchor"] == "m" else 0)
            for p, lines in laid:
                for ln in lines:
                    lh = max(r["size"] for _, r, _ in ln) * LINE / 72
                    lw = sum(w for _, _, w in ln) - (ln[-1][2] - renderer.get_text_width_height_descent(
                        ln[-1][0].rstrip(" "), fprop(ln[-1][1]["font"], ln[-1][1]["bold"], ln[-1][1]["size"]),
                        ismath=False)[0] / dpi)
                    x = e["x"] + INSET + {"l": 0, "c": (e["w"] - 2 * INSET - lw) / 2,
                                          "r": e["w"] - 2 * INSET - lw}[p["align"]]
                    base = y + lh * (1.0 / 1.3125) * 1.0    # baseline: ascender (1 em) of the 1.3125 em line
                    for wd, r, w in ln:
                        ax.text(x, base, wd.replace(" ", " "), fontproperties=fprop(r["font"], r["bold"], r["size"]),
                                color="#" + r["color"], style="italic" if r["italic"] else "normal", va="baseline",
                                ha="left", zorder=3)
                        x += w
                    y += lh
                y += p["after"] / 72
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return problems


# ---------------------------------------------------------------- main
TITLE = "Fade, Don't Forget"
SUBTITLE = "Low-bit attention memory for hybrid reasoning models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "slides/fade_slides.pptx"))
    ap.add_argument("--preview", default=str(BUILD / "preview"))
    a = ap.parse_args()
    cells, sl = load_numbers()
    pv = Path(a.preview)
    pv.mkdir(parents=True, exist_ok=True)
    fig_path = BUILD / "fig_slide_frontier.png"
    frontier_png(dict(cells), sl, fig_path)
    s1, n1 = slide_approach()
    s2, n2 = slide_results(cells, sl, fig_path)

    prs = Presentation(str(BUILD / "template_vuhaian.pptx"))
    t, ap_, rs = prs.slides[0], prs.slides[1], prs.slides[2]
    set_title_slide(t, TITLE, SUBTITLE)
    drop_shapes(ap_, ("2604;p147", "2607;p147", "2608;p147", "2616;p147", "2617;p147"))
    drop_shapes(rs, ("2664;p148", "2665;p148", "2666;p148"))
    add_pptx(ap_, s1)
    add_pptx(rs, s2)
    set_notes(t, "Title slide.")
    set_notes(ap_, n1)
    set_notes(rs, n2)
    prs.save(a.out)

    # title slide: the template's text box at (0.56, 1.99), 8.84 in wide, no inset, grown to 1.5 in
    tels = [T(0.56 - INSET, 1.99 - INSET, 8.84 + 2 * INSET, 1.5, para(run(TITLE, 42, True, "FFFFFF", semi=True)),
              para(run(SUBTITLE, 22, False, "FFFFFF")), name="project name")]
    probs = preview(BUILD / "bg_92.png", tels, pv / "slide-1.png", cover=[(0.5, 1.95, 8.0, 0.85, "8B2C33")])
    probs += preview(BUILD / "bg_93.png", s1, pv / "slide-2.png") + preview(BUILD / "bg_94.png", s2, pv / "slide-3.png")
    print(f"wrote {a.out}; previews in {pv}; StreamingLLM {'filled' if sl else 'pending'}")
    for p in probs:
        print("FIT:", p)


if __name__ == "__main__":
    main()
