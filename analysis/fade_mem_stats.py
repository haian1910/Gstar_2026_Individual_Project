#!/usr/bin/env python3
# Measured-memory benchmark check (prereg_fade.md: design ~23:38 UTC, code and refinement ~00:0x UTC, both before any GPU
# number). Rows come from the research bot's fade_mem.py --bench, one per (cell, L, B), L = total context P + g0.
# Per sample, for B = 1 and B = 32 (bytes / B):
#   measured = state_alloc_bytes      memory_allocated after building the state - weights-only allocated bytes
#   numel    = state_bytes            numel x element_size of the state tensors (capacity slack included)
#   slack    = the capacity fade_mem allocates beyond what the policy holds at L, from the logged capacities and token
#              counts, priced at the logged per-slot rates (exact slots; packed slots for fq)
#   analytic = fade_stats.sample_bytes(cell, g0) + P x 32 KiB for full, tl and fq; min(L, 8,196) x 32 KiB + GDN for sl
# Registered check: |measured - slack - analytic| / analytic <= 10%. Also reported, not tested: raw measured / analytic and
# measured / numel (allocator overhead). Ratios between cells are reported raw and slack-removed, next to the analytic ones.
# Usage: fade_mem_stats.py --rows out_fade/fade_mem.jsonl --out results/stats_fade_mem.json
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fade_stats as F  # noqa: E402
import tl_stats as T  # noqa: E402

TOL = 0.10
SINKS, SL_W = 4, 8192
CHECK_B = (1, 32)
RATIOS = [("Fade-2 / TL-W8K", "fade-kv2-had", "tl-w8k"), ("Fade-2 / SL-8K", "fade-kv2-had", "sl-8k"),
          ("Fade-2 / Full", "fade-kv2-had", "full"), ("Fade-4 / Full", "fade-kv4", "full")]
MIB = 2 ** 20


def analytic(cell, L, P):
    if cell.startswith("sl-"):
        return min(L, SINKS + SL_W) * T.FA_BYTES_PER_TOKEN + T.GDN_STATE_BYTES
    return F.sample_bytes(cell, L - P) + P * T.FA_BYTES_PER_TOKEN


def per_sample(r):
    B = r["B"]
    ex_rate = r["fa_exact_bytes"] / B / r["cap_exact_slots"]
    slack = (r["cap_exact_slots"] - r["n_exact_total"]) * ex_rate
    if r["cap_old_slots"]:
        pk_rate = (r["fa_packed_codes_bytes"] + r["fa_scales_bytes"]) / B / r["cap_old_slots"]
        slack += (r["cap_old_slots"] - r["n_quantized_tokens"]) * pk_rate
    cell = F.fcanon(r["cell"])
    an = analytic(cell, r["L"], r["P"])
    meas, numel = r["state_alloc_bytes"] / B, r["state_bytes"] / B
    return {"cell": cell, "L": r["L"], "P": r["P"], "B": B, "measured": meas, "numel": numel, "slack": slack,
            "analytic": an, "exact_rate_kib": ex_rate / 1024, "dev_policy": (meas - slack) / an - 1,
            "dev_raw": meas / an - 1, "meas_over_numel": meas / numel - 1}


def analyse(rows):
    ok = [r for r in rows if not r.get("oom")]
    out = {"oom": sorted((F.fcanon(r["cell"]), r["L"], r["B"]) for r in rows if r.get("oom")),
           "not_finite": sorted((F.fcanon(r["cell"]), r["L"], r["B"]) for r in ok if not r.get("finite", True)),
           "per_config": [per_sample(r) for r in sorted(ok, key=lambda r: (r["L"], r["B"], r["cell"]))]}
    checked = [c for c in out["per_config"] if c["B"] in CHECK_B]
    fails = [c for c in checked if abs(c["dev_policy"]) > TOL]
    out["check"] = {"n_checked": len(checked), "tolerance": TOL,
                    "fails": [(c["cell"], c["L"], c["B"], round(100 * c["dev_policy"], 2)) for c in fails],
                    "verdict": "PASS" if checked and not fails else "FAIL" if fails else "no rows"}
    by = {(c["cell"], c["L"], c["B"]): c for c in out["per_config"]}
    out["ratios"] = []
    for name, a, b in RATIOS:
        for L, B in sorted({(c["L"], c["B"]) for c in out["per_config"]}):
            if (a, L, B) in by and (b, L, B) in by:
                x, y = by[(a, L, B)], by[(b, L, B)]
                out["ratios"].append({"ratio": name, "L": L, "B": B, "measured": x["measured"] / y["measured"],
                                      "slack_removed": (x["measured"] - x["slack"]) / (y["measured"] - y["slack"]),
                                      "analytic": x["analytic"] / y["analytic"]})
    fits = {}
    for r in rows:
        if not r.get("oom"):
            k = (F.fcanon(r["cell"]), r["L"])
            fits[k] = max(fits.get(k, 0), r["B"])
    out["largest_B_run"] = {f"{c} L={L}": B for (c, L), B in sorted(fits.items())}
    out["ms_per_step"] = {f"{F.fcanon(r['cell'])} L={r['L']} B={r['B']}": r["ms_per_step"] for r in ok}
    return out


def fake_rows(bias=0.0, R=2048, G=128, steps=128, warm=4, P=150):
    """Rows laid out as fade_mem.py documents them (capacities (f) of the research bot's message), for the selftest."""
    rows, fa = [], T.FA_BYTES_PER_TOKEN
    for L in (8192, 16384, 32768):
        g0 = L - P
        for cell in ("full", "tl-8192", "sl-8192", "fq-kv4", "fq-kv2_had"):
            for B in (1, 8, 32):
                cap_old = nq = 0
                pk = 0.0
                if cell == "full":
                    cap, nex = L + steps + warm + G, L
                elif cell == "tl-8192":
                    cap, nex = P + SL_W + 16, P + min(g0, SL_W)
                elif cell == "sl-8192":
                    cap, nex = SINKS + SL_W + 16, SINKS + min(L - SINKS, SL_W)
                else:
                    nq = F.quantized_tokens(g0)
                    cap, nex, cap_old = P + R + G, P + g0 - nq, g0 + steps + warm + G
                    pk = cap_old * F.old_bits(F.fcanon(cell)) / 16 * fa * B
                ex = cap * fa * B
                gdn, oth = T.GDN_STATE_BYTES * B, cap * B
                numel = ex + pk + gdn + oth
                rows.append({"cell": cell, "L": L, "P": P, "g0": g0, "B": B, "oom": cell == "full" and L == 32768 and B == 32,
                             "cap_exact_slots": cap, "cap_old_slots": cap_old, "n_exact_total": nex, "n_quantized_tokens": nq,
                             "fa_exact_bytes": ex, "fa_packed_codes_bytes": pk * 16 / 17 if pk else 0,
                             "fa_scales_bytes": pk / 17 if pk else 0, "gdn_bytes": gdn, "other_bytes": oth,
                             "state_bytes": numel, "state_alloc_bytes": numel * (1 + bias) + 2 * MIB, "ms_per_step": 30.0,
                             "finite": True})
    return rows


def selftest():
    out = analyse(fake_rows())
    assert out["check"]["verdict"] == "PASS", out["check"]
    assert ("full", 32768, 32) in out["oom"] and out["largest_B_run"]["full L=32768"] == 8
    raw = {(c["cell"], c["L"], c["B"]): c["dev_raw"] for c in out["per_config"]}
    assert raw[("fade-kv4", 8192, 1)] > 0.10 > raw[("fade-kv4", 32768, 1)], raw[("fade-kv4", 8192, 1)]
    r = next(x for x in out["ratios"] if x["ratio"] == "Fade-2 / Full" and x["L"] == 32768 and x["B"] == 1)
    assert abs(r["slack_removed"] - r["analytic"]) < 0.01, r
    bad = analyse(fake_rows(bias=0.2))
    assert bad["check"]["verdict"] == "FAIL" and len(bad["check"]["fails"]) == bad["check"]["n_checked"]
    print(f"selftest ok (fake rows: kv4 raw +{100 * raw[('fade-kv4', 8192, 1)]:.1f}% at L = 8K from slack alone; "
          f"policy check {out['check']['verdict']}; +20% bias -> {bad['check']['verdict']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    out = analyse([json.loads(l) for l in open(a.rows) if l.strip()])
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("%-13s %6s %3s %5s %10s %10s %9s %10s %8s %8s %8s" % ("cell", "L", "B", "P", "meas MiB", "numel MiB", "slack MiB",
                                                                "anal MiB", "policy%", "raw%", "m/numel%"))
    for c in out["per_config"]:
        print("%-13s %6d %3d %5d %10.1f %10.1f %9.1f %10.1f %+8.2f %+8.2f %+8.2f" % (
            c["cell"], c["L"], c["B"], c["P"], c["measured"] / MIB, c["numel"] / MIB, c["slack"] / MIB, c["analytic"] / MIB,
            100 * c["dev_policy"], 100 * c["dev_raw"], 100 * c["meas_over_numel"]))
    print(f"check (B in {CHECK_B}, policy count within ±{TOL:.0%}): {out['check']['verdict']}; fails {out['check']['fails']}")
    for r in out["ratios"]:
        if r["B"] in CHECK_B:
            print(f"{r['ratio']} L={r['L']} B={r['B']}: measured x{r['measured']:.3f}, slack removed x{r['slack_removed']:.3f}, "
                  f"analytic x{r['analytic']:.3f}")
    print(f"OOM: {out['oom']}; non-finite: {out['not_finite']}")


if __name__ == "__main__":
    main()
