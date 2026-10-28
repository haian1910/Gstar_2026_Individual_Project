#!/usr/bin/env python3
# Exploratory hybrid-vs-dense comparison (prereg_fade.md, "Dense comparison"): teacher-forced KL of the Fade policy on
# Qwen3.5-9B (8 FA layers quantized) vs Qwen3-8B (36 attention layers; kv2_every4 = 9 of them), same texts per task.
# Row format (research bot's KL sweeps): {"problem_id", <acc>: {config: {bucket: [n_tok, kl_sum, top1_agree, dnll_sum]}},
# optional "chars": {bucket: n_chars}}. Per-token KL is not comparable across tokenizers (151k vs 248k vocab), so the
# cross-model ratio uses KL per character of scored text when "chars" is present, else per token (flagged).
# Usage: h1_stats.py --rows q35:math:a.jsonl q3:math:b.jsonl q35:countdown:c.jsonl q3:countdown:d.jsonl --out h1.json
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

CONTRASTS = [("kv4", "kv4"), ("kv2", "kv2"), ("kv1", "kv1"), ("kv2", "kv2_every4"), ("w2k", "w2k"), ("w8k", "w8k")]


def acc_of(row):
    for key in ("acc", "kl"):
        if isinstance(row.get(key), dict):
            return row[key]
    for v in row.values():
        if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
            return v
    raise KeyError(f"no per-config bucket dict in row {row.get('problem_id')}")


def load(spec):
    """spec = model:task:rows.jsonl[+chars.jsonl]; the optional side file maps problem_id -> chars{bucket} (post hoc)."""
    model, task, path = spec.split(":", 2)
    path, _, side = path.partition("+")
    extra = {}
    if side:
        for line in open(side):
            r = json.loads(line)
            assert r["problem_id"] not in extra, f"duplicate {r['problem_id']} in {side}"
            extra[r["problem_id"]] = r["chars"]
    rows = {}
    for line in open(path):
        r = json.loads(line)
        assert r["problem_id"] not in rows, f"duplicate {r['problem_id']} in {path}"
        per = {}
        for cfg, buckets in acc_of(r).items():
            n = sum(b[0] for b in buckets.values())
            per[cfg] = {"n": n, "kl": sum(b[1] for b in buckets.values()), "agree": sum(b[2] for b in buckets.values()),
                        "buckets": {k: (b[0], b[1]) for k, b in buckets.items() if b[0]}}
        chars = r.get("chars", extra.get(r["problem_id"]))
        assert not side or r["problem_id"] in extra, f"{r['problem_id']} missing from {side}"
        rows[r["problem_id"]] = {"cfg": per, "chars": sum(chars.values()) if isinstance(chars, dict) else chars,
                                 "full_nll": r.get("full_nll")}
    return model, task, rows


def boot(ids, stat, n_boot, seed=0):
    rng = random.Random(seed)
    est = stat(ids)
    sims = sorted(stat([ids[rng.randrange(len(ids))] for _ in ids]) for _ in range(n_boot))
    return est, sims[int(0.025 * n_boot)], sims[int(0.975 * n_boot) - 1]


def per_cell(rows, cfg, n_boot):
    ids = sorted(p for p in rows if cfg in rows[p]["cfg"])
    kl_tok = lambda s: sum(rows[p]["cfg"][cfg]["kl"] for p in s) / sum(rows[p]["cfg"][cfg]["n"] for p in s)
    est, lo, hi = boot(ids, kl_tok, n_boot)
    out = {"n_traces": len(ids), "n_tok": sum(rows[p]["cfg"][cfg]["n"] for p in ids), "kl_tok": est, "ci": [lo, hi],
           "top1": sum(rows[p]["cfg"][cfg]["agree"] for p in ids) / sum(rows[p]["cfg"][cfg]["n"] for p in ids)}
    if all(rows[p]["chars"] for p in ids):
        out["kl_kchar"] = 1000 * sum(rows[p]["cfg"][cfg]["kl"] for p in ids) / sum(rows[p]["chars"] for p in ids)
        out["chars_per_tok"] = sum(rows[p]["chars"] for p in ids) / out["n_tok"]
    if cfg != "w2k" and all("w2k" in rows[p]["cfg"] for p in ids):
        ratio = lambda s: sum(rows[p]["cfg"][cfg]["kl"] for p in s) / sum(rows[p]["cfg"]["w2k"]["kl"] for p in s)
        e, l, h = boot(ids, ratio, n_boot)
        out["frac_of_w2k"] = {"est": e, "ci": [l, h]}
    bk = defaultdict(lambda: [0, 0.0])
    for p in ids:
        for k, (n, kl) in rows[p]["cfg"][cfg]["buckets"].items():
            bk[k][0] += n
            bk[k][1] += kl
    out["by_bucket"] = {k: v[1] / v[0] for k, v in sorted(bk.items(), key=lambda kv: int(kv[0].split("-")[0]))}
    return out


def cross(hy, de, c_hy, c_de, n_boot):
    """Paired over the shared texts: KL(hybrid, c_hy) / KL(dense, c_de), per character if both have chars."""
    ids = sorted(p for p in hy if p in de and c_hy in hy[p]["cfg"] and c_de in de[p]["cfg"])
    per_char = all(hy[p]["chars"] and de[p]["chars"] for p in ids)

    def rate(rows, cfg, s):
        return sum(rows[p]["cfg"][cfg]["kl"] for p in s) / sum(rows[p]["chars"] if per_char else rows[p]["cfg"][cfg]["n"] for p in s)

    def stat(s):
        return rate(hy, c_hy, s) / rate(de, c_de, s)

    est, lo, hi = boot(ids, stat, n_boot)
    word = "hybrid lower" if hi < 1 else "hybrid higher" if lo > 1 else "not distinguishable"
    return {"hybrid": c_hy, "dense": c_de, "n_traces": len(ids), "unit": "char" if per_char else "token (tokenizers differ!)",
            "ratio": est, "ci": [lo, hi], "reading": word}


def analyse(specs, n_boot=10000, hybrid="q35", dense="q3"):
    data = defaultdict(dict)
    for s in specs:
        m, t, rows = load(s)
        data[t][m] = rows
    out = {}
    for task, models in sorted(data.items()):
        out[task] = {"cells": {m: {c: per_cell(rows, c, n_boot) for c in sorted({c for r in rows.values() for c in r["cfg"]})}
                               for m, rows in models.items()}}
        for m, rows in models.items():
            nll = [r["full_nll"] for r in rows.values() if r["full_nll"] is not None]
            if nll and all(isinstance(x, (int, float)) for x in nll):
                out[task]["cells"][m]["_full_nll_mean"] = sum(nll) / len(nll)
        if hybrid in models and dense in models:
            out[task]["contrasts"] = [cross(models[hybrid], models[dense], a, b, n_boot) for a, b in CONTRASTS
                                      if any(a in r["cfg"] for r in models[hybrid].values())
                                      and any(b in r["cfg"] for r in models[dense].values())]
            c = {(x["hybrid"], x["dense"]): x for x in out[task]["contrasts"]}
            a, b = c.get(("kv2", "kv2")), c.get(("kv2", "kv2_every4"))
            if a and b:
                out[task]["reading"] = (
                    "GDN layers appear to absorb the noise" if b["ci"][1] < 1 else
                    "explained by how many layers are quantized" if a["ci"][1] < 1 else
                    "hybrid not more tolerant at 2 bits")
    return out


def selftest():
    rng = random.Random(3)

    def rows(scale, cfgs, chars_per_tok):
        out = []
        for i in range(40):
            acc = {}
            for c, kl in cfgs.items():
                acc[c] = {f"{lo}-{2 * lo}": [lo, kl * scale * lo * rng.uniform(.5, 1.5), .95 * lo, 0.0] for lo in (2048, 4096, 8192)}
            out.append({"problem_id": f"p{i}", "acc": acc, "chars": {"all": int(14336 * chars_per_tok)}, "full_nll": .5})
        return out

    d = Path("/tmp") if not Path(".").is_dir() else Path(".")
    a, b = d / "_h1_q35.jsonl", d / "_h1_q3.jsonl"
    a.write_text("\n".join(json.dumps(r) for r in rows(1, {"w2k": .06, "w8k": .01, "kv4": .0004, "kv2": .007, "kv1": .03}, 4.0)))
    b.write_text("\n".join(json.dumps(r) for r in rows(1, {"w2k": .3, "w8k": .1, "kv4": .002, "kv2": .03, "kv1": .12,
                                                          "kv2_every4": .02}, 3.6)))
    out = analyse([f"q35:math:{a}", f"q3:math:{b}"], n_boot=400)
    a.unlink()
    b.unlink()
    m = out["math"]
    assert m["cells"]["q35"]["kv2"]["n_traces"] == 40 and abs(m["cells"]["q35"]["kv2"]["chars_per_tok"] - 4.0) < 1e-9
    k = {(x["hybrid"], x["dense"]): x for x in m["contrasts"]}
    assert k[("kv2", "kv2")]["unit"] == "char" and k[("kv2", "kv2_every4")]["reading"] == "hybrid lower"
    assert m["reading"] == "GDN layers appear to absorb the noise", m["reading"]
    print(f"selftest ok (kv2 ratio {k[('kv2', 'kv2')]['ratio']:.2f}, vs every4 {k[('kv2', 'kv2_every4')]['ratio']:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    out = analyse(args.rows, args.n_boot)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for task, t in out.items():
        for x in t.get("contrasts", []):
            print(f"{task} q35 {x['hybrid']} / q3 {x['dense']}: x{x['ratio']:.2f} ({x['ci'][0]:.2f}-{x['ci'][1]:.2f}) per "
                  f"{x['unit']} -> {x['reading']}")
        if "reading" in t:
            print(f"{task}: {t['reading']}")


if __name__ == "__main__":
    main()
