"""Characters of scored text per KL bucket (nti-e2's H1 request, prereg_fade.md c302e40d: cross-model KL per character, since
the two tokenizers split the same text differently). Position j's logits are scored against token j + 1 when
max(lo, score_from) <= j < hi (as run_trace); n_chars = sum of len(tok.decode([t])) over those target tokens.
Post hoc for a finished sweep: python fade_chars.py --rows 'out_fade/fd.s*.jsonl' --traces out/traces.jsonl --out O
(rows without "chars" get it from the trace tokens output_ids[:n_gen]; Qwen3.5 tokenizer)."""
import argparse, glob, json


def chars_by_bucket(tok, gen, buckets, score_from):
    memo, out = {}, {f"{lo}-{hi}": 0 for lo, hi in buckets}
    for j in range(len(gen) - 1):
        t = gen[j + 1]
        for lo, hi in buckets:
            if max(lo, score_from) <= j < hi:
                if t not in memo:
                    memo[t] = len(tok.decode([t]))
                out[f"{lo}-{hi}"] += memo[t]
    return out


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    BUCKETS = [(2048, 4096), (4096, 8192), (8192, 16384), (16384, 24576), (24576, 32768)]  # as kv_fade.BUCKETS
    rows = [json.loads(l) for f in sorted(glob.glob(a.rows)) for l in open(f) if l.strip()]
    want = {(r["problem_id"], r["sample_idx"]) for r in rows}
    tr = {}
    for l in open(a.traces):
        t = json.loads(l)
        if (t["problem_id"], t["sample_idx"]) in want:
            tr[(t["problem_id"], t["sample_idx"])] = t["output_ids"]
    tok = AutoTokenizer.from_pretrained(a.model)
    nc = nt = 0
    with open(a.out, "w") as f:
        for r in rows:
            gen = tr[(r["problem_id"], r["sample_idx"])][:r["n_gen"]]
            c = chars_by_bucket(tok, gen, BUCKETS, r["score_from"])
            n = next(iter(r["acc"].values()))
            assert all((c[b] > 0) == (n[b][0] > 0) for b in c), (r["problem_id"], c, n)  # same scored positions
            nt += sum(v[0] for v in n.values()); nc += sum(c.values())
            f.write(json.dumps({"problem_id": r["problem_id"], "sample_idx": r["sample_idx"], "chars": c,
                                "n_tok": {b: n[b][0] for b in c}}) + "\n")
    print(f"{len(rows)} rows, {nt} scored tokens, {nc} chars, {nc / max(1, nt):.3f} chars/token -> {a.out}")


if __name__ == "__main__":
    main()
