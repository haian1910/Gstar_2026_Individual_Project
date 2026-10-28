"""Full-attention NLL of the trace tokens under Qwen3.5-9B, per kv_fade bucket (the H1 NLL pair, prereg_fade.md:54).
Same rows, prompt, truncation (--max-gen) and chunked full path as kv_fade.py; only the full config runs.
Output per row: problem_id, sample_idx, prompt_len, n_gen, score_from, full_nll {bucket: [n, nll sum]} (the
kv_fade_dense.py full_nll format).
Usage: python fade_nll35.py --traces .. --problems .. --test-ids .. [--rows ..] --out X.jsonl [--shard g --nshards 2]
"""
import argparse, json, os, sys

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--traces", required=True)
ap.add_argument("--problems", required=True)
ap.add_argument("--test-ids", required=True)
ap.add_argument("--rows")
ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
a = ap.parse_args()
_argv, sys.argv = sys.argv, ["kv_fade.py", "--traces", a.traces, "--problems", a.problems, "--test-ids", a.test_ids] + \
    (["--rows", a.rows] if a.rows else [])
import kv_fade as kf  # noqa: E402  (parses the argv above; its own main does not run)
sys.argv = _argv
from transformers import AutoTokenizer  # noqa: E402


@torch.no_grad()
def full_nll(Rn, pr, gen, Hm):
    ka = kf.a
    st = kf.new_states(Rn, pr, len(gen), {"full": kf.CONFIGS["full"]})["full"]
    nll = {f"{lo}-{hi}": [0, 0.0] for lo, hi in kf.BUCKETS}
    g = torch.tensor(gen, device=Rn.device)
    for s0 in range(0, len(gen) - 1, ka.chunk):
        s1 = min(len(gen) - 1, s0 + ka.chunk)
        lf = kf.forward_cfg(Rn, st, g[None, s0:s1], kf.CONFIGS["full"], len(pr), ka.R, ka.G, Hm)[0]
        n = -F.log_softmax(lf, -1).gather(1, g[s0 + 1:s1 + 1, None])[:, 0]
        j = torch.arange(s0, s1, device=Rn.device)
        for lo, hi in kf.BUCKETS:
            m = (j >= max(lo, ka.score_from)) & (j < hi)
            if m.any():
                b = nll[f"{lo}-{hi}"]
                b[0] += int(m.sum()); b[1] += n[m].sum().item()
    return nll


def main():
    ka = kf.a
    tok = AutoTokenizer.from_pretrained(ka.model)
    probs = {p["problem_id"]: p for l in open(ka.problems) if l.strip() for p in [json.loads(l)]}
    test = set(l.strip() for l in open(ka.test_ids) if l.strip())
    rows = kf.load_rows(test)
    print(f"{len(rows)} rows", flush=True)
    Rn = kf.Runner(kf.load_model(ka.model))
    Hm = kf.hadamard(Rn.cfg.head_dim, Rn.device)
    done = {json.loads(l)["problem_id"] for l in open(a.out) if l.strip()} if os.path.exists(a.out) else set()
    for i, r in enumerate(rows):
        if i % a.nshards != a.shard or r["problem_id"] in done:
            continue
        pr = kf.prompt_ids(tok, probs[r["problem_id"]])
        assert len(pr) == r["prompt_len"], (r["problem_id"], len(pr), r["prompt_len"])
        gen = r["output_ids"][:ka.max_gen]
        nll = full_nll(Rn, pr, gen, Hm)
        with open(a.out, "a") as f:
            f.write(json.dumps({"problem_id": r["problem_id"], "sample_idx": r["sample_idx"], "prompt_len": len(pr),
                                "n_gen": len(gen), "score_from": ka.score_from, "full_nll": nll}) + "\n")
        n = sum(v[0] for v in nll.values())
        print(f"{r['problem_id']} n_gen {len(gen)}: full NLL/tok {sum(v[1] for v in nll.values()) / max(1, n):.4f}", flush=True)
    print("NLL35_DONE", flush=True)


if __name__ == "__main__":
    main()
