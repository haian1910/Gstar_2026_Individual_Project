"""Think-Local multi-wrap, batched check on the real model (nti-e2's request after the TL-W2K primary failed):
teacher-force two real Full traces (different problems, different prompt lengths) for N tokens through
  ring: one B=2 pool exactly as tl_gen.py builds it (Lmax = Pmax + W + 16, slot = P + i % W, lcur = Lmax - 16),
  ref:  a B=2 plain state where every slot older than the last W is blocked (the masked definition of TL),
  full: a B=2 plain state with nothing blocked (for scale),
and report per 1K-step bucket: KL(ref || ring), argmax agreement ring vs ref, max |dlogit|, and KL(full || ref).
PASS if ring tracks ref far more closely than ref differs from full after the first wrap.
Usage: python tests/test_tl_long.py --gen ../out_tl/gen_full.s0.jsonl --problems ../out/problems.jsonl \
           --problems ../out_pilot/problems.jsonl [--W 2048] [--N 10240]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("USE_HUB_KERNELS", "0")
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from fa_loop import Runner

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."  # as tl_gen.py
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--gen", required=True)
ap.add_argument("--problems", action="append", required=True)
ap.add_argument("--W", type=int, default=2048)
ap.add_argument("--N", type=int, default=10240)
a = ap.parse_args()


def load_model(name):  # as tl_gen.load_model
    from transformers import AutoModelForImageTextToText
    try:
        m = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16)
    except Exception:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    return m.cuda().eval()


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(a.model)
    probs = {p["problem_id"]: p for f in a.problems for l in open(f) if l.strip() for p in [json.loads(l)]}
    pids = lambda p: (lambda x: x if isinstance(x, list) else x["input_ids"])(tok.apply_chat_template(
        [{"role": "user", "content": p["problem"] + p.get("suffix", SUFFIX)}], add_generation_prompt=True,
        enable_thinking=True, tokenize=True))
    rows = []
    for l in open(a.gen):  # first two traces >= N tokens with different prompt lengths
        r = json.loads(l)
        if r["n_tokens"] >= a.N and r["problem_id"] in probs:
            pr = pids(probs[r["problem_id"]])
            if all(len(pr) != len(x[1]) for x in rows):
                rows.append((r["problem_id"], pr, r["gen_ids"][:a.N]))
        if len(rows) == 2:
            break
    assert len(rows) == 2, "need two traces >= N tokens"
    print("rows:", [(pid, len(pr)) for pid, pr, _ in rows], "W", a.W, "N", a.N, flush=True)
    R = Runner(load_model(a.model))
    P = [len(pr) for _, pr, _ in rows]
    Pmax, B = max(P), 2
    ring = R.new_state(B, Pmax + a.W + 16)
    ref = R.new_state(B, Pmax + a.N + 32)
    full = R.new_state(B, Pmax + a.N + 32)
    lg = []
    for b, (_, pr, _) in enumerate(rows):
        s = R.new_state(1, Pmax + 16)
        lg.append(R.prefill(s, torch.tensor([pr], device=R.device))[0])
        for st in (ring, ref, full):
            st.copy_row_from(b, s)
    lcur = ring.Lmax - 16
    stats, bucket = [], None
    for i in range(a.N):
        ids = torch.tensor([[g[i]] for _, _, g in rows], device=R.device)
        sl = torch.tensor([[P[b] + i % a.W] for b in range(B)], device=R.device)
        for b in range(B):
            p = P[b] + i
            if p - a.W >= P[b]:
                ref.block_slots(b, P[b], p - a.W + 1)  # visible: prompt + the last W tokens including the one fed
        lr = R.forward(ring, ids, slot=sl, lcur=lcur).float()
        lf = R.forward(ref, ids).float()
        lu = R.forward(full, ids).float()
        qr, qf, qu = (F.log_softmax(x, -1) for x in (lr, lf, lu))
        kl_rr = (qf.exp() * (qf - qr)).sum(-1)  # KL(ref || ring) per row
        kl_ur = (qu.exp() * (qu - qf)).sum(-1)  # KL(full || ref) per row: the size of the windowing effect
        agree = (lr.argmax(-1) == lf.argmax(-1)).float()
        dmax = (lr - lf).abs().max(-1).values
        if i % 1024 == 0:
            if bucket:
                stats.append(bucket)
            bucket = dict(lo=i, n=0, kl_ring_ref=0.0, agree=0.0, dmax=0.0, kl_full_ref=0.0, agree_full_ref=0.0)
        bucket["n"] += B
        bucket["kl_ring_ref"] += kl_rr.sum().item()
        bucket["agree"] += agree.sum().item()
        bucket["dmax"] = max(bucket["dmax"], dmax.max().item())
        bucket["kl_full_ref"] += kl_ur.sum().item()
        bucket["agree_full_ref"] += (lu.argmax(-1) == lf.argmax(-1)).float().sum().item()
        if (i + 1) % 1024 == 0:
            print(f"step {i + 1}", flush=True)
    stats.append(bucket)
    print(f"{'steps':>13} {'KL(ref||ring)':>14} {'agree':>7} {'max|dl|':>8} {'KL(full||ref)':>14} {'agree f/r':>9}")
    ok = True
    for s in stats:
        n = s["n"]
        kr, ag, kf = s["kl_ring_ref"] / n, s["agree"] / n, s["kl_full_ref"] / n
        print(f"{s['lo']:>6}-{s['lo'] + n // B - 1:<6} {kr:>14.5f} {ag:>7.4f} {s['dmax']:>8.3f} {kf:>14.5f} {s['agree_full_ref'] / n:>9.4f}")
        if s["lo"] >= a.W:  # after the first wrap: ring must match ref far better than ref matches full
            ok &= kr < 0.1 * kf and ag >= 0.97
    print("TEST LONG", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
