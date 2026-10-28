"""Correctness test of the fq decode cell (tl_gen.py + fade_q.py) against kv_fade.py, small R so that blocks age early.
  (1) fade_q's quantizer copies are bit-identical to kv_fade's on random data.
  (2) B = 1, teacher-forced, one token per step: Runner.forward + fade_q.age_rows == kv_fade.forward_cfg step mode
      (logits and every FA cache slot), for kv2, kv2_had, kv1; must be exact.
  (3) B = 2 pool with rows at different prompts and offsets (row 1 starts at generated index 100), as tl_gen batches:
      the aging schedule must be exactly {block m of a row at its generated index (m + 1) G - 1 + R}, and the logits vs
      each row's B = 1 run must differ no more than the same comparison without quantization (batching numerics).
Usage: python fade_dec_test.py --traces T --problems P --test-ids I"""
import sys
import torch
import torch.nn.functional as F
import kv_fade as kf  # parses sys.argv (--traces, --problems, --test-ids)
import fade_q
from transformers import AutoTokenizer
from fa_loop import Runner

Rs, G, N, OFF = 256, 128, 768, 100


def kl(lp, lq):
    p = F.log_softmax(lp, -1)
    return (p.exp() * (p - F.log_softmax(lq, -1))).sum(-1)


@torch.no_grad()
def main():
    ok = True
    dev = torch.device("cuda")
    Hm = fade_q.hadamard(256, dev)
    assert torch.equal(Hm, kf.hadamard(256, dev))
    gen = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(2, 4, 512, 256, device=dev, generator=gen) * 3
    same = all(torch.equal(fade_q.quant_k(x, b, h, Hm, G), kf.quant_k(x, b, h, Hm, G)) and
               torch.equal(fade_q.quant_v(x, b, h, Hm), kf.quant_v(x, b, h, Hm)) for b in (1, 2, 3, 4, 16) for h in (False, True))
    for n, c in fade_q.CONFIGS.items():
        same &= c == kf.CONFIGS[n]
    ok &= same
    print(f"test 1: fade_q quantizer and configs == kv_fade: {same}", flush=True)

    tok = AutoTokenizer.from_pretrained(kf.a.model)
    import json
    probs = {p["problem_id"]: p for p in map(json.loads, open(kf.a.problems))}
    test = {l.strip() for l in open(kf.a.test_ids) if l.strip()}
    tr = kf.select(kf.a.traces, test)[:2]
    prs = [kf.prompt_ids(tok, probs[t["problem_id"]]) for t in tr]
    gs = [torch.tensor(t["output_ids"][:N + OFF + 1], device=dev) for t in tr]
    Rn = Runner(kf.load_model(kf.a.model))

    def b1(pr, g, cfg, start=0):
        """B = 1 Runner decode with fade_q aging; prompt (+ g[:start]) prefilled. Returns logits [N, V], state."""
        st = Rn.new_state(1, len(pr) + N + OFF + 32)
        Rn.prefill(st, torch.tensor([pr], device=dev))
        if start:
            Rn.prefill(st, g[None, :start])
        out = []
        for i in range(start, start + N):
            if cfg is not None:
                fade_q.age_rows(st, [(0, len(pr), i)], cfg, Hm, Rs, G)
            out.append(Rn.forward(st, g[None, i:i + 1])[0])
        return torch.stack(out), st

    for n in ("kv2", "kv2_had", "kv1"):
        cfg = fade_q.CONFIGS[n]
        P = len(prs[0])
        sa = kf.new_states(Rn, prs[0], N, {n: cfg})[n]
        la = torch.stack([kf.forward_cfg(Rn, sa, gs[0][None, i:i + 1], cfg, P, Rs, G, Hm)[0, 0] for i in range(N)])
        lb, sb = b1(prs[0], gs[0], cfg)
        dl = (la - lb).abs().max().item()
        dc = max(max((sa.K[i][:, :, :P + N] - sb.K[i][:, :, :P + N]).abs().max().item(),
                      (sa.V[i][:, :, :P + N] - sb.V[i][:, :, :P + N]).abs().max().item()) for i in range(len(sa.K)))
        ok &= dl == 0 and dc == 0
        print(f"test 2 {n}: B=1 decode vs kv_fade step mode: max |dlogits| {dl:.3g}, max |dcache| {dc:.3g} (must be 0)", flush=True)

    for n in (None, "kv2_had"):
        cfg = None if n is None else fade_q.CONFIGS[n]
        Lm = max(len(p) for p in prs) + N + OFF + 32
        pool = Rn.new_state(2, Lm)
        starts = (0, OFF)
        for r in range(2):
            st = Rn.new_state(1, Lm)
            Rn.prefill(st, torch.tensor([prs[r]], device=dev))
            if starts[r]:
                Rn.prefill(st, gs[r][None, :starts[r]])
            pool.copy_row_from(r, st)
        out, ev = [], []
        for i in range(N):
            gi = [starts[r] + i for r in range(2)]
            if cfg is not None:
                ev += [(r, m, gi[r]) for r, m in fade_q.age_rows(pool, [(r, len(prs[r]), gi[r]) for r in range(2)], cfg, Hm, Rs, G)]
            out.append(Rn.forward(pool, torch.stack([gs[r][gi[r]] for r in range(2)])[:, None]))
        out = torch.stack(out, 1)  # [2, N, V]
        ref = [b1(prs[r], gs[r], cfg, starts[r])[0] for r in range(2)]
        k = [kl(ref[r], out[r]).max().item() for r in range(2)]
        if cfg is None:
            base = max(k)
            print(f"test 3 baseline (no quantization): B=2 pool vs B=1 rows: max KL/tok {k[0]:.2e}, {k[1]:.2e}", flush=True)
            continue
        want = sorted((r, m, (m + 1) * G - 1 + Rs) for r in range(2) for m in range(N + OFF) if starts[r] <= (m + 1) * G - 1 + Rs < starts[r] + N)
        sched = sorted(ev) == want
        ok &= sched and max(k) <= 2 * base + 1e-6
        print(f"test 3 {n}: aging events {sorted(ev)} == expected: {sched}; B=2 vs B=1 max KL/tok {k[0]:.2e}, {k[1]:.2e} "
              f"(must be <= 2 x baseline {base:.2e} + 1e-6)", flush=True)
    print("FADE_DEC_TEST", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
