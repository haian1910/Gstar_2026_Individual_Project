"""Exact unit check of fade_q.age_rows in a batched pool (no model; CPU): random bf16 K/V caches for B = 3 rows with different
prompt lengths P and generated indices g; feeding g = 0 .. n-1 step by step (rows offset), every call may change only the
slots [P + mG, P + (m+1)G) of a row whose block m ages at that g, in every FA layer, and those slots must equal
quant(previous content) bit-exactly; nothing else (other rows, prompt slots, other blocks) may change. All configs.
Usage: python fade_q_unit.py   (prints FADE_Q_UNIT PASS/FAIL)"""
import types
import torch
import fade_q

torch.manual_seed(0)
Rw, G, B, KVH, D, n_fa = 256, 128, 3, 4, 256, 8
Ps, offs, n = (37, 101, 64), (0, 100, 250), 900
Lmax = max(Ps) + max(offs) + n + 16
Hm = fade_q.hadamard(D, "cpu")
ok = True
for name, cfg in fade_q.CONFIGS.items():
    st = types.SimpleNamespace(K=[torch.randn(B, KVH, Lmax, D).bfloat16() for _ in range(n_fa)],
                               V=[torch.randn(B, KVH, Lmax, D).bfloat16() for _ in range(n_fa)])
    events = []
    for i in range(n):
        rows = [(r, Ps[r], offs[r] + i) for r in range(B)]
        K0, V0 = [k.clone() for k in st.K], [v.clone() for v in st.V]
        done = fade_q.age_rows(st, rows, cfg, Hm, Rw, G)
        want = [(r, (g - Rw + 1) // G - 1) for r, P, g in rows if g - Rw + 1 >= G and (g - Rw + 1) % G == 0]
        ok &= done == want
        for fi in range(n_fa):
            for r in range(B):
                for X0, X, f in ((K0, st.K, lambda x: fade_q.quant_k(x, cfg["kb"][fi], cfg["had"], Hm, G)),
                                 (V0, st.V, lambda x: fade_q.quant_v(x, cfg["vb"][fi], cfg["had"], Hm))):
                    exp = X0[fi][r].clone()
                    for rr, m in done:
                        if rr == r:
                            lo = Ps[r] + m * G
                            exp[:, lo:lo + G] = f(X0[fi][r:r + 1, :, lo:lo + G].float()).to(exp.dtype)[0]
                    ok &= torch.equal(X[fi][r], exp)
        events += [(r, m, offs[r] + i) for r, m in done]
    print(f"{name}: {len(events)} aging events, first {events[:3]}", flush=True)
print("FADE_Q_UNIT", "PASS" if ok else "FAIL", flush=True)
