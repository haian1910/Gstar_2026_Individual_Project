"""CPU unit tests of fade_mem.py (no model, no GPU). Usage: CUDA_VISIBLE_DEVICES= python fade_mem_test.py
  1 bit-packing round trip (b = 1, 2, 4, 8)
  2 dequant(packed) bitwise equal to fade_q.quant_k / quant_v (.to(bf16)): random fp32 blocks, and a B=2 store dequantized in chunks
    of several sizes and per [1,KVH,G,D] block, for kv4, kv2, kv2_had, kv1_had
  3 online-softmax chunked attention vs dense SDPA (fp32) on the dequantized K/V: GQA (16 q heads / 4 KV heads), prompt + old + recent
    split, odd chunk sizes (multiples of G, not dividing the lengths)
  4 block aging over many steps (R=256, G=128): the store matches fade_q.age_rows' schedule exactly (old part and exact ring equal the
    emulated cache at every step; n_quantized == policy_nq == G*floor((g0-R)/G)), and attention matches dense SDPA on the emulated cache
  5 the no-op variant (bf16 old blocks) equals dense attention on the exact cache"""
import sys
import types
import torch
import torch.nn.functional as F

import fade_q
import fade_mem as fm

torch.manual_seed(0)
dev = torch.device("cpu")
Hm = fade_q.hadamard(256, dev)
RESULTS = []


def report(name, ok):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


def test_pack():
    ok = True
    for b in (1, 2, 4, 8):
        c = torch.randint(0, 2 ** b, (2, 4, 33, 256), dtype=torch.uint8)
        p = fm.pack(c, b)
        ok &= p.shape == (2, 4, 33, 256 * b // 8) and p.dtype == torch.uint8 and torch.equal(fm.unpack(p, b), c)
    c = torch.tensor([[1, 2, 3, 0]], dtype=torch.uint8)  # known layout: code i in bits 2i
    ok &= int(fm.pack(c, 2)) == 1 | (2 << 2) | (3 << 4)
    report("1 bit-packing round trip (b=1,2,4,8) + 2-bit layout", ok)


def test_dequant_equal():
    for name in ("kv4", "kv2", "kv2_had", "kv1_had"):
        r1 = fm.random_block_equal(dev, name)
        report(f"2a random fp32 [1,4,128,256] blocks, dequant(packed) == fade_q ({name})", r1)
        r2 = fm.roundtrip_equal(dev, name, R=256, G=128, g0=256 + 7 * 128 + 21, chunks=(128, 384, 640, 2048))
        report(f"2b real-layout B=2 store, chunked + per-block dequant == fade_q per block ({name}; chunks 128/384/640/2048)", r2)


def dense_attn(q16, K, V, scale):
    """q16 [B,16,1,D], K/V [B,4,n,D] -> [B,16,1,D] via repeat_kv + SDPA, fp32."""
    rep = q16.shape[1] // K.shape[1]
    Kr, Vr = K.float().repeat_interleave(rep, 1), V.float().repeat_interleave(rep, 1)
    return F.scaled_dot_product_attention(q16.float(), Kr, Vr, scale=scale)


def test_attention():
    P, R, G, g0, B = 53, 256, 128, 256 + 5 * 128 + 70, 3
    for name in ("kv4", "kv2_had"):
        cfg = fade_q.CONFIGS[name]
        gen = torch.Generator().manual_seed(1)
        n = P + g0
        Kf = (torch.randn(1, 4, n, 256, generator=gen) * 2).to(torch.bfloat16)
        Vf = (torch.randn(1, 4, n, 256, generator=gen) * 0.5).to(torch.bfloat16)
        lay = fm.PackedLayer(1, 4, 256, P, R, G, g0 + 8, cfg["kb"][0], cfg["vb"][0], cfg["had"], Hm, dev)
        lay.load(Kf, Vf, g0)
        lb = lay.to_batch(B)
        nq = lb.nq
        assert nq == (g0 - R) // G == 5
        # dense reference: emulated cache = prompt exact, old blocks fade_q-quantized, recent exact
        Kd, Vd = Kf.clone(), Vf.clone()
        for m in range(nq):
            lo = P + m * G
            Kd[:, :, lo:lo + G] = fade_q.quant_k(Kf[:, :, lo:lo + G].float(), cfg["kb"][0], cfg["had"], Hm, G).to(torch.bfloat16)
            Vd[:, :, lo:lo + G] = fade_q.quant_v(Vf[:, :, lo:lo + G].float(), cfg["vb"][0], cfg["had"], Hm).to(torch.bfloat16)
        q16 = torch.randn(B, 16, 1, 256, generator=gen)
        ref = dense_attn(q16, Kd.expand(B, -1, -1, -1), Vd.expand(B, -1, -1, -1), 256 ** -0.5)
        ok, worst = True, 0.0
        for chunk in (128, 256, 384, 640, 896, 2048, 4096):
            out = lb.attend(q16.reshape(B, 4, 4, 256), 256 ** -0.5, chunk).reshape(B, 16, 1, 256)
            d = (out - ref).abs().max().item()
            worst = max(worst, d)
            ok &= d < 1e-5
        report(f"3 chunked online-softmax == dense SDPA, GQA 16/4, P={P} + {nq} old blocks + {g0 - nq * G} recent, {name} "
               f"(max |d| {worst:.2e} < 1e-5, 7 chunk sizes)", ok)


def test_aging():
    P, R, G, N = 10, 256, 128, 900
    for name in ("kv4", "kv2_had"):
        cfg = fade_q.CONFIGS[name]
        gen = torch.Generator().manual_seed(2)
        Lmax = P + N + 8
        fake = types.SimpleNamespace(K=[torch.zeros(1, 4, Lmax, 256, dtype=torch.bfloat16)],
                                     V=[torch.zeros(1, 4, Lmax, 256, dtype=torch.bfloat16)])
        Kp = (torch.randn(1, 4, P, 256, generator=gen) * 2).to(torch.bfloat16)
        Vp = (torch.randn(1, 4, P, 256, generator=gen) * 0.5).to(torch.bfloat16)
        fake.K[0][:, :, :P], fake.V[0][:, :, :P] = Kp, Vp
        lay = fm.PackedLayer(1, 4, 256, P, R, G, N, cfg["kb"][0], cfg["vb"][0], cfg["had"], Hm, dev)
        lay.load(Kp, Vp, 0)
        ok_sched = ok_store = ok_attn = True
        age_events = []
        for g in range(N):
            if g % 17 == 0:  # before the step feeding g: quantized blocks == policy_nq(g) == floor((g - R) / G)
                ok_sched &= lay.nq == fm.policy_nq(g, R, G) == max(0, (g - R) // G)
            done = fade_q.age_rows(fake, [(0, P, g)], cfg, Hm, R, G)
            aged = lay.age(g)
            ok_sched &= aged == bool(done) and (not aged or done[0][1] == lay.nq - 1)
            if aged:
                age_events.append(g)
            k = (torch.randn(1, 4, 1, 256, generator=gen) * 2).to(torch.bfloat16)
            v = (torch.randn(1, 4, 1, 256, generator=gen) * 0.5).to(torch.bfloat16)
            fake.K[0][:, :, P + g:P + g + 1], fake.V[0][:, :, P + g:P + g + 1] = k, v
            lay.append(g, k, v)
            nq = lay.nq
            # exact generated tokens still held: indices [nq*G, g]
            n_ex = lay.n_exact_generated()
            ok_store &= n_ex == g + 1 - nq * G
            if g % 37 == 0 or aged or g == N - 1:
                j = torch.arange(nq * G, g + 1)
                ok_store &= torch.equal(lay.Ke[:, :, P + j % (R + G)], fake.K[0][:, :, P + j])
                ok_store &= torch.equal(lay.Ve[:, :, P + j % (R + G)], fake.V[0][:, :, P + j])
                if nq:
                    ok_store &= torch.equal(lay.deqk(0, nq * G), fake.K[0][:, :, P:P + nq * G])
                    ok_store &= torch.equal(lay.deqv(0, nq * G), fake.V[0][:, :, P:P + nq * G])
                q16 = torch.randn(1, 16, 1, 256, generator=gen)
                out = lay.attend(q16.reshape(1, 4, 4, 256), 256 ** -0.5, 384).reshape(1, 16, 1, 256)
                ref = dense_attn(q16, fake.K[0][:, :, :P + g + 1], fake.V[0][:, :, :P + g + 1], 256 ** -0.5)
                ok_attn &= (out - ref).abs().max().item() < 1e-5
        sched_ref = [R + G - 1 + i * G for i in range((N - R) // G)]
        ok_sched &= age_events == sched_ref and lay.nq == (N - R) // G
        # n_quantized at a given g0 (before feeding index g0): G * floor((g0 - R) / G), from fade_q.age_rows' schedule
        for g0 in (0, R - 1, R, R + G - 1, R + G, R + G + 1, 3 * G + R, 900):
            ok_sched &= fm.policy_nq(g0, R, G) == max(0, (g0 - R) // G)
        report(f"4 aging over {N} steps (R={R}, G={G}) vs fade_q.age_rows, {name}: schedule {age_events} / store / attention", ok_sched and ok_store and ok_attn)
        if not (ok_sched and ok_store and ok_attn):
            print(f"   sched {ok_sched} store {ok_store} attn {ok_attn}")


def test_noop():
    P, R, G, g0 = 20, 256, 128, 256 + 3 * 128 + 5
    gen = torch.Generator().manual_seed(3)
    n = P + g0
    Kf = (torch.randn(1, 4, n, 256, generator=gen) * 2).to(torch.bfloat16)
    Vf = (torch.randn(1, 4, n, 256, generator=gen) * 0.5).to(torch.bfloat16)
    cfg = fade_q.CONFIGS["kv4"]
    lay = fm.PackedLayer(1, 4, 256, P, R, G, g0 + 200, 4, 4, False, Hm, dev, noop=True)
    lay.load(Kf, Vf, g0)
    ok = True
    for g in range(g0, g0 + 200):  # also ages new blocks through the noop path
        lay.age(g)
        k = (torch.randn(1, 4, 1, 256, generator=gen) * 2).to(torch.bfloat16)
        v = (torch.randn(1, 4, 1, 256, generator=gen) * 0.5).to(torch.bfloat16)
        Kf = torch.cat([Kf, k], 2)
        Vf = torch.cat([Vf, v], 2)
        lay.append(g, k, v)
    q16 = torch.randn(1, 16, 1, 256, generator=gen)
    out = lay.attend(q16.reshape(1, 4, 4, 256), 256 ** -0.5, 384).reshape(1, 16, 1, 256)
    ref = dense_attn(q16, Kf, Vf, 256 ** -0.5)
    d = (out - ref).abs().max().item()
    ok = d < 1e-5 and lay.nq == (g0 + 200 - R) // G and lay.nbytes()["fa_packed_codes"] == 0
    report(f"5 no-op variant (bf16 old blocks, chunked path) == dense SDPA on the exact cache (max |d| {d:.2e}), {lay.nq} blocks", ok)


if __name__ == "__main__":
    test_pack()
    test_dequant_equal()
    test_attention()
    test_aging()
    test_noop()
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if all(RESULTS) else 1)
