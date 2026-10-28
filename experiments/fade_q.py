"""Fade, don't forget (HA_own/experiments/2026-10-28_fade/PLAN.md) for decoding: in-place FA KV quantization of aged blocks.
Policy as kv_fade.py: the prompt stays exact; a generated key at index gk is quantized for query index g iff its block end
(gk // G) * G + G - 1 <= g - R. Decoding feeds one token per row per step, so block m of a row is quantized in place
(irreversibly) right before that row's query at index g = (m + 1) * G - 1 + R, which is kv_fade's step mode.
The quantizer (Q, qdq, quant_k, quant_v, hadamard, GD) is a copy of kv_fade.py (that file parses argv at import);
fade_dec_test.py checks the copies are bit-identical and that decoding matches kv_fade's step mode."""
import torch

GD = 128  # group size along head_dim (values; keys of the Hadamard variant)


def Q(kb, vb=None, mode="post", had=False, force=False, n_fa=8):
    kb = kb if isinstance(kb, list) else [kb] * n_fa
    vb = kb if vb is None else (vb if isinstance(vb, list) else [vb] * n_fa)
    return {"kind": "q", "kb": kb, "vb": vb, "pre": mode == "pre", "had": had, "force": force}


CONFIGS = {  # the decode candidates of the prereg cell rule (kv2_pre is not supported here: it needs pre-RoPE keys)
    "kv4": Q(4), "kv2": Q(2), "kv1": Q(1),
    "kv2_had": Q(2, had=True), "kv1_had": Q(1, had=True),
}


def hadamard(n, device):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    assert H.shape[0] == n, "head_dim must be a power of 2"
    return (H / n ** 0.5).to(device)


def qdq(x, b, dim):
    """Asymmetric min/max b-bit quantize-dequantize of fp32 x, one group per slice along `dim`; fp16 scale and zero."""
    if b >= 16:
        return x
    L = 2 ** b - 1
    mn, mx = x.amin(dim, keepdim=True), x.amax(dim, keepdim=True)
    s = ((mx - mn) / L).half().float()
    z = mn.half().float()
    s = torch.where(s > 0, s, torch.ones_like(s))
    return ((x - z) / s).round().clamp(0, L) * s + z


def quant_k(k, b, had, Hm, G):
    """k [B, KVH, n, D] fp32, n % G == 0."""
    if b >= 16:
        return k
    B, KVH, n, D = k.shape
    if had:
        y = k @ Hm
        return qdq(y.view(B, KVH, n, D // GD, GD), b, -1).view(B, KVH, n, D) @ Hm
    return qdq(k.view(B, KVH, n // G, G, D), b, 3).view(B, KVH, n, D)


def quant_v(v, b, had, Hm):
    if b >= 16:
        return v
    B, KVH, n, D = v.shape
    y = v @ Hm if had else v
    y = qdq(y.view(B, KVH, n, D // GD, GD), b, -1).view(B, KVH, n, D)
    return y @ Hm if had else y


def age_rows(st, rows, cfg, Hm, Rw, G):
    """rows: [(row, P, g)] with g = generated index of the token this step feeds. Quantizes, in place, the block that ages
    at query g (if any) in every FA layer. Returns [(row, block)] quantized now."""
    assert not cfg["pre"], "pre-RoPE keys are not stored in decoding"
    done = []
    for r, P, g in rows:
        e = g - Rw + 1
        if e < G or e % G:
            continue
        m = e // G - 1
        lo, hi = P + m * G, P + (m + 1) * G
        for fi in range(len(st.K)):
            kb, vb = cfg["kb"][fi], cfg["vb"][fi]
            K, V = st.K[fi], st.V[fi]
            if cfg["force"] or kb < 16:
                K[r:r + 1, :, lo:hi] = quant_k(K[r:r + 1, :, lo:hi].float(), kb, cfg["had"], Hm, G).to(K.dtype)
            if cfg["force"] or vb < 16:
                V[r:r + 1, :, lo:hi] = quant_v(V[r:r + 1, :, lo:hi].float(), vb, cfg["had"], Hm).to(V.dtype)
        done.append((r, m))
    return done
