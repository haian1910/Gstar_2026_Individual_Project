"""Fade, don't forget (NTI/individual_project/prereg_fade.md; HA_own/experiments/2026-10-27_fade/PLAN.md): KL(full || config)
of FA-only KV-cache quantization of OLD generated tokens in Qwen3.5-9B, teacher-forced on the layer-map traces (40 non-test
MATH-500 L5 E1 traces, <= 24,576 generated tokens; same selection as layer_map.py). GDN layers are untouched.
Policy of a quantized config: the prompt stays exact; a generated token's K/V stays exact until its 128-token block (generated
index // 128) has fully aged past R (block end <= current index - R), then the block's K/V are replaced by a b-bit
quantize-dequantize (irreversible, as real storage). So the last R .. R + 127 generated tokens are exact (KIVI-style residual).
Quantizer: asymmetric min/max, scale and zero stored as fp16 (effective bits b + 0.25 at groups of 128).
  plain: keys per channel over the 128-token block (post-RoPE, as cached), values per token in groups of 128 along head_dim;
  pre:   keys quantized before RoPE (per channel over the block), then rotated with their true positions;
  had:   rotate head_dim (256) with a normalised Hadamard matrix, per-token groups of 128 for K and V, rotate back.
b >= 16 is the identity and adds nothing: such a config runs the exact full-attention code path (bit-identical to full).
Windowed references (TL-W, all 8 FA layers): prompt + the last W generated tokens incl. the current one.
Chunked lockstep as layer_map.py (chunks aligned to generated index 0, chunk <= R, chunk % 128 == 0). Blocks that age inside a
chunk are attended through appended quantized copies with per-query masks, so every query sees exactly the policy above.
Per config and generated-index bucket (indices >= --score-from): sums of KL(full || config), top-1 agreement, NLL(config) - NLL(full).
Usage: python kv_fade.py --traces ../out/traces.jsonl --problems ../out/problems.jsonl --test-ids ../out/tl_ids.txt \
           --out ../out_fade/fd.s0.jsonl --shard 0 --nshards 2 [--configs kv2,w8k,...]
       python kv_fade.py ... --smoke   (correctness checks on one trace; no output file)"""
import argparse, json, os, random
os.environ.setdefault("USE_HUB_KERNELS", "0")
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from fa_loop import Runner, _apply_rope
from fade_chars import chars_by_bucket

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."  # as gen_traces.py (math rows)
BUCKETS = [(2048, 4096), (4096, 8192), (8192, 16384), (16384, 24576), (24576, 32768)]
GD = 128  # group size along head_dim (values; keys of the Hadamard variant)
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--traces", required=True)
ap.add_argument("--problems", required=True)
ap.add_argument("--test-ids", required=True)
ap.add_argument("--out")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--R", type=int, default=2048)
ap.add_argument("--G", type=int, default=128)
ap.add_argument("--score-from", type=int, default=2048)
ap.add_argument("--max-gen", type=int, default=24576)
ap.add_argument("--min-gen", type=int, default=16384)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--chunk", type=int, default=1024)
ap.add_argument("--configs", help="comma-separated subset of the default config names (full is always added)")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--rows", help="prepared trace rows (traces.jsonl format, e.g. mk_task_rows.py) used as-is instead of select()")
a = ap.parse_args()


def Q(kb, vb=None, mode="post", had=False, force=False, n_fa=8):
    kb = kb if isinstance(kb, list) else [kb] * n_fa
    vb = kb if vb is None else (vb if isinstance(vb, list) else [vb] * n_fa)
    return {"kind": "q", "kb": kb, "vb": vb, "pre": mode == "pre", "had": had, "force": force}


CONFIGS = {
    "full": {"kind": "full"},
    "w2k": {"kind": "win", "W": 2048},
    "w8k": {"kind": "win", "W": 8192},
    "kv16": Q(16),
    "kv4": Q(4), "kv3": Q(3), "kv2": Q(2), "kv1": Q(1),
    "kv2_pre": Q(2, mode="pre"),
    "k2only": Q(2, vb=16), "v2only": Q(16, vb=2),
    "kv2_had": Q(2, had=True), "kv1_had": Q(1, had=True),
    "mix": Q([2, 2, 4, 4, 2, 2, 2, 2]),  # FA idx 2, 3 = model layers 11, 15 (top layer-map recovery) at 4 bits
}


def active(cfg):
    return cfg["kind"] == "q" and (cfg["force"] or min(cfg["kb"] + cfg["vb"]) < 16)


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


def init_state(st, cfg):
    st.nq = st.nrep = 0  # generated tokens quantized (complete blocks) / replaced in place so far
    if active(cfg):
        st.Kq = [torch.zeros_like(k) for k in st.K]
        st.Vq = [torch.zeros_like(v) for v in st.V]
        st.Kpre = [torch.zeros_like(k) for k in st.K] if cfg["pre"] else None


def fa_q(Rn, sa, x, cos, sin, st, fi, slot, mask, Lcur, cfg, ctx, Hm):
    """Runner._fa (same ops, same order) + the fade policy when ctx is given and this layer is quantized."""
    B, T, _ = x.shape
    D = sa.head_dim
    q, gate = torch.chunk(sa.q_proj(x).view(B, T, -1, D * 2), 2, dim=-1)
    gate = gate.reshape(B, T, -1)
    q = sa.q_norm(q).transpose(1, 2)  # [B, H, T, D]
    k = sa.k_norm(sa.k_proj(x).view(B, T, -1, D)).transpose(1, 2)
    v = sa.v_proj(x).view(B, T, -1, D).transpose(1, 2)
    bidx = torch.arange(B, device=x.device)[:, None]
    if ctx is not None and cfg["pre"]:
        st.Kpre[fi].permute(0, 2, 1, 3)[bidx, slot] = k.permute(0, 2, 1, 3)
    q, k = _apply_rope(q, k, cos, sin)
    st.K[fi].permute(0, 2, 1, 3)[bidx, slot] = k.permute(0, 2, 1, 3)
    st.V[fi].permute(0, 2, 1, 3)[bidx, slot] = v.permute(0, 2, 1, 3)
    K = st.K[fi][:, :, :Lcur]
    V = st.V[fi][:, :, :Lcur]
    if ctx is not None and (cfg["force"] or min(cfg["kb"][fi], cfg["vb"][fi]) < 16):
        kb, vb = cfg["kb"][fi], cfg["vb"][fi]
        qa, qb = ctx["q"]
        if qb > qa:  # newly complete blocks: store their quantized copies
            if cfg["pre"]:
                kp = quant_k(st.Kpre[fi][:, :, qa:qb].float(), kb, cfg["had"], Hm, ctx["G"]).to(K.dtype)
                _, kq = _apply_rope(kp, kp, ctx["cq"], ctx["sq"])
            else:
                kq = quant_k(st.K[fi][:, :, qa:qb].float(), kb, cfg["had"], Hm, ctx["G"]).to(K.dtype)
            st.Kq[fi][:, :, qa:qb] = kq
            st.Vq[fi][:, :, qa:qb] = quant_v(st.V[fi][:, :, qa:qb].float(), vb, cfg["had"], Hm).to(V.dtype)
        ra, rb = ctx["r"]
        if rb > ra:  # blocks aged for every query of this call: replace in place (irreversible)
            st.K[fi][:, :, ra:rb] = st.Kq[fi][:, :, ra:rb]
            st.V[fi][:, :, ra:rb] = st.Vq[fi][:, :, ra:rb]
        mask = ctx["main"]
        ta, tb = ctx["t"]
        if tb > ta:  # blocks that age inside this call: quantized copies as extra keys, visible once aged
            K = torch.cat([K, st.Kq[fi][:, :, ta:tb]], 2)
            V = torch.cat([V, st.Vq[fi][:, :, ta:tb]], 2)
            mask = torch.cat([ctx["main"], ctx["extra"]], -1)
    # GQA by folding the G query heads of each KV head into the query axis (as Runner._fa)
    H, KVH = q.shape[1], K.shape[1]
    G = H // KVH
    qg = q.reshape(B, KVH, G * T, D)
    mg = mask if G == 1 else mask.repeat(1, 1, G, 1)
    o = F.scaled_dot_product_attention(qg, K, V, attn_mask=mg, scale=sa.scaling)
    o = o.reshape(B, H, T, D).transpose(1, 2).reshape(B, T, -1)
    o = o * torch.sigmoid(gate)
    return sa.o_proj(o)


def fade_masks(base, kv, slot, P, Rw, G, keep=False):
    """Masks of one call (queries at `slot`, B = 1) over [cache slots 0..Lcur) + [extra keys = quantized copies of the
    transitional slots P + nr .. P + nt). Generated key j (index gk = j - P) is quantized for query index g iff its block end
    (gk // G) * G + G - 1 <= g - Rw. Slots below P + nr are quantized for every query (replaced in place, visible in main).
    keep (kv_pulse.py): nothing is replaced in place (nr = 0), every aged block is read through its quantized copy, so the
    cache stays exact for later exact queries."""
    g_lo, g_hi = int(slot.min()) - P, int(slot.max()) - P
    nr = 0 if keep else max(0, (g_lo - Rw + 1) // G) * G  # blocks aged for every query of this call
    nt = max(0, (g_hi - Rw + 1) // G) * G  # blocks aged for at least one query
    qm = (kv >= P) & (((kv - P) // G) * G + G - 1 <= (slot - P)[:, None, :, None] - Rw)  # [B,1,T,Lcur]
    assert bool(qm[..., P:P + nr].all()), "replaced slots must be quantized for every query"
    in_tr = (kv >= P + nr) & (kv < P + nt)
    return base & ~(qm & in_tr), qm[..., P + nr:P + nt], nr, nt


@torch.no_grad()
def forward_cfg(Rn, st, ids, cfg, P, Rw, G, Hm):
    """Append ids [1, T]; logits [1, T, V] under config cfg (full / win / q)."""
    B, T = ids.shape
    dev = ids.device
    pos = st.pos_next[:, None] + torch.arange(T, device=dev)[None]
    slot = st.length[:, None] + torch.arange(T, device=dev)[None]
    assert int(slot.max()) < st.Lmax, "state storage full"
    Lcur = min(st.Lmax, (int(slot.max()) + 1 + 15) // 16 * 16)
    st.valid.scatter_(1, slot, True)
    kv = torch.arange(Lcur, device=dev)[None, None, None, :]
    base = st.valid[:, None, None, :Lcur] & (kv <= slot[:, None, :, None])  # [B,1,T,Lcur]
    h = Rn.lm.embed_tokens(ids)
    cos, sin = Rn.lm.rotary_emb(h, pos)
    mask, ctx = base, None
    if cfg["kind"] == "win":
        mask = base & ((kv < P) | (kv > slot[:, None, :, None] - cfg["W"]))
    elif active(cfg):
        assert B == 1 and bool((pos == slot).all()), "fade path: one row, no deleted slots"
        g_lo, g_hi = int(slot.min()) - P, int(slot.max()) - P  # generated indices of this call's tokens
        assert g_lo >= 0, "prompt must be prefilled before the fade path"
        nc = (g_hi + 1) // G * G  # complete blocks after this call
        main, extra, nr, nt = fade_masks(base, kv, slot, P, Rw, G, keep=getattr(st, "keep", False))
        assert nt <= nc and nr >= st.nrep
        ctx = {"G": G, "q": (P + st.nq, P + nc), "r": (P + st.nrep, P + nr), "t": (P + nr, P + nt), "main": main, "extra": extra}
        if cfg["pre"] and nc > st.nq:
            ctx["cq"], ctx["sq"] = Rn.lm.rotary_emb(h, torch.arange(P + st.nq, P + nc, device=dev)[None])
    for li, layer in enumerate(Rn.layers):
        r = h
        x = layer.input_layernorm(h)
        if li in Rn.gdn_of:
            x = Rn._gdn(layer.linear_attn, x, st, Rn.gdn_of[li])
        else:
            x = fa_q(Rn, layer.self_attn, x, cos, sin, st, Rn.fa_of[li], slot, mask, Lcur, cfg, ctx, Hm)
        h = r + x
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    st.length += T
    st.pos_next = pos[:, -1] + 1
    if ctx is not None:
        st.nq, st.nrep = max(st.nq, nc), nr
    return Rn.lm_head(Rn.lm.norm(h)).float()


def load_model(name):  # as tl_gen.load_model
    from transformers import AutoModelForImageTextToText
    try:
        m = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16)
    except Exception:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    return m.cuda().eval()


def select(traces, test):
    """As layer_map.select: one trace per non-test MATH-500 L5 problem, the longest with >= min_gen output tokens (ties:
    lowest sample_idx); problems sorted, shuffled with Random(0), first n."""
    best = {}
    for l in open(traces):
        r = json.loads(l)
        if r["dataset"] != "math500_l5" or r["problem_id"] in test or len(r["output_ids"]) < a.min_gen:
            continue
        k = (len(r["output_ids"]), -r["sample_idx"])
        if r["problem_id"] not in best or k > best[r["problem_id"]][0]:
            best[r["problem_id"]] = (k, r)
    pids = sorted(best)
    random.Random(0).shuffle(pids)
    return [best[p][1] for p in pids[:a.n]]


def load_rows(test):
    return [json.loads(l) for l in open(a.rows) if l.strip()] if a.rows else select(a.traces, test)


def prompt_ids(tok, p):
    x = tok.apply_chat_template([{"role": "user", "content": p["problem"] + p.get("suffix", SUFFIX)}],
                                add_generation_prompt=True, enable_thinking=True, tokenize=True)
    return x if isinstance(x, list) else x["input_ids"]


def new_states(Rn, pr, n_gen, cfgs):
    base = Rn.new_state(1, len(pr) + n_gen + 32)
    Rn.prefill(base, torch.tensor([pr], device=Rn.device))
    out = {}
    for name, cfg in cfgs.items():
        s = Rn.new_state(1, base.Lmax)
        s.copy_row_from(0, base)
        init_state(s, cfg)
        out[name] = s
    return out


@torch.no_grad()
def run_trace(Rn, pr, gen, cfgs, chunk, Hm):
    """Lockstep over chunks of the generated tokens. Returns {config: {bucket: [n, kl, agree, dnll]}}."""
    P = len(pr)
    states = new_states(Rn, pr, len(gen), cfgs)
    acc = {name: {f"{lo}-{hi}": [0, 0.0, 0.0, 0.0] for lo, hi in BUCKETS} for name in cfgs if name != "full"}
    g = torch.tensor(gen, device=Rn.device)
    for s0 in range(0, len(gen) - 1, chunk):  # feed tokens s0..s1-1; logits at j predict token j+1
        s1 = min(len(gen) - 1, s0 + chunk)
        ids = g[None, s0:s1]
        tgt = g[s0 + 1:s1 + 1]
        j = torch.arange(s0, s1, device=Rn.device)
        lf = forward_cfg(Rn, states["full"], ids, cfgs["full"], P, a.R, a.G, Hm)[0]
        qf = F.log_softmax(lf, -1)
        pf = qf.exp()
        nll_f = -qf.gather(1, tgt[:, None])[:, 0]
        top_f = lf.argmax(-1)
        del lf
        for name, cfg in cfgs.items():
            if name == "full":
                continue
            lc = forward_cfg(Rn, states[name], ids, cfg, P, a.R, a.G, Hm)[0]
            qc = F.log_softmax(lc, -1)
            kl = (pf * (qf - qc)).sum(-1)
            agree = (lc.argmax(-1) == top_f).float()
            dnll = -qc.gather(1, tgt[:, None])[:, 0] - nll_f
            del lc, qc
            for lo, hi in BUCKETS:
                m = (j >= max(lo, a.score_from)) & (j < hi)
                if m.any():
                    b = acc[name][f"{lo}-{hi}"]
                    b[0] += int(m.sum()); b[1] += kl[m].sum().item(); b[2] += agree[m].sum().item(); b[3] += dnll[m].sum().item()
        del qf, pf
    return acc


# --- smoke ----------------------------------------------------------------------------------------------------------
def np_qdq(x, b, axis):
    L = 2 ** b - 1
    mn, mx = x.min(axis, keepdims=True), x.max(axis, keepdims=True)
    s = ((mx - mn) / L).astype(np.float16).astype(np.float32)
    z = mn.astype(np.float16).astype(np.float32)
    s = np.where(s > 0, s, 1.0).astype(np.float32)
    return np.clip(np.round((x - z) / s), 0, L) * s + z


def np_quant(x, b, kind, H, G):
    """numpy reference: kind 'k' (per channel over G tokens), 'v' (per token, GD along head_dim), 'had' (rotate, per token)."""
    B, KVH, n, D = x.shape
    if kind == "k":
        return np_qdq(x.reshape(B, KVH, n // G, G, D), b, 3).reshape(x.shape)
    y = x @ H if kind == "had" else x
    y = np_qdq(y.reshape(B, KVH, n, D // GD, GD), b, -1).reshape(x.shape)
    return y @ H if kind == "had" else y


def kl_mean(lp, lq):
    p = F.log_softmax(lp, -1)
    return (p.exp() * (p - F.log_softmax(lq, -1))).sum(-1)


@torch.no_grad()
def oracle_masks(dev, P=37, Rw=256, G=128, chunk=256, n=1280, D=8):
    """fp64 check of fade_masks + in-place replacement + appended copies against a per-query oracle that builds each query's
    key set explicitly (exact K/V for unaged keys, a distinct 'quantized' K'/V' for aged ones). Returns the max abs error."""
    gen = torch.Generator().manual_seed(0)
    L = P + n
    K, V = torch.randn(L, D, generator=gen, dtype=torch.float64), torch.randn(L, D, generator=gen, dtype=torch.float64)
    Kq, Vq = K + torch.randn(L, D, generator=gen, dtype=torch.float64), V + torch.randn(L, D, generator=gen, dtype=torch.float64)
    Qy = torch.randn(L, D, generator=gen, dtype=torch.float64)
    K, V, Kq, Vq, Qy = (x.to(dev) for x in (K, V, Kq, Vq, Qy))
    Kc, Vc = K.clone(), V.clone()  # the cache, replaced in place over time
    nrep, err = 0, 0.0
    for s0 in range(0, n, chunk):
        s1 = min(n, s0 + chunk)
        slot = torch.arange(P + s0, P + s1, device=dev)[None]
        Lcur = P + s1
        kv = torch.arange(Lcur, device=dev)[None, None, None, :]
        base = kv <= slot[:, None, :, None]
        main, extra, nr, nt = fade_masks(base, kv, slot, P, Rw, G)
        Kc[P + nrep:P + nr], Vc[P + nrep:P + nr] = Kq[P + nrep:P + nr], Vq[P + nrep:P + nr]
        nrep = nr
        Kx, Vx = torch.cat([Kc[:Lcur], Kq[P + nr:P + nt]]), torch.cat([Vc[:Lcur], Vq[P + nr:P + nt]])
        m = torch.cat([main, extra], -1)[0, 0]
        out = torch.softmax((Qy[P + s0:P + s1] @ Kx.T).masked_fill(~m, float("-inf")), -1) @ Vx
        for i, t in enumerate(range(s0, s1)):  # oracle: query at generated index t
            js = torch.arange(P + t + 1, device=dev)
            aged = (js >= P) & (((js - P) // G) * G + G - 1 <= t - Rw)
            Kt, Vt = torch.where(aged[:, None], Kq[:P + t + 1], K[:P + t + 1]), torch.where(aged[:, None], Vq[:P + t + 1], V[:P + t + 1])
            ref = torch.softmax(Qy[P + t] @ Kt.T, -1) @ Vt
            err = max(err, (out[i] - ref).abs().max().item())
    return err


@torch.no_grad()
def smoke(Rn, pr, gen, Hm):
    P, G, N, Rs = len(pr), a.G, 1024, 256
    e0 = max(oracle_masks(Rn.device, chunk=c, Rw=r) for c, r in ((256, 256), (128, 256), (256, 512), (128, 128)))
    print(f"smoke 0: fade masks vs per-query oracle (fp64; chunks 128/256, R 128-512): max abs err {e0:.2e} (must be < 1e-10)", flush=True)
    g = torch.tensor(gen[:N], device=Rn.device)
    print(f"rotary dim {Rn.lm.rotary_emb(Rn.lm.embed_tokens(g[None, :1]), torch.zeros(1, 1, dtype=torch.long, device=Rn.device))[0].shape[-1]} of head_dim {Rn.cfg.head_dim}", flush=True)
    cf = {"full": CONFIGS["full"], "kv16": CONFIGS["kv16"], "kv16f": Q(16, force=True), "kv2": CONFIGS["kv2"],
          "kv2_pre": CONFIGS["kv2_pre"], "kv2_had": CONFIGS["kv2_had"]}
    # (1) full and kv16 through forward_cfg vs Runner.forward: bit-identical
    st = new_states(Rn, pr, N, {"ref": CONFIGS["full"], **cf})
    ch = 256
    L = {n: [] for n in st}
    for s in range(0, N, ch):
        L["ref"].append(Rn.forward(st["ref"], g[None, s:s + ch], logits="all")[0])
        for n, c in cf.items():
            L[n].append(forward_cfg(Rn, st[n], g[None, s:s + ch], c, P, Rs, G, Hm)[0])
    L = {n: torch.cat(v) for n, v in L.items()}
    print(f"smoke 1: max|dlogit| full(forward_cfg) vs Runner.forward {(L['full'] - L['ref']).abs().max().item():.6f}; "
          f"kv16 vs full {(L['kv16'] - L['full']).abs().max().item():.6f} (both must be 0)", flush=True)
    kf = kl_mean(L["full"], L["kv16f"])
    print(f"smoke 2: forced identity quantizer (append path, R={Rs}): KL(full||kv16f) mean {kf.mean().item():.2e} max {kf.max().item():.2e}", flush=True)
    # (3) chunked (with transitional blocks) vs step-by-step decode (T=1, no transitional blocks), R=256
    sb = new_states(Rn, pr, N, {"full": CONFIGS["full"]})["full"]
    lb = torch.stack([forward_cfg(Rn, sb, g[None, i:i + 1], CONFIGS["full"], P, Rs, G, Hm)[0, 0] for i in range(N)])
    kb = kl_mean(lb, L["full"])[Rs + G:]
    print(f"smoke 3 numerics baseline (full, no quantization): j >= {Rs + G}: KL(step || chunked) mean {kb.mean().item():.5f} "
          f"max {kb.max().item():.4f}; argmax agree {(lb.argmax(-1) == L['full'].argmax(-1))[Rs + G:].float().mean().item():.4f}", flush=True)
    ok = kf.mean().item() <= 2 * kb.mean().item()
    for n in ("kv2", "kv2_pre", "kv2_had"):
        sd = new_states(Rn, pr, N, {n: cf[n]})[n]
        ld = torch.stack([forward_cfg(Rn, sd, g[None, i:i + 1], cf[n], P, Rs, G, Hm)[0, 0] for i in range(N)])
        k_step = kl_mean(ld, L[n])[Rs + G:]
        k_ful = kl_mean(L["full"], ld)[Rs + G:]
        r = k_step.mean().item() / max(k_ful.mean().item(), 1e-12)
        print(f"smoke 3 {n}: j >= {Rs + G}: KL(step || chunked) mean {k_step.mean().item():.5f} max {k_step.max().item():.4f}; "
              f"KL(full || step) mean {k_ful.mean().item():.5f}; ratio {r:.3f}; argmax agree "
              f"{(ld.argmax(-1) == L[n].argmax(-1))[Rs + G:].float().mean().item():.4f}", flush=True)
    # (4) quantizer vs numpy reference, and (5) relative MSE per bits, on this trace's real keys/values (full state)
    Hn = Hm.double().cpu().numpy().astype(np.float32)
    s_full = st["full"]
    n_blk = (N // G) * G
    for fi in (0, 3, 7):
        Kx = s_full.K[fi][:, :, P:P + n_blk].float()
        Vx = s_full.V[fi][:, :, P:P + n_blk].float()
        for b in (1, 2, 4):
            for kind, x, tq in (("k", Kx, quant_k(Kx, b, False, Hm, G)), ("v", Vx, quant_v(Vx, b, False, Hm)),
                                ("had", Kx, quant_k(Kx, b, True, Hm, G))):
                ref = np_quant(x.cpu().numpy(), b, kind, Hn, G)
                d = np.abs(tq.cpu().numpy() - ref)
                step = (np.abs(ref).max() + 1e-12) / (2 ** b - 1)
                print(f"smoke 4 fa{fi} b{b} {kind}: max|torch-numpy| {d.max():.2e}; frac > 1e-3 x range {(d > 1e-3 * step).mean():.2e}", flush=True)
    for b in (1, 2, 3, 4):
        row = []
        for nm, f in (("K", lambda x: quant_k(x, b, False, Hm, G)), ("K_had", lambda x: quant_k(x, b, True, Hm, G)),
                      ("V", lambda x: quant_v(x, b, False, Hm)), ("V_had", lambda x: quant_v(x, b, True, Hm))):
            src = s_full.K if nm.startswith("K") else s_full.V
            e = [((f(src[fi][:, :, P:P + n_blk].float()) - src[fi][:, :, P:P + n_blk].float()).pow(2).sum()
                  / src[fi][:, :, P:P + n_blk].float().pow(2).sum()).item() for fi in range(len(src))]
            row.append(f"{nm} {np.mean(e):.4f}")
        print(f"smoke 5 b{b}: relative MSE (mean over 8 FA layers): " + ", ".join(row), flush=True)
    sp = new_states(Rn, pr, N, {"p": cf["kv2_pre"]})["p"]
    for s in range(0, N, ch):
        forward_cfg(Rn, sp, g[None, s:s + ch], cf["kv2_pre"], P, Rs, G, Hm)
    e = [((sp.Kq[fi][:, :, P:P + n_blk].float() - st["full"].K[fi][:, :, P:P + n_blk].float()).pow(2).sum()
          / st["full"].K[fi][:, :, P:P + n_blk].float().pow(2).sum()).item() for fi in range(len(sp.K))]
    print(f"smoke 5 b2: relative MSE K_pre (post-RoPE space, own trajectory) {np.mean(e):.4f}", flush=True)
    ok &= (L["full"] - L["ref"]).abs().max().item() == 0 and (L["kv16"] - L["full"]).abs().max().item() == 0
    ok &= e0 < 1e-10
    # gates (PLAN Changes 18:1x): oracle exact, no-op bit-identical, forced identity append path within 2x the bf16 step-vs-chunk
    # noise; smoke 3 (step vs chunked with 2-bit codes) is reported, since bf16 differences flip 2-bit codes and cannot be separated
    print("SMOKE", "PASS" if ok else "FAIL", flush=True)


def main():
    assert a.chunk % a.G == 0 and a.chunk <= a.R and a.R % a.G == 0
    tok = AutoTokenizer.from_pretrained(a.model)
    probs = {p["problem_id"]: p for l in open(a.problems) if l.strip() for p in [json.loads(l)]}
    test = set(l.strip() for l in open(a.test_ids) if l.strip())
    rows = load_rows(test)
    print(f"selected {len(rows)} traces " + (f"(as given in {a.rows})" if a.rows else f"(non-test MATH-500 L5, >= {a.min_gen} output tokens)"), flush=True)
    Rn = Runner(load_model(a.model))
    Hm = hadamard(Rn.cfg.head_dim, Rn.device)
    print(f"FA layers (model index): {sorted(Rn.fa_of)}; kv heads {Rn.cfg.num_key_value_heads}, head_dim {Rn.cfg.head_dim}; "
          f"R {a.R}, G {a.G}, chunk {a.chunk}", flush=True)
    if a.smoke:
        r = rows[0]
        pr = prompt_ids(tok, probs[r["problem_id"]])
        assert len(pr) == r["prompt_len"], (len(pr), r["prompt_len"])
        smoke(Rn, pr, r["output_ids"], Hm)
        return
    cfgs = dict(CONFIGS) if not a.configs else {"full": CONFIGS["full"], **{k: CONFIGS[k] for k in a.configs.split(",")}}
    print("configs", json.dumps(cfgs), flush=True)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["problem_id"] for l in open(a.out) if l.strip()}
    for i, r in enumerate(rows):
        if i % a.nshards != a.shard or r["problem_id"] in done:
            continue
        pr = prompt_ids(tok, probs[r["problem_id"]])
        assert len(pr) == r["prompt_len"], (r["problem_id"], len(pr), r["prompt_len"])
        gen = r["output_ids"][:a.max_gen]
        acc = run_trace(Rn, pr, gen, cfgs, a.chunk, Hm)
        with open(a.out, "a") as f:
            f.write(json.dumps({"problem_id": r["problem_id"], "sample_idx": r["sample_idx"], "prompt_len": len(pr),
                                "n_gen": len(gen), "R": a.R, "G": a.G, "score_from": a.score_from,
                                "configs": cfgs, "acc": acc, "chars": chars_by_bucket(tok, gen, BUCKETS, a.score_from)}) + "\n")
        msg = []
        for name in ("w2k", "w8k", "kv2", "kv4"):
            if name in acc:
                n = sum(v[0] for v in acc[name].values())
                msg.append(f"{name} {sum(v[1] for v in acc[name].values()) / max(1, n):.4f}")
        print(f"{r['problem_id']} n_gen {len(gen)}: KL/tok " + ", ".join(msg), flush=True)
    print("FADE_DONE", flush=True)


if __name__ == "__main__":
    main()
