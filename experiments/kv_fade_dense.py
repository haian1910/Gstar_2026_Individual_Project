"""Fade KL sweep on a DENSE model (Qwen3-8B), the hybrid-vs-dense control of HA_own/experiments/2026-10-28_fade
(nti-e2's spec in NTI/individual_project/prereg_fade.md, exploratory). Same 40 held-out E1 traces as kv_fade.py: each Qwen3.5
trace (first --max-gen Qwen3.5 tokens) is decoded to text and re-tokenized for the dense model as "<think>\\n" + text after
the dense model's own thinking prompt (off-policy for the dense model). Same policy and quantizer as kv_fade.py (prompt
exact, last R generated tokens exact, aged G-token blocks quantized in place, KIVI K per channel / V per token in groups of
GD), applied to EVERY attention layer, or to every 4th (kv2_every4: layers 3, 7, ..., the hybrid's FA layout) as the
layer-count control. Scored from generated index --score-from in the dense model's tokens. Shared code (fade_masks,
quantizer, oracle, select, prompt_ids, Q) is imported from kv_fade.py, so the two sweeps run the same functions.
Usage: python kv_fade_dense.py --traces T --problems P --test-ids I --out O [--shard i --nshards n] | --smoke | --tiny"""
import argparse, json, os, sys, types
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--ref-model", default="Qwen/Qwen3.5-9B", help="tokenizer of the traces")
ap.add_argument("--traces")
ap.add_argument("--problems")
ap.add_argument("--test-ids")
ap.add_argument("--out")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--R", type=int, default=2048)
ap.add_argument("--G", type=int, default=128)
ap.add_argument("--score-from", type=int, default=2048)
ap.add_argument("--max-gen", type=int, default=24576, help="in trace (Qwen3.5) tokens, as kv_fade.py")
ap.add_argument("--min-gen", type=int, default=16384)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--rows", help="prepared trace rows, as kv_fade.py --rows")
ap.add_argument("--chunk", type=int, default=1024)
ap.add_argument("--configs", help="comma-separated subset (full is always added)")
ap.add_argument("--smoke", action="store_true", help="correctness gates on the real model (GPU)")
ap.add_argument("--tiny", action="store_true", help="correctness gates on a tiny random Qwen3 (CPU, fp32)")
a = ap.parse_args()
_argv, sys.argv = sys.argv, ["kv_fade.py", "--traces", a.traces or "-", "--problems", a.problems or "-",
                             "--test-ids", a.test_ids or "-", "--min-gen", str(a.min_gen), "--n", str(a.n)] + (["--rows", a.rows] if a.rows else [])
import kv_fade as kf  # noqa: E402  (parses the argv above; its own main does not run)
from fade_chars import chars_by_bucket  # noqa: E402
sys.argv = _argv


def configs(nl):
    Q = lambda kb, **k: kf.Q(kb, n_fa=nl, **k)
    return {"full": {"kind": "full"}, "w2k": {"kind": "win", "W": 2048}, "w8k": {"kind": "win", "W": 8192},
            "kv16": Q(16), "kv4": Q(4), "kv2": Q(2), "kv1": Q(1),
            "kv2_every4": Q([2 if i % 4 == 3 else 16 for i in range(nl)]), "kv2_had": Q(2, had=True)}


class Dense:
    def __init__(self, model):
        self.model, self.lm, self.cfg = model, model.model, model.config
        self.layers = self.lm.layers
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    def new_state(self, Lmax):
        c, z = self.cfg, lambda: torch.zeros(1, c.num_key_value_heads, Lmax, c.head_dim, device=self.device, dtype=self.dtype)
        return types.SimpleNamespace(Lmax=Lmax, K=[z() for _ in self.layers], V=[z() for _ in self.layers],
                                     valid=torch.zeros(1, Lmax, dtype=torch.bool, device=self.device),
                                     length=torch.zeros(1, dtype=torch.long, device=self.device),
                                     pos_next=torch.zeros(1, dtype=torch.long, device=self.device))


def attn_q(sa, x, cos, sin, st, li, slot, mask, Lcur, cfg, ctx, Hm):
    """kv_fade.fa_q for a Qwen3 attention layer (no output gate; keys quantized post-RoPE only)."""
    B, T, _ = x.shape
    D = sa.head_dim
    q = sa.q_norm(sa.q_proj(x).view(B, T, -1, D)).transpose(1, 2)
    k = sa.k_norm(sa.k_proj(x).view(B, T, -1, D)).transpose(1, 2)
    v = sa.v_proj(x).view(B, T, -1, D).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    bidx = torch.arange(B, device=x.device)[:, None]
    st.K[li].permute(0, 2, 1, 3)[bidx, slot] = k.permute(0, 2, 1, 3)
    st.V[li].permute(0, 2, 1, 3)[bidx, slot] = v.permute(0, 2, 1, 3)
    K = st.K[li][:, :, :Lcur]
    V = st.V[li][:, :, :Lcur]
    if ctx is not None and (cfg["force"] or min(cfg["kb"][li], cfg["vb"][li]) < 16):
        kb, vb = cfg["kb"][li], cfg["vb"][li]
        qa, qb = ctx["q"]
        if qb > qa:  # newly complete blocks: store their quantized copies
            st.Kq[li][:, :, qa:qb] = kf.quant_k(st.K[li][:, :, qa:qb].float(), kb, cfg["had"], Hm, ctx["G"]).to(K.dtype)
            st.Vq[li][:, :, qa:qb] = kf.quant_v(st.V[li][:, :, qa:qb].float(), vb, cfg["had"], Hm).to(V.dtype)
        ra, rb = ctx["r"]
        if rb > ra:  # blocks aged for every query of this call: replace in place (irreversible)
            st.K[li][:, :, ra:rb] = st.Kq[li][:, :, ra:rb]
            st.V[li][:, :, ra:rb] = st.Vq[li][:, :, ra:rb]
        mask = ctx["main"]
        ta, tb = ctx["t"]
        if tb > ta:  # blocks that age inside this call: quantized copies as extra keys, visible once aged
            K = torch.cat([K, st.Kq[li][:, :, ta:tb]], 2)
            V = torch.cat([V, st.Vq[li][:, :, ta:tb]], 2)
            mask = torch.cat([ctx["main"], ctx["extra"]], -1)
    H, KVH = q.shape[1], K.shape[1]
    G = H // KVH
    qg = q.reshape(B, KVH, G * T, D)
    mg = mask if G == 1 else mask.repeat(1, 1, G, 1)
    o = F.scaled_dot_product_attention(qg, K, V, attn_mask=mg, scale=sa.scaling)
    return sa.o_proj(o.reshape(B, H, T, D).transpose(1, 2).reshape(B, T, -1))


@torch.no_grad()
def forward_cfg(Rn, st, ids, cfg, P, Rw, G, Hm):
    """kv_fade.forward_cfg for the dense model: append ids [1, T]; logits [1, T, V] under cfg."""
    B, T = ids.shape
    dev = ids.device
    pos = st.pos_next[:, None] + torch.arange(T, device=dev)[None]
    slot = st.length[:, None] + torch.arange(T, device=dev)[None]
    assert int(slot.max()) < st.Lmax, "state storage full"
    Lcur = min(st.Lmax, (int(slot.max()) + 1 + 15) // 16 * 16)
    st.valid.scatter_(1, slot, True)
    kv = torch.arange(Lcur, device=dev)[None, None, None, :]
    base = st.valid[:, None, None, :Lcur] & (kv <= slot[:, None, :, None])
    h = Rn.lm.embed_tokens(ids)
    cos, sin = Rn.lm.rotary_emb(h, pos)
    mask, ctx = base, None
    if cfg["kind"] == "win":
        mask = base & ((kv < P) | (kv > slot[:, None, :, None] - cfg["W"]))
    elif kf.active(cfg):
        assert not cfg["pre"] and bool((pos == slot).all())
        g_lo, g_hi = int(slot.min()) - P, int(slot.max()) - P
        assert g_lo >= 0, "prompt must be prefilled before the fade path"
        nc = (g_hi + 1) // G * G
        main, extra, nr, nt = kf.fade_masks(base, kv, slot, P, Rw, G, keep=getattr(st, "keep", False))
        assert nt <= nc and nr >= st.nrep
        ctx = {"G": G, "q": (P + st.nq, P + nc), "r": (P + st.nrep, P + nr), "t": (P + nr, P + nt), "main": main, "extra": extra}
    for li, layer in enumerate(Rn.layers):
        r = h
        h = r + attn_q(layer.self_attn, layer.input_layernorm(h), cos, sin, st, li, slot, mask, Lcur, cfg, ctx, Hm)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    st.length += T
    st.pos_next = pos[:, -1] + 1
    if ctx is not None:
        st.nq, st.nrep = max(st.nq, nc), nr
    return Rn.model.lm_head(Rn.lm.norm(h)).float()


def new_states(Rn, pr, n_gen, cfgs, chunk=4096):
    base = Rn.new_state(len(pr) + n_gen + 32)
    ids = torch.tensor([pr], device=Rn.device)
    for s in range(0, len(pr), chunk):
        forward_cfg(Rn, base, ids[:, s:s + chunk], {"kind": "full"}, len(pr), a.R, a.G, None)
    out = {}
    for name, cfg in cfgs.items():
        s = types.SimpleNamespace(**{k: ([x.clone() for x in v] if isinstance(v, list) else (v.clone() if torch.is_tensor(v) else v))
                                     for k, v in vars(base).items()})
        kf.init_state(s, cfg)
        out[name] = s
    return out


@torch.no_grad()
def run_trace(Rn, pr, gen, cfgs, chunk, Hm, Rw, G, score_from):
    """kv_fade.run_trace (lockstep over chunks). Returns {config: {bucket: [n, kl, agree, dnll]}}, {bucket: [n, full nll]}."""
    P = len(pr)
    states = new_states(Rn, pr, len(gen), cfgs)
    acc = {name: {f"{lo}-{hi}": [0, 0.0, 0.0, 0.0] for lo, hi in kf.BUCKETS} for name in cfgs if name != "full"}
    nll = {f"{lo}-{hi}": [0, 0.0] for lo, hi in kf.BUCKETS}  # full's NLL of the trace tokens (how off-policy the text is)
    g = torch.tensor(gen, device=Rn.device)
    for s0 in range(0, len(gen) - 1, chunk):
        s1 = min(len(gen) - 1, s0 + chunk)
        ids, tgt = g[None, s0:s1], g[s0 + 1:s1 + 1]
        j = torch.arange(s0, s1, device=Rn.device)
        lf = forward_cfg(Rn, states["full"], ids, cfgs["full"], P, Rw, G, Hm)[0]
        qf = F.log_softmax(lf, -1)
        pf = qf.exp()
        nll_f = -qf.gather(1, tgt[:, None])[:, 0]
        top_f = lf.argmax(-1)
        del lf
        for lo, hi in kf.BUCKETS:
            m = (j >= max(lo, score_from)) & (j < hi)
            nll[f"{lo}-{hi}"][0] += int(m.sum()); nll[f"{lo}-{hi}"][1] += nll_f[m].sum().item()
        for name, cfg in cfgs.items():
            if name == "full":
                continue
            lc = forward_cfg(Rn, states[name], ids, cfg, P, Rw, G, Hm)[0]
            qc = F.log_softmax(lc, -1)
            kl = (pf * (qf - qc)).sum(-1)
            agree = (lc.argmax(-1) == top_f).float()
            dnll = -qc.gather(1, tgt[:, None])[:, 0] - nll_f
            del lc, qc
            for lo, hi in kf.BUCKETS:
                m = (j >= max(lo, score_from)) & (j < hi)
                if m.any():
                    b = acc[name][f"{lo}-{hi}"]
                    b[0] += int(m.sum()); b[1] += kl[m].sum().item(); b[2] += agree[m].sum().item(); b[3] += dnll[m].sum().item()
        del qf, pf
    return acc, nll


@torch.no_grad()
def smoke(Rn, pr, gen, Hm, Rs=256, G=128, N=1024, chunk=256):
    """Gates: (0) fade_masks vs the fp64 per-query oracle (< 1e-10); (1) full path vs the HF forward (reported; must be
    exact in --tiny fp32 up to 1e-4 abs) and kv16 == full bit-identically; (2) blocks 0-1 (computed before anything ages)
    of every quantized layer equal quant(full's cache) bit-exactly after the chunked run, unquantized layers equal full's;
    (3) step vs chunked KL for full (numerics baseline) and kv2, reported."""
    ok = True
    e0 = max(kf.oracle_masks(Rn.device, chunk=c, Rw=r) for c, r in ((256, 256), (128, 256), (256, 512), (128, 128)))
    ok &= e0 < 1e-10
    print(f"smoke 0: fade masks vs per-query oracle (fp64): max abs err {e0:.2e} (must be < 1e-10)", flush=True)
    P, g = len(pr), torch.tensor(gen[:N + 1], device=Rn.device)
    cf = {k: v for k, v in configs(len(Rn.layers)).items() if k in ("full", "kv16", "kv2", "kv2_every4", "kv2_had")}
    st = new_states(Rn, pr, N, cf)
    L = {n: [] for n in cf}
    for s0 in range(0, N, chunk):
        for n, c in cf.items():
            L[n].append(forward_cfg(Rn, st[n], g[None, s0:s0 + chunk], c, P, Rs, G, Hm)[0])
    L = {n: torch.cat(v) for n, v in L.items()}
    ref = Rn.model(torch.cat([torch.tensor(pr, device=Rn.device), g[:N]])[None]).logits[0, P:].float()
    d_ref = (L["full"] - ref).abs().max().item()
    d16 = (L["kv16"] - L["full"]).abs().max().item()
    ok &= d16 == 0 and (d_ref < 1e-4 if a.tiny else True)
    print(f"smoke 1: max|dlogit| full path vs HF forward {d_ref:.3g} (tiny fp32: must be < 1e-4); KL mean "
          f"{kf.kl_mean(ref, L['full']).mean().item():.2e}; kv16 vs full {d16} (must be 0)", flush=True)
    for n in ("kv2", "kv2_every4", "kv2_had"):
        c, worst = cf[n], 0.0
        for li in range(len(Rn.layers)):
            lo, hi = P, P + 2 * G
            Kf, Vf = st["full"].K[li][:, :, lo:hi].float(), st["full"].V[li][:, :, lo:hi].float()
            ek = kf.quant_k(Kf, c["kb"][li], c["had"], Hm, G).to(Kf.dtype if c["kb"][li] >= 16 else st[n].K[li].dtype)
            ev = kf.quant_v(Vf, c["vb"][li], c["had"], Hm)
            dk = (st[n].K[li][:, :, lo:hi].float() - ek.to(st[n].K[li].dtype).float()).abs().max().item()
            dv = (st[n].V[li][:, :, lo:hi].float() - ev.to(st[n].V[li].dtype).float()).abs().max().item()
            worst = max(worst, dk, dv)
        ok &= worst == 0
        print(f"smoke 2 {n}: blocks 0-1 vs quant(full cache), all {len(Rn.layers)} layers: max abs diff {worst} (must be 0); "
              f"quantized layers {[li for li in range(len(Rn.layers)) if c['kb'][li] < 16]}", flush=True)
    for n in ("full", "kv2"):
        s1 = new_states(Rn, pr, N, {n: cf[n]})[n]
        ls = torch.stack([forward_cfg(Rn, s1, g[None, i:i + 1], cf[n], P, Rs, G, Hm)[0, 0] for i in range(N)])
        k = kf.kl_mean(ls, L[n])[Rs + G:]
        print(f"smoke 3 {n}: j >= {Rs + G}: KL(step || chunked) mean {k.mean().item():.2e} max {k.max().item():.2e}; "
              f"KL(full || {n}) mean {kf.kl_mean(L['full'], L[n])[Rs + G:].mean().item():.2e}", flush=True)
    print("SMOKE", "PASS" if ok else "FAIL", flush=True)


def tiny_model():
    from transformers import Qwen3Config
    torch.manual_seed(0)
    c = Qwen3Config(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=128, max_position_embeddings=4096, tie_word_embeddings=False)
    c._attn_implementation = "sdpa"
    return AutoModelForCausalLM.from_config(c, dtype=torch.float32).eval()


def main():
    if a.tiny:
        Rn = Dense(tiny_model())
        gen = torch.Generator().manual_seed(1)
        pr = torch.randint(0, 512, (37,), generator=gen).tolist()
        smoke(Rn, pr, torch.randint(0, 512, (1100,), generator=gen).tolist(), kf.hadamard(128, Rn.device))
        return
    assert a.chunk % a.G == 0 and a.chunk <= a.R and a.R % a.G == 0
    tok, tref = AutoTokenizer.from_pretrained(a.model), AutoTokenizer.from_pretrained(a.ref_model)
    probs = {p["problem_id"]: p for l in open(a.problems) if l.strip() for p in [json.loads(l)]}
    test = set(l.strip() for l in open(a.test_ids) if l.strip())
    rows = kf.load_rows(test)
    print(f"selected {len(rows)} traces (as kv_fade.py{', from ' + a.rows if a.rows else ''})", flush=True)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    Rn = Dense(m)
    Hm = kf.hadamard(Rn.cfg.head_dim, Rn.device)
    print(f"{a.model}: {len(Rn.layers)} attention layers, kv heads {Rn.cfg.num_key_value_heads}, head_dim {Rn.cfg.head_dim}; "
          f"R {a.R}, G {a.G}, chunk {a.chunk}", flush=True)

    def texts(r):
        pr = kf.prompt_ids(tok, probs[r["problem_id"]])
        gen = tok("<think>\n" + tref.decode(r["output_ids"][:a.max_gen]), add_special_tokens=False)["input_ids"]
        return pr, gen

    if a.smoke:
        pr, gen = texts(rows[0])
        smoke(Rn, pr, gen, Hm)
        return
    allc = configs(len(Rn.layers))
    names = a.configs.split(",") if a.configs else ["w2k", "w8k", "kv4", "kv2", "kv1", "kv2_every4", "kv2_had"]
    cfgs = {"full": allc["full"], **{k: allc[k] for k in names}}
    print("configs", json.dumps(cfgs), flush=True)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["problem_id"] for l in open(a.out) if l.strip()}
    for i, r in enumerate(rows):
        if i % a.nshards != a.shard or r["problem_id"] in done:
            continue
        pr, gen = texts(r)
        acc, nll = run_trace(Rn, pr, gen, cfgs, a.chunk, Hm, a.R, a.G, a.score_from)
        with open(a.out, "a") as f:
            f.write(json.dumps({"problem_id": r["problem_id"], "sample_idx": r["sample_idx"], "model": a.model,
                                "prompt_len": len(pr), "n_gen": len(gen), "n_gen_ref": min(a.max_gen, len(r["output_ids"])),
                                "R": a.R, "G": a.G, "score_from": a.score_from, "configs": cfgs, "acc": acc, "full_nll": nll,
                                "chars": chars_by_bucket(tok, gen, kf.BUCKETS, a.score_from)}) + "\n")
        n = {c: sum(v[0] for v in acc[c].values()) for c in acc}
        print(f"{r['problem_id']} n_gen {len(gen)}: KL/tok " + ", ".join(
            f"{c} {sum(v[1] for v in acc[c].values()) / max(1, n[c]):.4f}" for c in acc), flush=True)
    print("FADE_DENSE_DONE", flush=True)


if __name__ == "__main__":
    main()
