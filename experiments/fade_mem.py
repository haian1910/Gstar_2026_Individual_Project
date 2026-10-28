"""Fade with a truly packed KV store: real GPU memory + decode speed of "Fade" (quantized old FA KV) on Qwen3.5-9B, and a
check that the packed values are bit-identical to the emulated quantizer of fade_q.py (so the decode accuracies measured with
the emulation carry over). HA_own, see ../../2026-10-28_fade/PLAN.md.

README
------
Policy (as fade_q.age_rows / tl_gen fq cell): prompt exact; generated tokens exact for R=2048 .. R+G-1 steps; then each block of
G=128 generated tokens is quantized once (irreversibly) right before the step that feeds generated index g = (m+1)G-1+R.
At g0 generated tokens fed: n_quantized = G * floor((g0 - R) / G) (>= 0), n_exact_generated = g0 - n_quantized in [R, R+G-1]
once g0 >= R. The bench derives n_quantized by running fade_q.age_rows' own schedule and asserts the store holds exactly that.
Packed store per FA layer (PackedLayer):
  exact part  bf16 [B, KVH, P + R + G, D]: the prompt + a ring of R+G slots (slot of generated index j = P + j % (R+G); the
              ring is a multiple of G so a block never wraps). Slots of aged blocks are masked out (they are stale copies).
  old part    uint8 codes bit-packed along D (kv4: 2 codes/byte, kv2: 4/byte, kv1: 8/byte) [B, KVH, cap, D*b/8], preallocated
              for the context of the config (g0 + steps + warm-up + G slack), + fp16 scale and zero with fade_q's grouping:
              K plain = per channel per block [B,KVH,cap/G,D]; K had / V = per token per group of GD=128 along D [B,KVH,cap,D/128].
  dequant     code.float() * s + z (then @ Hm for the Hadamard variant), then .to(bf16): the same ops as fade_q.qdq.
Decode (T=1) attention (PackedLayer.attend): exact part and old part (dequantized in chunks of --chunk=2048 tokens, never a
  full-layer bf16 copy) merged with an online softmax (fp32 running max / sum / acc). GQA folded into the query axis and the
  output gate as Runner._fa. PackedRunner subclasses fa_loop.Runner; prefill uses the normal path on a temporary full State,
  then PackedState.from_base converts (blocks quantized exactly as fade_q would have, one [1,KVH,G,D] slice at a time).
  A "no-op" variant (noop=True: aged blocks kept in bf16 in the old part, unquantized, attended through the same chunked
  online-softmax path) gives the bf16 noise floor of the check.
Usage (GPU; run --check first, it exits non-zero on failure):
  python fade_mem.py --check [--model Qwen/Qwen3.5-9B]            (tee to out_fade/mem_check.log)
  python fade_mem.py --bench --cells full,tl-8192,sl-8192,fq-kv4,fq-kv2_had --ctx 8192,16384,32768 --batch 1,8,32 \
         --steps 128 --out fade_mem.jsonl
Cells: full | tl-W | sl-W | fq-NAME (PACKED fade, this file) | fqe-NAME (emulated fade: bf16 cache + fade_q.age_rows, for
  comparison). --ctx is the TOTAL context L = P + g0 at the start of decoding (add --ctx-is-gen to read it as g0 generated
  tokens, so L = P + ctx). Per (cell, L, B) one JSON line (OOM -> {"oom": true}); resumable: lines already in --out are skipped.
  Fields: problem_id, P, L, g0, B, steps, warmup; capacities in tokens (cap_exact_slots, cap_old_slots, slack_tokens);
  n_exact_tokens (generated, still exact), n_quantized_tokens, n_exact_total (incl. prompt); weights_bytes (memory_allocated
  right after model load), weights_numel_bytes; resident_allocated (after building the state, before decoding),
  state_alloc_bytes (= resident_allocated - weights_bytes), peak_allocated, reserved_at_peak (max_memory_reserved),
  numel-analytic fa_exact_bytes, fa_packed_codes_bytes, fa_scales_bytes, gdn_bytes, other_bytes (valid masks), state_bytes
  (their sum); ms_per_step, tok_s.
What is measured: prefill one real row (first row of out/traces.jsonl with prompt_len + len(output_ids) >= L; prompt rebuilt
  from out/problems.jsonl with the chat template as kv_fade.py, prompt_len asserted) at full precision on a temporary B=1
  State, converted to the cell's representation, replicated to B rows, then 4 untimed warm-up + --steps timed steps incl.
  fa_loop.sample (per-row seeds, history = the real trace) and a host sync per step (as tl_gen). The first-step logits are the
  full-precision prefill logits for every cell. Capacity is per config (not a global 32K): full/fqe Lmax = L + steps + warm-up
  + G; tl = P + W + 16, sl = sinks + W + 16 (as tl_gen); fq exact P + R + G slots + packed old slots for g0 + steps + warm-up + G.
Known limitations
  * Lockstep batch: all rows share P and the generated index g (true in the bench and the check; continuous batching with
    per-row offsets as tl_gen would need per-row block indices).
  * Attention for the packed path is plain PyTorch (fp32 matmuls on chunks, unfused dequant): it is slower than a fused
    dequant-attention kernel would be; ms/step is an upper bound for Fade, memory is the real resident/peak number.
  * Packed vs emulated path agree bit-for-bit on the stored values (check i); logits differ at bf16-noise level because the
    attention kernels differ (check ii compares to that floor). Per-row Hadamard quantization is looped per row so that the
    matmul shape equals fade_q's [1,KVH,G,D]; dequant of a whole chunk is one matmul (check i tests it is bit-equal, and
    --dequant-per-block falls back to fade_q's exact per-block shape if it is not).
  * Weights include the vision tower that Qwen3.5-9B ships with (not used).
"""
import argparse
import gc
import json
import os
import sys
import types
import time

os.environ.setdefault("USE_HUB_KERNELS", "0")
import torch
import torch.nn.functional as F

from fa_loop import Runner, State, sample, _apply_rope
import fade_q
from fade_q import GD

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


# --- bit packing -------------------------------------------------------------------------------------------------------
def pack(c, b):
    """c uint8 codes [..., D] with values < 2**b -> uint8 [..., D * b // 8] (code i of each group of 8//b in bits b*i)."""
    assert b in (1, 2, 4, 8)
    per = 8 // b
    c = c.reshape(*c.shape[:-1], c.shape[-1] // per, per)
    out = torch.zeros(c.shape[:-1], dtype=torch.uint8, device=c.device)
    for i in range(per):
        out |= (c[..., i] << (b * i))
    return out


def unpack(p, b):
    per = 8 // b
    sh = torch.arange(per, device=p.device, dtype=torch.uint8) * b
    return ((p.unsqueeze(-1) >> sh) & (2 ** b - 1)).reshape(*p.shape[:-1], p.shape[-1] * per)


# --- quantizer returning codes (same math as fade_q.qdq / quant_k / quant_v) -------------------------------------------
def _codes(x, b, dim):
    L = 2 ** b - 1
    mn, mx = x.amin(dim, keepdim=True), x.amax(dim, keepdim=True)
    s = ((mx - mn) / L).half().float()
    z = mn.half().float()
    s = torch.where(s > 0, s, torch.ones_like(s))
    c = ((x - z) / s).round().clamp(0, L)
    return c.to(torch.uint8), s.half(), z.half()  # s, z are fp16-representable: the casts are lossless


def quantk_codes(k, b, had, Hm, G):
    """k [B,KVH,n,D] fp32, n % G == 0 -> codes uint8 [B,KVH,n,D], s, z fp16 ([B,KVH,n//G,D] plain; [B,KVH,n,D//GD] had)."""
    B, KVH, n, D = k.shape
    if had:
        y = k @ Hm
        c, s, z = _codes(y.view(B, KVH, n, D // GD, GD), b, -1)
        return c.view(B, KVH, n, D), s.squeeze(-1), z.squeeze(-1)
    c, s, z = _codes(k.view(B, KVH, n // G, G, D), b, 3)
    return c.view(B, KVH, n, D), s.squeeze(3), z.squeeze(3)


def quantv_codes(v, b, had, Hm):
    B, KVH, n, D = v.shape
    y = v @ Hm if had else v
    c, s, z = _codes(y.view(B, KVH, n, D // GD, GD), b, -1)
    return c.view(B, KVH, n, D), s.squeeze(-1), z.squeeze(-1)


def dequantk(c, s, z, had, Hm, G):
    """c uint8 [B,KVH,n,D] (unpacked) -> fp32 (caller casts to bf16). Same ops as fade_q.quant_k's dequant."""
    B, KVH, n, D = c.shape
    if had:
        y = c.view(B, KVH, n, D // GD, GD).float() * s.float().unsqueeze(-1) + z.float().unsqueeze(-1)
        return y.view(B, KVH, n, D) @ Hm
    y = c.view(B, KVH, n // G, G, D).float() * s.float().unsqueeze(3) + z.float().unsqueeze(3)
    return y.view(B, KVH, n, D)


def dequantv(c, s, z, had, Hm):
    B, KVH, n, D = c.shape
    y = c.view(B, KVH, n, D // GD, GD).float() * s.float().unsqueeze(-1) + z.float().unsqueeze(-1)
    y = y.view(B, KVH, n, D)
    return y @ Hm if had else y


def _per_row(fn, x, had):
    """The Hadamard matmul is looped per row so its shape is fade_q's [1,KVH,G,D]; elementwise/min/max are batch invariant."""
    if had and x.shape[0] > 1:
        outs = [fn(x[i:i + 1]) for i in range(x.shape[0])]
        return tuple(torch.cat(t, 0) for t in zip(*outs))
    return fn(x)


def policy_nq(g0, R, G):
    """Blocks quantized before the step feeding generated index g0, derived from fade_q.age_rows' own schedule: run it for
    g = 0 .. g0-1 on a stateless dummy (no FA layers, so it only reports which block ages when)."""
    dummy = types.SimpleNamespace(K=[], V=[])
    cfg = fade_q.CONFIGS["kv4"]
    n = 0
    for g in range(g0):
        n += len(fade_q.age_rows(dummy, [(0, 0, g)], cfg, None, R, G))
    return n


# --- packed layer -----------------------------------------------------------------------------------------------------
class PackedLayer:
    def __init__(self, B, KVH, D, P, R, G, gen_cap, kb, vb, had, Hm, device, dtype=torch.bfloat16, per_block=False, noop=False):
        """gen_cap: number of generated tokens the store must be able to hold (sizes the old part). noop: keep aged blocks in bf16
        (no quantization) in the old part: the noise-floor variant of --check."""
        assert R % G == 0 and D % GD == 0 and kb in (1, 2, 4, 8) and vb in (1, 2, 4, 8)
        self.B, self.KVH, self.D, self.P, self.R, self.G = B, KVH, D, P, R, G
        self.kb, self.vb, self.had, self.Hm, self.per_block, self.dtype, self.noop = kb, vb, had, Hm, per_block, dtype, noop
        self.Le = P + R + G
        self.ncap = max(0, (gen_cap - R) // G)  # blocks that can be quantized by the last step (g = gen_cap - 1)
        cap = self.ncap * G
        self.cap = cap
        self.Ke = torch.zeros(B, KVH, self.Le, D, device=device, dtype=dtype)
        self.Ve = torch.zeros(B, KVH, self.Le, D, device=device, dtype=dtype)
        self.valid = torch.zeros(self.Le, dtype=torch.bool, device=device)
        c2 = 0 if noop else cap  # a noop layer keeps bf16 blocks in Kraw / Vraw instead of codes
        n2 = 0 if noop else self.ncap
        self.Kc = torch.zeros(B, KVH, c2, D * kb // 8, dtype=torch.uint8, device=device)
        self.Vc = torch.zeros(B, KVH, c2, D * vb // 8, dtype=torch.uint8, device=device)
        ks = (B, KVH, c2, D // GD) if had else (B, KVH, n2, D)
        self.Ks = torch.zeros(ks, dtype=torch.float16, device=device)
        self.Kz = torch.zeros(ks, dtype=torch.float16, device=device)
        self.Vs = torch.zeros(B, KVH, c2, D // GD, dtype=torch.float16, device=device)
        self.Vz = torch.zeros(B, KVH, c2, D // GD, dtype=torch.float16, device=device)
        self.Kraw = torch.zeros(B, KVH, cap if noop else 0, D, device=device, dtype=dtype)
        self.Vraw = torch.zeros(B, KVH, cap if noop else 0, D, device=device, dtype=dtype)
        self.nq = 0  # blocks quantized so far

    # storage -------------------------------------------------------------------------------------------------------
    def tensors(self):
        return dict(Ke=self.Ke, Ve=self.Ve, Kc=self.Kc, Vc=self.Vc, Ks=self.Ks, Kz=self.Kz, Vs=self.Vs, Vz=self.Vz,
                    Kraw=self.Kraw, Vraw=self.Vraw)

    def nbytes(self):
        """analytic (numel * element_size) bytes: exact bf16, packed codes, scales/zeros (+ raw bf16 old blocks of a noop layer
        counted as exact)."""
        t = self.tensors()
        sz = lambda ks: sum(t[k].numel() * t[k].element_size() for k in ks)
        return dict(fa_exact=sz(("Ke", "Ve", "Kraw", "Vraw")), fa_packed_codes=sz(("Kc", "Vc")),
                    fa_scales=sz(("Ks", "Kz", "Vs", "Vz")), other=self.valid.numel())

    def slot(self, j):
        return self.P + j % (self.R + self.G)

    def _store_block(self, m, kblk, vblk):
        """Quantize the exact block m (bf16 [B,KVH,G,D]) into the old part."""
        G = self.G
        sl = slice(m * G, (m + 1) * G)
        if self.noop:
            self.Kraw[:, :, sl], self.Vraw[:, :, sl] = kblk, vblk
            return
        kq = _per_row(lambda x: quantk_codes(x.float(), self.kb, self.had, self.Hm, G), kblk, self.had)
        vq = _per_row(lambda x: quantv_codes(x.float(), self.vb, self.had, self.Hm), vblk, self.had)
        self.Kc[:, :, sl] = pack(kq[0], self.kb)
        self.Vc[:, :, sl] = pack(vq[0], self.vb)
        ks = sl if self.had else slice(m, m + 1)
        self.Ks[:, :, ks], self.Kz[:, :, ks] = kq[1], kq[2]
        self.Vs[:, :, sl], self.Vz[:, :, sl] = vq[1], vq[2]

    def age(self, g):
        """Called before the step feeding generated index g (as fade_q.age_rows): quantize the block that ages now, if any."""
        e = g - self.R + 1
        if e < self.G or e % self.G:
            return False
        m = e // self.G - 1
        assert m == self.nq and m < self.ncap, (m, self.nq, self.ncap)
        s0 = self.P + (m * self.G) % (self.R + self.G)
        self._store_block(m, self.Ke[:, :, s0:s0 + self.G], self.Ve[:, :, s0:s0 + self.G])
        self.valid[s0:s0 + self.G] = False
        self.nq += 1
        return True

    def append(self, g, k, v):
        """k, v [B,KVH,1,D]: the K/V of generated index g (after aging at g)."""
        i = self.slot(g)
        self.Ke[:, :, i] = k[:, :, 0]
        self.Ve[:, :, i] = v[:, :, 0]
        self.valid[i] = True

    def load(self, Kf, Vf, g0):
        """B == 1: fill from the exact prefill K/V [1,KVH,>=P+g0,D] (g0 generated tokens already fed) as the policy would hold
        them before the step feeding index g0: blocks 0 .. nq-1 quantized, nq = max(0, (g0 - R) // G)."""
        assert self.B == 1
        P, G, R = self.P, self.G, self.R
        dev = self.Ke.device
        Kf, Vf = Kf[:, :, :P + g0].to(dev), Vf[:, :, :P + g0].to(dev)
        self.Ke[:, :, :P], self.Ve[:, :, :P] = Kf[:, :, :P], Vf[:, :, :P]
        self.valid[:P] = True
        nq = max(0, (g0 - R) // G)
        assert nq <= self.ncap
        for m in range(nq):
            self._store_block(m, Kf[:, :, P + m * G:P + (m + 1) * G], Vf[:, :, P + m * G:P + (m + 1) * G])
        self.nq = nq
        j = torch.arange(nq * G, g0, device=dev)
        sl = P + j % (R + G)
        self.Ke[:, :, sl], self.Ve[:, :, sl] = Kf[:, :, P + j], Vf[:, :, P + j]
        self.valid[sl] = True

    def n_exact_generated(self):
        return int(self.valid[self.P:].sum())

    def to_batch(self, B):
        assert self.B == 1
        o = object.__new__(PackedLayer)
        o.__dict__.update(self.__dict__)
        o.B = B
        for k, t in self.tensors().items():
            setattr(o, k, t.expand(B, *t.shape[1:]).clone())
        o.valid = self.valid.clone()
        return o

    # dequant (old part, tokens [t0, t1), t0 and t1 multiples of G) ---------------------------------------------------
    def _deq(self, t0, t1, which):
        G = self.G
        if self.noop:
            return (self.Kraw if which == "k" else self.Vraw)[:, :, t0:t1]
        if which == "k":
            c = unpack(self.Kc[:, :, t0:t1], self.kb)
            ssl = slice(t0, t1) if self.had else slice(t0 // G, t1 // G)
            s, z = self.Ks[:, :, ssl], self.Kz[:, :, ssl]
            f = lambda c_, s_, z_: dequantk(c_, s_, z_, self.had, self.Hm, G)
        else:
            c = unpack(self.Vc[:, :, t0:t1], self.vb)
            s, z = self.Vs[:, :, t0:t1], self.Vz[:, :, t0:t1]
            f = lambda c_, s_, z_: dequantv(c_, s_, z_, self.had, self.Hm)
        if not self.per_block:
            return f(c, s, z).to(self.dtype)
        outs = []  # fade_q's exact shapes: one [1,KVH,G,D] block of one row at a time
        for r in range(c.shape[0]):
            row = []
            for a in range(0, t1 - t0, G):
                ss = slice(a, a + G) if (which == "v" or self.had) else slice(a // G, a // G + 1)
                row.append(f(c[r:r + 1, :, a:a + G], s[r:r + 1, :, ss], z[r:r + 1, :, ss]).to(self.dtype))
            outs.append(torch.cat(row, 2))
        return torch.cat(outs, 0)

    def deqk(self, t0, t1):
        return self._deq(t0, t1, "k")

    def deqv(self, t0, t1):
        return self._deq(t0, t1, "v")

    # attention -------------------------------------------------------------------------------------------------------
    def attend(self, qg, scale, chunk=2048):
        """qg [B,KVH,Gq,D] (T=1, GQA folded into the query axis) -> fp32 [B,KVH,Gq,D]; softmax over exact + old tokens with
        an online merge: no full-layer bf16 copy of the cache is ever built."""
        assert chunk % self.G == 0
        qf = qg.float()
        B, KVH, Gq, D = qf.shape
        m = torch.full((B, KVH, Gq, 1), float("-inf"), device=qf.device)
        l = torch.zeros(B, KVH, Gq, 1, device=qf.device)
        acc = torch.zeros(B, KVH, Gq, D, device=qf.device)

        def merge(m, l, acc, K, V, valid):
            s = torch.matmul(qf, K.float().transpose(-1, -2)) * scale  # [B,KVH,Gq,n] fp32
            if valid is not None:
                s = s.masked_fill(~valid, float("-inf"))
            mn = torch.maximum(m, s.amax(-1, keepdim=True))
            mn_s = torch.where(torch.isinf(mn), torch.zeros_like(mn), mn)
            p = torch.exp(s - mn_s)
            alpha = torch.exp(m - mn_s)
            return mn, l * alpha + p.sum(-1, keepdim=True), acc * alpha + torch.matmul(p, V.float())

        for a in range(0, self.Le, chunk):
            b = min(self.Le, a + chunk)
            m, l, acc = merge(m, l, acc, self.Ke[:, :, a:b], self.Ve[:, :, a:b], self.valid[a:b])
        for a in range(0, self.nq * self.G, chunk):
            b = min(self.nq * self.G, a + chunk)
            m, l, acc = merge(m, l, acc, self.deqk(a, b), self.deqv(a, b), None)
        return acc / l


class PackedState:
    def __init__(self, layers, conv, rec, pos_next, g, P, chunk=2048):
        self.layers, self.conv, self.rec, self.pos_next, self.g, self.P, self.chunk = layers, conv, rec, pos_next, g, P, chunk
        self.B = layers[0].B

    @classmethod
    def from_base(cls, base, cfg, Hm, R, G, gen_cap, chunk=2048, per_block=False, device=None, noop=False):
        """base: .K/.V lists [1,KVH,>=L,D], .conv/.rec lists (B=1), .L context length, .P prompt length."""
        assert cfg["kind"] == "q" and not cfg["pre"] and min(cfg["kb"] + cfg["vb"]) < 16
        dev = device or base.K[0].device
        g0 = base.L - base.P
        layers = []
        for fi in range(len(base.K)):
            KVH, D = base.K[fi].shape[1], base.K[fi].shape[3]
            lay = PackedLayer(1, KVH, D, base.P, R, G, gen_cap, cfg["kb"][fi], cfg["vb"][fi], cfg["had"], Hm, dev,
                              per_block=per_block, noop=noop)
            lay.load(base.K[fi], base.V[fi], g0)
            layers.append(lay)
        return cls(layers, [c.to(dev) for c in base.conv], [r.to(dev) for r in base.rec],
                   torch.tensor([base.L], device=dev), g0, base.P, chunk)

    def to_batch(self, B):
        return PackedState([l.to_batch(B) for l in self.layers], [c.expand(B, *c.shape[1:]).clone() for c in self.conv],
                           [r.expand(B, *r.shape[1:]).clone() for r in self.rec], self.pos_next.expand(B).clone(), self.g,
                           self.P, self.chunk)

    def age(self):
        return [l.age(self.g) for l in self.layers][0]

    def sizes(self):
        tot = dict(fa_exact=0, fa_packed_codes=0, fa_scales=0, other=0)
        for l in self.layers:
            for k, v in l.nbytes().items():
                tot[k] += v
        gdn = sum(t.numel() * t.element_size() for t in self.conv + self.rec)
        return dict(fa_exact_bytes=tot["fa_exact"], fa_packed_codes_bytes=tot["fa_packed_codes"], fa_scales_bytes=tot["fa_scales"],
                    gdn_bytes=gdn, other_bytes=tot["other"], state_bytes=sum(tot.values()) + gdn)


class Base:
    """Prefill result of one row (B=1), K/V/GDN tensors trimmed to the L tokens fed."""

    def __init__(self, K, V, conv, rec, L, P, logits=None, gen_ids=None):
        self.K, self.V, self.conv, self.rec, self.L, self.P, self.logits, self.gen_ids = K, V, conv, rec, L, P, logits, gen_ids

    @classmethod
    def from_state(cls, st, L, P, to_cpu=False, **kw):
        f = (lambda t: t.cpu()) if to_cpu else (lambda t: t.clone())
        return cls([f(k[:, :, :L]) for k in st.K], [f(v[:, :, :L]) for v in st.V], [f(c) for c in st.conv],
                   [f(r) for r in st.rec], L, P, **kw)


class PackedRunner(Runner):
    """Runner whose forward also accepts a PackedState (T=1 decode); any other State goes to Runner.forward."""

    @torch.no_grad()
    def forward(self, st, ids, pos=None, logits="last", slot=None, lcur=None):
        if not isinstance(st, PackedState):
            return super().forward(st, ids, pos, logits, slot, lcur)
        assert ids.shape[1] == 1 and slot is None
        st.age()
        pos = st.pos_next[:, None]
        h = self.lm.embed_tokens(ids)
        cos, sin = self.lm.rotary_emb(h, pos)
        for li, layer in enumerate(self.layers):
            r = h
            x = layer.input_layernorm(h)
            if li in self.gdn_of:
                x = self._gdn(layer.linear_attn, x, st, self.gdn_of[li])
            else:
                x = self._fa_packed(layer.self_attn, x, cos, sin, st, self.fa_of[li])
            h = r + x
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        st.g += 1
        st.pos_next = pos[:, -1] + 1
        h = self.lm.norm(h[:, -1:])
        out = self.lm_head(h).float()
        return out[:, -1]

    def _fa_packed(self, sa, x, cos, sin, st, fi):
        B, T, _ = x.shape
        D = sa.head_dim
        q, gate = torch.chunk(sa.q_proj(x).view(B, T, -1, D * 2), 2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = sa.q_norm(q).transpose(1, 2)
        k = sa.k_norm(sa.k_proj(x).view(B, T, -1, D)).transpose(1, 2)
        v = sa.v_proj(x).view(B, T, -1, D).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        lay = st.layers[fi]
        lay.append(st.g, k, v)
        H, KVH = q.shape[1], k.shape[1]
        qg = q.reshape(B, KVH, (H // KVH) * T, D)  # GQA folded as Runner._fa
        o = lay.attend(qg, sa.scaling, st.chunk).to(x.dtype)
        o = o.reshape(B, H, T, D).transpose(1, 2).reshape(B, T, -1)
        o = o * torch.sigmoid(gate)
        return sa.o_proj(o)


# --- equivalence of stored values (shared by --check and the CPU unit test) --------------------------------------------
def roundtrip_equal(device, cfgname, R=256, G=128, g0=1300, P=37, seed=0, chunks=(128, 384, 2048), B=2, log=print):
    """Random bf16 prefill K/V -> PackedLayer.load -> chunked dequant (B rows) == fade_q applied per [1,KVH,G,D] block, torch.equal.
    Also checks the exact ring part and that per-block dequant equals chunked dequant. Returns True if all equal."""
    cfg = fade_q.CONFIGS[cfgname]
    Hm = fade_q.hadamard(256, device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = P + g0
    Kf = (torch.randn(1, 4, n, 256, generator=gen) * 3).to(torch.bfloat16).to(device)
    Vf = (torch.randn(1, 4, n, 256, generator=gen) * 0.5 + 0.2).to(torch.bfloat16).to(device)
    Kf[:, :, P + 5, :] = 1.25  # degenerate group (max == min): scale falls back to 1
    nq = (g0 - R) // G
    ok = True
    for per_block in (False, True):
        lay = PackedLayer(1, 4, 256, P, R, G, g0 + 64, cfg["kb"][0], cfg["vb"][0], cfg["had"], Hm, device, per_block=per_block)
        lay.load(Kf, Vf, g0)
        lb = lay.to_batch(B)
        assert lb.nq == nq
        for ch in chunks:
            for a in range(0, nq * G, ch):
                b = min(nq * G, a + ch)
                dk, dv = lb.deqk(a, b), lb.deqv(a, b)
                for m in range(a // G, b // G):
                    lo = P + m * G
                    ek = fade_q.quant_k(Kf[:, :, lo:lo + G].float(), cfg["kb"][0], cfg["had"], Hm, G).to(torch.bfloat16)
                    ev = fade_q.quant_v(Vf[:, :, lo:lo + G].float(), cfg["vb"][0], cfg["had"], Hm).to(torch.bfloat16)
                    s = slice((m - a // G) * G, (m - a // G + 1) * G)
                    for r in range(B):
                        ok_k, ok_v = torch.equal(dk[r:r + 1, :, s], ek), torch.equal(dv[r:r + 1, :, s], ev)
                        if not (ok_k and ok_v):
                            log(f"  MISMATCH {cfgname} per_block={per_block} chunk={ch} block={m} row={r} K {ok_k} V {ok_v}")
                        ok &= ok_k and ok_v
    j = torch.arange(nq * G, g0, device=device)
    ok &= torch.equal(lay.Ke[:, :, P + j % (R + G)], Kf[:, :, P + j]) and torch.equal(lay.Ke[:, :, :P], Kf[:, :, :P])
    return bool(ok)


def random_block_equal(device, cfgname, seed=0, log=print):
    """Random fp32 [1,4,128,256] blocks (several scales): pack -> unpack -> dequant == fade_q.quant_k/quant_v .to(bf16)."""
    cfg = fade_q.CONFIGS[cfgname]
    Hm = fade_q.hadamard(256, device)
    G, kb, vb, had = 128, cfg["kb"][0], cfg["vb"][0], cfg["had"]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ok = True
    for sc in (0.01, 1.0, 50.0):
        x = (torch.randn(1, 4, G, 256, generator=gen) * sc + sc * 0.3).to(device)
        for kind in ("k", "v"):
            if kind == "k":
                c, s, z = quantk_codes(x, kb, had, Hm, G)
                d = dequantk(unpack(pack(c, kb), kb), s, z, had, Hm, G).to(torch.bfloat16)
                e = fade_q.quant_k(x, kb, had, Hm, G).to(torch.bfloat16)
            else:
                c, s, z = quantv_codes(x, vb, had, Hm)
                d = dequantv(unpack(pack(c, vb), vb), s, z, had, Hm).to(torch.bfloat16)
                e = fade_q.quant_v(x, vb, had, Hm).to(torch.bfloat16)
            good = torch.equal(d, e)
            if not good:
                log(f"  MISMATCH random {cfgname} {kind} scale {sc}")
            ok &= good
    return bool(ok)


# --- model / data helpers ----------------------------------------------------------------------------------------------
def load_model(name):  # as tl_gen.load_model
    from transformers import AutoModelForImageTextToText
    try:
        m = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16)
    except Exception:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    return m.cuda().eval()


def prompt_ids(tok, p):  # as kv_fade.prompt_ids
    x = tok.apply_chat_template([{"role": "user", "content": p["problem"] + p.get("suffix", SUFFIX)}],
                                add_generation_prompt=True, enable_thinking=True, tokenize=True)
    return x if isinstance(x, list) else x["input_ids"]


def pick_row(traces, problems, tok, min_total, min_gen=0):
    """First row of traces.jsonl (file order) with prompt_len + len(output_ids) >= min_total and >= min_gen output tokens.
    The prompt ids are rebuilt from problems.jsonl (traces do not store them); prompt_len is asserted."""
    probs = {p["problem_id"]: p for p in map(json.loads, (l for l in open(problems) if l.strip()))}
    for l in open(traces):
        r = json.loads(l)
        if r["prompt_len"] + len(r["output_ids"]) >= min_total and len(r["output_ids"]) >= min_gen and r["problem_id"] in probs:
            pr = prompt_ids(tok, probs[r["problem_id"]])
            assert len(pr) == r["prompt_len"], (r["problem_id"], len(pr), r["prompt_len"])
            return r, pr
    raise SystemExit(f"no trace row with prompt_len + n_out >= {min_total} (n_out >= {min_gen})")


def kl(lp, lq):
    """KL(p || q) of logits [1,V] -> float."""
    a, b = F.log_softmax(lp.double(), -1), F.log_softmax(lq.double(), -1)
    return float((a.exp() * (a - b)).sum())


# --- --check ----------------------------------------------------------------------------------------------------------
@torch.no_grad()
def check(a):
    from transformers import AutoTokenizer
    dev = torch.device("cuda")
    R, G = a.R, a.G
    ok = True
    print("== (i) stored values: dequant(packed) vs fade_q (torch.equal) ==", flush=True)
    for name in ("kv4", "kv2_had"):
        r1 = random_block_equal(dev, name)
        r2 = roundtrip_equal(dev, name, R=R, G=G, g0=R + 6 * G + 37, chunks=(G, 3 * G, 2048))
        print(f"  random fp32 blocks {name}: {'PASS' if r1 else 'FAIL'}; chunked/per-block dequant of B=2 store {name}: "
              f"{'PASS' if r2 else 'FAIL'}", flush=True)
        ok &= r1 and r2
    tok = AutoTokenizer.from_pretrained(a.model)
    row, pr = pick_row(a.traces, a.problems, tok, 0, min_gen=a.gen_end)
    gen = row["output_ids"][:a.gen_end]
    P = len(pr)
    print(f"row {row['problem_id']} sample {row['sample_idx']} prompt {P} gen {len(gen)} (prompt rebuilt from {a.problems})", flush=True)
    PR = PackedRunner(load_model(a.model))
    Hm = fade_q.hadamard(PR.cfg.head_dim, PR.device)
    g_i = a.g_real
    st = PR.new_state(1, P + g_i + 8)
    PR.prefill(st, torch.tensor([pr + gen[:g_i]], device=dev), logits=None)
    for name in ("kv4", "kv2_had"):
        cfg = fade_q.CONFIGS[name]
        base = Base.from_state(st, P + g_i, P)
        ps = PackedState.from_base(base, cfg, Hm, R, G, g_i + 16)
        em = PR.new_state(1, st.Lmax)
        em.copy_row_from(0, st)
        for g in range(R + G - 1, g_i, 1):
            fade_q.age_rows(em, [(0, P, g)], cfg, Hm, R, G)
        nq = ps.layers[0].nq
        eq = nq == policy_nq(g_i, R, G)
        for fi, lay in enumerate(ps.layers):
            n = nq * G
            ek, ev = em.K[fi][:, :, P:P + n], em.V[fi][:, :, P:P + n]
            e1 = torch.equal(lay.deqk(0, n), ek) and torch.equal(lay.deqv(0, n), ev)
            e2 = torch.equal(lay.Ke[:, :, :P], em.K[fi][:, :, :P])
            if not (e1 and e2):
                print(f"  MISMATCH real K/V {name} layer {fi}: old {e1} prompt {e2}", flush=True)
            eq &= e1 and e2
        print(f"  real prefill K/V ({P + g_i} tok, {nq} aged blocks x 8 layers) {name}: {'PASS' if eq else 'FAIL'}", flush=True)
        ok &= eq
        del ps, em, base
    del st
    torch.cuda.empty_cache()
    # (ii) teacher forcing
    g0, g1 = a.g_start, len(gen)
    print(f"== (ii) teacher-forced steps g={g0}..{g1 - 1} (R={R}, G={G}): packed vs emulated, noise floor vs full SDPA ==", flush=True)
    base_st = PR.new_state(1, P + g1 + 16)
    PR.prefill(base_st, torch.tensor([pr + gen[:g0]], device=dev), logits=None)
    base = Base.from_state(base_st, P + g0, P)
    names = ("kv4", "kv2_had")
    full = PR.new_state(1, base_st.Lmax)  # full-cache SDPA step decode
    full.copy_row_from(0, base_st)
    # no-op variant: old blocks stay bf16 (unquantized) but sit in the old part and are attended by the chunked online-softmax path
    noop = PackedState.from_base(base, fade_q.CONFIGS["kv4"], Hm, R, G, g1 + 16, chunk=a.chunk, noop=True)
    emu = {n: PR.new_state(1, base_st.Lmax) for n in names}
    for n in names:
        emu[n].copy_row_from(0, base_st)
    pk = {n: PackedState.from_base(base, fade_q.CONFIGS[n], Hm, R, G, g1 + 16, chunk=a.chunk, per_block=a.dequant_per_block)
          for n in names}
    del base_st
    acc = {k: dict(kl=0.0, mx=0.0, top1=0) for k in ("floor", "pk_kv4", "pk_kv2_had", "q_kv4", "q_kv2_had")}
    t0 = time.time()
    for g in range(g0, g1):
        ids = torch.tensor([[gen[g]]], device=dev)
        lf = PR.forward(full, ids)
        lc = PR.forward(noop, ids)
        rec = [("floor", lf, lc)]  # KL(full-cache SDPA || no-op chunked path)
        for n in names:
            fade_q.age_rows(emu[n], [(0, P, g)], fade_q.CONFIGS[n], Hm, R, G)
            le = PR.forward(emu[n], ids)
            lp = PR.forward(pk[n], ids)
            rec += [(f"pk_{n}", le, lp), (f"q_{n}", lf, le)]  # pk: KL(emulated || packed); q: quantization effect KL(full || emulated)
        for k, x, y in rec:
            acc[k]["kl"] += kl(x, y)
            acc[k]["mx"] = max(acc[k]["mx"], float((x - y).abs().max()))
            acc[k]["top1"] += int(x.argmax() == y.argmax())
        if (g - g0) % 500 == 0:
            print(f"  step {g - g0}/{g1 - g0} {time.time() - t0:.0f}s", flush=True)
    N = g1 - g0
    floor = acc["floor"]["kl"] / N
    fmx = acc["floor"]["mx"]
    nq_end = noop.layers[0].nq
    print(f"  floor = KL(full-cache SDPA || no-op chunked path), {N} steps, {nq_end} blocks aged in the no-op old part: "
          f"mean KL {floor:.3e} max|dlogit| {fmx:.3g} top1 agree {acc['floor']['top1'] / N:.4f}", flush=True)
    res = {"floor_kl": floor, "floor_maxdl": fmx, "N": N}
    for n in names:
        kp, qq = acc[f"pk_{n}"], acc[f"q_{n}"]
        mkl = kp["kl"] / N
        passed = mkl <= 2 * floor or mkl < 1e-4
        ok &= passed
        res[n] = dict(floor_kl=floor, floor_maxdl=fmx, kl_emulated_vs_packed=mkl, maxdl=kp["mx"], top1=kp["top1"] / N,
                      kl_full_vs_emulated=qq["kl"] / N, pass_=passed)
        print(f"  {n}: floor KL {floor:.3e} (max|dlogit| {fmx:.3g}) | KL(emulated || packed) mean {mkl:.3e} max|dlogit| {kp['mx']:.3g} "
              f"top1 {kp['top1'] / N:.4f} | [quantization effect KL(full || emulated) {qq['kl'] / N:.3e}] "
              f"gate (<= 2 x floor {2 * floor:.3e} or < 1e-4): {'PASS' if passed else 'FAIL'}", flush=True)
    print("CHECK_JSON " + json.dumps(res), flush=True)
    print("CHECK", "PASS" if ok else "FAIL", flush=True)
    return ok


# --- --bench -----------------------------------------------------------------------------------------------------------
def parse_cell(cell):
    if cell == "full":
        return "full", None, []
    k, _, rest = cell.partition("-")
    if k in ("fq", "fqe"):
        return k, fade_q.CONFIGS[rest], []
    assert k in ("tl", "sl"), cell
    return k, None, [int(rest)]


def build_state(PR, base, kind, cfg, par, B, total, a, Hm):
    """State of the cell at context L = base.L, B rows (lockstep), room for `total` more steps. Returns (state, Lmax, info);
    info = capacities in tokens and the exact / quantized token counts."""
    L, P = base.L, base.P
    g0 = L - P
    dev = PR.device
    info = dict(slack_tokens=a.G, cap_old_slots=0)
    if kind == "fq":
        nq = policy_nq(g0, a.R, a.G)  # from fade_q.age_rows' schedule
        ps = PackedState.from_base(base, cfg, Hm, a.R, a.G, g0 + total + a.G, chunk=a.chunk, per_block=a.dequant_per_block, device=dev)
        lay = ps.layers[0]
        assert all(l.nq == nq for l in ps.layers), (nq, [l.nq for l in ps.layers])
        assert all(l.n_exact_generated() == g0 - nq * a.G for l in ps.layers)
        out = ps.to_batch(B)
        info.update(cap_exact_slots=lay.Le, cap_old_slots=lay.cap, n_quantized_tokens=nq * a.G, n_exact_tokens=g0 - nq * a.G,
                    n_exact_total=P + g0 - nq * a.G)
        del ps
        return out, None, info
    if kind in ("full", "fqe"):
        Lmax = L + total + a.G
    elif kind == "tl":
        Lmax, info["slack_tokens"] = P + par[0] + 16, 16
    else:
        Lmax, info["slack_tokens"] = a.sinks + par[0] + 16, 16
    st1 = PR.new_state(1, Lmax)
    if kind in ("full", "fqe"):
        n = L
        for fi in range(len(base.K)):
            st1.K[fi][:, :, :n] = base.K[fi].to(dev)
            st1.V[fi][:, :, :n] = base.V[fi].to(dev)
        st1.valid[0, :n] = True
        nq = policy_nq(g0, a.R, a.G) if kind == "fqe" else 0
        info.update(n_quantized_tokens=nq * a.G, n_exact_tokens=g0 - nq * a.G, n_exact_total=P + g0 - nq * a.G)
    else:
        W = par[0]
        kb = P if kind == "tl" else a.sinks
        pp = torch.arange(max(kb, L - W), L, device=dev)  # absolute positions kept in the ring
        slots = kb + (pp - kb) % W
        for fi in range(len(base.K)):
            K, V = base.K[fi].to(dev), base.V[fi].to(dev)
            st1.K[fi][:, :, :kb], st1.V[fi][:, :, :kb] = K[:, :, :kb], V[:, :, :kb]
            st1.K[fi][:, :, slots], st1.V[fi][:, :, slots] = K[:, :, pp], V[:, :, pp]
        st1.valid[0, :kb] = True
        st1.valid[0, slots] = True
        info.update(n_quantized_tokens=0, n_exact_tokens=len(pp), n_exact_total=kb + len(pp))
    info["cap_exact_slots"] = Lmax
    st1.length[:] = L
    st1.pos_next[:] = L
    for i in range(len(st1.conv)):
        st1.conv[i].copy_(base.conv[i].to(dev))
        st1.rec[i].copy_(base.rec[i].to(dev))
    st = st1.replicate(B)
    del st1
    return st, Lmax, info


def state_sizes(st):
    if isinstance(st, PackedState):
        return st.sizes()
    ex = sum(t.numel() * t.element_size() for t in st.K + st.V)
    gdn = sum(t.numel() * t.element_size() for t in st.conv + st.rec)
    oth = st.valid.numel() * st.valid.element_size()
    return dict(fa_exact_bytes=ex, fa_packed_codes_bytes=0, fa_scales_bytes=0, gdn_bytes=gdn, other_bytes=oth,
                state_bytes=ex + gdn + oth)


@torch.no_grad()
def run_one(PR, base, cell, B, a, Hm, wb, V):
    kind, cfg, par = parse_cell(cell)
    dev = PR.device
    L, P = base.L, base.P
    warm = a.warmup
    total = a.steps + warm
    st, Lmax, info = build_state(PR, base, kind, cfg, par, B, total, a, Hm)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    sizes = state_sizes(st)
    seen = torch.zeros(B, V, dtype=torch.bool, device=dev)
    seen[:, torch.tensor(base.gen_ids, device=dev)] = True
    logits = base.logits.to(dev)[None].expand(B, -1).clone()
    gens = [torch.Generator(device=dev).manual_seed(a.seed + b) for b in range(B)]
    ar = torch.arange(B, device=dev)
    g0 = L - P
    for i in range(total):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        nt = sample(logits, seen, gens)
        seen[ar, nt] = True
        nt.tolist()  # host sync per step, as tl_gen
        if kind in ("tl", "sl"):
            kb = P if kind == "tl" else a.sinks
            sl = torch.full((B, 1), kb + (L + i - kb) % par[0], dtype=torch.long, device=dev)
            logits = PR.forward(st, nt[:, None], slot=sl, lcur=Lmax - 16)
        elif kind == "fqe":
            fade_q.age_rows(st, [(r, P, g0 + i) for r in range(B)], cfg, Hm, a.R, a.G)
            logits = PR.forward(st, nt[:, None])
        else:
            logits = PR.forward(st, nt[:, None])
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    fin = bool(torch.isfinite(logits).all())
    if not fin:
        print(f"WARNING non-finite logits {cell} L={L} B={B}", flush=True)
    ms = dt / a.steps * 1000
    rec = dict(cell=cell, problem_id=a.row_id, P=P, L=L, g0=g0, B=B, steps=a.steps, warmup=warm, oom=False, **info,
               weights_bytes=wb[0], weights_numel_bytes=wb[1], resident_allocated=resident, state_alloc_bytes=resident - wb[0],
               peak_allocated=torch.cuda.max_memory_allocated(), reserved_at_peak=torch.cuda.max_memory_reserved(),
               ms_per_step=ms, tok_s=B * 1000.0 / ms, finite=fin, **sizes)
    if kind == "fq":
        nq_end = st.layers[0].nq
        assert nq_end == policy_nq(g0 + total, a.R, a.G), (nq_end, g0 + total)
        rec["aged_blocks_end"] = nq_end
    del st, seen, logits
    return rec


@torch.no_grad()
def bench(a):
    from transformers import AutoTokenizer
    dev = torch.device("cuda")
    cells = a.cells.split(",")
    ctxs = [int(x) for x in a.ctx.split(",")]
    Bs = [int(x) for x in a.batch.split(",")]
    for c in cells:
        parse_cell(c)
    done = set()
    if os.path.exists(a.out):
        done = {(r["cell"], r.get("ctx_arg"), r["B"]) for r in map(json.loads, (l for l in open(a.out) if l.strip()))}
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_model(a.model)
    gc.collect()
    torch.cuda.empty_cache()
    w_alloc = torch.cuda.memory_allocated()  # weights only: right after model load, before any state
    w_numel = sum(p.numel() * p.element_size() for p in model.parameters())
    wb = (w_alloc, w_numel)
    PR = PackedRunner(model)
    V = model.lm_head.out_features
    Hm = fade_q.hadamard(PR.cfg.head_dim, dev)
    gpu = torch.cuda.get_device_name(0)
    print(f"weights: allocated {w_alloc / 2**30:.3f} GiB, numel-analytic {w_numel / 2**30:.3f} GiB on {gpu}", flush=True)
    fo = open(a.out, "a")
    for ctx in ctxs:
        todo = [(c, B) for c in cells for B in Bs if (c, ctx, B) not in done]
        if not todo:
            continue
        if a.ctx_is_gen:
            row, pr = pick_row(a.traces, a.problems, tok, 0, min_gen=ctx)
            L = len(pr) + ctx
        else:
            row, pr = pick_row(a.traces, a.problems, tok, ctx)
            L = ctx
        P = len(pr)
        gen = row["output_ids"][:L - P]
        assert P + len(gen) == L
        a.row_id = f"{row['problem_id']}#{row['sample_idx']}"
        print(f"ctx {ctx} (L={L}): row {a.row_id} P={P} gen={len(gen)}; prefill ...", flush=True)
        st = PR.new_state(1, L + 8)
        t0 = time.time()
        lg = PR.prefill(st, torch.tensor([pr + gen], device=dev), chunk=4096)
        base = Base.from_state(st, L, P, to_cpu=True, logits=lg[0].cpu(), gen_ids=pr + gen)
        del st, lg
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  prefill {time.time() - t0:.0f}s", flush=True)
        for c, B in todo:
            try:
                rec = run_one(PR, base, c, B, a, Hm, wb, V)
            except torch.OutOfMemoryError:
                rec = dict(cell=c, problem_id=a.row_id, P=P, L=L, B=B, steps=a.steps, oom=True)
            gc.collect()
            torch.cuda.empty_cache()
            rec.update(ctx_arg=ctx, ctx_mode="gen" if a.ctx_is_gen else "total", gpu=gpu, torch=torch.__version__, R=a.R, G=a.G,
                       chunk=a.chunk)
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            print("  " + (f"{c} B={B}: OOM" if rec["oom"] else f"{c} B={B}: resident {rec['resident_allocated'] / 2**30:.2f} GiB peak "
                          f"{rec['peak_allocated'] / 2**30:.2f} GiB {rec['ms_per_step']:.1f} ms/step {rec['tok_s']:.1f} tok/s"), flush=True)
    allrows = [json.loads(l) for l in open(a.out) if l.strip()]
    print("\n%-14s %6s %3s %10s %10s %10s %10s %9s %8s" % ("cell", "L", "B", "state GiB", "resid GiB", "peak GiB", "resv GiB", "ms/step", "tok/s"))
    for r in sorted(allrows, key=lambda r: (r["L"], r["B"], r["cell"])):
        if r.get("oom"):
            print("%-14s %6d %3d %10s" % (r["cell"], r["L"], r["B"], "OOM"))
        else:
            print("%-14s %6d %3d %10.2f %10.2f %10.2f %10.2f %9.1f %8.1f" % (r["cell"], r["L"], r["B"], r["state_bytes"] / 2**30,
                  r["resident_allocated"] / 2**30, r["peak_allocated"] / 2**30, r["reserved_at_peak"] / 2**30, r["ms_per_step"], r["tok_s"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("README")[0].strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--traces", default="../out/traces.jsonl")
    ap.add_argument("--problems", default="../out/problems.jsonl")
    ap.add_argument("--cells", default="full,tl-8192,sl-8192,fq-kv4,fq-kv2_had")
    ap.add_argument("--ctx", default="8192,16384,32768", help="total context L (prompt + generated) at the start of decoding")
    ap.add_argument("--ctx-is-gen", action="store_true", help="read --ctx as the number of GENERATED tokens g0 (L = P + ctx)")
    ap.add_argument("--batch", default="1,8,32")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--out", default="fade_mem.jsonl")
    ap.add_argument("--R", type=int, default=2048)
    ap.add_argument("--G", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=2048, help="old-part dequant chunk (tokens, multiple of G)")
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dequant-per-block", action="store_true", help="dequant one [1,KVH,G,D] block at a time (fade_q's exact shapes)")
    ap.add_argument("--g-start", type=int, default=2048, help="--check (ii): first teacher-forced generated index")
    ap.add_argument("--gen-end", type=int, default=4800, help="--check (ii): last generated index + 1")
    ap.add_argument("--g-real", type=int, default=4000, help="--check (i): generated tokens prefilled for the real K/V comparison")
    a = ap.parse_args()
    if a.check == a.bench:
        ap.error("give exactly one of --check / --bench")
    if a.check:
        sys.exit(0 if check(a) else 1)
    bench(a)


if __name__ == "__main__":
    main()
