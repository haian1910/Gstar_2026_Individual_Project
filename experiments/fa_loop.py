"""Attempt-scoped attention for Qwen3.5 (hybrid Gated DeltaNet + gated full attention): our own forward over the HF
modules, with explicit state so we can (i) block / delete FA KV entries of a span, (ii) snapshot / restore the GDN
recurrent + short-conv states, (iii) keep true RoPE positions after a deletion (keys are cached post-RoPE).

State per batch row: FA keys/values in slot storage [B, KVH, Lmax, D] + a validity mask [B, Lmax] (deleting a span =
clearing its valid bits; the slot index never changes, so positions stay true), GDN conv state [B, C, K-1] and recurrent
state [B, H, Dk, Dv] (fp32). Every forward call appends T tokens per row at slots length[b] .. length[b]+T-1.
"""
import copy
import torch
import torch.nn.functional as F

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
except ImportError:  # pragma: no cover
    chunk_gated_delta_rule = fused_recurrent_gated_delta_rule = None


def text_model(model):
    """(text decoder, lm_head) for Qwen3_5ForConditionalGeneration or Qwen3_5ForCausalLM."""
    m = model.model
    lm = getattr(m, "language_model", m)
    return lm, model.lm_head


class State:
    def __init__(self, B, Lmax, cfg, device, dtype=torch.bfloat16):
        self.B, self.Lmax = B, Lmax
        n_fa = sum(t == "full_attention" for t in cfg.layer_types)
        n_gdn = len(cfg.layer_types) - n_fa
        kvh, d = cfg.num_key_value_heads, cfg.head_dim
        conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2 + cfg.linear_value_head_dim * cfg.linear_num_value_heads
        self.K = [torch.zeros(B, kvh, Lmax, d, device=device, dtype=dtype) for _ in range(n_fa)]
        self.V = [torch.zeros(B, kvh, Lmax, d, device=device, dtype=dtype) for _ in range(n_fa)]
        self.valid = torch.zeros(B, Lmax, dtype=torch.bool, device=device)
        self.length = torch.zeros(B, dtype=torch.long, device=device)  # slots used
        self.pos_next = torch.zeros(B, dtype=torch.long, device=device)  # next RoPE position
        self.conv = [torch.zeros(B, conv_dim, cfg.linear_conv_kernel_dim - 1, device=device, dtype=dtype) for _ in range(n_gdn)]
        self.rec = [torch.zeros(B, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim,
                                device=device, dtype=torch.float32) for _ in range(n_gdn)]

    # --- GDN snapshot / restore -------------------------------------------------------------------------------------
    def gdn_snapshot(self):
        return ([c.clone() for c in self.conv], [r.clone() for r in self.rec])

    def gdn_restore(self, snap, rows=None):
        conv, rec = snap
        for i in range(len(self.conv)):
            if rows is None:
                self.conv[i].copy_(conv[i]); self.rec[i].copy_(rec[i])
            else:
                self.conv[i][rows] = conv[i][rows] if conv[i].shape[0] == self.B else conv[i].expand(len(rows), *conv[i].shape[1:])
                self.rec[i][rows] = rec[i][rows] if rec[i].shape[0] == self.B else rec[i].expand(len(rows), *rec[i].shape[1:])

    def gdn_zero(self, rows=None):
        for i in range(len(self.conv)):
            if rows is None:
                self.conv[i].zero_(); self.rec[i].zero_()
            else:
                self.conv[i][rows] = 0; self.rec[i][rows] = 0

    # --- FA span blocking -------------------------------------------------------------------------------------------
    def block_slots(self, row, lo, hi):
        """Delete FA entries in slots [lo, hi) of one row (all 8 FA layers share the mask). Positions are untouched."""
        self.valid[row, lo:hi] = False

    def fa_tokens_live(self):
        return self.valid.sum(1)

    # --- row copy (for continuous batching) -------------------------------------------------------------------------
    def copy_row_from(self, dst_row, src, src_row=0):
        L = int(src.length[src_row])
        assert L <= self.Lmax
        for i in range(len(self.K)):
            self.K[i][dst_row, :, :L] = src.K[i][src_row, :, :L]
            self.V[i][dst_row, :, :L] = src.V[i][src_row, :, :L]
        self.valid[dst_row] = False
        self.valid[dst_row, :L] = src.valid[src_row, :L]
        self.length[dst_row] = L
        self.pos_next[dst_row] = src.pos_next[src_row]
        for i in range(len(self.conv)):
            self.conv[i][dst_row] = src.conv[i][src_row]
            self.rec[i][dst_row] = src.rec[i][src_row]

    def clone(self):
        return copy.deepcopy(self)

    def replicate(self, n):
        """B=1 state -> B=n state (independent copies)."""
        assert self.B == 1
        out = copy.copy(self)
        out.B = n
        out.K = [k.expand(n, *k.shape[1:]).clone() for k in self.K]
        out.V = [v.expand(n, *v.shape[1:]).clone() for v in self.V]
        out.valid = self.valid.expand(n, -1).clone()
        out.length = self.length.expand(n).clone()
        out.pos_next = self.pos_next.expand(n).clone()
        out.conv = [c.expand(n, *c.shape[1:]).clone() for c in self.conv]
        out.rec = [r.expand(n, *r.shape[1:]).clone() for r in self.rec]
        return out


class Runner:
    def __init__(self, model):
        self.model = model
        self.lm, self.lm_head = text_model(model)
        self.cfg = self.lm.config
        self.layers = self.lm.layers
        self.fa_of, self.gdn_of = {}, {}
        for i, t in enumerate(self.cfg.layer_types):
            if t == "full_attention":
                self.fa_of[i] = len(self.fa_of)
            else:
                self.gdn_of[i] = len(self.gdn_of)
        assert chunk_gated_delta_rule is not None, "flash-linear-attention is required"
        self.device = next(model.parameters()).device

    def new_state(self, B, Lmax):
        return State(B, Lmax, self.cfg, self.device, dtype=next(self.model.parameters()).dtype)

    # --- layers -----------------------------------------------------------------------------------------------------
    def _gdn(self, la, x, st, gi):
        B, T, _ = x.shape
        qkv = la.in_proj_qkv(x).transpose(1, 2)  # [B, C, T]
        xin = torch.cat([st.conv[gi].to(qkv.dtype), qkv], dim=-1)  # [B, C, K-1+T]
        st.conv[gi] = xin[:, :, -(la.conv_kernel_size - 1):].clone()
        w = la.conv1d.weight  # [C, 1, K]
        out = F.conv1d(xin.to(w.dtype), w, la.conv1d.bias, groups=xin.shape[1])  # [B, C, T]
        out = F.silu(out).to(x.dtype).transpose(1, 2)
        q, k, v = torch.split(out, [la.key_dim, la.key_dim, la.value_dim], dim=-1)
        q = q.reshape(B, T, -1, la.head_k_dim)
        k = k.reshape(B, T, -1, la.head_k_dim)
        v = v.reshape(B, T, -1, la.head_v_dim)
        z = la.in_proj_z(x).reshape(B, T, -1, la.head_v_dim)
        beta = la.in_proj_b(x).sigmoid()
        g = -la.A_log.float().exp() * F.softplus(la.in_proj_a(x).float() + la.dt_bias)
        rep = la.num_v_heads // la.num_k_heads
        if rep > 1:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        fn = fused_recurrent_gated_delta_rule if T == 1 else chunk_gated_delta_rule
        o, s = fn(q, k, v, g=g, beta=beta, initial_state=st.rec[gi], output_final_state=True, use_qk_l2norm_in_kernel=True)
        st.rec[gi] = s.float()
        o = la.norm(o.reshape(-1, la.head_v_dim), z.reshape(-1, la.head_v_dim)).reshape(B, T, -1)
        return la.out_proj(o)

    def _fa(self, sa, x, cos, sin, st, fi, slot, mask, Lcur):
        B, T, _ = x.shape
        D = sa.head_dim
        q, gate = torch.chunk(sa.q_proj(x).view(B, T, -1, D * 2), 2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = sa.q_norm(q).transpose(1, 2)  # [B, H, T, D]
        k = sa.k_norm(sa.k_proj(x).view(B, T, -1, D)).transpose(1, 2)
        v = sa.v_proj(x).view(B, T, -1, D).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        bidx = torch.arange(B, device=x.device)[:, None]
        st.K[fi].permute(0, 2, 1, 3)[bidx, slot] = k.permute(0, 2, 1, 3)
        st.V[fi].permute(0, 2, 1, 3)[bidx, slot] = v.permute(0, 2, 1, 3)
        K = st.K[fi][:, :, :Lcur]
        V = st.V[fi][:, :, :Lcur]
        # GQA by folding the G query heads of each KV head into the query axis (q head h uses KV head h // G, as HF
        # repeat_kv); avoids SDPA's enable_gqa path, which can fall back to the math kernel and expand K/V G times
        H, KVH = q.shape[1], K.shape[1]
        G = H // KVH
        qg = q.reshape(B, KVH, G * T, D)
        mg = mask if G == 1 else mask.repeat(1, 1, G, 1)
        o = F.scaled_dot_product_attention(qg, K, V, attn_mask=mg, scale=sa.scaling)
        o = o.reshape(B, H, T, D).transpose(1, 2).reshape(B, T, -1)
        o = o * torch.sigmoid(gate)
        return sa.o_proj(o)

    # --- forward ----------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, st, ids, pos=None, logits="last", slot=None, lcur=None):
        """Append ids [B, T] (same T for all rows) at each row's next slots. pos [B, T] are true RoPE positions
        (default: continue from pos_next). Returns logits [B, V] ('last'), [B, T, V] ('all'), final-normed hidden
        [B, T, H] ('hidden'; project with Runner.project in chunks to save memory) or None.
        slot [B, 1] (decode only, T=1): write into these slots instead of length[b] (ring-buffer FA windows, tl_gen.py);
        a slot may overwrite an older entry, and every valid slot is visible (the slot order is not the time order).
        lcur: attend over slots [0, lcur) at least (ring rows keep a fixed slot range)."""
        B, T = ids.shape
        dev = ids.device
        if pos is None:
            pos = st.pos_next[:, None] + torch.arange(T, device=dev)[None]
        ring = slot is not None
        if ring:
            assert T == 1, "slot override is for single-token decode steps"
        else:
            slot = st.length[:, None] + torch.arange(T, device=dev)[None]  # [B, T]
        assert int(slot.max()) < st.Lmax, "state storage full"
        Lcur = min(st.Lmax, (max(int(slot.max()) + 1, lcur or 0) + 15) // 16 * 16)  # padded to 16 for the mem-efficient SDPA kernel; extra slots are invalid
        st.valid.scatter_(1, slot, True)
        kv = torch.arange(Lcur, device=dev)
        if ring:
            mask = st.valid[:, None, None, :Lcur]  # [B,1,1,Lcur]: everything cached is in the past of a decode step
        else:
            mask = st.valid[:, None, None, :Lcur] & (kv[None, None, None, :] <= slot[:, None, :, None])  # [B,1,T,Lcur]
        h = self.lm.embed_tokens(ids)
        cos, sin = self.lm.rotary_emb(h, pos)
        for li, layer in enumerate(self.layers):
            r = h
            x = layer.input_layernorm(h)
            if li in self.gdn_of:
                x = self._gdn(layer.linear_attn, x, st, self.gdn_of[li])
            else:
                x = self._fa(layer.self_attn, x, cos, sin, st, self.fa_of[li], slot, mask, Lcur)
            h = r + x
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        st.length += T
        st.pos_next = pos[:, -1] + 1
        if logits is None:
            return None
        h = self.lm.norm(h if logits in ("all", "hidden") else h[:, -1:])
        if logits == "hidden":
            return h
        out = self.lm_head(h).float()
        return out if logits == "all" else out[:, -1]

    def project(self, h):
        return self.lm_head(h).float()

    @torch.no_grad()
    def prefill(self, st, ids, pos=None, chunk=4096, logits="last"):
        """Multi-chunk prefill of ids [B, T]; returns the logits of the final chunk call."""
        T = ids.shape[1]
        out = None
        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            p = None if pos is None else pos[:, s:e]
            out = self.forward(st, ids[:, s:e], p, logits=logits if e == T else None)
        return out


def _apply_rope(q, k, cos, sin):
    # same as HF apply_rotary_pos_emb (partial rotary: first cos.shape[-1] dims), unsqueeze over heads
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rd = cos.shape[-1]

    def rot(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    qr, qp = q[..., :rd], q[..., rd:]
    kr, kp = k[..., :rd], k[..., rd:]
    return torch.cat([qr * cos + rot(qr) * sin, qp], -1), torch.cat([kr * cos + rot(kr) * sin, kp], -1)


# --- sampling (vLLM order: penalties -> temperature -> top-k -> top-p -> min-p -> sample) ----------------------------
def sample(logits, seen, gens, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5):
    """logits [B, V] fp32; seen [B, V] bool = token appeared in the output so far (TEXT history, identical across cells);
    gens: list of per-row torch.Generator (paired seeds across cells)."""
    if presence_penalty:
        logits = logits - presence_penalty * seen.to(logits.dtype)
    logits = logits / max(temperature, 1e-6)
    vals, idx = logits.topk(top_k, dim=-1)  # sorted desc
    probs = torch.softmax(vals, dim=-1)
    if top_p < 1.0:
        cum = probs.cumsum(-1)
        keep = (cum - probs) < top_p  # always keeps the first
        probs = probs * keep
    if min_p > 0:
        probs = probs * (probs >= min_p * probs[:, :1])
    probs = probs / probs.sum(-1, keepdim=True)
    out = torch.empty(logits.shape[0], dtype=torch.long, device=logits.device)
    for b, g in enumerate(gens):
        out[b] = idx[b, torch.multinomial(probs[b], 1, generator=g)]
    return out
