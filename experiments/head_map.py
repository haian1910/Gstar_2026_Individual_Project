"""Step 0 of nti-e2's "few heads remember" check (request 2026-10-27 ~16:35 UTC; layer-map PLAN Changes). Per FA KV head of
Qwen3.5-9B (8 FA layers x 4 KV heads), how much of the attention output at a query comes from FAR generated keys?
Teacher-forces each trace (prompt + generated ids) with full attention, capturing post-RoPE q and the output gate at
sampled query positions (64 per trace, uniform over [P + W, end), Random(seed of problem_id)). After the prefill, per FA
layer and query t, softmax over keys 0..t per q head; keys split into prompt (j < P), near (j > t - W, generated) and far
(P <= j <= t - W). Per q head: far_mass = sum of far weights; gated_far = ||sigmoid(gate_t) * sum_far a_tj v_j||;
far_share = gated_far / ||sigmoid(gate_t) * o_t||. Per KV head = mean over its G q heads (q head h -> KV head h // G).
One JSON line per trace with [8, 4] means over its queries. Aggregate with head_map_report.py.
Sources: --source tl  = Full TL runs (out_tl/gen_full.s*.jsonl), sample_k 0, math, >= --min-gen tokens; up to --n spread
                         round-robin over the 4 math datasets (Random(0) order within each);
          --source e1  = E1 traces, non-test MATH-500 L5, the layer-map selection (longest >= 16K per problem, Random(0)),
                         truncated to 24,576 tokens (held out from the TL test problems).
Usage: python head_map.py --source tl|e1 --out ../out_hm/hm_<source>.jsonl [--n 30] [--q 64] [--W 2048]"""
import argparse, glob, json, os, random, types, zlib
os.environ.setdefault("USE_HUB_KERNELS", "0")
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from fa_loop import Runner, _apply_rope

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
MATH = ["aime25", "aime26", "hmmt25", "math500_l5"]
R_ = os.environ.get("FADE_ROOT", ".")  # run directory holding out/ and out_tl/
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--source", choices=["tl", "e1"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--q", type=int, default=64)
ap.add_argument("--W", type=int, default=2048)
ap.add_argument("--min-gen", type=int, default=12288)
a = ap.parse_args()


def load_model(name):  # as tl_gen.load_model
    from transformers import AutoModelForImageTextToText
    try:
        m = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16)
    except Exception:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    return m.cuda().eval()


def traces():
    probs = {}
    for f in (f"{R_}/out/problems.jsonl", f"{R_}/out_pilot/problems.jsonl"):
        for l in open(f):
            if l.strip():
                p = json.loads(l)
                probs.setdefault(p["problem_id"], p)
    if a.source == "tl":
        by = {d: [] for d in MATH}
        for f in sorted(glob.glob(f"{R_}/out_tl/gen_full.s*.jsonl")):
            for l in open(f):
                r = json.loads(l)
                if r["dataset"] in by and r["sample_k"] == 0 and len(r["gen_ids"]) >= a.min_gen:
                    by[r["dataset"]].append(r)
        for d in MATH:
            by[d].sort(key=lambda r: r["problem_id"])
            random.Random(0).shuffle(by[d])
        out = []
        while len(out) < a.n and any(by.values()):
            for d in MATH:
                if by[d] and len(out) < a.n:
                    r = by[d].pop(0)
                    out.append((r["problem_id"], r["dataset"], r["gen_ids"], r["max_fa_slots"] - r["n_tokens"] + 1))  # Full: the last token is never fed
        return probs, out
    test = set(l.strip() for l in open(f"{R_}/out/tl_ids.txt") if l.strip())
    best = {}
    for l in open(f"{R_}/out/traces.jsonl"):
        r = json.loads(l)
        if r["dataset"] != "math500_l5" or r["problem_id"] in test or len(r["output_ids"]) < 16384:
            continue
        k = (len(r["output_ids"]), -r["sample_idx"])
        if r["problem_id"] not in best or k > best[r["problem_id"]][0]:
            best[r["problem_id"]] = (k, r)
    pids = sorted(best)
    random.Random(0).shuffle(pids)
    return probs, [(p, "math500_l5", best[p][1]["output_ids"][:24576], best[p][1]["prompt_len"]) for p in pids[:a.n]]


def fa_capture(self, sa, x, cos, sin, st, fi, slot, mask, Lcur):
    """Runner._fa plus: store post-RoPE q [H, D] and gate [H, D] at the sampled slots in this chunk."""
    B, T, _ = x.shape
    D = sa.head_dim
    q, gate = torch.chunk(sa.q_proj(x).view(B, T, -1, D * 2), 2, dim=-1)
    want = [(i, s) for i, s in enumerate(slot[0].tolist()) if s in self.cap_slots]
    gate_h = gate[0]  # [T, H, D]
    gate = gate.reshape(B, T, -1)
    q = sa.q_norm(q).transpose(1, 2)
    k = sa.k_norm(sa.k_proj(x).view(B, T, -1, D)).transpose(1, 2)
    v = sa.v_proj(x).view(B, T, -1, D).transpose(1, 2)
    q, k = _apply_rope(q, k, cos, sin)
    bidx = torch.arange(B, device=x.device)[:, None]
    st.K[fi].permute(0, 2, 1, 3)[bidx, slot] = k.permute(0, 2, 1, 3)
    st.V[fi].permute(0, 2, 1, 3)[bidx, slot] = v.permute(0, 2, 1, 3)
    K = st.K[fi][:, :, :Lcur]
    V = st.V[fi][:, :, :Lcur]
    H, KVH = q.shape[1], K.shape[1]
    G = H // KVH
    qg = q.reshape(B, KVH, G * T, D)
    mg = mask if G == 1 else mask.repeat(1, 1, G, 1)
    o = F.scaled_dot_product_attention(qg, K, V, attn_mask=mg, scale=sa.scaling)
    o = o.reshape(B, H, T, D).transpose(1, 2)  # [B, T, H, D]
    for i, s in want:  # q, gate and the SDPA output (pre-gate) at the sampled query, for the self-check in measure
        self.cap[(fi, s)] = (q[0, :, i].clone(), gate_h[i].clone(), o[0, i].clone())
    o = o.reshape(B, T, -1) * torch.sigmoid(gate)
    return sa.o_proj(o)


@torch.no_grad()
def measure(R, st, P, slots, scaling):
    """-> far_mass, gated_far, far_share, prompt_mass: each [n_fa, KVH] means over the sampled queries."""
    n_fa, KVH = len(st.K), st.K[0].shape[1]
    acc = torch.zeros(4, n_fa, KVH, device=R.device, dtype=torch.float64)
    err = 0.0
    for fi in range(n_fa):
        for t in slots:
            q, g, o_sdpa = R.cap[(fi, t)]  # [H, D]
            H, D = q.shape
            G = H // KVH
            K = st.K[fi][0, :, :t + 1].float()  # [KVH, t+1, D]
            V = st.V[fi][0, :, :t + 1].float()
            Kh = K.repeat_interleave(G, 0)  # q head h -> KV head h // G
            Vh = V.repeat_interleave(G, 0)
            att = torch.softmax(torch.einsum("hd,hjd->hj", q.float(), Kh) * scaling, -1)  # [H, t+1]
            far = torch.zeros(t + 1, dtype=torch.bool, device=R.device)
            far[P:max(P, t - a.W + 1)] = True  # P <= j <= t - W
            sg = torch.sigmoid(g.float())
            o = torch.einsum("hj,hjd->hd", att, Vh)
            err = max(err, ((o - o_sdpa.float()).norm(dim=-1) / o.norm(dim=-1).clamp_min(1e-12)).max().item())
            of = torch.einsum("hj,hjd->hd", att[:, far], Vh[:, far])
            fm = att[:, far].sum(-1)
            gf = (sg * of).norm(dim=-1)
            fs = gf / (sg * o).norm(dim=-1).clamp_min(1e-12)
            pm = att[:, :P].sum(-1)
            for i, x in enumerate((fm, gf, fs, pm)):
                acc[i, fi] += x.view(KVH, G).mean(-1).double()
    return (acc / len(slots)).tolist(), err


def main():
    tok = AutoTokenizer.from_pretrained(a.model)
    probs, rows = traces()
    print(f"source {a.source}: {len(rows)} traces; datasets {sorted({d for _, d, _, _ in rows})}", flush=True)
    R = Runner(load_model(a.model))
    R._fa = types.MethodType(fa_capture, R)
    scaling = R.layers[sorted(R.fa_of)[0]].self_attn.scaling
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["problem_id"] for l in open(a.out) if l.strip()}
    for pid, ds, gen, P_rec in rows:
        if pid in done:
            continue
        p = probs[pid]
        x = tok.apply_chat_template([{"role": "user", "content": p["problem"] + p.get("suffix", SUFFIX)}],
                                    add_generation_prompt=True, enable_thinking=True, tokenize=True)
        pr = x if isinstance(x, list) else x["input_ids"]
        P = len(pr)
        assert P == P_rec, (pid, P, P_rec)
        ids = pr + list(gen)
        rng = random.Random(zlib.crc32(pid.encode()))
        slots = sorted(rng.sample(range(P + a.W, len(ids)), a.q))
        R.cap_slots, R.cap = set(slots), {}
        st = R.new_state(1, len(ids) + 16)
        R.prefill(st, torch.tensor([ids], device=R.device), logits=None)
        (fm, gf, fs, pm), err = measure(R, st, P, slots, scaling)
        with open(a.out, "a") as f:
            f.write(json.dumps({"problem_id": pid, "dataset": ds, "source": a.source, "P": P, "n_gen": len(gen), "W": a.W,
                                "n_q": len(slots), "fa_layers": sorted(R.fa_of), "far_mass": fm, "gated_far": gf,
                                "far_share": fs, "prompt_mass": pm, "max_rel_err_vs_sdpa": err}) + "\n")
        print(f"{pid} ({ds}) n_gen {len(gen)}: mean far_mass {sum(map(sum, fm)) / 32:.3f}; max rel err vs SDPA {err:.2e}", flush=True)
        del st, R.cap
        torch.cuda.empty_cache()
    print("HEAD_MAP_DONE", flush=True)


if __name__ == "__main__":
    main()
