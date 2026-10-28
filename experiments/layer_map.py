"""Layer-sensitivity map for Think-Local (HA_own/experiments/2026-10-27_layer-map/PLAN.md): which of the 8 FA layers of
Qwen3.5-9B need the far trace? Teacher-forces long full-attention traces (forget-attempt E1, MATH-500 L5 problems NOT in
tl_ids) through 18 configs in lockstep, chunk by chunk:
  full (reference), all8 (= TL-W), only_l (FA layer l windowed, rest full), except_l (all windowed except FA layer l).
Windowed = the masked TL definition: prompt pinned + the last W generated tokens incl. the current one (as tests/test_tl_long.py's
ref, which matched the ring buffer). Per config and generated-index bucket (only indices j >= W, where something was dropped):
sum of KL(full || config), top-1 agreement with full, and NLL(config) - NLL(full) of the trace's own next token.
One JSON line per trace; resumable per problem_id. Aggregate with layer_map_report.py.
Usage: python layer_map.py --traces ../out/traces.jsonl --problems ../out/problems.jsonl --test-ids ../out/tl_ids.txt \
           --out ../out_lm/lm.s0.jsonl --shard 0 --nshards 2 [--W 2048] [--max-gen 24576] [--n 40] [--chunk 1024]
       python layer_map.py ... --smoke   (correctness checks on one short trace; no output file)"""
import argparse, json, os, random
os.environ.setdefault("USE_HUB_KERNELS", "0")
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from fa_loop import Runner

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."  # as gen_traces.py (math rows)
BUCKETS = [(2048, 4096), (4096, 8192), (8192, 16384), (16384, 24576), (24576, 32768)]
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--traces", required=True)
ap.add_argument("--problems", required=True)
ap.add_argument("--test-ids", required=True)
ap.add_argument("--out")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--W", type=int, default=2048)
ap.add_argument("--max-gen", type=int, default=24576)
ap.add_argument("--min-gen", type=int, default=16384)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--chunk", type=int, default=1024)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--configs", help="custom windowed sets instead of the 18 default configs, e.g. \"win_0567=0,5,6,7 all8=0,1,2,3,4,5,6,7\" (FA layer indices 0..7); full is always added")
a = ap.parse_args()


def load_model(name):  # as tl_gen.load_model
    from transformers import AutoModelForImageTextToText
    try:
        m = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16)
    except Exception:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    return m.cuda().eval()


def select(traces, test):
    """One trace per non-test MATH-500 L5 problem: the longest with >= min_gen output tokens (ties: lowest sample_idx);
    problems sorted, shuffled with Random(0), first n."""
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


@torch.no_grad()
def forward_win(R, st, ids, win, W, P):
    """Runner.forward (append, T >= 1, logits 'all') with FA layers whose index (0..7) is in `win` limited to slots < P
    (prompt) or > slot - W (the last W tokens incl. the current one). win = empty set -> identical to Runner.forward.
    win may also be a dict {FA index: set of windowed KV heads} (per-head windows; the other heads of that layer see all)."""
    B, T = ids.shape
    dev = ids.device
    pos = st.pos_next[:, None] + torch.arange(T, device=dev)[None]
    slot = st.length[:, None] + torch.arange(T, device=dev)[None]
    assert int(slot.max()) < st.Lmax, "state storage full"
    Lcur = min(st.Lmax, (int(slot.max()) + 1 + 15) // 16 * 16)
    st.valid.scatter_(1, slot, True)
    kv = torch.arange(Lcur, device=dev)[None, None, None, :]
    base = st.valid[:, None, None, :Lcur] & (kv <= slot[:, None, :, None])  # [B,1,T,Lcur]
    wmask = base & ((kv < P) | (kv > slot[:, None, :, None] - W))
    h = R.lm.embed_tokens(ids)
    cos, sin = R.lm.rotary_emb(h, pos)
    for li, layer in enumerate(R.layers):
        r = h
        x = layer.input_layernorm(h)
        if li in R.gdn_of:
            x = R._gdn(layer.linear_attn, x, st, R.gdn_of[li])
        else:
            fi = R.fa_of[li]
            x = R._fa(layer.self_attn, x, cos, sin, st, fi, slot, layer_mask(win, fi, base, wmask, st.K[fi].shape[1]), Lcur)
        h = r + x
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    st.length += T
    st.pos_next = pos[:, -1] + 1
    return R.lm_head(R.lm.norm(h)).float()


def layer_mask(win, fi, base, wmask, kvh):
    """Set win: the whole layer windowed or not. Dict win: per-KV-head [B, KVH, T, Lcur] mask (_fa repeats it over the G q
    heads of each KV head, as it does the [B, 1, T, Lcur] mask)."""
    if not isinstance(win, dict):
        return wmask if fi in win else base
    hs = win.get(fi, set())
    if not hs:
        return base
    if len(hs) == kvh:
        return wmask
    sel = torch.zeros(1, kvh, 1, 1, dtype=torch.bool, device=base.device)
    sel[0, sorted(hs)] = True
    return torch.where(sel, wmask, base)


def parse_cfg(v, n_fa, kvh=4):
    """'0,5,6,7' -> windowed FA layers (set). 'keep:7.1,6.3' -> per-head dict: every KV head windowed except the listed
    <FA index>.<KV head> ones ('keep:' alone = all 32 heads windowed through the per-head path)."""
    if not v.startswith("keep:"):
        return {int(x) for x in v.split(",")}
    keep = {tuple(map(int, x.split("."))) for x in v[5:].split(",") if x}
    return {fi: {h for h in range(kvh) if (fi, h) not in keep} for fi in range(n_fa)}


def configs(n_fa):
    c = {"full": set(), "all8": set(range(n_fa))}
    for l in range(n_fa):
        c[f"only_{l}"] = {l}
    for l in range(n_fa):
        c[f"except_{l}"] = set(range(n_fa)) - {l}
    return c


def prompt_ids(tok, p):
    x = tok.apply_chat_template([{"role": "user", "content": p["problem"] + p.get("suffix", SUFFIX)}],
                                add_generation_prompt=True, enable_thinking=True, tokenize=True)
    return x if isinstance(x, list) else x["input_ids"]


@torch.no_grad()
def run_trace(R, pr, gen, W, cfgs, chunk):
    """Lockstep over chunks of the generated tokens. Returns {config: {bucket: [n, kl, agree, dnll]}}."""
    P = len(pr)
    base = R.new_state(1, P + len(gen) + 32)
    R.prefill(base, torch.tensor([pr], device=R.device))
    states = {}
    for name in cfgs:
        s = R.new_state(1, base.Lmax)
        s.copy_row_from(0, base)
        states[name] = s
    del base
    acc = {name: {f"{lo}-{hi}": [0, 0.0, 0.0, 0.0] for lo, hi in BUCKETS} for name in cfgs if name != "full"}
    g = torch.tensor(gen, device=R.device)
    for s0 in range(0, len(gen) - 1, chunk):  # feed tokens s0..s1-1; logits at j predict token j+1
        s1 = min(len(gen) - 1, s0 + chunk)
        ids = g[None, s0:s1]
        tgt = g[s0 + 1:s1 + 1]
        j = torch.arange(s0, s1, device=R.device)
        lf = forward_win(R, states["full"], ids, set(), W, P)[0]
        qf = F.log_softmax(lf, -1)
        pf = qf.exp()
        nll_f = -qf.gather(1, tgt[:, None])[:, 0]
        top_f = lf.argmax(-1)
        del lf
        for name, win in cfgs.items():
            if name == "full":
                continue
            lc = forward_win(R, states[name], ids, win, W, P)[0]
            qc = F.log_softmax(lc, -1)
            kl = (pf * (qf - qc)).sum(-1)
            agree = (lc.argmax(-1) == top_f).float()
            dnll = -qc.gather(1, tgt[:, None])[:, 0] - nll_f
            del lc, qc
            for lo, hi in BUCKETS:
                m = (j >= max(lo, W)) & (j < hi)
                if m.any():
                    b = acc[name][f"{lo}-{hi}"]
                    b[0] += int(m.sum()); b[1] += kl[m].sum().item(); b[2] += agree[m].sum().item(); b[3] += dnll[m].sum().item()
        del qf, pf
    return acc


@torch.no_grad()
def smoke(R, pr, gen):
    """(1) forward_win with no windowed layer == Runner.forward on the same chunks; (2) all8 chunked == step-by-step
    block_slots reference (the masked TL definition of tests/test_tl_long.py) at W=256 over 1,024 tokens."""
    P, W, N = len(pr), 256, 1024
    g = torch.tensor(gen[:N], device=R.device)
    st0 = R.new_state(1, P + N + 32); R.prefill(st0, torch.tensor([pr], device=R.device))
    sa, sb = R.new_state(1, st0.Lmax), R.new_state(1, st0.Lmax)
    sa.copy_row_from(0, st0); sb.copy_row_from(0, st0)
    la = torch.cat([forward_win(R, sa, g[None, s:s + 256], set(), W, P)[0] for s in range(0, N, 256)])
    lb = torch.cat([R.forward(sb, g[None, s:s + 256], logits="all")[0] for s in range(0, N, 256)])
    print(f"smoke 1: max|dlogit| win=() vs Runner.forward = {(la - lb).abs().max().item():.4f}", flush=True)
    sc, sd = R.new_state(1, st0.Lmax), R.new_state(1, st0.Lmax)
    sc.copy_row_from(0, st0); sd.copy_row_from(0, st0)
    lc = torch.cat([forward_win(R, sc, g[None, s:s + 128], set(range(len(R.fa_of))), W, P)[0] for s in range(0, N, 128)])
    ld = []
    for i in range(N):
        p = P + i
        if p - W >= P:
            sd.block_slots(0, P, p - W + 1)
        ld.append(R.forward(sd, g[None, i:i + 1])[0])
    ld = torch.stack(ld)
    qc, qd = F.log_softmax(lc, -1), F.log_softmax(ld, -1)
    kl = (qd.exp() * (qd - qc)).sum(-1)
    lu = la  # full logits over the same tokens, for scale
    qu = F.log_softmax(lu, -1)
    klu = (qu.exp() * (qu - qd)).sum(-1)
    print(f"smoke 2: after first wrap (j>={W}): KL(stepref || chunked all8) mean {kl[W:].mean().item():.5f} max "
          f"{kl[W:].max().item():.4f}; argmax agree {(lc.argmax(-1) == ld.argmax(-1))[W:].float().mean().item():.4f}; "
          f"KL(full || stepref) mean {klu[W:].mean().item():.5f}", flush=True)
    ok = kl[W:].mean().item() < 0.1 * klu[W:].mean().item()
    print("SMOKE", "PASS" if ok else "FAIL", flush=True)


def main():
    tok = AutoTokenizer.from_pretrained(a.model)
    probs = {p["problem_id"]: p for l in open(a.problems) if l.strip() for p in [json.loads(l)]}
    test = set(l.strip() for l in open(a.test_ids) if l.strip())
    rows = select(a.traces, test)
    print(f"selected {len(rows)} traces (non-test MATH-500 L5, >= {a.min_gen} output tokens)", flush=True)
    R = Runner(load_model(a.model))
    n_fa = len(R.fa_of)
    print(f"FA layers (model index): {sorted(R.fa_of)}; kv heads {R.cfg.num_key_value_heads}, head_dim {R.cfg.head_dim}", flush=True)
    if a.smoke:
        r = rows[0]
        pr = prompt_ids(tok, probs[r["problem_id"]])
        assert len(pr) == r["prompt_len"], (len(pr), r["prompt_len"])
        smoke(R, pr, r["output_ids"])
        return
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["problem_id"] for l in open(a.out) if l.strip()}
    cfgs = configs(n_fa) if not a.configs else {"full": set(), **{k: parse_cfg(v, n_fa, R.cfg.num_key_value_heads) for k, v in (c.split("=") for c in a.configs.split())}}
    print("configs", {k: (sorted(v) if isinstance(v, set) else {f: sorted(h) for f, h in v.items()}) for k, v in cfgs.items()}, flush=True)
    for i, r in enumerate(rows):
        if i % a.nshards != a.shard or r["problem_id"] in done:
            continue
        pr = prompt_ids(tok, probs[r["problem_id"]])
        assert len(pr) == r["prompt_len"], (r["problem_id"], len(pr), r["prompt_len"])
        gen = r["output_ids"][:a.max_gen]
        acc = run_trace(R, pr, gen, a.W, cfgs, a.chunk)
        with open(a.out, "a") as f:
            f.write(json.dumps({"problem_id": r["problem_id"], "sample_idx": r["sample_idx"], "prompt_len": len(pr),
                                "n_gen": len(gen), "W": a.W, "fa_layers": sorted(R.fa_of), "acc": acc}) + "\n")
        a8 = acc["all8"]
        n = sum(v[0] for v in a8.values())
        print(f"{r['problem_id']} n_gen {len(gen)}: all8 KL/tok {sum(v[1] for v in a8.values()) / max(1, n):.4f}", flush=True)
    print("LAYER_MAP_DONE", flush=True)


if __name__ == "__main__":
    main()
