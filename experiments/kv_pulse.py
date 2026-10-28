"""H2 pulse test (NTI/individual_project/prereg_fade.md, "Where does a memory error live", exploratory): does an FA-cache error
persist in the FA cache or in the GDN recurrent state? Teacher-forced on Qwen3.5 E1 traces (the first --n by (problem_id,
sample_idx) with >= --min-gen generated tokens). Queries at generated positions [at, at + len) see old tokens (the Fade
policy, R / G) at --bits through quantized copies (kv_fade keep mode: the cache itself is never modified); every other
query is exact (bits: plain kv2 by default, prereg change ~18:39 UTC). Runs from the same prefix (E's state cloned at `at`):
  E   exact everywhere (reference);           P   the pulse;
  Pc  P, at at + len GDN conv + recurrent states replaced by E's (only the FA K/V written during the pulse carry error);
  Ps  P, at at + len the FA K/V written during the pulse replaced by E's (only the GDN states carry error);
  Pb  both replaced: must equal E exactly (gate, with "P's cache before the pulse == E's");
  P0  the pulse path with a no-op quantizer (16 bits, forced): the bf16 floor of the keep path (nti-e2, prereg 0814451f).
KL(E || X)/tok of positions after the pulse by distance d = j - (at + len) in [0,64), [64,256), [256,1024), [1024,4096), plus
a 64-token curve. --dense: Qwen3-8B (no state: only P, and P_every4 = layers 3, 7, ... as the layer-count control) on the
re-tokenized trace text ("<think>\\n" + decoded Qwen3.5 tokens, as kv_fade_dense.py); its pulse and distances are the dense
tokens whose text starts inside the same Qwen3.5 token spans (character offsets), so both models see the same text span.
Usage: python kv_pulse.py [--dense] --traces out/traces.jsonl --problems out/problems.jsonl --out O [--smoke]"""
import argparse, bisect, copy, json, os, sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--dense", action="store_true")
ap.add_argument("--model")
ap.add_argument("--ref-model", default="Qwen/Qwen3.5-9B", help="tokenizer of the traces")
ap.add_argument("--traces", required=True)
ap.add_argument("--problems", required=True)
ap.add_argument("--out")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--min-gen", type=int, default=12288)
ap.add_argument("--at", type=int, default=8192)
ap.add_argument("--len", type=int, default=1024)
ap.add_argument("--after", type=int, default=4096)
ap.add_argument("--bits", type=int, default=2, help="plain KIVI (prereg b06aebfb: kv2; kv1 = optional stress variant)")
ap.add_argument("--R", type=int, default=2048)
ap.add_argument("--G", type=int, default=128)
ap.add_argument("--chunk", type=int, default=1024)
ap.add_argument("--bin", type=int, default=64)
ap.add_argument("--smoke", action="store_true", help="gates on 1 trace with at 4096, len 1024, after 1024")
ap.add_argument("--distinct", action="store_true", help="exploratory (prereg ~22:19): sample 0 of each problem in ID order, "
                "skipping problems whose sample 0 has < --min-gen tokens; first --n")
a = ap.parse_args()
if a.smoke:
    a.at, a.len, a.after, a.n = 4096, 1024, 1024, 1
_argv = sys.argv
if a.dense:
    sys.argv = ["kv_fade_dense.py", "--model", a.model or "Qwen/Qwen3-8B", "--ref-model", a.ref_model, "--R", str(a.R), "--G", str(a.G)]
    import kv_fade_dense as H  # noqa: E402
    kf = H.kf
else:
    sys.argv = ["kv_fade.py", "--traces", "-", "--problems", "-", "--test-ids", "-", "--R", str(a.R), "--G", str(a.G)]
    import kv_fade as kf  # noqa: E402
sys.argv = _argv
from fade_chars import chars_by_bucket  # noqa: E402

BUCKETS = [(0, 64), (64, 256), (256, 1024), (1024, 4096)]
THINK = "<think>\n"


def select():
    rows = sorted((r for r in map(json.loads, open(a.traces)) if len(r["output_ids"]) >= a.min_gen
                   and (not a.distinct or r["sample_idx"] == 0)), key=lambda r: (r["problem_id"], r["sample_idx"]))
    return rows[:a.n]


def kl_agree(lE, lX):
    qE, qX = F.log_softmax(lE, -1), F.log_softmax(lX, -1)
    return (qE.exp() * (qE - qX)).sum(-1), (lE.argmax(-1) == lX.argmax(-1)).float()


@torch.no_grad()
def run_trace(M, tok, tref, pr, ref):
    """M: model adapter. ref: Qwen3.5 output ids. Returns the row dict (without ids)."""
    end_ref = a.at + a.len + a.after
    ref = ref[:end_ref + 1]
    edges = [a.at, a.at + a.len] + [a.at + a.len + i * a.bin for i in range(1, a.after // a.bin + 1)]
    if a.dense:  # dense tokens of the same text; ref position of token j = by its first character
        text = THINK + tref.decode(ref)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        gen, st0 = enc["input_ids"], [s for s, _ in enc["offset_mapping"]]
        cb = [len(THINK) + len(tref.decode(ref[:k])) for k in edges]
        pa, pb = bisect.bisect_left(st0, cb[0]), bisect.bisect_left(st0, cb[1])
        cut = [bisect.bisect_left(st0, c) for c in cb[1:]]  # dense index where each 64-token ref bin starts
    else:
        gen = ref
        pa, pb = a.at, a.at + a.len
        cut = edges[1:]
    je = min(len(gen) - 1, cut[-1])  # positions j < je are scored (target j + 1 exists)

    def dbin(j):  # 64-token bin index after the pulse (ref tokens)
        return bisect.bisect_right(cut, j) - 1

    P0 = len(pr)
    E = M.new_state(pr, len(gen))
    for s in range(0, pa, a.chunk):
        M.fwd(E, gen[s:min(pa, s + a.chunk)], None)
    runs = {"P": M.qcfg, "P0": M.q0}
    if a.dense:
        runs["P_every4"] = M.qcfg4
    X = {}
    for n, c in runs.items():
        X[n] = copy.deepcopy(E)
        kf.init_state(X[n], c)
        X[n].keep = True
    acc = {n: {"pulse": [0, 0.0, 0.0], **{f"{lo}-{hi}": [0, 0.0, 0.0] for lo, hi in BUCKETS}} for n in list(runs) + ["Pc", "Ps", "Pb"]}
    curve = {n: [[0, 0.0] for _ in range(a.after // a.bin)] for n in acc}
    gates = {}
    smoke_ref = None
    for s in range(pa, pb, a.chunk):  # the pulse
        ids = gen[s:min(pb, s + a.chunk)]
        if a.smoke and s == pa:  # keep mode vs in-place replacement on the first pulse chunk (key order differs only)
            Q = copy.deepcopy(E)
            kf.init_state(Q, M.qcfg)
            smoke_ref = M.fwd(Q, ids, M.qcfg)
            del Q
        lE = M.fwd(E, ids, None)
        for n, c in runs.items():
            lX = M.fwd(X[n], ids, c)
            k, ag = kl_agree(lE, lX)
            b = acc[n]["pulse"]
            b[0] += len(ids); b[1] += k.sum().item(); b[2] += ag.sum().item()
            if smoke_ref is not None and n == "P":
                gates["keep_vs_inplace_kl"] = kl_agree(smoke_ref, lX)[0].mean().item()
                gates["pulse_kl_first_chunk"] = k.mean().item()
            if a.smoke and s == pa and n == "P0":  # no-op through the keep path vs E (= no-op in place): key order only
                gates["noop_keep_vs_E_kl"] = k.mean().item()
            del lX
        smoke_ref = None
    for n in runs:
        X[n].keep = False
        gates[f"{n}_prefix_cache_equal"] = all(torch.equal(X[n].K[i][:, :, :P0 + pa], E.K[i][:, :, :P0 + pa]) and
                                               torch.equal(X[n].V[i][:, :, :P0 + pa], E.V[i][:, :, :P0 + pa]) for i in range(len(E.K)))
    if not a.dense:
        P = X["P"]
        X["Pc"] = copy.deepcopy(P); X["Pc"].gdn_restore(E.gdn_snapshot())
        X["Ps"] = copy.deepcopy(P)
        for i in range(len(E.K)):
            X["Ps"].K[i][:, :, P0 + pa:P0 + pb] = E.K[i][:, :, P0 + pa:P0 + pb]
            X["Ps"].V[i][:, :, P0 + pa:P0 + pb] = E.V[i][:, :, P0 + pa:P0 + pb]
        X["Pb"] = copy.deepcopy(X["Ps"]); X["Pb"].gdn_restore(E.gdn_snapshot())
    else:
        for n in ("Pc", "Ps", "Pb"):
            del acc[n], curve[n]
    pb_diff = 0.0
    js = torch.arange(len(gen))
    for s in range(pb, je, a.chunk):  # after the pulse: everyone exact
        e = min(je, s + a.chunk)
        ids = gen[s:e]
        lE = M.fwd(E, ids, None)
        bins = torch.tensor([dbin(j) for j in range(s, e)])
        for n in X:
            lX = M.fwd(X[n], ids, None)
            if n == "Pb":
                pb_diff = max(pb_diff, (lX - lE).abs().max().item())
            k, ag = kl_agree(lE, lX)
            k, ag = k.cpu(), ag.cpu()
            for i in range(len(curve[n])):
                m = bins == i
                if m.any():
                    curve[n][i][0] += int(m.sum()); curve[n][i][1] += k[m].sum().item()
            for lo, hi in BUCKETS:
                m = (bins >= lo // a.bin) & (bins < hi // a.bin)
                if m.any():
                    b = acc[n][f"{lo}-{hi}"]
                    b[0] += int(m.sum()); b[1] += k[m].sum().item(); b[2] += ag[m].sum().item()
            del lX
    if not a.dense:
        gates["Pb_max_abs_dlogit"] = pb_diff  # must be 0
    # characters of scored text per bucket (target of position j is token j + 1), for per-character comparisons
    chars = {}
    for lo, hi in BUCKETS:
        c = lambda i: cut[min(i, len(cut) - 1)]
        j0, j1 = min(je, c(lo // a.bin)), min(je, c(hi // a.bin))  # positions [j0, j1) of the bucket
        chars[f"{lo}-{hi}"] = sum(chars_by_bucket(tok, gen[j0:j1 + 1], [(0, 1 << 30)], 0).values()) if j1 > j0 else 0
    chars["pulse"] = sum(chars_by_bucket(tok, gen[pa:pb + 1], [(0, 1 << 30)], 0).values())
    return {"n_gen": len(gen), "pulse": [pa, pb], "end": je, "acc": acc, "curve": curve, "chars": chars, "gates": gates}


class Hybrid:
    def __init__(self):
        from fa_loop import Runner
        self.Rn = Runner(kf.load_model(a.model or "Qwen/Qwen3.5-9B"))
        self.Hm = kf.hadamard(self.Rn.cfg.head_dim, self.Rn.device)
        self.qcfg = kf.Q(a.bits)
        self.q0 = kf.Q(16, force=True)

    def new_state(self, pr, n_gen):
        self.P = len(pr)
        return kf.new_states(self.Rn, pr, n_gen, {"E": kf.CONFIGS["full"]})["E"]

    def fwd(self, st, ids, cfg):
        t = torch.tensor([ids], device=self.Rn.device)
        return kf.forward_cfg(self.Rn, st, t, cfg or kf.CONFIGS["full"], self.P, a.R, a.G, self.Hm)[0]


class Dense:
    def __init__(self):
        from transformers import AutoModelForCausalLM
        self.Rn = H.Dense(AutoModelForCausalLM.from_pretrained(a.model or "Qwen/Qwen3-8B", dtype=torch.bfloat16).cuda().eval())
        self.Hm = kf.hadamard(self.Rn.cfg.head_dim, self.Rn.device)
        nl = len(self.Rn.layers)
        self.qcfg = kf.Q(a.bits, n_fa=nl)
        self.qcfg4 = kf.Q([a.bits if i % 4 == 3 else 16 for i in range(nl)], n_fa=nl)
        self.q0 = kf.Q(16, force=True, n_fa=nl)  # all layers through the keep path: also the (upper) floor of P_every4

    def new_state(self, pr, n_gen):
        self.P = len(pr)
        return H.new_states(self.Rn, pr, n_gen, {"E": {"kind": "full"}})["E"]

    def fwd(self, st, ids, cfg):
        t = torch.tensor([ids], device=self.Rn.device)
        return H.forward_cfg(self.Rn, st, t, cfg or {"kind": "full"}, self.P, a.R, a.G, self.Hm)[0]


def main():
    assert a.chunk % a.G == 0 and a.chunk <= a.R and a.after % a.bin == 0 and all(h % a.bin == 0 for _, h in BUCKETS)
    name = a.model or ("Qwen/Qwen3-8B" if a.dense else "Qwen/Qwen3.5-9B")
    tok, tref = AutoTokenizer.from_pretrained(name), AutoTokenizer.from_pretrained(a.ref_model)
    probs = {p["problem_id"]: p for p in map(json.loads, open(a.problems))}
    rows = select()
    print(f"{len(rows)} traces (first by (problem_id, sample_idx) with >= {a.min_gen} tokens): "
          f"{[(r['problem_id'], r['sample_idx']) for r in rows]}", flush=True)
    M = Dense() if a.dense else Hybrid()
    print(f"{name}: pulse [{a.at}, {a.at + a.len}) at {a.bits} bits, after {a.after}, R {a.R}, G {a.G}, chunk {a.chunk}", flush=True)
    done = set()
    if a.out and os.path.exists(a.out):
        done = {(r["problem_id"], r["sample_idx"]) for r in map(json.loads, open(a.out))}
    ok = True
    for r in rows:
        if (r["problem_id"], r["sample_idx"]) in done:
            continue
        pr = kf.prompt_ids(tok, probs[r["problem_id"]])
        if not a.dense:
            assert len(pr) == r["prompt_len"], (r["problem_id"], len(pr), r["prompt_len"])
        out = run_trace(M, tok, tref, pr, r["output_ids"])
        g = out["gates"]
        good = all(v for k, v in g.items() if k.endswith("_equal")) and g.get("Pb_max_abs_dlogit", 0.0) == 0.0
        if a.smoke:
            good &= g["keep_vs_inplace_kl"] <= 2 * g["noop_keep_vs_E_kl"]  # nti-e2, prereg 0814451f
        ok &= good
        msg = ", ".join(f"{n} " + " ".join(f"{v[1] / max(1, v[0]):.4f}" for k, v in acc.items()) for n, acc in out["acc"].items())
        print(f"{r['problem_id']}#{r['sample_idx']} n_gen {out['n_gen']} pulse {out['pulse']}: KL/tok [pulse, d buckets] {msg}; "
              f"gates {json.dumps(g)}", flush=True)
        if a.out and not a.smoke:
            with open(a.out, "a") as f:
                f.write(json.dumps({"problem_id": r["problem_id"], "sample_idx": r["sample_idx"], "model": name, "prompt_len": len(pr),
                                    "at": a.at, "len": a.len, "after": a.after, "bits": a.bits, "R": a.R, "G": a.G, "bin": a.bin,
                                    **out}) + "\n")
    print("PULSE_SMOKE" if a.smoke else "PULSE_DONE", "PASS" if ok else "FAIL (gates)", flush=True)


if __name__ == "__main__":
    main()
