"""Think-Local (HA_own/experiments/2026-10-27_think-local/PLAN.md): sample k reasoning traces per problem from Qwen3.5
in one of these FA-memory cells, as rows of one continuously refilled batch (same sampler and settings as E3):
  full        FA attends to every token (reference)
  tl-W        FA attends to the prompt + the last W generated tokens (ring buffer of W slots after the prompt); GDN untouched
  sl-W        StreamingLLM: the first 4 tokens (sinks) + the last W tokens of everything after them; the prompt is NOT kept
  dt-C-M      training-free Delethink (Markovian Thinker, arXiv 2510.06557): when the thinking context after the prompt
              reaches C tokens, rebuild the whole state (FA and GDN) from the prompt + the last M generated tokens,
              re-encoded right after the prompt (positions restart there); C, M in tokens
  fq-NAME     full FA memory, but every generated block of G tokens is quantized in place (fade_q.py config NAME, e.g.
              kv2_had) once all of it is older than --fade-R tokens (HA_own/experiments/2026-10-28_fade/PLAN.md)
GDN layers always run on every token they are fed (tl/sl: all generated tokens; dt: since the last rebuild).
Pairing: sample k uses torch seed base_seed + k in every cell. Presence penalty: every generated token so far.
Output --out: problem_id, dataset, cell, sample_k, seed, gen_ids, finished, n_tokens, n_rebuilds, max_fa_slots
(fq: also n_quant_blocks, fade_R, fade_G).
Resumable per (problem_id, cell, sample_k). Usage:
  python tl_gen.py --problems P.jsonl [--problems Q.jsonl] --cell tl-2048 --out tl_gen.jsonl [--k 4] [--max-new 32768]
                   [--bmax 32] [--shard i --nshards n] [--limit N] [--select-ids ids.txt]"""
import argparse, collections, json, os, time
os.environ.setdefault("USE_HUB_KERNELS", "0")
import torch
from transformers import AutoTokenizer
from fa_loop import Runner, sample
import fade_q

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."  # as gen_traces.py (math rows)
ap = argparse.ArgumentParser()
ap.add_argument("--problems", action="append", required=True)
ap.add_argument("--select-ids", default=None, help="file with one problem_id per line (keeps that order)")
ap.add_argument("--cell", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--max-new", type=int, default=32768)
ap.add_argument("--bmax", type=int, default=32)
ap.add_argument("--base-seed", type=int, default=0)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--sinks", type=int, default=4)
ap.add_argument("--fade-R", type=int, default=2048)
ap.add_argument("--fade-G", type=int, default=128)
a = ap.parse_args()

if a.cell.startswith("fq-"):
    KIND, PAR, FQ = "fq", [], fade_q.CONFIGS[a.cell[3:]]
else:
    KIND, *PAR = a.cell.split("-")
    PAR = [int(x) for x in PAR]
assert (KIND, len(PAR)) in {("full", 0), ("tl", 1), ("sl", 1), ("dt", 2), ("fq", 0)}, a.cell
if KIND == "dt":
    assert 0 < PAR[1] < PAR[0], "dt-C-M needs 0 < M < C"


def load_model(name):  # as e3_branch.load_model / eos_ids
    from transformers import AutoModelForImageTextToText
    try:
        m = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16)
    except Exception:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    return m.cuda().eval()


def eos_ids(tok, model):
    e = model.generation_config.eos_token_id
    e = set([e] if isinstance(e, int) else list(e or []))
    e |= {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    return sorted(x for x in e if x is not None)


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(a.model)
    probs = [json.loads(l) for f in a.problems for l in open(f) if l.strip()]
    if a.select_ids:
        byid = {p["problem_id"]: p for p in probs}
        probs = [byid[x.strip()] for x in open(a.select_ids) if x.strip()]
    probs = probs[a.shard::a.nshards]
    if a.limit:
        probs = probs[:a.limit]
    have = set()
    if os.path.exists(a.out):
        have = {(r["problem_id"], r["cell"], r["sample_k"]) for r in map(json.loads, open(a.out))}
    todo = [(p, k) for p in probs for k in range(a.k) if (p["problem_id"], a.cell, k) not in have]
    if not todo:
        print("TL DONE (nothing to do)", flush=True)
        return
    pids = {}
    for p in probs:
        ids = tok.apply_chat_template([{"role": "user", "content": p["problem"] + p.get("suffix", SUFFIX)}],
                                      add_generation_prompt=True, enable_thinking=True, tokenize=True)
        pids[p["problem_id"]] = ids if isinstance(ids, list) else ids["input_ids"]
    Pmax = max(len(pids[p["problem_id"]]) for p, _ in todo)
    model = load_model(a.model)
    R = Runner(model)
    V = model.lm_head.out_features
    EOS = eos_ids(tok, model)
    if KIND in ("full", "fq"):
        Lmax = Pmax + a.max_new + 32
    elif KIND == "tl":
        Lmax = Pmax + PAR[0] + 16
    elif KIND == "sl":
        Lmax = max(Pmax, a.sinks + PAR[0]) + 16
        assert Pmax <= a.sinks + PAR[0], "sl: the prompt must fit in sinks + window at the start"
    else:
        Lmax = Pmax + PAR[0] + 32
    print(f"cell {a.cell} shard {a.shard}/{a.nshards}: {len(todo)} samples of {len(probs)} problems, pool B={a.bmax} "
          f"Lmax={Lmax}, eos {EOS}", flush=True)
    pool = R.new_state(a.bmax, Lmax)
    Hm = fade_q.hadamard(R.cfg.head_dim, R.device) if KIND == "fq" else None
    seen = torch.zeros(a.bmax, V, dtype=torch.bool, device=R.device)
    logits = torch.zeros(a.bmax, V, dtype=torch.float32, device=R.device)
    gens = [torch.Generator(device=R.device) for _ in range(a.bmax)]
    slot_job = [None] * a.bmax
    jobs = collections.deque(todo)
    pstate = {}  # problem_id -> (B=1 state after the prompt, logits); one problem's samples are queued together
    fo = open(a.out, "a")
    done, t0, steps = 0, time.time(), 0

    def prompt_state(pid):
        if pid not in pstate:
            pstate.clear()  # todo is ordered by problem, so one cached prompt state is enough
            st = R.new_state(1, Lmax)
            lg = R.prefill(st, torch.tensor([pids[pid]], device=R.device))
            pstate[pid] = (st, lg[0])
        return pstate[pid]

    def rebuild(r, j):
        """dt: state = prompt + the last M generated tokens, re-encoded right after the prompt"""
        st = R.new_state(1, Lmax)
        carry = j["gen"][-PAR[1]:]
        R.prefill(st, torch.tensor([pids[j["pid"]]], device=R.device), logits=None)
        lg = R.prefill(st, torch.tensor([carry], device=R.device))
        pool.copy_row_from(r, st)
        logits[r] = lg[0]
        j["ctx"] = len(carry)
        j["rebuilds"] += 1
        del st

    def start(r):
        if not jobs:
            slot_job[r] = None
            return
        p, k = jobs.popleft()
        st, lg = prompt_state(p["problem_id"])
        pool.copy_row_from(r, st)
        logits[r] = lg
        seen[r] = False
        gens[r].manual_seed(a.base_seed + k)
        slot_job[r] = dict(pid=p["problem_id"], ds=p.get("dataset"), k=k, gen=[], P=len(pids[p["problem_id"]]), ctx=0,
                           rebuilds=0, max_slots=len(pids[p["problem_id"]]), nq=0)

    for r in range(a.bmax):
        start(r)
    dummy = torch.full((a.bmax,), EOS[0], dtype=torch.long, device=R.device)
    while any(j is not None for j in slot_job):
        nt = sample(logits, seen, gens)
        active = torch.tensor([j is not None for j in slot_job], device=R.device)
        nt = torch.where(active, nt, dummy)
        seen[torch.arange(a.bmax, device=R.device)[active], nt[active]] = True
        nt_l = nt.tolist()
        to_free = []
        for r, j in enumerate(slot_job):
            if j is None:
                continue
            j["gen"].append(nt_l[r])
            fin = nt_l[r] in EOS
            if fin or len(j["gen"]) >= a.max_new:
                fo.write(json.dumps({"problem_id": j["pid"], "dataset": j["ds"], "cell": a.cell, "sample_k": j["k"],
                                     "seed": a.base_seed + j["k"], "finished": fin, "n_tokens": len(j["gen"]),
                                     "n_rebuilds": j["rebuilds"], "max_fa_slots": j["max_slots"], "gen_ids": j["gen"],
                                     **({"n_quant_blocks": j["nq"], "fade_R": a.fade_R, "fade_G": a.fade_G} if KIND == "fq" else {})})
                     + "\n")
                fo.flush()
                done += 1
                to_free.append(r)
        for r in to_free:
            slot_job[r] = None
        for r, j in enumerate(slot_job):
            if j is None:
                pool.length[r] = 0; pool.pos_next[r] = 0; pool.valid[r] = False
        if any(j is not None for j in slot_job):
            if KIND in ("tl", "sl"):
                # ring slot of the token being fed: tl keeps slots [0, P) for the prompt, sl keeps slots [0, sinks)
                sl = torch.zeros(a.bmax, 1, dtype=torch.long, device=R.device)
                for r, j in enumerate(slot_job):
                    if j is None:
                        sl[r, 0] = Lmax - 1  # parked row: write into a scratch slot of an empty row
                        continue
                    base = j["P"] if KIND == "tl" else a.sinks
                    p = j["P"] + len(j["gen"]) - 1  # absolute position of the token being fed
                    sl[r, 0] = base + (p - base) % PAR[0]
                    j["max_slots"] = max(j["max_slots"], min(base + PAR[0], p + 1))
                logits = R.forward(pool, nt[:, None], slot=sl, lcur=Lmax - 16)
            else:
                if KIND == "fq":  # quantize the block that ages at this step's query, per row (generated index len(gen) - 1)
                    rows = [(r, j["P"], len(j["gen"]) - 1) for r, j in enumerate(slot_job) if j is not None]
                    for r, _ in fade_q.age_rows(pool, rows, FQ, Hm, a.fade_R, a.fade_G):
                        slot_job[r]["nq"] += 1
                logits = R.forward(pool, nt[:, None])
                for r, j in enumerate(slot_job):
                    if j is not None:
                        j["ctx"] += 1
                        j["max_slots"] = max(j["max_slots"], j["P"] + j["ctx"])
            steps += 1
        if KIND == "dt":
            for r, j in enumerate(slot_job):
                if j is not None and r not in to_free and j["ctx"] >= PAR[0]:
                    rebuild(r, j)
        for r in to_free:  # refill AFTER the forward (PLAN #23 of the forget-attempt run)
            start(r)
        if steps % 500 == 0:
            print(f"step {steps} active {sum(j is not None for j in slot_job)} done {done} queued {len(jobs)} "
                  f"{steps / (time.time() - t0):.1f} steps/s", flush=True)
    print(f"TL DONE {done} samples, {steps} steps, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
