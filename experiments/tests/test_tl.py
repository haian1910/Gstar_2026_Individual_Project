"""Think-Local checks on a small Qwen3.5 (run on any GPU): python tests/test_tl.py [--model Qwen/Qwen3.5-0.8B]
1. ring window == mask reference: teacher-force one token sequence through (a) the ring-buffer path of fa_loop (slot
   override, prompt pinned, W slots) and (b) the plain path with every slot older than W blocked; logits must match.
2. ring with W larger than the run == full attention (no wrap): logits must match.
3. tl_gen.py end to end for full / tl / sl / dt on two problems (short budgets): runs, fields, dt rebuild count and
   peak FA slots."""
import argparse, json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("USE_HUB_KERNELS", "0")
import torch
from transformers import AutoTokenizer
from fa_loop import Runner

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
ap.add_argument("--steps", type=int, default=160)
ap.add_argument("--W", type=int, default=48)
ap.add_argument("--fp32", action="store_true", help="tests 1-2 in float32 (reordering noise ~0), then stop")
a = ap.parse_args()
code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model(name, dtype=torch.bfloat16):  # as tl_gen.load_model (tl_gen parses its argv at import)
    from transformers import AutoModelForImageTextToText
    try:
        m = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype)
    except Exception:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    return m.cuda().eval()


def run_ring(R, prompt, toks, W, Lmax):
    st = R.new_state(1, Lmax)
    R.prefill(st, torch.tensor([prompt], device=R.device), logits=None)
    P, out = len(prompt), []
    for i, t in enumerate(toks):
        p = P + i
        sl = torch.tensor([[P + (p - P) % W]], device=R.device)
        out.append(R.forward(st, torch.tensor([[t]], device=R.device), slot=sl, lcur=P + W)[0])
    return torch.stack(out)


def run_ref(R, prompt, toks, W, Lmax):
    st = R.new_state(1, Lmax)
    R.prefill(st, torch.tensor([prompt], device=R.device), logits=None)
    P, out = len(prompt), []
    for i, t in enumerate(toks):
        p = P + i
        if p - W >= P:
            st.block_slots(0, P, p - W + 1)  # visible: prompt + the last W tokens including the one being fed
        out.append(R.forward(st, torch.tensor([[t]], device=R.device))[0])
    return torch.stack(out)


tok = AutoTokenizer.from_pretrained(a.model)
model = load_model(a.model, torch.float32 if a.fp32 else torch.bfloat16)
R = Runner(model)
prompt = tok.apply_chat_template([{"role": "user", "content": "What is 17*23? Think step by step."}],
                                 add_generation_prompt=True, enable_thinking=True, tokenize=True)
prompt = prompt if isinstance(prompt, list) else prompt["input_ids"]
g = torch.Generator(device="cpu").manual_seed(0)
toks = torch.randint(100, 20000, (a.steps,), generator=g).tolist()
Lmax = len(prompt) + a.steps + 64
ring = run_ring(R, prompt, toks, a.W, Lmax)
ref = run_ref(R, prompt, toks, a.W, Lmax)
d1 = (ring - ref).abs().max().item()
same1 = (ring.argmax(-1) == ref.argmax(-1)).float().mean().item()
full = run_ref(R, prompt, toks, 10 ** 9, Lmax)
big = run_ring(R, prompt, toks, a.steps + 8, Lmax)
d2 = (big - full).abs().max().item()
dfw = (full - ref).abs().max().item()
print(f"1. ring W={a.W} vs mask reference: max|dlogit| {d1:.4f}, argmax agree {same1:.3f}")
print(f"2. ring W>run vs full: max|dlogit| {d2:.4f}   (full vs window, for scale: {dfw:.3f})")
ok12 = d1 < 0.05 * max(dfw, 1e-3) + 0.05 and d2 < 0.05 + 0.05 * dfw and dfw > 0.5
print("TEST 1-2", "PASS" if ok12 else "FAIL")
if a.fp32:
    first = next(i for i in range(a.steps) if i >= a.W)  # first step after the window starts dropping tokens
    print("fp32 per-step max|dlogit| around the first wrap:", [round((ring[i] - ref[i]).abs().max().item(), 5) for i in range(first - 2, first + 4)])
    raise SystemExit(0)
del R, model
torch.cuda.empty_cache()

tmp = tempfile.mkdtemp()
pf = os.path.join(tmp, "p.jsonl")
with open(pf, "w") as f:
    for i, (q, s) in enumerate([("What is the sum of the first 30 odd numbers?", None),
                                ("Using the numbers [3, 7, 8, 9], create an equation that equals 24. Each number must be "
                                 "used exactly once.", "\n\nPut your final equation within \\boxed{}.")]):
        f.write(json.dumps({"problem_id": f"t{i}", "dataset": "test", "problem": q, **({"suffix": s} if s else {})}) + "\n")
res = {}
for cell in ["full", "tl-64", "sl-64", "dt-96-48"]:
    out = os.path.join(tmp, f"{cell}.jsonl")
    cmd = [sys.executable, os.path.join(code, "tl_gen.py"), "--problems", pf, "--cell", cell, "--out", out, "--k", "2",
           "--max-new", "400", "--bmax", "3", "--model", a.model]
    rc = subprocess.run(cmd, cwd=code, capture_output=True, text=True)
    rows = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
    res[cell] = rows
    print(cell, "rc", rc.returncode, "rows", len(rows), [(r["n_tokens"], r["finished"], r["n_rebuilds"], r["max_fa_slots"]) for r in rows])
    if rc.returncode:
        print(rc.stderr[-2000:])
P = {r["problem_id"]: None for r in res["full"]}
ok3 = all(len(v) == 4 for v in res.values())
for r in res["dt-96-48"]:
    if r["n_tokens"] > 96:
        exp = 1 + (r["n_tokens"] - 1 - 96) // 48  # first rebuild after 96 fed tokens, then every C - M = 48
        ok3 &= abs(r["n_rebuilds"] - exp) <= 1
for c, W in (("tl-64", 64), ("sl-64", 64)):
    for r in res[c]:
        ok3 &= r["max_fa_slots"] <= 400 + W  # prompt (< 400 tokens) + W
print("TEST 3", "PASS" if ok3 else "FAIL")
