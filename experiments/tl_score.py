"""Think-Local scoring: tl_gen.py outputs -> runs file for NTI/individual_project/src/tl_stats.py (nti-e2's prereg_tl.md).
Per run: problem_id, dataset, cell (canonical: full, tl-w1k, tl-w2k, tl-w4k, dt-c4k-m2k, sl-2k; raw name in cell_raw),
sample_k, correct, finished, n_tokens (every generated token incl. the final EOS: the count the cap applies to),
max_tokens, answer (normalised final answer, or None), rep_score, repetitive, rep_threshold, n_rebuilds, n_reencoded,
max_fa_slots, correct16, finished16 (finished within 16,384 tokens; correct16 needs both), n_triggers (trigger sentences
in the thinking: sentence-initial Wait/Alternatively/Hmm/Actually, negations, restarts; boundaries.classify) and
n_triggers_per_1k (per 1,000 thinking tokens).
Final answer: the last \\boxed{} after </think>; if the run finished without one there, the last \\boxed{} in its thinking
(as postprocess.py / e3_score.py). Math: answers_equal with the reference. Countdown: the boxed expression must use
each given number exactly once and equal the target (pilot_check.countdown_ok; no exec).
rep_score: share of the run's token 64-grams already seen earlier in the same run; repetitive = rep_score >= REP_T
(fixed 2026-10-26 before any Think-Local run was generated).
Usage: python tl_score.py --gen 'tl_gen.*.jsonl' --problems problems.jsonl --problems countdown.jsonl --out tl_runs.jsonl
       [--max-tokens 32768] [--text-out tl_text.jsonl]"""
import argparse, glob, json, re
from boundaries import classify, sentences  # trigger lexicon of the forget-attempt run (n_triggers: exploratory moderator)
from fta_common import TokBytes, answers_equal, last_boxed, load_jsonl, load_tokenizer, normalize_answer, write_jsonl

REP_N, REP_T = 64, 0.30
ap = argparse.ArgumentParser()
ap.add_argument("--gen", action="append", required=True, help="glob, repeatable")
ap.add_argument("--problems", action="append", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=32768)
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--text-out", default=None)
a = ap.parse_args()


def canon(cell):
    kind, *p = cell.split("-")
    k = lambda x: f"{int(x) // 1024}k" if int(x) % 1024 == 0 else str(x)
    return {"full": lambda: "full", "tl": lambda: f"tl-w{k(p[0])}", "sl": lambda: f"sl-{k(p[0])}",
            "dt": lambda: f"dt-c{k(p[0])}-m{k(p[1])}", "fq": lambda: cell}[kind]()


def rep_score(ids):
    if len(ids) <= REP_N:
        return 0.0
    import array
    b = array.array("i", ids).tobytes()
    w = 4 * REP_N
    seen, rep = set(), 0
    for i in range(0, len(b) - w + 4, 4):
        s = b[i:i + w]
        if s in seen:
            rep += 1
        else:
            seen.add(s)
    return rep / (len(ids) - REP_N + 1)


def countdown_ok(expr, nums, target):  # same checker as pilot_check.py (unit-tested there)
    import ast
    from fractions import Fraction
    if not expr:
        return False
    e = expr.replace("\\times", "*").replace("\\cdot", "*").replace("×", "*").replace("\\div", "/").replace("÷", "/")
    for _ in range(4):
        e = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", e)
    e = re.sub(r"\\left|\\right|\$|\\,|\\!", "", e).replace("{", "(").replace("}", ")").replace("−", "-")
    if "=" in e:
        sides = [s.strip() for s in e.split("=")]
        e = next((s for s in sides if s and s != str(target)), sides[0])
    used, OPS = [], {ast.Add: lambda x, y: x + y, ast.Sub: lambda x, y: x - y, ast.Mult: lambda x, y: x * y,
                     ast.Div: lambda x, y: x / y}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            x, y = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Div) and y == 0:
                raise ZeroDivisionError
            return OPS[type(n.op)](x, y)
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            used.append(n.value)
            return Fraction(n.value)
        raise ValueError(type(n).__name__)
    try:
        v = ev(ast.parse(e, mode="eval"))
    except Exception:
        return False
    return v == target and sorted(used) == sorted(nums)


tok = load_tokenizer(a.model)
tb = TokBytes(tok)
probs = {p["problem_id"]: p for f in a.problems for p in load_jsonl(f)}
files = sorted({f for g in a.gen for f in glob.glob(g)})
assert files, a.gen
rows, texts, seen_keys = [], [], set()
for f in files:
    for g in load_jsonl(f):
        key = (g["problem_id"], g["cell"], g["sample_k"])
        assert key not in seen_keys, f"duplicate run {key} ({f})"
        seen_keys.add(key)
        p = probs[g["problem_id"]]
        ids = list(g["gen_ids"])
        while ids and ids[-1] in tb.eos:
            ids.pop()
        te = next((i for i, x in enumerate(ids) if x == tb.think_close), None)
        think = tb.text(ids[:te] if te is not None else ids)
        answer_text = tb.text(ids[te + 1:]) if te is not None else ""
        final = last_boxed(answer_text) or None
        if final is None and g["finished"]:
            final = last_boxed(think) or None
        if p.get("dataset") == "countdown":
            ok = countdown_ok(final, p["meta"]["nums"], p["meta"]["target"])
            ans = final if ok else None
        else:
            ok = bool(answers_equal(final, p["answer"]))
            ans = normalize_answer(final) if final is not None else None
        rs = rep_score(g["gen_ids"])
        kind = g["cell"].split("-")
        n_re = g.get("n_rebuilds", 0) * int(kind[2]) if kind[0] == "dt" else 0
        fin16 = bool(g["finished"]) and g["n_tokens"] <= 16384
        n_trig = sum(classify(think[s0:s1]) is not None for s0, s1 in sentences(think))
        n_think = te if te is not None else len(ids)
        rows.append({"problem_id": g["problem_id"], "dataset": p.get("dataset"), "cell": canon(g["cell"]), "cell_raw": g["cell"],
                     "sample_k": g["sample_k"], "correct": ok, "finished": bool(g["finished"]), "n_tokens": g["n_tokens"],
                     "max_tokens": a.max_tokens, "answer": ans, "rep_score": round(rs, 4), "repetitive": rs >= REP_T,
                     "rep_threshold": REP_T, "n_rebuilds": g.get("n_rebuilds", 0), "n_reencoded": n_re,
                     "max_fa_slots": g.get("max_fa_slots"), "closed_think": te is not None,
                     "finished16": fin16, "correct16": ok and fin16, "n_triggers": n_trig,
                     "n_triggers_per_1k": round(1000 * n_trig / max(1, n_think), 3)})
        if a.text_out:
            texts.append({"problem_id": g["problem_id"], "cell": g["cell"], "sample_k": g["sample_k"], "think": think,
                          "answer": answer_text})
write_jsonl(a.out, rows)
if a.text_out:
    write_jsonl(a.text_out, texts)
from collections import defaultdict
agg = defaultdict(list)
for r in rows:
    agg[(r["cell"], "countdown" if r["dataset"] == "countdown" else "math")].append(r)
for (c, d), rs in sorted(agg.items()):
    n = len(rs)
    print(f"{c:12s} {d:9s} n={n:4d} acc={sum(r['correct'] for r in rs) / n:.3f} fin={sum(r['finished'] for r in rs) / n:.3f} "
          f"len={sum(r['n_tokens'] for r in rs) / n:7.0f} rep={sum(r['repetitive'] for r in rs) / n:.3f}")
print(f"{len(rows)} runs from {len(files)} files -> {a.out}")
