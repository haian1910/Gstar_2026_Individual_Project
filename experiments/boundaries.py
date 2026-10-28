"""E1 steps 2-5 (spec §2.1, §2.2, §7). Subcommands, run in this order:
  candidates  traces.jsonl -> candidates.jsonl      trigger sentences + exact token spans (think-relative), prints counts
  prompts     traces.jsonl + candidates -> judge_prompts.jsonl   spec §2.2 prompt, non-thinking chat template, prefix only
  judge       judge_prompts.jsonl -> judge_raw[.shardK].jsonl    --backend vllm | dry (dry writes nothing but a note)
  assemble    traces + prompts + judge_raw* -> boundaries.jsonl  parses the JSON, resolves attempt_start_tok, w_is_wrong
Token indices are over the ORIGINAL output_ids, counted from think_start_tok; b_tok/c_tok = sentence start / exclusive end."""
import argparse, glob, json, random, re
from collections import Counter
from fta_common import TokBytes, answers_equal, boxed_all, is_numeric_value, load_jsonl, load_tokenizer, split_paragraphs, write_jsonl

NEG = ["this is wrong", "that's wrong", "that is wrong", "can't be right", "cannot be right", "doesn't work", "does not work", "won't work",
       "contradiction", "impossible", "not possible", "doesn't make sense", "I made a mistake", "made an error", "is incorrect", "not correct",
       "inconsistent", "doesn't match"]
RESTART = ["different approach", "another approach", "alternative approach", "different method", "another method", "another way", "start over",
           "start again", "let me reconsider", "let me rethink", "scrap that", "go back to"]
WEAK = ["Wait", "Alternatively", "Hmm", "Actually"]
_pat = lambda L: re.compile(r"\b(" + "|".join(re.escape(x).replace("'", "['’]") for x in L) + ")", re.I)   # leading \b only: high recall
NEG_RE, RESTART_RE = _pat(NEG), _pat(RESTART)
WEAK_RE = re.compile(r"^[\W_]*(" + "|".join(WEAK) + r")\b", re.I)                                           # sentence start only
SENT_SEP = re.compile(r"(?<=[.!?])\s+|\n+")
LABELS = ("abandon", "local_fix", "verify_confirm", "case_split")
NUM = r"(-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?)"
VAL_PATS = [re.compile(r"the answer is\s*:?\s*\$*\s*([^\n$]{1,40}?)\s*(?:\$|\.\s|\n|$)", re.I), re.compile(r"\bso\b[^\n=]{0,100}=\s*\$*\s*" + NUM, re.I),
            re.compile(r"\bwe get\s*:?\s*\$*\s*" + NUM, re.I)]

INSTR = """You are labelling one moment in a math solution written by a model. You see the problem and the
solution up to and including a marked sentence. Do NOT solve the problem. Decide what the marked
sentence does to the work that precedes it:
- abandon: the writer concludes the preceding approach or attempt is wrong or unworkable and leaves it
  (e.g. "this gives a contradiction, so this approach fails", "that's wrong, let me try another method").
- local_fix: the writer corrects a small error (arithmetic, sign, one step) and keeps the same approach.
- verify_confirm: the writer checks, or is about to check, an existing result, and does not reject it.
- case_split: the writer moves to another case or branch of the same overall approach.
Return JSON only:
{"label": "abandon|local_fix|verify_confirm|case_split",
 "rejected_value": "<value the writer rejected, or null>",
 "attempt_start_paragraph": <index of the paragraph where the abandoned approach began, or null>,
 "approach_summary": "<=15 words on the abandoned approach, or null>"}
"""


# ---------- (a) candidates ----------

def sentences(text):
    """[(start, end)] char spans: split on [.!?] + whitespace, or on newline; whitespace trimmed, empties dropped"""
    out, pos = [], 0
    for m in list(SENT_SEP.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        seg = text[pos:end]
        if seg.strip():
            out.append((pos + len(seg) - len(seg.lstrip()), pos + len(seg.rstrip())))
        pos = m.end() if m else len(text)
    return out


def classify(sent):
    """-> (trigger_type, trigger, all_triggers) for one sentence, priority negation > restart > weak; None if no trigger"""
    hits = [("negation", m.group(1)) for m in NEG_RE.finditer(sent)] + [("restart", m.group(1)) for m in RESTART_RE.finditer(sent)]
    w = WEAK_RE.match(sent)
    if w:
        hits.append(("weak", w.group(1)))
    return (hits[0][0], hits[0][1], [h[1] for h in hits]) if hits else None


def intermediate_values(text):
    vals = boxed_all(text)
    for p in VAL_PATS:
        vals += [m.group(1).strip().rstrip(".,") for m in p.finditer(text)]
    return vals


def think_view(tb, t):
    return tb.view(t["output_ids"][t["think_start_tok"]:t["think_end_tok"]])


def cmd_candidates(a):
    tb = TokBytes(load_tokenizer(a.model))
    cands = []
    for t in load_jsonl(a.traces):
        v = think_view(tb, t)
        for s, e in sentences(v.text):
            c = classify(v.text[s:e])
            if not c:
                continue
            b_tok, c_tok = v.span(s, e)
            w0 = v.char_of_tok(max(0, b_tok - 400))
            cands.append({"boundary_id": f"{t['problem_id']}|{t['sample_idx']}|{b_tok}", "problem_id": t["problem_id"], "sample_idx": t["sample_idx"],
                          "trigger": c[1], "trigger_type": c[0], "triggers": c[2], "b_tok": b_tok, "c_tok": c_tok, "sentence": v.clean(v.text[s:e]),
                          "sent_chars": [s, e], "intermediate_values": intermediate_values(v.text[w0:s])})
    ids = [c["boundary_id"] for c in cands]
    assert len(ids) == len(set(ids)), "duplicate boundary_id (two sentences share a start token)"
    weak = sorted(c["boundary_id"] for c in cands if c["trigger_type"] == "weak")
    drop = set()
    if a.weak == "none":
        drop = set(weak)
    elif a.weak.startswith("sample:"):
        keep = set(random.Random(0).sample(weak, min(int(a.weak.split(":")[1]), len(weak))))
        drop = set(weak) - keep
    n_all = Counter(c["trigger_type"] for c in cands)
    cands = [c for c in cands if c["boundary_id"] not in drop]
    write_jsonl(a.out, cands)
    print(f"candidates before --weak {a.weak}: {dict(n_all)}")
    print(f"written {len(cands)} to {a.out}: {dict(Counter(c['trigger_type'] for c in cands))} | "
          f"traces with >=1: {len({(c['problem_id'], c['sample_idx']) for c in cands})}/{len(load_jsonl(a.traces))}")


# ---------- (b) judge prompts ----------

def build_prompt(tok, problem, solution_block, sentence):
    user = f"{INSTR}\nPROBLEM: {problem}\nSOLUTION SO FAR (numbered paragraphs):\n{solution_block}\nMARKED SENTENCE: <<<{sentence}>>>"
    chat = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    return user, chat


def cmd_prompts(a):
    tok = load_tokenizer(a.model)
    tb = TokBytes(tok)
    cands = {}
    for c in load_jsonl(a.candidates):
        cands.setdefault((c["problem_id"], c["sample_idx"]), []).append(c)
    rows = []
    for t in load_jsonl(a.traces):
        cl = cands.get((t["problem_id"], t["sample_idx"]))
        if not cl:
            continue
        v = think_view(tb, t)
        paras = split_paragraphs(v.text)                       # global numbering: index in this list, from 0 at the start of thinking
        p_tok = [v.span(s, e) for s, e in paras]
        for c in cl:
            s = c["sent_chars"][0]
            k = next(i for i, (ps, pe) in enumerate(paras) if ps <= s < pe or ps > s) if paras else 0
            k = min(k, len(paras) - 1)
            win0 = max(0, c["b_tok"] - a.window)
            wc = v.char_of_tok(win0)
            shown, start_tok = [], {}
            for j in range(k + 1):
                ps, pe = paras[j]
                if j < k and p_tok[j][1] <= win0:
                    continue
                txt = v.text[ps:pe] if j < k else v.text[ps:s]
                if j == k and not txt.strip():
                    continue                                   # marked sentence opens its paragraph: nothing of it precedes the sentence
                if ps < wc:
                    txt = "[...] " + v.text[wc:ps + len(txt)]
                shown.append(f"[{j}] {v.clean(txt.rstrip())}")
                start_tok[str(j)] = p_tok[j][0]
            user, chat = build_prompt(tok, t["problem"], "\n\n".join(shown), c["sentence"])
            rows.append({"boundary_id": c["boundary_id"], "problem_id": t["problem_id"], "sample_idx": t["sample_idx"], "problem": t["problem"],
                         "solution_block": "\n\n".join(shown), "sentence": c["sentence"], "boundary_paragraph": k, "window_start_tok": win0,
                         "para_start_tok": start_tok, "user": user, "prompt": chat})
    write_jsonl(a.out, rows)
    print(f"{len(rows)} judge prompts -> {a.out}")


# ---------- (c) judge runner ----------

def first_json(raw):
    """first balanced {...} in raw that parses as JSON (strings with braces handled), else None"""
    d = None
    i = raw.find("{")
    while i >= 0 and d is None:
        depth, j, in_s, esc = 0, i, False, False
        for j in range(i, len(raw)):
            ch = raw[j]
            if in_s:
                esc, in_s = (not esc and ch == "\\"), not (ch == '"' and not esc)
            elif ch == '"':
                in_s = True
            elif ch in "{}":
                depth += 1 if ch == "{" else -1
                if depth == 0:
                    break
        if depth == 0:
            try:
                d = json.loads(raw[i:j + 1])
            except ValueError:
                pass
        i = raw.find("{", i + 1)
    return d


def parse_judge(raw):
    """-> dict(label, rejected_value, attempt_start_paragraph, approach_summary) or None if nothing usable"""
    if not raw:
        return None
    d = first_json(raw)
    if not isinstance(d, dict):                                # regex fallback
        m = re.search(r'"?label"?\s*:\s*"?(abandon|local_fix|verify_confirm|case_split)', raw)
        if not m:
            return None
        g = lambda k: (re.search(r'"?%s"?\s*:\s*"([^"]*)"' % k, raw) or [None, None])[1]
        pm = re.search(r'attempt_start_paragraph"?\s*:\s*(\d+)', raw)
        d = {"label": m.group(1), "rejected_value": g("rejected_value"), "attempt_start_paragraph": int(pm.group(1)) if pm else None,
             "approach_summary": g("approach_summary")}
    label = d.get("label")
    label = label.strip().lower() if isinstance(label, str) else None
    nul = lambda x: None if x is None or str(x).strip().lower() in ("", "null", "none") else str(x).strip()
    p = d.get("attempt_start_paragraph")
    try:
        p = int(str(p).strip("[] ")) if nul(p) is not None else None
    except ValueError:
        p = None
    return {"label": label if label in LABELS else None, "rejected_value": nul(d.get("rejected_value")),
            "attempt_start_paragraph": p, "approach_summary": nul(d.get("approach_summary"))}


def cmd_judge(a):
    rows = sorted(load_jsonl(a.prompts), key=lambda r: r["boundary_id"])
    rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
    if a.backend == "dry":
        print(f"dry: {len(rows)} prompts for shard {a.shard}/{a.nshards}; nothing generated (prompts are in {a.prompts})")
        return
    out = a.out or f"judge_raw.shard{a.shard}.jsonl"
    import os
    done = {json.loads(l)["boundary_id"] for l in open(out)} if os.path.exists(out) else set()
    rows = [r for r in rows if r["boundary_id"] not in done]
    print(f"shard {a.shard}/{a.nshards}: {len(rows)} to judge ({len(done)} done)", flush=True)
    if not rows:
        return
    from vllm import LLM, SamplingParams
    llm = LLM(a.model, tensor_parallel_size=1, max_model_len=a.max_len, gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True,
              seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, max_tokens=a.max_tokens, seed=0)  # default greedy, 256
    for s in range(0, len(rows), a.chunk):
        ch = rows[s:s + a.chunk]
        outs = llm.generate([r["prompt"] for r in ch], sp, use_tqdm=False)
        with open(out, "a") as f:
            for r, o in zip(ch, outs):
                f.write(json.dumps({"boundary_id": r["boundary_id"], "judge_raw": o.outputs[0].text}) + "\n")
        print(f"{s + len(ch)}/{len(rows)}", flush=True)


# ---------- (d) assemble ----------

def cmd_assemble(a):
    traces = {(t["problem_id"], t["sample_idx"]): t for t in load_jsonl(a.traces)}
    cands = {c["boundary_id"]: c for c in load_jsonl(a.candidates)}
    prompts = {r["boundary_id"]: r for r in load_jsonl(a.prompts)}
    raw = {}
    for fn in sorted(glob.glob(a.judge)):
        for r in load_jsonl(fn):
            raw[r["boundary_id"]] = r["judge_raw"]
    by_trace = {}
    for bid, c in cands.items():
        by_trace.setdefault((c["problem_id"], c["sample_idx"]), []).append(c)
    out = []
    for key, cl in by_trace.items():
        prev_abandon_end = 0
        for c in sorted(cl, key=lambda x: x["b_tok"]):
            bid, p, jr = c["boundary_id"], prompts[c["boundary_id"]], raw.get(c["boundary_id"])
            j = parse_judge(jr) or {"label": None, "rejected_value": None, "attempt_start_paragraph": None, "approach_summary": None}
            start = src = None
            clipped = first_vis = False
            if j["label"] == "abandon":
                st = p["para_start_tok"].get(str(j["attempt_start_paragraph"]))
                if st is not None and st < c["b_tok"]:
                    # a paragraph cut by the judge window starts at the window: the judge never saw text before it
                    clipped = st < p["window_start_tok"]
                    first_vis = str(j["attempt_start_paragraph"]) == str(min(int(k) for k in p["para_start_tok"]))  # the earliest paragraph shown
                    start, src = max(st, p["window_start_tok"]), "judge_paragraph"
                elif prev_abandon_end:
                    start, src = prev_abandon_end, "fallback_prev_abandon"
                else:
                    start, src = 0, "fallback_think_start"
                prev_abandon_end = c["c_tok"]
            w, ref = j["rejected_value"], traces[key]["reference"]
            out.append({"problem_id": c["problem_id"], "sample_idx": c["sample_idx"], "boundary_id": bid, "trigger": c["trigger"],
                        "trigger_type": c["trigger_type"], "attempt_start_tok": start, "attempt_end_tok": c["b_tok"], "b_tok": c["b_tok"],
                        "c_tok": c["c_tok"], "judge_label": j["label"], "rejected_value": w, "approach_summary": j["approach_summary"],
                        "w_is_wrong": bool(w is not None and not answers_equal(w, ref)), "w_is_numeric": is_numeric_value(w), "judge_raw": jr,
                        "attempt_start_paragraph": j["attempt_start_paragraph"], "attempt_start_source": src, "attempt_start_clipped": clipped,
                        "attempt_start_first_visible": first_vis,
                        "boundary_paragraph": p["boundary_paragraph"], "sentence": c["sentence"], "intermediate_values": c["intermediate_values"]})
    out.sort(key=lambda r: (r["problem_id"], r["sample_idx"], r["b_tok"]))
    write_jsonl(a.out, out)
    print(f"{len(out)} boundaries -> {a.out} | labels {dict(Counter(r['judge_label'] for r in out))} | "
          f"unparsed/missing {sum(r['judge_label'] is None for r in out)} | w_is_wrong {sum(r['w_is_wrong'] for r in out)} | "
          f"attempt_start {dict(Counter(r['attempt_start_source'] for r in out if r['judge_label'] == 'abandon'))} | "
          f"clipped to the judge window {sum(r['attempt_start_clipped'] for r in out)} | "
          f"judge picked the first visible paragraph {sum(r['attempt_start_first_visible'] for r in out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    par = argparse.ArgumentParser(add_help=False)
    par.add_argument("--model", default="Qwen/Qwen3.5-9B")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("candidates", parents=[par]); p.set_defaults(f=cmd_candidates)
    p.add_argument("--traces", default="traces.jsonl"); p.add_argument("--out", default="candidates.jsonl")
    p.add_argument("--weak", default="all", help="all | none | sample:N (N weak candidates, seed 0, over the whole file)")
    p = sub.add_parser("prompts", parents=[par]); p.set_defaults(f=cmd_prompts)
    p.add_argument("--traces", default="traces.jsonl"); p.add_argument("--candidates", default="candidates.jsonl")
    p.add_argument("--out", default="judge_prompts.jsonl"); p.add_argument("--window", type=int, default=4096)
    p = sub.add_parser("judge", parents=[par]); p.set_defaults(f=cmd_judge)
    p.add_argument("--prompts", default="judge_prompts.jsonl"); p.add_argument("--out", default=None)
    p.add_argument("--backend", choices=["vllm", "dry"], default="dry")
    p.add_argument("--shard", type=int, default=0); p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--max-len", type=int, default=10240); p.add_argument("--gpu-mem", type=float, default=0.85); p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=256); p.add_argument("--temperature", type=float, default=0.0)  # v3 confirmer: thinking, 0.6/0.95/20
    p.add_argument("--top-p", type=float, default=1.0); p.add_argument("--top-k", type=int, default=-1)
    p = sub.add_parser("assemble", parents=[par]); p.set_defaults(f=cmd_assemble)
    p.add_argument("--traces", default="traces.jsonl"); p.add_argument("--candidates", default="candidates.jsonl")
    p.add_argument("--prompts", default="judge_prompts.jsonl"); p.add_argument("--judge", default="judge_raw*.jsonl")
    p.add_argument("--out", default="boundaries.jsonl")
    a = ap.parse_args()
    a.f(a)
