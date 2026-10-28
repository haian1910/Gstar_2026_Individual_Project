"""Run: <venv>/bin/python -m pytest code/tests -q   (needs the Qwen3.5 tokenizer; the 0.8B one is identical)"""
import json, os, subprocess, sys
from pathlib import Path
import pytest

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
from fta_common import SUFFIX, TokBytes, answers_equal, boxed_all, last_boxed, load_jsonl, load_tokenizer, split_paragraphs
import boundaries as B

MODEL = os.environ.get("FTA_TOK", "Qwen/Qwen3.5-9B")
TEXTS = [
    "Okay, let me parse the problem.\n\nWe need $x^2 + y^2 = 25$ with $x, y \\in \\mathbb{Z}$.\n\nTry x = 3, y = 4. Then $\\frac{3}{4}$ works. "
    "Wait, that gives a contradiction with $x > y$.\n\nThis is wrong. Let me try a different approach: set $x = \\sqrt{25 - y^2}$.\n\n"
    "So the sum = 12 and we get 7 as the count. Hmm, let me check the case $n = 3.5$. Actually, caf\u00e9 na\u00efve \u4e2d\u6587 \u6d4b\u8bd5 \U0001F600 \u2014 that's wrong too! "
    "The answer is 42.\n\nAlternatively, another way: brute force over $\\{1, 2, 3\\}$ gives \\boxed{\\frac{1}{2}} which is not correct.\n",
    "First, \u03b1 + \u03b2 = \u221a2 \u2260 0.\n\n\n\nWait... the value 3.14 is impossible here? I made a mistake: \\[\n\\sum_{k=1}^{n} k = \\frac{n(n+1)}{2}\n\\]\nand Wait, "
    "this doesn\u2019t work. Hmm\nStart over. Let me reconsider: $a_1 = 1$.\n\nSo x = 17.\n",
    "No triggers in this one at all. Just arithmetic: 2 + 2 = 4.\n\nDone.",
]
ANSWERS = ["The final answer is $\\boxed{\\frac{1}{2}}$.", "\\boxed{17}", "\\boxed{4}"]


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(MODEL)


@pytest.fixture(scope="module")
def tb(tok):
    return TokBytes(tok)


def make_raw(tok, path, thinks=TEXTS, answers=ANSWERS, truncate_last=True):
    prompt = tok.apply_chat_template([{"role": "user", "content": "Compute things." + SUFFIX}], add_generation_prompt=True, enable_thinking=True, tokenize=True)
    prompt = list(prompt["input_ids"] if not isinstance(prompt, list) else prompt)
    close, eos = tok.convert_tokens_to_ids("</think>"), tok.convert_tokens_to_ids("<|im_end|>")
    rows = []
    for i, (t, ans) in enumerate(zip(thinks, answers)):
        ids = tok(t)["input_ids"]
        trunc = truncate_last and i == len(thinks) - 1
        out = ids if trunc else ids + [close] + tok("\n\n" + ans)["input_ids"] + [eos]
        rows.append({"problem_id": f"p{i % 2}", "dataset": "fake", "sample_idx": i // 2, "reference": "17" if i == 1 else "1/2", "prompt_ids": prompt,
                     "prompt_len": len(prompt), "output_ids": out, "text": tok.decode(out, skip_special_tokens=True),
                     "finish_reason": "length" if trunc else "stop", "n_tokens": len(out), "seed": 0})
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def test_tokbytes_matches_decode(tok, tb):
    for t in TEXTS + ["\U0001F600\U0001F600 \u4e2d\u6587"]:
        ids = tok(t)["input_ids"]
        assert tb.text(ids) == t == tok.decode(ids)


@pytest.mark.parametrize("ti", range(len(TEXTS)))
def test_candidate_spans_exact(tok, tb, ti):
    ids = tok(TEXTS[ti])["input_ids"]
    v = tb.view(ids)
    n = 0
    for s, e in B.sentences(v.text):
        c = B.classify(v.text[s:e])
        if not c:
            continue
        n += 1
        b, cc = v.span(s, e)
        assert 0 <= b < cc <= len(ids)
        dec = tok.decode(ids[b:cc])
        assert v.text[s:e] in dec, (v.text[s:e], dec)
        assert tb.text(ids[b:cc]).strip().startswith(v.text[s:s + 3]) or not v.text[s:s + 3].isascii() or v.text[s:e] in dec
        assert len(dec) < len(v.text[s:e]) + 40                      # tight: no more than boundary tokens' slack
    assert n >= (0 if ti == 2 else 3)


def test_char_token_roundtrip_multibyte(tok, tb):
    ids = tok("a \U0001F600 \u4e2d\u6587 b")["input_ids"]
    v = tb.view(ids)
    for ch in range(len(v.text)):
        t = v.tok_of_char(ch)
        assert v.text[ch] in tb.text(ids[t:t + 1]) or ord(v.text[ch]) > 0x7F   # char starts inside token t
    assert v.span(0, len(v.text)) == (0, len(ids))


def test_segmentation_and_classify():
    t = "Value is 3.5 here. Wait, that is wrong!\nNew line? Yes"
    assert [t[s:e] for s, e in B.sentences(t)] == ["Value is 3.5 here.", "Wait, that is wrong!", "New line?", "Yes"]
    assert B.classify("Wait, that is wrong")[:2] == ("negation", "that is wrong")
    assert B.classify("**Hmm**, let me see")[:2] == ("weak", "Hmm")
    assert B.classify("I did not wait here") is None and B.classify("Then, wait") is None       # weak only at sentence start
    assert B.classify("Let me try ANOTHER WAY.")[:2] == ("restart", "ANOTHER WAY")
    assert B.classify("this doesn\u2019t work")[0] == "negation"


def test_boxed_and_paragraphs():
    assert last_boxed("a \\boxed{1} b \\boxed{\\frac{1}{2}} c") == "\\frac{1}{2}"
    assert last_boxed("\\boxed{\\{1,2\\}}") == "\\{1,2\\}" and last_boxed("\\boxed{3") is None and boxed_all("no") == []
    t = "a\n\nb\nc\n \n\nd"
    assert [t[s:e] for s, e in split_paragraphs(t)] == ["a", "b\nc", "d"]


def test_answers_equal():
    assert answers_equal("\\frac{1}{2}", "0.5") and answers_equal("17", "17.0") and not answers_equal("17", "18") and not answers_equal(None, "1")


def test_parse_judge():
    p = B.parse_judge('Sure.\n{"label": "abandon", "rejected_value": "12", "attempt_start_paragraph": 3, "approach_summary": "tried {brute} force"}')
    assert p == {"label": "abandon", "rejected_value": "12", "attempt_start_paragraph": 3, "approach_summary": "tried {brute} force"}
    p = B.parse_judge('{"label": "local_fix", "rejected_value": null, "attempt_start_paragraph": null, "approach_summary": null}')
    assert p["rejected_value"] is None and p["attempt_start_paragraph"] is None
    assert B.parse_judge('{"label": "abandon", "rejected_value": "5", "attempt_start_paragraph": 2,')["label"] == "abandon"       # truncated -> regex
    assert B.parse_judge("garbage")is None and B.parse_judge("") is None
    assert B.parse_judge('{"label": "bogus"}')["label"] is None


def run(args, cwd):
    r = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, env={**os.environ, "FTA_TOK": MODEL})
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_end_to_end_dry(tok, tb, tmp_path):
    make_raw(tok, tmp_path / "raw_0.jsonl")
    out = run([str(CODE / "postprocess.py"), str(tmp_path / "raw_*.jsonl"), "--out", "traces.jsonl", "--model", MODEL], tmp_path)
    tr = load_jsonl(tmp_path / "traces.jsonl")
    assert [t["truncated"] for t in tr] == [False, False, True] and tr[0]["correct"] and tr[1]["correct"] and tr[0]["final_answer"] == "\\frac{1}{2}"
    assert tr[2]["final_answer"] is None and tr[2]["answer_text"] == "" and tr[0]["think_start_tok"] == 0
    assert tr[0]["think_text"] == TEXTS[0] and tr[0]["n_think_tokens"] == len(tok(TEXTS[0])["input_ids"]) and tr[0]["problem"] == "Compute things."
    cmd = lambda *a: run([str(CODE / "boundaries.py"), a[0], "--model", MODEL, *a[1:]], tmp_path)
    o = cmd("candidates", "--weak", "all")
    c1 = load_jsonl(tmp_path / "candidates.jsonl")
    cmd("candidates", "--weak", "all", "--out", "c2.jsonl")
    assert c1 == load_jsonl(tmp_path / "c2.jsonl")                                           # deterministic
    types = {c["trigger_type"] for c in c1}
    assert types == {"negation", "restart", "weak"}, types
    cmd("candidates", "--weak", "none", "--out", "c_none.jsonl")
    assert all(c["trigger_type"] != "weak" for c in load_jsonl(tmp_path / "c_none.jsonl"))
    cmd("candidates", "--weak", "sample:2", "--out", "c_s2.jsonl")
    assert sum(c["trigger_type"] == "weak" for c in load_jsonl(tmp_path / "c_s2.jsonl")) == 2
    # exact spans over output_ids
    by = {(t["problem_id"], t["sample_idx"]): t for t in tr}
    for c in c1:
        ids = by[(c["problem_id"], c["sample_idx"])]["output_ids"]
        assert c["sentence"] in tok.decode(ids[c["b_tok"]:c["c_tok"]]), c
    assert any(c["intermediate_values"] for c in c1)
    cmd("prompts")
    pr = load_jsonl(tmp_path / "judge_prompts.jsonl")
    assert len(pr) == len(c1)
    for p in pr:
        assert p["prompt"].endswith("<think>\n\n</think>\n\n") and f"MARKED SENTENCE: <<<{p['sentence']}>>>" in p["prompt"]
        assert p["prompt"].count("MARKED SENTENCE") == 1 and p["prompt"].rstrip().endswith("<think>\n\n</think>") is True
    o = cmd("judge", "--backend", "dry")
    # no look-ahead: "brute force" first appears in the LAST candidate sentence of trace p0|0, so no earlier prompt may contain it
    t0 = sorted([c for c in c1 if (c["problem_id"], c["sample_idx"]) == ("p0", 0)], key=lambda c: c["b_tok"])
    assert "brute" in t0[-1]["sentence"]
    by_id = {p["boundary_id"]: p for p in pr}
    assert all("brute" not in by_id[c["boundary_id"]]["prompt"] for c in t0[:-1])
    for kw in ("correct answer", "reference"):
        assert kw not in " ".join(p["solution_block"] for p in pr)
    # fake judge: first candidate of trace p0|0 -> abandon w/ paragraph, all others abandon w/ bad paragraph, one garbage
    jr = []
    for i, p in enumerate(sorted(pr, key=lambda p: p["boundary_id"])):
        if i == 0:
            raw = "garbage"
        else:
            k = sorted(p["para_start_tok"], key=int)[0]
            raw = json.dumps({"label": "abandon" if i % 2 else "verify_confirm", "rejected_value": "12" if i % 2 else None,
                              "attempt_start_paragraph": int(k) if i % 4 == 1 else 999, "approach_summary": "x"})
        jr.append({"boundary_id": p["boundary_id"], "judge_raw": raw})
    with open(tmp_path / "judge_raw.shard0.jsonl", "w") as f:
        for r in jr:
            f.write(json.dumps(r) + "\n")
    o = cmd("assemble")
    bd = load_jsonl(tmp_path / "boundaries.jsonl")
    assert len(bd) == len(c1) and sum(b["judge_label"] is None for b in bd) == 1
    keys = {"problem_id", "sample_idx", "boundary_id", "trigger", "attempt_start_tok", "b_tok", "c_tok", "judge_label", "rejected_value", "approach_summary", "w_is_wrong",
            "judge_raw", "trigger_type", "attempt_end_tok", "attempt_start_paragraph"}
    assert keys <= set(bd[0])
    for b in bd:
        assert b["boundary_id"] == f"{b['problem_id']}|{b['sample_idx']}|{b['b_tok']}" and b["attempt_end_tok"] == b["b_tok"]
        if b["judge_label"] == "abandon":
            assert 0 <= b["attempt_start_tok"] <= b["b_tok"] and b["w_is_wrong"] is True
        else:
            assert b["attempt_start_tok"] is None and b["w_is_wrong"] is False
        if b["attempt_start_source"] == "judge_paragraph":  # clipped to the judge window
            p = by_id[b["boundary_id"]]
            st = p["para_start_tok"][str(b["attempt_start_paragraph"])]
            assert b["attempt_start_tok"] == max(st, p["window_start_tok"]) and b["attempt_start_clipped"] == (st < p["window_start_tok"])
            assert b["attempt_start_first_visible"] == (int(b["attempt_start_paragraph"]) == min(int(k) for k in p["para_start_tok"]))
            assert not b["attempt_start_clipped"] or b["attempt_start_first_visible"]  # only the first shown paragraph can be cut
    src = {b["attempt_start_source"] for b in bd if b["judge_label"] == "abandon"}
    assert "judge_paragraph" in src and src & {"fallback_prev_abandon", "fallback_think_start"}
    prev = None
    for b in sorted([b for b in bd if (b["problem_id"], b["sample_idx"]) == ("p0", 0)], key=lambda b: b["b_tok"]):   # fallback = end of previous abandon
        if b["attempt_start_source"] == "fallback_prev_abandon":
            assert b["attempt_start_tok"] == prev
        if b["judge_label"] == "abandon":
            prev = b["c_tok"]
    # annotation export: deterministic, no labels
    o = run([str(CODE / "export_annotation.py"), "--n", "6"], tmp_path)
    a1 = (tmp_path / "boundaries_for_annotation.jsonl").read_text()
    run([str(CODE / "export_annotation.py"), "--n", "6"], tmp_path)
    assert a1 == (tmp_path / "boundaries_for_annotation.jsonl").read_text()
    ann = load_jsonl(tmp_path / "boundaries_for_annotation.jsonl")
    assert len(ann) == 6 and all(set(r) == {"boundary_id", "problem", "prefix", "sentence", "judge_input"} for r in ann)
    for r in ann:  # the judge's input text (instructions + schema): no reference / correctness
        assert r["prefix"] in r["judge_input"] and not any(x in r["judge_input"] for x in ("reference", "correct answer"))
    # confirmer (PLAN #16): prompts for numeric judge-abandons only, continuation = original think tokens [c, c + n)
    conf = lambda *a: run([str(CODE / "confirm.py"), a[0], "--model", MODEL, *a[1:]], tmp_path)
    conf("prompts", "--n-tokens", "20")
    cp = load_jsonl(tmp_path / "confirm_prompts.jsonl")
    num_ab = sorted(b["boundary_id"] for b in bd if b["judge_label"] == "abandon" and b["w_is_numeric"])
    assert [r["boundary_id"] for r in cp] == num_ab and len(cp) >= 3
    trs = {(t["problem_id"], t["sample_idx"]): t for t in tr}
    bd_id = {b["boundary_id"]: b for b in bd}
    for r in cp:
        b = bd_id[r["boundary_id"]]
        t = trs[(b["problem_id"], b["sample_idx"])]
        c = t["think_start_tok"] + b["c_tok"]
        cont = t["output_ids"][c:min(c + 20, t["think_end_tok"])]
        assert r["n_continuation_tokens"] == len(cont) and r["user"].endswith("WHAT THE WRITER WROTE NEXT:\n" + tok.decode(cont))
        assert r["prompt"].rstrip().endswith("<think>\n\n</think>") and by_id[r["boundary_id"]]["solution_block"] in r["user"]
        assert not any(x in r["user"] for x in ("reference", "correct answer")) and "</think>" not in tok.decode(cont)
    fake = {}
    for i, r in enumerate(cp):
        k = sorted(r["para_start_tok"], key=int)[0]
        fake[r["boundary_id"]] = [json.dumps({"label": "abandon", "rejected_value": "7", "new_value": "9", "attempt_start_paragraph": int(k)}),
                                  'x {"label": "abandon", "rejected_value": null, "attempt_start_paragraph": 999}',
                                  '{"label": "no_change", "rejected_value": null}'][i % 3]
    with open(tmp_path / "confirm_raw.shard0.jsonl", "w") as f:
        for bid, raw in fake.items():
            f.write(json.dumps({"boundary_id": bid, "judge_raw": raw}) + "\n")
    conf("assemble")
    bc = load_jsonl(tmp_path / "boundaries_confirmed.jsonl")
    assert [b["boundary_id"] for b in bc] == [b["boundary_id"] for b in bd]
    cpi = {r["boundary_id"]: r for r in cp}
    for b0, b in zip(bd, bc):
        if b0["boundary_id"] not in cpi:
            assert b == {**b0, "confirm_label": None}
            continue
        i = num_ab.index(b["boundary_id"]) % 3
        assert b["confirm_label"] == ["abandon", "abandon", "no_change"][i]
        if i == 0:   # confirmer's value and paragraph (window-clipped) replace the judge's; judge's kept
            r = cpi[b["boundary_id"]]
            st = r["para_start_tok"][str(b["confirm_attempt_start_paragraph"])]
            assert b["rejected_value"] == "7" and b["w_is_wrong"] is True and b["judge_rejected_value"] == "12" and b["rejected_value_source"] == "confirm"
            assert b["attempt_start_source"] == "confirm_paragraph" and b["attempt_start_tok"] == max(st, r["window_start_tok"])
            assert b["judge_attempt_start_tok"] == b0["attempt_start_tok"] and b["confirm_new_value"] == "9"
        else:        # no usable value / paragraph, or not an abandon: the judge's fields stay
            assert all(b[f] == b0[f] for f in ("rejected_value", "attempt_start_tok", "attempt_start_source", "w_is_wrong"))
            assert i == 2 or b["rejected_value_source"] == "judge"
    run([str(CODE / "select_e3.py"), "--traces", "traces.jsonl", "--boundaries", "boundaries_confirmed.jsonl", "--out", "e3c.jsonl",
         "--confirmed", "--min-len", "1"], tmp_path)
    sel = load_jsonl(tmp_path / "e3c.jsonl")
    assert sel and all(s["confirm_label"] == "abandon" and s["confirm_source"] == "judge_abandon" for s in sel)
    # v2 strict prompt, other pools, --ids
    conf("prompts", "--strict", "--pool", "negrestart", "--out", "cp_nr.jsonl")
    nr = load_jsonl(tmp_path / "cp_nr.jsonl")
    assert sorted(r["boundary_id"] for r in nr) == sorted(b["boundary_id"] for b in bd if b["trigger_type"] in ("negation", "restart"))
    assert nr and all(r["version"] == "v2" and "If you are unsure whether it is abandon" in r["user"] for r in nr)
    conf("prompts", "--pool", "both", "--ids", "e3c.jsonl", "--out", "cp_ids.jsonl")
    assert sorted(r["boundary_id"] for r in load_jsonl(tmp_path / "cp_ids.jsonl")) == sorted(s["boundary_id"] for s in sel)
    assert all(r["version"] == "v1" and "If you are unsure" not in r["user"] for r in cp)
    conf("prompts", "--pool", "both", "--out", "cp_both.jsonl")
    both = load_jsonl(tmp_path / "cp_both.jsonl")
    other = [r for r in both if r["boundary_id"] not in cpi]
    assert other
    with open(tmp_path / "craw_both.jsonl", "w") as f:
        for i, r in enumerate(both):
            k = sorted(r["para_start_tok"], key=int)[0]
            f.write(json.dumps({"boundary_id": r["boundary_id"], "judge_raw": json.dumps(
                {"label": "abandon", "rejected_value": ["5", "x + y"][i % 2], "attempt_start_paragraph": int(k)})}) + "\n")
    conf("assemble", "--prompts", "cp_both.jsonl", "--confirm", "craw_both.jsonl", "--out", "bc_both.jsonl")
    for b in load_jsonl(tmp_path / "bc_both.jsonl"):
        if b["confirm_label"] == "abandon":
            assert b["confirm_source"] == ("judge_abandon" if b["boundary_id"] in cpi else "candidate_other")
            if b["confirm_source"] == "candidate_other":
                assert b["rejected_value"] == (b["confirm_rejected_value"] if b["confirm_rejected_value"] == "5" else None)
    # verify batches (PLAN #18): blind rows of confirmer positives, excluding --exclude and earlier batches; labels -> raw
    vb = lambda *a: run([str(CODE / "verify_batch.py"), *a], tmp_path)
    va = ["--confirm-prompts", "cp_both.jsonl", "--confirm", "craw_both.jsonl", "--exclude", "e3c.jsonl", "--per-problem", "9"]
    vb("build", "--batch", "1", "--n", "2", *va)
    vb("build", "--batch", "2", "--n", "99", *va)
    b1, b2 = load_jsonl(tmp_path / "results/verify_batch_01.jsonl"), load_jsonl(tmp_path / "results/verify_batch_02.jsonl")
    k1 = load_jsonl(tmp_path / "verify_batch_01_key.jsonl")
    ids1, ids2 = {r["boundary_id"] for r in b1}, {r["boundary_id"] for r in b2}
    elig = {r["boundary_id"] for i, r in enumerate(both) if i % 2 == 0 or r["boundary_id"] in cpi} - {s["boundary_id"] for s in sel}
    assert len(b1) == 2 and not ids1 & ids2 and ids1 | ids2 == elig and [k["boundary_id"] for k in k1] == [r["boundary_id"] for r in b1]
    cpb = {r["boundary_id"]: r for r in both}
    for r in b1 + b2:  # blind: posthoc fields only; the continuation is exactly what the confirmer saw
        assert set(r) == {"row", "boundary_id", "problem", "prefix", "sentence", "continuation", "n_continuation_tokens"}
        assert cpb[r["boundary_id"]]["user"].endswith("WHAT THE WRITER WROTE NEXT:\n" + r["continuation"])
        assert by_id[r["boundary_id"]]["solution_block"] == r["prefix"]
    lab = [{"boundary_id": r["boundary_id"], "label": "abandon", "w": "13", "attempt_start_paragraph": int(sorted(cpb[r["boundary_id"]]["para_start_tok"], key=int)[0])} for r in b1]
    (tmp_path / "results/verify_labels_01.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lab))
    vb("to_raw")
    conf("assemble", "--prompts", "cp_both.jsonl", "--confirm", "verify_raw.jsonl", "--out", "bv.jsonl")
    bv = {b["boundary_id"]: b for b in load_jsonl(tmp_path / "bv.jsonl")}
    assert all(bv[i]["confirm_label"] == "abandon" and bv[i]["rejected_value"] == "13" for i in ids1)
    assert all(b["confirm_label"] != "abandon" for i, b in bv.items() if i not in ids1)
    # held-out mix (PLAN #22): --negatives adds random confirmer negatives, shuffled in unmarked among the positives
    with open(tmp_path / "craw_neg.jsonl", "w") as f:
        for i, r in enumerate(both):
            f.write(json.dumps({"boundary_id": r["boundary_id"], "judge_raw": json.dumps(
                {"label": "abandon" if i % 2 == 0 else "no_change", "rejected_value": "5" if i % 2 == 0 else None})}) + "\n")
    (tmp_path / "k3").mkdir()
    vb("build", "--batch", "3", "--n", "99", "--negatives", "2", "--confirm-prompts", "cp_both.jsonl", "--confirm", "craw_neg.jsonl",
       "--exclude", "e3c.jsonl", "--per-problem", "99", "--key-dir", "k3")
    b3, k3 = load_jsonl(tmp_path / "results/verify_batch_03.jsonl"), load_jsonl(tmp_path / "k3/verify_batch_03_key.jsonl")
    neg = [k for k in k3 if k["confirm_source"] == "negative"]
    sel_ids = {s["boundary_id"] for s in sel}
    assert len(neg) == 2 and all(k["confirm_label"] == "no_change" and k["boundary_id"] not in sel_ids for k in neg)
    assert {k["boundary_id"] for k in k3 if k["confirm_source"] != "negative"} == {r["boundary_id"] for i, r in enumerate(both) if i % 2 == 0} - sel_ids
    assert [r["row"] for r in b3] == [k["row"] for k in k3] == list(range(len(b3))) and [r["boundary_id"] for r in b3] == [k["boundary_id"] for k in k3]
    assert all(set(r) == {"row", "boundary_id", "problem", "prefix", "sentence", "continuation", "n_continuation_tokens"} and r["problem"] for r in b3)
    # base-rate batch (PLAN #19): model hidden behind opaque ids, <= 1 per trace, continuation = trace tokens [c, c + n)
    run([str(CODE / "base_batch.py"), "--src", "m1:.", "--src", "m2:.", "--n", "2", "--n-tokens", "20", "--exclude", "e3c.jsonl",
         "--out", "results/base_batch_01.jsonl", "--key", "base_key.jsonl", "--model", MODEL], tmp_path)
    bb, bk = load_jsonl(tmp_path / "results/base_batch_01.jsonl"), load_jsonl(tmp_path / "base_key.jsonl")
    assert len(bb) == len(bk) == 4 and [r["item_id"] for r in bb] == [k["item_id"] for k in bk]
    assert all(set(r) == {"row", "item_id", "problem", "prefix", "sentence", "continuation", "n_continuation_tokens"} for r in bb)
    for m in ("m1", "m2"):
        km = [k for k in bk if k["model"] == m]
        assert len(km) == 2 and len({(k["problem_id"], k["boundary_id"].split("|")[1]) for k in km}) == 2
    assert not {k["boundary_id"] for k in bk if k["model"] == "m1"} & {s["boundary_id"] for s in sel}
    for r, k in zip(bb, bk):
        b = bd_id.get(k["boundary_id"]) or next(c for c in load_jsonl(tmp_path / "candidates.jsonl") if c["boundary_id"] == k["boundary_id"])
        t = trs[(b["problem_id"], int(b["sample_idx"]))]
        c = t["think_start_tok"] + int(b["c_tok"])
        assert r["continuation"] == tok.decode(t["output_ids"][c:min(c + 20, t["think_end_tok"])]) and r["prefix"] == by_id[k["boundary_id"]]["solution_block"]
    # strata (PLAN #21): fixed count per dataset group; an empty group yields nothing; earlier base keys excluded for the first model only
    run([str(CODE / "base_batch.py"), "--src", "m1:.", "--strata", "f=fake:1;z=nope:3", "--n-tokens", "20", "--exclude", "e3c.jsonl",
         "--exclude-glob", "base_key.jsonl", "--seed", "1", "--out", "results/base_batch_02.jsonl", "--key", "base_key2.jsonl", "--model", MODEL], tmp_path)
    bk2 = load_jsonl(tmp_path / "base_key2.jsonl")
    assert len(bk2) == 1 and bk2[0]["stratum"] == "f" and bk2[0]["dataset"] == "fake" and 0 <= bk2[0]["rel_pos"] <= 1
    assert bk2[0]["boundary_id"] not in {k["boundary_id"] for k in bk if k["model"] == "m1"} | {s["boundary_id"] for s in sel}
    print(out, o)


def test_dry_export_without_labels(tok, tmp_path):
    make_raw(tok, tmp_path / "raw_0.jsonl")
    run([str(CODE / "postprocess.py"), str(tmp_path / "raw_*.jsonl"), "--model", MODEL], tmp_path)
    for a in (["candidates"], ["prompts"], ["assemble"]):
        run([str(CODE / "boundaries.py"), *a, "--model", MODEL], tmp_path)
    assert all(b["judge_label"] is None for b in load_jsonl(tmp_path / "boundaries.jsonl"))
    run([str(CODE / "export_annotation.py"), "--n", "5"], tmp_path)
    assert len(load_jsonl(tmp_path / "boundaries_for_annotation.jsonl")) == 5


def test_long_trace_window_and_global_numbering(tok, tb, tmp_path):
    think = "".join(f"Step {i}: compute ${i}^2 = {i * i}$ and note the value {i * i}.\n\n" for i in range(700)) + "Wait, this is wrong. Let me start over.\n"
    make_raw(tok, tmp_path / "raw_0.jsonl", thinks=[think], answers=["\\boxed{1}"], truncate_last=False)
    run([str(CODE / "postprocess.py"), str(tmp_path / "raw_*.jsonl"), "--model", MODEL], tmp_path)
    run([str(CODE / "boundaries.py"), "candidates", "--model", MODEL], tmp_path)
    run([str(CODE / "boundaries.py"), "prompts", "--model", MODEL], tmp_path)
    pr = load_jsonl(tmp_path / "judge_prompts.jsonl")
    assert len(pr) == 2 and pr[0]["boundary_paragraph"] == 700                           # global index, 0-based
    p = pr[0]
    shown = sorted(int(k) for k in p["para_start_tok"])
    assert shown[0] > 100 and shown[-1] == 699 and p["window_start_tok"] == int(p["boundary_id"].split("|")[-1]) - 4096
    n_prompt_tok = len(tok(p["solution_block"])["input_ids"])
    assert 4096 * 0.9 < n_prompt_tok < 4096 * 1.15, n_prompt_tok
    ids = load_jsonl(tmp_path / "traces.jsonl")[0]["output_ids"]
    j = shown[1]                                                                          # a fully shown paragraph: its start token decodes to its own text
    assert tb.text(ids[p["para_start_tok"][str(j)]:][:8]).startswith(f"Step {j}:")
    assert p["solution_block"].startswith(f"[{shown[0]}] ")
