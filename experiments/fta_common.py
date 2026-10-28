"""Shared helpers for the E1 pipeline: exact token<->char maps over the ORIGINAL output_ids (no re-tokenizing),
\\boxed extraction, answer equivalence. Token offsets everywhere are relative to the start of the thinking text."""
import bisect, json, re
from itertools import accumulate

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."  # same as gen_traces.py
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
except ImportError:
    _mv_parse = _mv_verify = None


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_tokenizer(name="Qwen/Qwen3.5-9B"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


# ---------- token -> bytes ----------

def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


class TokBytes:
    """id -> raw bytes. Byte-level BPE pieces go through the inverse of GPT-2 bytes_to_unicode; added/special tokens use UTF-8 of their content."""
    def __init__(self, tok):
        self.tok = tok
        self.inv = {c: b for b, c in _bytes_to_unicode().items()}
        self.added = {i: t for t, i in tok.get_added_vocab().items()}
        self.cache = {}
        self.think_open = tok.convert_tokens_to_ids("<think>")
        self.think_close = tok.convert_tokens_to_ids("</think>")
        self.eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}

    def one(self, i):
        r = self.cache.get(i)
        if r is None:
            if i in self.added:
                r = self.added[i].encode("utf-8")
            else:
                r = bytes(self.inv[c] for c in self.tok.convert_ids_to_tokens(i))
            self.cache[i] = r
        return r

    def view(self, ids):
        return TextView(self, ids)

    def text(self, ids):
        return b"".join(self.one(i) for i in ids).decode("utf-8", "replace")


def _clen(c):
    o = ord(c)
    return 1 if o < 0x80 or 0xDC80 <= o <= 0xDCFF else 2 if o < 0x800 else 3 if o < 0x10000 else 4


class TextView:
    """Decoded text of a token list plus exact char<->token maps. Invalid UTF-8 bytes become lone surrogates (1 byte each) so offsets stay exact."""
    def __init__(self, tb, ids):
        self.ids = list(ids)
        bl = [tb.one(i) for i in self.ids]
        self.tok_off = [0] + list(accumulate(len(b) for b in bl))        # byte offset of each token start, len n+1
        self.text = b"".join(bl).decode("utf-8", "surrogateescape")
        self.char2byte = None if self.text.isascii() else [0] + list(accumulate(_clen(c) for c in self.text))

    def _byte(self, ch):
        return ch if self.char2byte is None else self.char2byte[ch]

    def tok_of_char(self, ch):
        """index of the token containing char offset ch (ch < len(text)); len(ids) if ch >= len(text)"""
        if ch >= len(self.text):
            return len(self.ids)
        return bisect.bisect_right(self.tok_off, self._byte(ch)) - 1

    def tok_end_of_char(self, ch):
        """exclusive end token index covering all chars before char offset ch"""
        if ch <= 0:
            return 0
        return min(bisect.bisect_right(self.tok_off, self._byte(ch) - 1), len(self.ids))   # last token containing byte ch-1, +1

    def char_of_tok(self, t):
        """char offset where token t starts (first char starting at or after that byte)"""
        b = self.tok_off[min(t, len(self.ids))]
        return b if self.char2byte is None else bisect.bisect_left(self.char2byte, b)

    def span(self, c0, c1):
        return self.tok_of_char(c0), self.tok_end_of_char(c1)

    def clean(self, s):
        return s.encode("utf-8", "surrogateescape").decode("utf-8", "replace")


# ---------- text utilities ----------

def boxed_all(s):
    """contents of every \\boxed{...} (balanced braces, backslash-escaped chars skipped), in order"""
    out = []
    for m in re.finditer(r"\\boxed\s*\{", s):
        i, depth, j = m.end(), 1, m.end()
        while j < len(s) and depth:
            if s[j] == "\\":
                j += 2
                continue
            depth += (s[j] == "{") - (s[j] == "}")
            j += 1
        if depth == 0:
            out.append(s[i:j - 1].strip())
    return out


def last_boxed(s):
    b = boxed_all(s)
    return b[-1] if b else None


def split_paragraphs(text):
    """[(char_start, char_end)] of blocks separated by blank lines (whitespace-only lines); empty blocks dropped"""
    out, pos = [], 0
    for m in list(re.finditer(r"\n[ \t\r]*\n\s*", text)) + [None]:
        end = m.start() if m else len(text)
        seg = text[pos:end]
        if seg.strip():
            out.append((pos, end))
        pos = m.end() if m else len(text)
    return out


# ---------- answer equivalence (same conventions as fta_stats.answers_equal) ----------

def _unwrap(s, cmd):
    while True:
        i = s.find(cmd + "{")
        if i < 0:
            return s
        depth, j = 0, i + len(cmd)
        for j in range(i + len(cmd), len(s)):
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            if depth == 0:
                break
        s = s[:i] + s[i + len(cmd) + 1:j] + s[j + 1:]


def normalize_answer(a):
    s = str(a).strip().strip("$").strip()
    for cmd in ("\\boxed", "\\text", "\\mathrm", "\\textbf"):
        s = _unwrap(s, cmd)
    s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("^\\circ", "").replace("°", "").replace("\\!", "")
    if "=" in s and not s.startswith("("):
        s = s.split("=")[-1]
    s = s.strip().rstrip(".").replace(" ", "")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    return s


def _num(s):
    from fractions import Fraction
    t = s.replace("(", "").replace(")", "")
    try:
        if re.fullmatch(r"-?\d+/-?\d+", t):
            n, d = t.split("/")
            return Fraction(int(n), int(d))
        return Fraction(t)
    except (ValueError, ZeroDivisionError):
        return None


_LATEX_NUM = {"sqrt": "", "pi": "", "cdot": "*", "times": "*"}


def is_numeric_value(w):
    """spec §3/§6: a rejected value counts only if it is a number or a closed-form numeric expression (6^9, 3\\sqrt{2},
    \\frac{1}{2}, (1,2)); free text and expressions with variables do not"""
    if w is None or str(w).strip() == "":
        return False
    s = re.sub(r"\\(sqrt|pi|cdot|times)", lambda m: _LATEX_NUM[m.group(1)], normalize_answer(w))
    return bool(re.search(r"\d", s)) and re.fullmatch(r"[0-9+\-*/^().{}\[\],]+", s) is not None


def answers_equal(a, b):
    """math_verify first (parse both as LaTeX in $..$), string/number normaliser as fallback; missing never matches"""
    if a is None or b is None or str(a).strip() == "" or str(b).strip() == "":
        return False
    if _mv_verify is not None:
        try:
            if _mv_verify(_mv_parse(f"${b}$"), _mv_parse(f"${a}$")):
                return True
        except Exception:
            pass
    na, nb = normalize_answer(a), normalize_answer(b)
    if na == nb:
        return True
    xa, xb = _num(na), _num(nb)
    return xa is not None and xb is not None and xa == xb
