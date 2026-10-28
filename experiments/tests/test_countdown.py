"""Countdown validator check (added 2026-10-27 after the exp-auditor's fix 7; the function itself is unchanged):
python tests/test_countdown.py. Runs countdown_ok exactly as scored, i.e. the copy in tl_score.py (extracted from the
source, because tl_score.py parses its arguments on import), and checks that pilot_check.py holds the same logic."""
import ast, os, re

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(fn):
    src = open(os.path.join(CODE, fn)).read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "countdown_ok")
    ns = {"re": re, "ast": ast}
    if fn == "pilot_check.py":  # pilot_check's copy uses module-level imports and OPS
        from fractions import Fraction
        ns["Fraction"] = Fraction
        ops = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "OPS")
        exec(compile(ast.Module([ops], []), fn, "exec"), ns)
    exec(compile(ast.Module([node], []), fn, "exec"), ns)
    return ns["countdown_ok"]


CASES = [  # (expr, nums, target, expected)
    ("3*7 + 5", [3, 5, 7], 26, True),  # plain
    ("3 \\times 7 + 5", [3, 5, 7], 26, True),  # LaTeX times
    ("(7 - 5) \\div 2 + 3", [2, 3, 5, 7], 4, True),  # LaTeX div
    ("\\frac{8}{2} + 6", [2, 6, 8], 10, True),  # \frac
    ("3*7 + 5 = 26", [3, 5, 7], 26, True),  # "= target" on the right
    ("26 = 3*7 + 5", [3, 5, 7], 26, True),  # "target =" on the left
    ("3*7 + 5", [3, 5, 7], 27, False),  # wrong target
    ("3*7 + 6", [3, 5, 7], 27, False),  # number not given
    ("3*7 + 5 + 5", [3, 5, 7], 31, False),  # number used twice
    ("3*7", [3, 5, 7], 21, False),  # number left unused
    ("-3 + 7*5", [3, 5, 7], 32, False),  # unary minus is not an allowed operation
    ("3**2 + 5", [2, 3, 5], 14, False),  # power is not allowed
    ("__import__('os').getpid()", [1], 1, False),  # calls are rejected, nothing is executed
    ("7 / (5 - 5) + 3", [3, 5, 5, 7], 3, False),  # division by zero
    (None, [3, 5, 7], 26, False),  # no answer
]

if __name__ == "__main__":
    fails = 0
    for fn in ("tl_score.py", "pilot_check.py"):
        f = load(fn)
        for expr, nums, target, want in CASES:
            got = f(expr, nums, target)
            if got != want:
                fails += 1
                print(f"FAIL {fn}: {expr!r} {nums} -> {target}: got {got}, want {want}")
    print(f"{len(CASES)} cases x 2 copies: {'OK' if not fails else f'{fails} FAILED'}")
    raise SystemExit(1 if fails else 0)
