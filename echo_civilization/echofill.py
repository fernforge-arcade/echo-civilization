"""echofill — program-by-example data wrangling, no LLM.

The civilization's synthesis engine (synthesis.py, frontier.py, parametric.py)
learns string->string transforms from input/output examples. This module turns
that capability into a practical tool: give it a few examples of how one column
of a spreadsheet should be cleaned or reformatted, and it searches for a
deterministic program that reproduces every example, then applies that program
to a whole column.

This is the thing people currently reach an LLM for — "clean this messy column",
"pull the domain out of these emails", "flip these `LAST, First` names" — one
row at a time, paying per token and tolerating the occasional hallucination.
echofill infers the rule ONCE from 2-4 examples, verifies it against all of them,
then runs it over the rest of the file offline: zero marginal cost, microseconds
per row, and the same answer every time.

The engine is a bounded search over a library of string primitives. Two ideas
carry most of the weight, both borrowed from the research code and from FlashFill:

  * a program is a short PIPELINE of ops, each op transforming the cell string;
  * the parametric ops (split delimiter/index, replace, extract) have their
    arguments INDUCED from the examples rather than blindly searched — we look at
    how the inputs differ from the outputs and read the argument straight off.

Argument induction is what makes this fast and high-coverage. Blindly searching
`split(delim, idx)` over every delimiter and index is wasteful; instead we ask
"which (delim, idx) reproduces the output on every example?" and only keep that.

No numpy, no external anything — pure standard library, so the tool drops into
any environment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Op library.
#
# Every op is (name, arity, fn). fn takes (cell, arg) and returns a string or
# None (None == "not applicable to this input", which prunes the candidate).
# Non-parametric ops ignore arg. Parametric ops read a value out of arg.
#
# The DELIMS list is the small, high-value set of separators real data uses.
# --------------------------------------------------------------------------- #

DELIMS = [" ", ", ", ",", " - ", "-", "_", "/", "@", ".", ":", "|", "\t", ";"]


def _lower(c, a):        return c.lower()
def _upper(c, a):        return c.upper()
def _title(c, a):        return c.title()
def _capitalize(c, a):   return c.capitalize()
def _strip(c, a):        return c.strip()
def _collapse_ws(c, a):  return re.sub(r"\s+", " ", c).strip()
def _keep_digits(c, a):  return re.sub(r"\D", "", c)
def _keep_alpha(c, a):   return re.sub(r"[^A-Za-z]", "", c)
def _keep_alnum(c, a):   return re.sub(r"[^A-Za-z0-9]", "", c)


def _split_take(c, a):
    """a = (delim, idx). Split on delim, take field idx (idx may be negative)."""
    delim, idx = a
    parts = c.split(delim)
    if -len(parts) <= idx < len(parts):
        return parts[idx]
    return None


def _before(c, a):
    """a = delim. Substring before the FIRST occurrence of delim."""
    i = c.find(a)
    return c[:i] if i != -1 else None


def _after(c, a):
    """a = delim. Substring after the FIRST occurrence of delim."""
    i = c.find(a)
    return c[i + len(a):] if i != -1 else None


def _before_last(c, a):
    i = c.rfind(a)
    return c[:i] if i != -1 else None


def _after_last(c, a):
    i = c.rfind(a)
    return c[i + len(a):] if i != -1 else None


def _replace(c, a):
    """a = (find, repl). Replace all occurrences."""
    find, repl = a
    return c.replace(find, repl)


def _prepend(c, a):      return a + c
def _append(c, a):       return c + a


def _reorder(c, a):
    """a = (in_delim, out_delim, perm). Split on in_delim, reorder fields by
    perm (a tuple of source indices), rejoin with out_delim. Covers `LAST, First`
    -> `First LAST` and date field reordering in one op."""
    in_delim, out_delim, perm = a
    parts = c.split(in_delim)
    if any(not (-len(parts) <= i < len(parts)) for i in perm):
        return None
    return out_delim.join(parts[i].strip() for i in perm)


def _zfill(c, a):        return c.zfill(a)          # a = width
def _slice(c, a):        return c[a[0]:a[1]]        # a = (i, j)


# name -> (fn, is_parametric)
OPS = {
    "lower":        (_lower, False),
    "upper":        (_upper, False),
    "title":        (_title, False),
    "capitalize":   (_capitalize, False),
    "strip":        (_strip, False),
    "collapse_ws":  (_collapse_ws, False),
    "keep_digits":  (_keep_digits, False),
    "keep_alpha":   (_keep_alpha, False),
    "keep_alnum":   (_keep_alnum, False),
    "split_take":   (_split_take, True),
    "before":       (_before, True),
    "after":        (_after, True),
    "before_last":  (_before_last, True),
    "after_last":   (_after_last, True),
    "replace":      (_replace, True),
    "prepend":      (_prepend, True),
    "append":       (_append, True),
    "reorder":      (_reorder, True),
    "zfill":        (_zfill, True),
    "slice":        (_slice, True),
}

# Non-parametric ops are cheap to try blindly.
NONPARAM = [n for n, (_, p) in OPS.items() if not p]


# --------------------------------------------------------------------------- #
# A program is a list of (op_name, arg) steps applied left to right.
# --------------------------------------------------------------------------- #

def apply_program(program, cell):
    """Run a program on one cell. Returns the transformed string, or None if any
    step was inapplicable (e.g. a delimiter that isn't present)."""
    s = cell
    for name, arg in program:
        fn, _ = OPS[name]
        s = fn(s, arg)
        if s is None:
            return None
    return s


def program_str(program):
    """Human-readable one-liner for a program — shown to the user so the inferred
    rule is auditable, not a black box."""
    out = []
    for name, arg in program:
        if arg is None:
            out.append(name)
        else:
            out.append(f"{name}{_fmt_arg(arg)}")
    return " | ".join(out) if out else "identity"


def _fmt_arg(a):
    if isinstance(a, tuple):
        return "(" + ", ".join(repr(x) for x in a) + ")"
    return f"({a!r})"


# --------------------------------------------------------------------------- #
# Argument induction. For each parametric op, read plausible args straight off
# the examples instead of blind search. examples = [(input, output), ...].
# --------------------------------------------------------------------------- #

def _induce_split_take(examples):
    """Find (delim, idx) such that input.split(delim)[idx] == output for all."""
    cands = []
    for delim in DELIMS:
        # how many fields does the first input have? bound idx by that.
        n = len(examples[0][0].split(delim))
        for idx in list(range(n)) + [-1, -2]:
            if all(_split_take(i, (delim, idx)) == o for i, o in examples):
                cands.append((delim, idx))
    return cands


def _induce_delim_op(examples, fn):
    """For before/after style ops: any delim in DELIMS that reproduces output."""
    return [d for d in DELIMS if all(fn(i, d) == o for i, o in examples)]


def _induce_replace(examples):
    """Infer (find, repl) by diffing an input against its output. We try each
    DELIM as `find` and read what it maps to, then verify across all examples.
    Also try single-character replacements discovered from the first pair."""
    cands = []
    for find in DELIMS:
        if find not in examples[0][0]:
            continue
        # candidate replacements: try common targets + empty + space.
        for repl in ["", " ", "-", "_", ", ", "/"]:
            if all(i.replace(find, repl) == o for i, o in examples):
                cands.append((find, repl))
    return cands


def _induce_reorder(examples):
    """Infer (in_delim, out_delim, perm). For each in/out delimiter pair, check
    whether the output fields are a permutation of the (stripped) input fields,
    consistent across all examples."""
    cands = []
    for ind in DELIMS:
        for outd in DELIMS:
            perm = None
            ok = True
            for i, o in examples:
                iparts = [p.strip() for p in i.split(ind)]
                oparts = [p.strip() for p in o.split(outd)]
                if len(iparts) < 2 or len(oparts) != len(iparts):
                    ok = False
                    break
                # find a permutation mapping oparts -> iparts positions
                p = []
                used = set()
                for op in oparts:
                    match = next((k for k, ip in enumerate(iparts)
                                  if ip == op and k not in used), None)
                    if match is None:
                        ok = False
                        break
                    p.append(match)
                    used.add(match)
                if not ok:
                    break
                if perm is None:
                    perm = tuple(p)
                elif perm != tuple(p):
                    ok = False
                    break
            if ok and perm is not None and perm != tuple(range(len(perm))):
                cands.append((ind, outd, perm))
    return cands


def _induce_zfill(examples):
    widths = {len(o) for _, o in examples}
    if len(widths) == 1:
        w = next(iter(widths))
        if all(i.zfill(w) == o for i, o in examples):
            return [w]
    return []


def induce_args(op_name, examples):
    """Return a list of candidate args for a parametric op given the examples.
    Non-parametric ops return [None]."""
    fn, is_param = OPS[op_name]
    if not is_param:
        return [None]
    if op_name == "split_take":
        return _induce_split_take(examples)
    if op_name == "before":
        return _induce_delim_op(examples, _before)
    if op_name == "after":
        return _induce_delim_op(examples, _after)
    if op_name == "before_last":
        return _induce_delim_op(examples, _before_last)
    if op_name == "after_last":
        return _induce_delim_op(examples, _after_last)
    if op_name == "replace":
        return _induce_replace(examples)
    if op_name == "reorder":
        return _induce_reorder(examples)
    if op_name == "zfill":
        return _induce_zfill(examples)
    # prepend/append/slice: induce from the shortest common fix.
    if op_name == "prepend":
        pre = _common_prefix_added(examples)
        return [pre] if pre else []
    if op_name == "append":
        suf = _common_suffix_added(examples)
        return [suf] if suf else []
    if op_name == "slice":
        return _induce_slice(examples)
    return []


def _common_prefix_added(examples):
    """If every output is input with the same literal prepended, return it."""
    pres = set()
    for i, o in examples:
        if not o.endswith(i) or o == i:
            return None
        pres.add(o[:len(o) - len(i)])
    return next(iter(pres)) if len(pres) == 1 else None


def _common_suffix_added(examples):
    sufs = set()
    for i, o in examples:
        if not o.startswith(i) or o == i:
            return None
        sufs.add(o[len(i):])
    return next(iter(sufs)) if len(sufs) == 1 else None


def _induce_slice(examples):
    """Fixed [i:j] slice consistent across examples (small windows only)."""
    cands = []
    L = min(len(i) for i, _ in examples)
    for i0 in range(0, min(L, 8)):
        for j0 in range(i0 + 1, min(L, 12) + 1):
            if all(inp[i0:j0] == out for inp, out in examples):
                cands.append((i0, j0))
    return cands


# --------------------------------------------------------------------------- #
# The synthesizer. Staged search, cheapest first:
#   0) identity (the column is already correct)
#   1) single induced op            (before('@'), keep_digits, ...)
#   2) single non-param op
#   3) depth-2: non-param then induced, and induced then non-param
#   4) depth-2/3 combinations over a pruned op set
# First program that reproduces ALL examples exactly wins. Ties broken by
# shortness (fewer ops = more likely to generalize).
# --------------------------------------------------------------------------- #

@dataclass
class SynthResult:
    program: list          # list of (op, arg) or None if unsolved
    solved: bool
    tried: int             # candidate programs evaluated (the "search cost")


def _fits(program, examples):
    return all(apply_program(program, i) == o for i, o in examples)


def synthesize(examples, max_depth: int = 3):
    """Search for a program reproducing every (input, output) example."""
    tried = 0

    def consider(program):
        nonlocal tried
        tried += 1
        return _fits(program, examples)

    # 0) identity
    if consider([]):
        return SynthResult([], True, tried)

    # Build the per-op candidate arg lists once.
    param_ops = [(n, induce_args(n, examples)) for n in OPS
                 if OPS[n][1]]
    param_ops = [(n, args) for n, args in param_ops if args]

    # 1) single induced parametric op
    for name, args in param_ops:
        for arg in args:
            prog = [(name, arg)]
            if consider(prog):
                return SynthResult(prog, True, tried)

    # 2) single non-param op
    for name in NONPARAM:
        prog = [(name, None)]
        if consider(prog):
            return SynthResult(prog, True, tried)

    if max_depth < 2:
        return SynthResult(None, False, tried)

    # 3) depth-2: non-param FIRST then re-induced parametric.
    # After applying a non-param op to the inputs, re-induce args against the
    # transformed inputs (this is what lets `replace('-',' ') | title` etc work).
    for pre in NONPARAM:
        mid = [(apply_program([(pre, None)], i), o) for i, o in examples]
        if any(m is None for m, _ in mid):
            continue
        for name in OPS:
            for arg in induce_args(name, mid):
                prog = [(pre, None), (name, arg)]
                if consider(prog):
                    return SynthResult(prog, True, tried)

    # 4) depth-2: induced parametric FIRST then non-param.
    for name, args in param_ops:
        for arg in args:
            transformed = [(apply_program([(name, arg)], i), o) for i, o in examples]
            if any(t is None for t, _ in transformed):
                continue
            for post in NONPARAM:
                prog = [(name, arg), (post, None)]
                if consider(prog):
                    return SynthResult(prog, True, tried)

    if max_depth < 3:
        return SynthResult(None, False, tried)

    # 5) depth-3: non-param, induced parametric, non-param.
    for pre in NONPARAM:
        s1 = [(apply_program([(pre, None)], i), o) for i, o in examples]
        if any(m is None for m, _ in s1):
            continue
        for name in OPS:
            for arg in induce_args(name, s1):
                s2 = [(apply_program([(name, arg)], m), o) for m, o in s1]
                if any(t is None for t, _ in s2):
                    continue
                for post in NONPARAM:
                    prog = [(pre, None), (name, arg), (post, None)]
                    if consider(prog):
                        return SynthResult(prog, True, tried)

    return SynthResult(None, False, tried)


# --------------------------------------------------------------------------- #
# Public API: learn a rule from examples, apply to a column.
# --------------------------------------------------------------------------- #

class Rule:
    """A learned, verified transformation ready to apply to a whole column."""

    def __init__(self, program, examples, tried):
        self.program = program
        self.examples = examples
        self.tried = tried

    @property
    def solved(self):
        return self.program is not None

    def apply(self, value):
        if not self.solved:
            raise ValueError("no rule was learned")
        out = apply_program(self.program, value)
        return out if out is not None else value  # fall back to original on N/A

    def apply_column(self, values):
        return [self.apply(v) for v in values]

    def describe(self):
        return program_str(self.program) if self.solved else "(no rule found)"


def learn(examples, max_depth: int = 3):
    """Learn a Rule from [(input, output), ...] examples.

    The returned Rule is guaranteed to reproduce every example exactly (that's
    the search's acceptance test), so a solved Rule never silently disagrees with
    the demonstration the user gave."""
    res = synthesize(list(examples), max_depth=max_depth)
    return Rule(res.program, list(examples), res.tried)
