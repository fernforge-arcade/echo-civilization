"""Environment 5 — Computer World (a simulated VM the agents learn to operate).

This is the project's reach toward more capable, *tool-using* agents. Instead of
mapping a string to a string, an agent must operate a tiny simulated computer —
a virtual filesystem plus a working register (a clipboard / stdout) — by issuing
a sequence of shell-like commands to achieve a goal state (e.g. *"find the file
about the treasure, keep only the lines mentioning the key, sort them, and write
the result to output.txt"*).

A solution is a **program**: a tuple of operation names. Operations auto-bind
their arguments from the task context (which file is the input, which is the
output, which keyword to grep for), so the search space is over op-sequences —
the same complexity class as the string skills, but the behaviours are genuine
multi-step computer use. Learned programs become reusable **macros** (skills)
that are shared, taught and inherited exactly like every other skill.

Crucially, the canonical solution at curriculum level *k* extends a level *(k-1)*
solution, so macros accumulated at lower levels compose into higher ones. Paired
with the auto-curriculum (`ComputerWorld.sample`), this lets a civilization climb
an open-ended ladder of task sophistication that no single un-aided lifetime
could reach — agents *evolve to match increasingly sophisticated tasks*.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------------ machine
@dataclass
class Machine:
    """The simulated computer's state."""
    files: dict          # filename -> content (text, newline-separated "lines")
    reg: str = ""        # working register (current value / clipboard / stdout)

    def clone(self) -> "Machine":
        return Machine(files=dict(self.files), reg=self.reg)


@dataclass
class Context:
    """Task-supplied bindings the agent 'perceives' (which files matter)."""
    input_file: str = ""
    output_file: str = "output.txt"
    keyword: str = ""


def _lines(s: str):
    return [ln for ln in s.split("\n") if ln != ""]


# ----------------------------------------------------- computer operations
# Each op: (machine, ctx) -> machine. They are total (never raise) so blind
# search is safe; nonsensical steps simply leave the register empty.
def _op_ls(m, ctx):
    m.reg = "\n".join(sorted(m.files))
    return m

def _op_read_input(m, ctx):
    m.reg = m.files.get(ctx.input_file, "")
    return m

def _op_find(m, ctx):
    # locate a file whose name OR content mentions the keyword; register = path
    hits = [f for f in sorted(m.files)
            if ctx.keyword and (ctx.keyword in f or ctx.keyword in m.files[f])]
    m.reg = hits[0] if hits else ""
    return m

def _op_read_found(m, ctx):
    m.reg = m.files.get(m.reg, "")
    return m

def _op_grep(m, ctx):
    if not ctx.keyword:
        return m
    m.reg = "\n".join(ln for ln in _lines(m.reg) if ctx.keyword in ln)
    return m

def _op_grep_v(m, ctx):
    if not ctx.keyword:
        return m
    m.reg = "\n".join(ln for ln in _lines(m.reg) if ctx.keyword not in ln)
    return m

def _op_upper(m, ctx):
    m.reg = m.reg.upper()
    return m

def _op_lower(m, ctx):
    m.reg = m.reg.lower()
    return m

def _op_sort(m, ctx):
    m.reg = "\n".join(sorted(_lines(m.reg)))
    return m

def _op_reverse_lines(m, ctx):
    m.reg = "\n".join(reversed(_lines(m.reg)))
    return m

def _op_uniq(m, ctx):
    out, prev = [], object()
    for ln in _lines(m.reg):
        if ln != prev:
            out.append(ln)
        prev = ln
    m.reg = "\n".join(out)
    return m

def _op_count_lines(m, ctx):
    m.reg = str(len(_lines(m.reg)))
    return m

def _op_head(m, ctx):
    ls = _lines(m.reg)
    m.reg = ls[0] if ls else ""
    return m

def _op_tail(m, ctx):
    ls = _lines(m.reg)
    m.reg = ls[-1] if ls else ""
    return m

def _op_concat_all(m, ctx):
    m.reg = "\n".join(m.files[f] for f in sorted(m.files))
    return m

def _op_write_output(m, ctx):
    m.files[ctx.output_file] = m.reg
    return m

def _op_append_output(m, ctx):
    m.files[ctx.output_file] = m.files.get(ctx.output_file, "") + \
        ("\n" if m.files.get(ctx.output_file) else "") + m.reg
    return m


COMPUTER_PRIMITIVES = {
    "ls": _op_ls,
    "read_input": _op_read_input,
    "find": _op_find,
    "read_found": _op_read_found,
    "grep": _op_grep,
    "grep_v": _op_grep_v,
    "upper": _op_upper,
    "lower": _op_lower,
    "sort": _op_sort,
    "reverse_lines": _op_reverse_lines,
    "uniq": _op_uniq,
    "count_lines": _op_count_lines,
    "head": _op_head,
    "tail": _op_tail,
    "concat_all": _op_concat_all,
    "write_output": _op_write_output,
    "append_output": _op_append_output,
}


def run_computer_program(program, machine: Machine, ctx: Context) -> Machine:
    m = machine.clone()
    for op in program:
        fn = COMPUTER_PRIMITIVES.get(op)
        if fn is not None:
            m = fn(m, ctx)
    return m


# ------------------------------------------------------------------- tasks
@dataclass
class ComputerTask:
    name: str
    level: int
    machine: Machine            # initial state
    ctx: Context
    canonical: tuple            # ground-truth solution program (hidden)
    expected_output: str        # expected content of ctx.output_file

    def grade(self, final: Machine) -> tuple[bool, float]:
        got = final.files.get(self.ctx.output_file, "")
        if got == self.expected_output:
            return True, 1.0
        # partial credit: positional character overlap of the output file
        target = self.expected_output
        n = max(len(got), len(target), 1)
        matches = sum(1 for i in range(min(len(got), len(target)))
                      if got[i] == target[i])
        return False, matches / n


# A curriculum of pipelines of increasing depth. Each level's canonical program is
# (deliberately) a one-operation extension of a shallower one, so a macro learned
# at level k-1 reaches level k by a single modification — the smooth ladder the
# civilization climbs. (`find`/`read_found` "locate" tasks are an extra skill
# family, not part of the frontier ladder, so climbing stays monotonic.)
CURRICULUM = {
    1: [
        ("copy_file", ("read_input", "write_output")),
    ],
    2: [
        ("upper_file", ("read_input", "upper", "write_output")),
        ("grep_file", ("read_input", "grep", "write_output")),
        ("sort_file", ("read_input", "sort", "write_output")),
    ],
    3: [
        ("grep_sort", ("read_input", "grep", "sort", "write_output")),
        ("upper_reverse", ("read_input", "upper", "reverse_lines", "write_output")),
        ("grep_count", ("read_input", "grep", "count_lines", "write_output")),
    ],
    4: [
        ("grep_sort_uniq", ("read_input", "grep", "sort", "uniq", "write_output")),
        ("grep_sort_count", ("read_input", "grep", "sort", "count_lines", "write_output")),
    ],
    5: [
        ("grep_sort_uniq_count",
         ("read_input", "grep", "sort", "uniq", "count_lines", "write_output")),
        ("grep_upper_reverse_count",
         ("read_input", "grep", "upper", "reverse_lines", "count_lines", "write_output")),
    ],
}
MAX_LEVEL = max(CURRICULUM)

# Bonus "locate" skill family (uses find/read_found) — demonstrates a second tool-use
# motif; reachable once read_found-based macros are in the culture.
LOCATE_TASKS = [
    ("locate_copy", ("find", "read_found", "write_output")),
    ("locate_grep", ("find", "read_found", "grep", "write_output")),
]

WORDS = ["treasure", "door", "key", "north", "river", "cave", "gold", "map",
         "blue", "tower", "forest", "bridge", "stone", "water", "fire"]


class ComputerWorld:
    name = "computer"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    # -- random filesystem -------------------------------------------------
    def _random_files(self, keyword: str):
        n = int(self.rng.integers(3, 6))
        files = {}
        names = self.rng.choice(WORDS, size=n, replace=False)
        for nm in names:
            nlines = int(self.rng.integers(3, 7))
            lines = []
            for _ in range(nlines):
                w = list(self.rng.choice(WORDS, size=int(self.rng.integers(2, 4))))
                # ensure the keyword shows up in some lines
                if self.rng.random() < 0.45:
                    w.append(keyword)
                lines.append(" ".join(w))
            files[f"{nm}.txt"] = "\n".join(lines)
        return files

    def make_task(self, level: int, name=None, program=None) -> ComputerTask:
        if program is None:
            name, program = CURRICULUM[level][int(self.rng.integers(0, len(CURRICULUM[level])))]
        keyword = str(self.rng.choice(WORDS))
        files = self._random_files(keyword)
        input_file = sorted(files)[int(self.rng.integers(0, len(files)))]
        # make the keyword appear in a file name sometimes (for `find`)
        if self.rng.random() < 0.5:
            renamed = f"{keyword}_notes.txt"
            files[renamed] = files[input_file]
        ctx = Context(input_file=input_file, output_file="output.txt", keyword=keyword)
        machine = Machine(files=files)
        expected = run_computer_program(program, machine, ctx).files.get(ctx.output_file, "")
        return ComputerTask(name=name, level=level, machine=machine, ctx=ctx,
                            canonical=tuple(program), expected_output=expected)

    def sample(self, level: int) -> ComputerTask:
        level = max(1, min(MAX_LEVEL, level))
        return self.make_task(level)
