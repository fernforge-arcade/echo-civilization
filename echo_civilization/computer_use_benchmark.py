"""Computer-Use Benchmark — "do they actually become computer-use agents?"

The operator asked for a *graded ladder of projects*, from the trivial ("move
this file") all the way up to the open-ended ("write this website/app"), run by
the civilization's agents, to find out **how far up the ladder they actually
get** — i.e. whether the simple learning agents this project breeds are genuine
computer-use agents.

This module builds that ladder and runs two agents against every rung:

  * CULTURED  — an agent carrying the macro library a Computer-World civilization
                accumulated over generations (the end product of the cultural
                process).
  * FRESH     — a generation-0 agent with an empty library (the control).

Every solvable rung is graded by **executing the agent's synthesised program as
real shell commands** in a throwaway sandbox (via `RealComputerWorld`), so a
"solve" means real files changed on a real disk — not a simulation.

The top rungs are *deliberately* out of reach. We prove it: an oracle does a
bounded exhaustive search over the entire operation vocabulary and still cannot
hit the target. Those rungs map the **capability ceiling** — the honest boundary
between what these agents are (bounded multi-step file/text tool-users) and what
a "write me an app" agent would need (open-ended code generation).
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field

import numpy as np

from .environments.computer_world import (COMPUTER_PRIMITIVES, Context, Machine,
                                          run_computer_program)
from .environments.real_computer_world import RealComputerWorld, RealTask, ALLOWED_OPS
from .synthesis import synthesize


# --------------------------------------------------------------------------
# Extended operation table: the base Computer-World primitives PLUS two genuine
# whole-file operations (move / copy) so the trivial "move this file" rung is a
# real, single-command file operation rather than a content round-trip.
# --------------------------------------------------------------------------
def _op_move_file(m, ctx):
    m.files[ctx.output_file] = m.files.get(ctx.input_file, "")
    if ctx.input_file in m.files and ctx.input_file != ctx.output_file:
        del m.files[ctx.input_file]
    m.reg = m.files[ctx.output_file]
    return m


def _op_copy_file(m, ctx):
    m.files[ctx.output_file] = m.files.get(ctx.input_file, "")
    m.reg = m.files[ctx.output_file]
    return m


BENCH_OPS = dict(COMPUTER_PRIMITIVES)
BENCH_OPS["move_file"] = _op_move_file
BENCH_OPS["copy_file"] = _op_copy_file
BENCH_PRIMITIVES = list(BENCH_OPS)


def _run(program, machine, ctx):
    m = machine.clone()
    for op in program:
        fn = BENCH_OPS.get(op)
        if fn is not None:
            m = fn(m, ctx)
    return m


# --------------------------------------------------------------------------
# Task ladder
# --------------------------------------------------------------------------
@dataclass
class Rung:
    tier: int
    name: str
    blurb: str                 # human description, the "project brief"
    canonical: tuple | None    # ground-truth op program, or None if unrepresentable
    reachable: bool            # is there ANY op-program that solves it?
    runnable: bool = True      # can we even execute/grade it in this world?


# Reachable rungs: each is a real file/text project expressible as an op pipeline.
LADDER = [
    Rung(1, "move_file",
         "Move a file to output.txt (rename on disk).",
         ("move_file",), True),
    Rung(1, "copy_file",
         "Copy a file to output.txt.",
         ("copy_file",), True),
    Rung(2, "dump_file",
         "Read a file and write its contents out.",
         ("read_input", "write_output"), True),
    Rung(2, "uppercase_file",
         "Uppercase every line of a file.",
         ("read_input", "upper", "write_output"), True),
    Rung(2, "filter_lines",
         "Keep only the lines mentioning the keyword.",
         ("read_input", "grep", "write_output"), True),
    Rung(2, "sort_file",
         "Sort the lines of a file.",
         ("read_input", "sort", "write_output"), True),
    Rung(3, "grep_then_sort",
         "Keep keyword lines, then sort them.",
         ("read_input", "grep", "sort", "write_output"), True),
    Rung(3, "count_matches",
         "Count how many lines mention the keyword.",
         ("read_input", "grep", "count_lines", "write_output"), True),
    Rung(3, "locate_and_dump",
         "Find the file about the keyword, then write its contents.",
         ("find", "read_found", "write_output"), True),
    Rung(4, "grep_sort_uniq",
         "Keep keyword lines, sort, drop adjacent duplicates.",
         ("read_input", "grep", "sort", "uniq", "write_output"), True),
    Rung(4, "grep_sort_count",
         "Keep keyword lines, sort, then count them.",
         ("read_input", "grep", "sort", "count_lines", "write_output"), True),
    Rung(5, "report_pipeline",
         "Keep keyword lines, sort, dedup, then count — a 'mini report'.",
         ("read_input", "grep", "sort", "uniq", "count_lines", "write_output"), True),
    Rung(5, "format_report",
         "Keep keyword lines, uppercase, reverse order, then count.",
         ("read_input", "grep", "upper", "reverse_lines", "count_lines", "write_output"),
         True),
]

# Beyond the vocabulary: real computer tasks that NO op-program can express.
# We still attempt them (and prove unreachability with an oracle search).
BEYOND = [
    Rung(6, "find_and_replace",
         "Replace every occurrence of the keyword with the word 'REDACTED'.",
         None, False),
    Rung(6, "word_frequency",
         "Produce a word-frequency table (each distinct word and its count).",
         None, False),
    Rung(6, "sum_numbers",
         "A file lists numbers; write their arithmetic sum.",
         None, False),
]

# Open-ended software projects: not even a single-file text transform — these
# need open-ended code generation across an unbounded action space.
OPEN_ENDED = [
    Rung(7, "write_python_script",
         "Write a Python script that reads a CSV and prints column averages.",
         None, False, runnable=False),
    Rung(7, "build_todo_webapp",
         "Build a Flask to-do web app (routes, templates, persistence).",
         None, False, runnable=False),
    Rung(7, "refactor_repo",
         "Refactor this multi-file repository and keep the tests green.",
         None, False, runnable=False),
]


# --------------------------------------------------------------------------
# Oracle reachability: bounded exhaustive search over the whole vocabulary.
# If even this cannot hit score 1.0, the task is out of representational class.
# --------------------------------------------------------------------------
def oracle_best_score(rung: Rung, world: RealComputerWorld, n_tasks=6,
                      max_depth=4, rng=None):
    """Exhaustively search all op-programs up to max_depth on simulated tasks.
    Returns the best mean score achievable by ANY program (the representational
    ceiling for this rung)."""
    rng = rng or np.random.default_rng(0)
    # build a handful of simulated tasks with a known target for this rung
    tasks = [_sim_task_for(rung, world, rng) for _ in range(n_tasks)]
    tasks = [t for t in tasks if t is not None]
    if not tasks:
        return 0.0, ()
    best_score, best_prog = 0.0, ()
    ops = BENCH_PRIMITIVES
    for depth in range(1, max_depth + 1):
        for combo in itertools.product(ops, repeat=depth):
            sc = 0.0
            for (machine, ctx, target, outf) in tasks:
                final = _run(combo, machine, ctx)
                got = final.files.get(outf, "")
                sc += 1.0 if got == target else 0.0
            sc /= len(tasks)
            if sc > best_score:
                best_score, best_prog = sc, combo
                if best_score >= 1.0:
                    return best_score, best_prog
    return best_score, best_prog


def _sim_task_for(rung: Rung, world: RealComputerWorld, rng):
    """Construct a simulated (Machine, ctx, target_output) for a rung. For
    reachable rungs the target is the canonical program's output; for BEYOND
    rungs we compute the *true* intended target directly in Python."""
    keyword = str(rng.choice(list(_WORDS())))
    files = _random_files(rng, keyword)
    input_file = sorted(files)[int(rng.integers(0, len(files)))]
    ctx = Context(input_file=input_file, output_file="output.txt", keyword=keyword)
    machine = Machine(files=dict(files))
    outf = ctx.output_file
    if rung.canonical is not None:
        target = _run(rung.canonical, machine.clone(), ctx).files.get(outf, "")
        return (machine, ctx, target, outf)
    # BEYOND rungs: define the genuine intended output in plain Python
    content = files[input_file]
    if rung.name == "find_and_replace":
        target = content.replace(keyword, "REDACTED")
    elif rung.name == "word_frequency":
        from collections import Counter
        words = content.replace("\n", " ").split()
        c = Counter(words)
        target = "\n".join(f"{w} {c[w]}" for w in sorted(c))
    elif rung.name == "sum_numbers":
        nums = list(range(int(rng.integers(2, 6))))
        files = {input_file: "\n".join(str(n) for n in nums)}
        machine = Machine(files=files)
        target = str(sum(nums))
    else:
        return None
    return (machine, ctx, target, outf)


# --------------------------------------------------------------------------
# A single benchmark instance: concrete files + the genuine intended target.
# Used both for the synthesiser's (simulated) search and for real-shell grading.
# --------------------------------------------------------------------------
def make_instance(rung: Rung, rng):
    keyword = str(rng.choice(list(_WORDS())))
    files = _random_files(rng, keyword)
    input_file = sorted(files)[int(rng.integers(0, len(files)))]
    ctx = Context(input_file=input_file, output_file="output.txt", keyword=keyword)

    if rung.name == "sum_numbers":
        nums = [int(rng.integers(0, 9)) for _ in range(int(rng.integers(2, 6)))]
        files = {input_file: "\n".join(str(n) for n in nums)}
        target = str(sum(nums))
    elif rung.canonical is not None:
        target = _run(rung.canonical, Machine(files=dict(files)), ctx).files.get("output.txt", "")
    elif rung.name == "find_and_replace":
        target = files[input_file].replace(keyword, "REDACTED")
    elif rung.name == "word_frequency":
        from collections import Counter
        c = Counter(files[input_file].replace("\n", " ").split())
        target = "\n".join(f"{w} {c[w]}" for w in sorted(c))
    else:
        target = ""
    return {"files": files, "input_file": input_file, "keyword": keyword,
            "ctx": ctx, "target": target}


def sim_evaluate_factory(inst):
    """A grade function over the simulated machine for the synthesiser."""
    machine = Machine(files=dict(inst["files"]))
    ctx, target = inst["ctx"], inst["target"]

    def evaluate(program):
        got = _run(program, machine.clone(), ctx).files.get("output.txt", "")
        if got == target:
            return True, 1.0
        n = max(len(got), len(target), 1)
        matches = sum(1 for i in range(min(len(got), len(target)))
                      if got[i] == target[i])
        return False, matches / n
    return evaluate


def grade_on_real_shell(program, inst, world: RealComputerWorld):
    """Execute `program` as REAL shell commands in a throwaway sandbox and grade
    the real output file against the intended target. Returns (solved, n_calls)."""
    import shutil
    import tempfile
    sandbox = tempfile.mkdtemp(prefix="echo_bench_")
    try:
        for fn, content in inst["files"].items():
            with open(os.path.join(sandbox, fn), "w") as fh:
                fh.write(content + ("\n" if not content.endswith("\n") else ""))
        task = RealTask(name="bench", level=0, sandbox=sandbox,
                        input_file=inst["input_file"], keyword=inst["keyword"],
                        canonical=tuple(program), expected_output=inst["target"])
        output, calls, _ = world.execute([op for op in program if op in ALLOWED_OPS], task)
        return output.strip() == inst["target"].strip(), calls
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def _WORDS():
    from .environments.computer_world import WORDS
    return WORDS


def _random_files(rng, keyword):
    from .environments.computer_world import WORDS
    n = int(rng.integers(3, 6))
    files = {}
    names = rng.choice(WORDS, size=n, replace=False)
    for nm in names:
        lines = []
        for _ in range(int(rng.integers(3, 7))):
            w = list(rng.choice(WORDS, size=int(rng.integers(2, 4))))
            if rng.random() < 0.5:
                w.append(keyword)
            lines.append(" ".join(w))
        files[f"{nm}.txt"] = "\n".join(lines)
    return files
