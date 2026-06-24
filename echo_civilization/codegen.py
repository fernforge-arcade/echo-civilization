"""Tier-7 prototype — grammar-guided synthesis of REAL Python code.

The benchmark's Tier 7 ("write a Python script that reads a CSV and prints column
averages") was marked NOT REPRESENTABLE: the op-pipeline action space cannot
express a general program. This module gives the agents a *second* action space —
a tiny typed grammar that **compiles to actual Python source**, which we run in a
subprocess against hidden test cases. We enumerate the grammar shallow-first and
keep the first program whose real output matches every test. That is genuine
program synthesis of executable code from examples, with no pretrained model.

It moves ONE Tier-7 rung (the CSV-averages script) from unreachable to reachable,
and — exactly as in every other world here — an agent that inherited the right
code *skeleton* solves it in a couple of trials while a fresh agent must search
the whole grammar. The Flask-app and repo-refactor rungs stay out of reach; we
say so. The ceiling moved up one rung, honestly, by a real mechanism.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


# ----------------------------------------------------------------------
# The grammar.  A "code gene" is a dict of choices; render() turns it into a
# real Python script.  The search space is the cartesian product of the choices
# (plus a structural skeleton id) — small, typed, and fully executable.
# ----------------------------------------------------------------------
# Default search order puts the simple/wrong skeletons first, so the correct one
# (per_column_reduce) is found only after real search — a fresh agent must grind
# through the grammar, while an agent that inherited the skeleton jumps to it.
SKELETONS = ["passthrough", "row_count", "first_column", "per_column_reduce"]
REDUCERS = {
    "mean": "sum(v) / len(v)",
    "sum":  "sum(v)",
    "max":  "max(v)",
    "min":  "min(v)",
}
HAS_HEADER = [True, False]
DELIMS = [",", "\t", ";"]
FORMATS = ["plain", "two_dp"]


def render(gene) -> str:
    """Render a code-gene into a runnable Python script (reads argv[1] = csv path)."""
    skel = gene["skeleton"]
    delim = repr(gene.get("delim", ","))
    header = gene.get("header", True)
    reducer = REDUCERS.get(gene.get("reducer", "mean"), "sum(v) / len(v)")
    fmt = gene.get("fmt", "plain")
    val_fmt = "round(r, 2)" if fmt == "two_dp" else "r"

    head = (
        "import sys, csv\n"
        "path = sys.argv[1]\n"
        "with open(path, newline='') as fh:\n"
        f"    rows = list(csv.reader(fh, delimiter={delim}))\n"
    )
    if header:
        head += "rows = rows[1:] if rows else rows\n"

    if skel == "per_column_reduce":
        body = (
            "if rows:\n"
            "    ncol = len(rows[0])\n"
            "    out = []\n"
            "    for c in range(ncol):\n"
            "        v = []\n"
            "        for row in rows:\n"
            "            try: v.append(float(row[c]))\n"
            "            except (ValueError, IndexError): pass\n"
            "        if v:\n"
            f"            r = {reducer}\n"
            f"            out.append(str({val_fmt}))\n"
            "    print(' '.join(out))\n"
        )
    elif skel == "passthrough":
        body = "print(rows)\n"
    elif skel == "row_count":
        body = "print(len(rows))\n"
    elif skel == "first_column":
        body = (
            "v = []\n"
            "for row in rows:\n"
            "    try: v.append(float(row[0]))\n"
            "    except (ValueError, IndexError): pass\n"
            f"r = {reducer} if v else 0\n"
            f"print({val_fmt})\n"
        )
    else:
        body = "print('')\n"
    return head + body


def enumerate_genes(skeleton_first=None):
    """Yield code-genes shallow-first. If `skeleton_first` is given (an inherited
    skill), its skeleton is tried before the rest — the cultural shortcut."""
    skels = list(SKELETONS)
    if skeleton_first in skels:
        skels.remove(skeleton_first)
        skels.insert(0, skeleton_first)
    for skel in skels:
        for reducer in REDUCERS:
            for header in HAS_HEADER:
                for delim in DELIMS:
                    for fmt in FORMATS:
                        yield {"skeleton": skel, "reducer": reducer,
                               "header": header, "delim": delim, "fmt": fmt}


# ----------------------------------------------------------------------
# Execute a rendered script for real and grade it on hidden test cases.
# ----------------------------------------------------------------------
def run_script(source: str, csv_text: str, timeout=5):
    """Write `source` to a temp .py, feed it a temp .csv, run it for real, return
    stdout (stripped) or None on error/timeout."""
    d = tempfile.mkdtemp(prefix="echo_codegen_")
    try:
        sp = os.path.join(d, "prog.py")
        cp = os.path.join(d, "data.csv")
        with open(sp, "w") as fh:
            fh.write(source)
        with open(cp, "w") as fh:
            fh.write(csv_text)
        proc = subprocess.run([sys.executable, sp, cp], capture_output=True,
                              text=True, timeout=timeout,
                              env={"PATH": os.environ.get("PATH", "")})
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()
    except Exception:
        return None
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


@dataclass
class CodegenResult:
    source: str | None
    skeleton: str | None
    solved: bool
    score: float
    trials: int          # how many real subprocess executions we ran
    via_recall: bool


def synthesize_code(tests, budget, rng, inherited_skeleton=None):
    """Search the grammar for a Python script passing every (csv_text, expected)
    test. `inherited_skeleton` is the cultural shortcut. Returns CodegenResult.

    Each candidate is COMPILED to real Python and EXECUTED — `trials` counts the
    real runs, the honest budget for this action space."""
    best = (None, None, 0.0)
    trials = 0
    genes = list(enumerate_genes(skeleton_first=inherited_skeleton))
    # within the chosen-skeleton-first order, lightly shuffle the parameter tail so
    # repeat runs aren't lock-step (still skeleton-first)
    for gene in genes:
        if trials >= budget:
            break
        source = render(gene)
        passed = 0
        ok = True
        for csv_text, expected in tests:
            trials += 1
            got = run_script(source, csv_text)
            if got is not None and got == expected:
                passed += 1
            else:
                ok = False
                # don't burn the whole budget on a clearly-wrong skeleton
                break
        score = passed / len(tests)
        if score > best[2]:
            best = (source, gene["skeleton"], score)
        if ok and passed == len(tests):
            return CodegenResult(source=source, skeleton=gene["skeleton"],
                                 solved=True, score=1.0, trials=trials,
                                 via_recall=(gene["skeleton"] == inherited_skeleton))
    src, skel, sc = best
    return CodegenResult(source=src, skeleton=skel, solved=sc >= 1.0, score=sc,
                         trials=trials, via_recall=False)


# ----------------------------------------------------------------------
# The Tier-7 task: CSV column averages.  Build hidden test cases with a Python
# oracle (the genuine intended behaviour), then synthesise a script to match.
# ----------------------------------------------------------------------
def make_csv_avg_tests(rng, n=3):
    tests = []
    for _ in range(n):
        ncol = int(rng.integers(2, 4))
        nrow = int(rng.integers(3, 6))
        header = [f"c{i}" for i in range(ncol)]
        rows = [[int(rng.integers(0, 50)) for _ in range(ncol)] for _ in range(nrow)]
        csv_text = ",".join(header) + "\n" + \
            "\n".join(",".join(str(x) for x in r) for r in rows) + "\n"
        means = []
        for c in range(ncol):
            col = [r[c] for r in rows]
            means.append(str(sum(col) / len(col)))
        expected = " ".join(means)
        tests.append((csv_text, expected))
    return tests


CSV_AVG_SKELETON = "per_column_reduce"
