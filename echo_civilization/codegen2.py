"""Tier 8 — the next open-ended programming rung above Tier 7.

Tier 7 (codegen.py) synthesised a *flat* per-column reduction: read a CSV, reduce
each column, print. Real computer-use work climbs past that into programs that
build and iterate a **data structure**. This module adds the next rung:

    GROUP-BY AGGREGATION
    "read a CSV, group rows by a key column, aggregate a value column per group,
     print  key:value  pairs sorted by key."

That is a genuinely harder program than Tier 7: it needs a dict accumulator, a
two-pass shape (accumulate, then reduce-and-emit per key), and it must discover
*which* column is the key, *which* is the value, and *which* reduction — none of
which it is told. We synthesise it the same honest way as Tier 7: emit a program
in a tiny typed grammar that COMPILES TO REAL PYTHON, run it in a subprocess
against hidden tests, keep the first program that passes every test. No
pretrained model anywhere.

The thesis holds exactly as before. The expensive thing is discovering the
*structural skeleton* (`group_by_aggregate`). An agent that inherited that
skeleton from culture jumps straight to it and only has to search the cheap
parameter tail (key col / value col / reducer); a fresh agent must grind through
every wrong skeleton's parameter grid first and runs out of a tight budget. The
ceiling moved up one more rung — and culture still decides who clears it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


# ----------------------------------------------------------------------
# Grammar.  Every skeleton iterates the SAME parameter grid (key/val/reducer),
# so a fresh search must pay for each wrong skeleton's whole grid before it
# reaches the correct one — which is placed LAST. An inherited skeleton is tried
# first, collapsing that cost to the parameter tail.
# ----------------------------------------------------------------------
SKELETONS = ["row_count", "first_column", "per_column_reduce", "group_by_aggregate"]
KEY_COLS = [0, 1]
VAL_COLS = [1, 2]
REDUCERS = {
    "sum":   "sum(v)",
    "mean":  "sum(v) / len(v)",
    "max":   "max(v)",
    "min":   "min(v)",
    "count": "len(v)",
}

GROUP_BY_SKELETON = "group_by_aggregate"

_HEAD = (
    "import sys, csv\n"
    "path = sys.argv[1]\n"
    "with open(path, newline='') as fh:\n"
    "    rows = list(csv.reader(fh))\n"
    "rows = rows[1:] if rows else rows   # drop header\n"
)


def render(gene) -> str:
    """Render a code-gene into a runnable Python script (argv[1] = csv path)."""
    skel = gene["skeleton"]
    kc = gene.get("key_col", 0)
    vc = gene.get("val_col", 1)
    reducer = REDUCERS.get(gene.get("reducer", "sum"), "sum(v)")

    if skel == "group_by_aggregate":
        body = (
            "groups = {}\n"
            "for row in rows:\n"
            "    try:\n"
            f"        k = row[{kc}]\n"
            f"        x = float(row[{vc}])\n"
            "    except (IndexError, ValueError):\n"
            "        continue\n"
            "    groups.setdefault(k, []).append(x)\n"
            "out = []\n"
            "for k in sorted(groups):\n"
            "    v = groups[k]\n"
            f"    r = {reducer}\n"
            "    rs = str(int(r)) if float(r).is_integer() else str(r)\n"
            "    out.append(k + ':' + rs)\n"
            "print(' '.join(out))\n"
        )
    elif skel == "per_column_reduce":
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
            "            out.append(str(r))\n"
            "    print(' '.join(out))\n"
        )
    elif skel == "first_column":
        body = (
            "v = []\n"
            "for row in rows:\n"
            f"    try: v.append(float(row[{vc}]))\n"
            "    except (ValueError, IndexError): pass\n"
            f"r = {reducer} if v else 0\n"
            "print(r)\n"
        )
    elif skel == "row_count":
        body = "print(len(rows))\n"
    else:
        body = "print('')\n"
    return _HEAD + body


def enumerate_genes(skeleton_first=None):
    """Yield code-genes shallow-first. If `skeleton_first` is given (an inherited
    skill), that skeleton is tried before the rest — the cultural shortcut."""
    skels = list(SKELETONS)
    if skeleton_first in skels:
        skels.remove(skeleton_first)
        skels.insert(0, skeleton_first)
    for skel in skels:
        for kc in KEY_COLS:
            for vc in VAL_COLS:
                for reducer in REDUCERS:
                    yield {"skeleton": skel, "key_col": kc,
                           "val_col": vc, "reducer": reducer}


# ----------------------------------------------------------------------
# Execute a rendered script for real and grade it on hidden test cases.
# ----------------------------------------------------------------------
def run_script(source: str, csv_text: str, timeout=5):
    """Write `source` to a temp .py, feed it a temp .csv, run it for real, return
    stdout (stripped) or None on error/timeout."""
    d = tempfile.mkdtemp(prefix="echo_codegen2_")
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
    gene: dict | None
    solved: bool
    score: float
    trials: int          # real subprocess executions
    via_recall: bool


def synthesize_code(tests, budget, rng, inherited_skeleton=None):
    """Search the grammar for a Python script passing every (csv_text, expected)
    test. `inherited_skeleton` is the cultural shortcut. Each candidate is COMPILED
    to real Python and EXECUTED — `trials` counts the real runs."""
    best = (None, None, None, 0.0)
    trials = 0
    for gene in enumerate_genes(skeleton_first=inherited_skeleton):
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
                break   # don't burn budget on a clearly-wrong candidate
        score = passed / len(tests)
        if score > best[3]:
            best = (source, gene["skeleton"], gene, score)
        if ok and passed == len(tests):
            return CodegenResult(source=source, skeleton=gene["skeleton"], gene=gene,
                                 solved=True, score=1.0, trials=trials,
                                 via_recall=(gene["skeleton"] == inherited_skeleton))
    src, skel, gene, sc = best
    return CodegenResult(source=src, skeleton=skel, gene=gene, solved=sc >= 1.0,
                         score=sc, trials=trials, via_recall=False)


# ----------------------------------------------------------------------
# The Tier-8 task: group-by aggregation. Build hidden test cases with a Python
# oracle (the genuine intended behaviour), then synthesise a script to match.
# ----------------------------------------------------------------------
_KEYS = ["red", "green", "blue", "gold", "grey"]


def make_group_by_tests(rng, n=3, reducer="sum", key_col=0, val_col=1):
    """Hidden tests for: group rows by `key_col`, reduce `val_col` per group,
    print  k:v  pairs sorted by key. The reducer/columns are the hidden transform
    a correct program must recover."""
    fn = {
        "sum": lambda v: sum(v),
        "mean": lambda v: sum(v) / len(v),
        "max": lambda v: max(v),
        "min": lambda v: min(v),
        "count": lambda v: len(v),
    }[reducer]
    tests = []
    for _ in range(n):
        ncol = max(key_col, val_col) + 1
        nrow = int(rng.integers(5, 9))
        keyset = list(rng.choice(_KEYS, size=int(rng.integers(2, 4)), replace=False))
        header = ",".join(f"c{i}" for i in range(ncol))
        rows = []
        groups = {}
        for _ in range(nrow):
            row = [str(int(rng.integers(0, 50))) for _ in range(ncol)]
            k = str(rng.choice(keyset))
            row[key_col] = k
            val = int(rng.integers(0, 50))
            row[val_col] = str(val)
            rows.append(row)
            groups.setdefault(k, []).append(float(val))
        csv_text = header + "\n" + "\n".join(",".join(r) for r in rows) + "\n"
        parts = []
        for k in sorted(groups):
            r = fn(groups[k])
            rs = str(int(r)) if float(r).is_integer() else str(r)
            parts.append(k + ":" + rs)
        expected = " ".join(parts)
        tests.append((csv_text, expected))
    return tests


def make_tier8_task(rng):
    """Pick a random concrete group-by task instance (hidden columns + reducer)."""
    reducer = str(rng.choice(list(REDUCERS)))
    key_col = int(rng.choice(KEY_COLS))
    # value column must differ from key column
    val_col = int(rng.choice([c for c in VAL_COLS if c != key_col] or VAL_COLS))
    tests = make_group_by_tests(rng, n=3, reducer=reducer,
                                key_col=key_col, val_col=val_col)
    return tests, {"reducer": reducer, "key_col": key_col, "val_col": val_col}
