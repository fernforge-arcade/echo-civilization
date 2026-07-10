"""Abstraction invention: learning the vocabulary in which programs are written.

The string world learned *programs* (compositions of a fixed primitive set). This
module lets the civilization learn the *concepts* -- it invents new named ops out
of recurring program fragments, adds them to a growing hierarchical library, and
then composes higher-level abstractions on top of those. This is the DreamCoder
"wake / sleep" idea, kept deliberately small and legible, and with no pretrained
model anywhere.

The loop, once per round of experience:

    wake   : solve as many tasks as the search budget allows, using the current
             library (base ops + everything invented so far).
    sleep  : mine the successful programs for frequently-recurring contiguous
             fragments; keep the ones whose MDL value is positive
             (value = search/description saved - cost); name them; add to the
             library. Re-encode past solutions with the new ops so that *pairs of
             abstractions* can themselves be abstracted next round -> hierarchy.

The payoff is measurable and is the whole point: an invented abstraction collapses
a depth-k base program into a single token, so tasks that were unreachable within a
fixed search budget become reachable, and the description length of solutions keeps
shrinking (compression). Capability accumulates by improving the language, not by
spending more compute.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .gridworld_arc import BASE_OP_NAMES, grids_equal, run_program


@dataclass
class Abstraction:
    """An invented op: a name that expands to a program over lower-level ops."""
    name: str
    expansion: tuple            # tuple of op names (base ops or lower abstractions)
    level: int                  # 0 = base; invented ops get 1 + max(level of parts)
    base_length: int            # length after fully expanding to base ops
    invented_round: int
    uses: int = 0               # how many solved tasks used it (culture: adoption)
    mdl_value: float = 0.0


class Library:
    """The growing, hierarchical concept library -- the civilization's culture."""

    def __init__(self):
        self.ops: dict[str, Abstraction] = {}     # only invented ops (base ops implicit)
        self._counter = 0

    # -- vocabulary ------------------------------------------------------------
    def op_names(self):
        """All ops available to search: base primitives + invented abstractions."""
        return BASE_OP_NAMES + list(self.ops.keys())

    def base_length(self, op) -> int:
        if op in self.ops:
            return self.ops[op].base_length
        return 1

    def expand_to_base(self, program) -> tuple:
        out = []
        for op in program:
            if op in self.ops:
                out.extend(self.expand_to_base(self.ops[op].expansion))
            else:
                out.append(op)
        return tuple(out)

    def levels(self):
        lv = Counter()
        for a in self.ops.values():
            lv[a.level] += 1
        return dict(lv)

    def max_level(self):
        return max([a.level for a in self.ops.values()], default=0)

    # -- invention -------------------------------------------------------------
    def add(self, expansion, invented_round) -> Abstraction:
        # level & base length computed from the (already-known) parts
        level = 1 + max((self.ops[o].level if o in self.ops else 0) for o in expansion)
        base_len = sum(self.base_length(o) for o in expansion)
        name = f"C{self._counter}_L{level}"
        self._counter += 1
        a = Abstraction(name=name, expansion=tuple(expansion), level=level,
                        base_length=base_len, invented_round=invented_round)
        self.ops[name] = a
        return a

    def encode(self, program) -> tuple:
        """Greedily rewrite a program to use the highest-level abstractions that
        match, longest-first. Idempotent-ish; used so higher abstractions can form
        over already-abstracted solutions (the hierarchy mechanism)."""
        # try longer / higher-level expansions first
        subs = sorted(self.ops.values(), key=lambda a: (-len(a.expansion), -a.level))
        prog = list(program)
        changed = True
        while changed:
            changed = False
            for a in subs:
                exp = list(a.expansion)
                L = len(exp)
                i = 0
                while i + L <= len(prog):
                    if prog[i:i + L] == exp:
                        prog[i:i + L] = [a.name]
                        changed = True
                    else:
                        i += 1
        return tuple(prog)


# --------------------------------------------------------------------------- #
# Search: solve one task with the current library, bounded by a budget
# --------------------------------------------------------------------------- #
@dataclass
class SolveResult:
    program: tuple | None
    solved: bool
    evals: int
    base_length: int            # length of the solution measured in base ops


def _consistent(program, examples, library) -> bool:
    for inp, out in examples:
        pred = run_program(program, inp, library)
        if not grids_equal(pred, out):
            return False
    return True


def solve_task(examples, library, budget, max_len=4, proposer=None):
    """Enumerate programs over the current vocabulary (increasing length) until one
    reproduces every I/O example or the budget of full-program evaluations runs out.

    Because an invented abstraction is a single token, the *effective* depth needed
    shrinks as the library grows -- that is how accumulated concepts make deep tasks
    reachable inside a fixed budget. `max_len` bounds length in *library tokens*, not
    base ops.
    """
    ops = library.op_names()
    if proposer is not None:
        ops = proposer.order(ops, examples, library)
    evals = 0

    # length-1 first, then grow. Standard iterative-deepening enumeration.
    frontier = [()]
    for length in range(1, max_len + 1):
        new_frontier = []
        for prefix in frontier:
            for op in ops:
                cand = prefix + (op,)
                if length < max_len:
                    new_frontier.append(cand)
                if evals >= budget:
                    return SolveResult(None, False, evals, 0)
                evals += 1
                if _consistent(cand, examples, library):
                    blen = sum(library.base_length(o) for o in cand)
                    return SolveResult(cand, True, evals, blen)
        frontier = new_frontier
        if evals >= budget:
            break
    return SolveResult(None, False, evals, 0)


# --------------------------------------------------------------------------- #
# Sleep: mine successful programs -> MDL-scored abstractions
# --------------------------------------------------------------------------- #
def mine_abstractions(solutions, library, invented_round, min_uses=2,
                      max_fragment=3, top_k=3):
    """Find recurring contiguous fragments in solved programs and promote the ones
    with positive MDL value.

    solutions : list[tuple] of solved programs, already encoded in current library
                tokens.
    value(fragment) = occurrences * (len(fragment) - 1)   # tokens saved at each use
                    - (len(fragment) + 1)                   # cost to store its definition
    A fragment is worth a name only if using it shortens the corpus by more than the
    definition costs -- classic minimum-description-length compression.
    """
    counts = Counter()
    for prog in solutions:
        n = len(prog)
        for L in range(2, max_fragment + 1):
            for i in range(n - L + 1):
                counts[tuple(prog[i:i + L])] += 1

    scored = []
    for frag, occ in counts.items():
        if occ < min_uses:
            continue
        saved = occ * (len(frag) - 1)
        cost = len(frag) + 1
        value = saved - cost
        if value > 0:
            scored.append((value, occ, frag))
    # prefer highest value; break ties by more occurrences then shorter fragment
    scored.sort(key=lambda t: (-t[0], -t[1], len(t[2])))

    invented = []
    for value, occ, frag in scored[:top_k]:
        # skip if this exact expansion already exists
        if any(a.expansion == frag for a in library.ops.values()):
            continue
        a = library.add(frag, invented_round)
        a.mdl_value = float(value)
        a.uses = occ
        invented.append(a)
    return invented
