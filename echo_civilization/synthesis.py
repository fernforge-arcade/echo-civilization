"""Domain-agnostic staged program synthesis.

This is the generalised version of the search that the string-task solver in
`agent.py` performs, lifted so it works for *any* domain whose solutions are
programs (tuples of primitive-operation names). It is the engine the Computer
World uses: an agent solves a multi-step computer task by

  1. recalling whole macros it already knows (own + inherited culture),
  2. recombining known macros (concatenating two learned pipelines),
  3. only then discovering from scratch — bounded, complexity-ordered blind
     search over the primitive operations.

The asymmetry between (1)/(2) and (3) is what makes accumulated culture decisive:
a depth-L pipeline costs ~|ops|^L to find blind, but is a one-step recombination
for an agent that inherited the constituent macros. This lets later generations
climb a complexity curriculum that earlier generations could never reach within
the same budget.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass
class SynthResult:
    program: tuple        # best program found (may be partial)
    solved: bool
    score: float          # best graded score in [0, 1]
    evals: int            # evaluation calls consumed
    discovered: bool      # True if found by from-scratch discovery (novel)
    via_composition: bool # True if found by recombining known macros


def synthesize(known_programs, primitives, evaluate, budget, rng,
               max_depth: int = 4, discovery_sample: int = 250):
    """Search for a program that solves the task.

    Args:
      known_programs : list[tuple] of macros the agent already knows (ordered by
                       priority — best first).
      primitives     : list[str] of all primitive op names available for discovery.
      evaluate       : callable(program) -> (solved: bool, score: float). Runs the
                       program in the domain and grades it against the goal.
      budget         : max number of `evaluate` calls.
      rng            : numpy Generator (for shuffling the discovery pool).
      max_depth      : deepest from-scratch composition to attempt.
      discovery_sample : cap on how many depth>=3 candidates are sampled.
    """
    evals = 0
    best = (tuple(), False, 0.0)  # (program, solved, score)

    def consider(program):
        nonlocal evals, best
        evals += 1
        solved, score = evaluate(program)
        if score > best[2]:
            best = (program, solved, score)
        return solved

    # 1) recall known macros -------------------------------------------------
    for prog in known_programs:
        if evals >= budget:
            return _result(best, evals, False, False)
        if consider(tuple(prog)):
            return _result(best, evals, False, False)

    # 2) recombine known macros (concatenate two learned pipelines) ----------
    for a, b in itertools.product(known_programs, known_programs):
        if evals >= budget:
            return _result(best, evals, False, False)
        if consider(tuple(a) + tuple(b)):
            return _result(best, evals, False, True)

    # 2b) MODIFY known macros: insert / prepend / append one primitive. This is
    #     "skills can be modified" — the engine of incremental climbing. A level-k
    #     pipeline is typically a level-(k-1) macro with one extra operation, so an
    #     agent that inherited the lower macro reaches the next rung in a cheap,
    #     bounded local search instead of a blind depth-k search.
    for prog in known_programs:
        for op in primitives:
            for pos in range(len(prog) + 1):
                if evals >= budget:
                    return _result(best, evals, False, True)
                variant = tuple(prog[:pos]) + (op,) + tuple(prog[pos:])
                if consider(variant):
                    return _result(best, evals, False, True)

    # 3) discovery from scratch, ordered by increasing complexity ------------
    # singles first (cheap -> primitives are learnable)
    for op in primitives:
        if evals >= budget:
            return _result(best, evals, True, False)
        if consider((op,)):
            return _result(best, evals, True, False)
    # then shuffled compositions of growing depth (composites are not learnable
    # from scratch within a tight budget => culture is required for deep tasks)
    for depth in range(2, max_depth + 1):
        if depth == 2:
            pool = [(a, b) for a in primitives for b in primitives]
        else:
            pool = [tuple(c) for c in itertools.product(primitives, repeat=depth)]
        rng.shuffle(pool)
        if depth >= 3:
            pool = pool[:discovery_sample]
        for combo in pool:
            if evals >= budget:
                return _result(best, evals, True, False)
            if consider(combo):
                return _result(best, evals, True, False)

    return _result(best, evals, True, False)


def _result(best, evals, discovered, via_composition):
    program, solved, score = best
    return SynthResult(program=program, solved=solved, score=score, evals=evals,
                       discovered=discovered and solved,
                       via_composition=via_composition and solved)
