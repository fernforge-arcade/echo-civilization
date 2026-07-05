"""Wrangling as a civilization task — the evolved agents (not a standalone tool)
do the LLM-style, per-row data work, and it is the ACCUMULATED CULTURE that lets
them.

This module puts the practical `echofill` synthesiser inside the same
recall -> recombine -> modify -> discover loop the rest of the civilization uses
(see agent.solve_task / synthesis.synthesize). The point is not "here is another
program-by-example tool" — those exist (Excel FlashFill, Microsoft PROSE). The
point is the thesis of this whole project, now aimed at real work:

    a from-scratch agent has a hard ceiling — the bounded search cannot compose
    two *parametric* ops (split, extract, replace with induced arguments), so
    tasks like "email -> company name" (split on '@', then split on '.') are
    unreachable no matter how long it searches. A cultured agent that inherited
    the two single-op pieces from earlier discoveries solves the same held-out
    task by RECOMBINING them, in a handful of checks.

So the accumulated skill library is doing genuine capability lifting, on tasks a
company would today pay an LLM to do one row at a time. Cost: ~$0 and a few
microseconds per row, deterministic and auditable.

The per-agent discovery engine is `echofill.synthesize` (staged, argument-induced
search). The culture is the set of grounded pieces agents discovered. Nothing
here touches the A-L research code; it reuses the *pattern*, not the classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .echofill import apply_program, program_str, synthesize


# --------------------------------------------------------------------------- #
# A wrangling skill: a grounded echofill program (list of (op, arg) steps) plus
# the civilization metadata the brief asks for. Mirrors skills.Skill, but for the
# echofill domain (whose programs carry induced arguments, unlike the a..h string
# primitives) so the two never collide.
# --------------------------------------------------------------------------- #

@dataclass
class WrangleSkill:
    program: tuple                      # tuple of (op, arg) steps
    creator: str                        # agent id that discovered it
    generation: int
    examples: list = field(default_factory=list)
    usage_count: int = 0
    success_count: int = 0
    reputation: float = 0.0
    adoption: int = 0

    def key(self):
        return self.program

    @property
    def depth(self):
        return len(self.program)

    def describe(self):
        return program_str(list(self.program))


class WrangleCulture:
    """Shared civilization store of discovered wrangling pieces. Successful
    pieces gain reputation and are offered to new agents; unused ones decay."""

    def __init__(self):
        self.skills: dict[tuple, WrangleSkill] = {}
        self.propagation: list[tuple] = []   # (program_str, from_id, to_id, gen)

    def contribute(self, skill: WrangleSkill) -> WrangleSkill:
        key = skill.key()
        if key in self.skills:
            s = self.skills[key]
            s.usage_count += skill.usage_count
            s.success_count += skill.success_count
            s.reputation += 1.0
            return s
        skill.reputation += 1.0
        self.skills[key] = skill
        return skill

    def top(self, n=None):
        ranked = sorted(self.skills.values(),
                        key=lambda s: (-s.reputation, s.depth))
        return ranked[:n] if n else ranked

    def size(self):
        return len(self.skills)


# --------------------------------------------------------------------------- #
# The agent. Solves a wrangling task from a few example rows, under an evaluation
# BUDGET (number of candidate programs it may test). Order, exactly like the rest
# of the civilization:
#   1) RECALL   — try each inherited/own grounded piece directly.
#   2) RECOMBINE— concatenate two known pieces (this is the stage that reaches
#                 param+param programs the from-scratch search cannot).
#   3) MODIFY   — append/prepend one cheap non-param op to a known piece.
#   4) DISCOVER — fall back to echofill's staged synthesis from scratch, its
#                 `tried` count charged against the remaining budget.
# A naive agent (empty library) skips straight to (4) and hits the ceiling.
# --------------------------------------------------------------------------- #

# The non-parametric "finishing" ops used by the cheap MODIFY stage.
_MODIFY_OPS = ["title", "upper", "lower", "strip", "collapse_ws",
               "keep_digits", "capitalize"]


@dataclass
class SolveResult:
    program: tuple | None
    solved: bool
    evals: int
    via: str          # "recall" | "recombine" | "modify" | "discover" | "none"


class EchofillAgent:
    _counter = 0

    def __init__(self, generation: int, parents=None):
        EchofillAgent._counter += 1
        self.id = f"W{EchofillAgent._counter:05d}"
        self.generation = generation
        self.parents = parents or []
        self.known: dict[tuple, WrangleSkill] = {}   # grounded programs
        self.contributions: list = []
        self.tasks_solved = 0

    # -- skill library ------------------------------------------------------
    def learn(self, skill: WrangleSkill) -> bool:
        if skill.key() in self.known:
            return False
        self.known[skill.key()] = skill
        return True

    def _known_programs(self):
        # best culture first (high reputation), shortest first
        return [s.program for s in sorted(
            self.known.values(), key=lambda s: (-s.reputation, s.depth))]

    # -- solving ------------------------------------------------------------
    def solve(self, examples, query_input, budget=200, generation=0,
              max_depth=3, allow_discovery=True, learn_at_solve=True):
        """Return (prediction, SolveResult). `examples` demonstrate the rule;
        `query_input` is a held-out row to transform."""
        evals = 0
        rows = list(examples)

        def fits(prog):
            return all(apply_program(list(prog), i) == o for i, o in rows)

        known = self._known_programs()

        # 1) RECALL
        for prog in known:
            evals += 1
            if fits(prog):
                self.known[tuple(prog)].usage_count += 1
                self.known[tuple(prog)].success_count += 1
                return apply_program(list(prog), query_input), \
                    SolveResult(tuple(prog), True, evals, "recall")
            if evals >= budget:
                return query_input, SolveResult(None, False, evals, "none")

        # 2) RECOMBINE — concatenate two known pieces
        for a in known:
            for b in known:
                combo = tuple(a) + tuple(b)
                evals += 1
                if fits(combo):
                    disc = self._abstract(combo, rows, generation) if learn_at_solve else None
                    return apply_program(list(combo), query_input), \
                        SolveResult(combo, True, evals, "recombine")
                if evals >= budget:
                    return query_input, SolveResult(None, False, evals, "none")

        # 3) MODIFY — append / prepend one non-param op to a known piece
        for a in known:
            for op in _MODIFY_OPS:
                for variant in (tuple(a) + ((op, None),),
                                ((op, None),) + tuple(a)):
                    evals += 1
                    if fits(variant):
                        disc = self._abstract(variant, rows, generation) if learn_at_solve else None
                        return apply_program(list(variant), query_input), \
                            SolveResult(variant, True, evals, "modify")
                    if evals >= budget:
                        return query_input, SolveResult(None, False, evals, "none")

        if not allow_discovery:
            return query_input, SolveResult(None, False, evals, "none")

        # 4) DISCOVER from scratch (bounded by remaining budget)
        res = synthesize(rows, max_depth=max_depth)
        evals += res.tried
        if res.solved and evals <= budget:
            prog = tuple(res.program)
            disc = self._abstract(prog, rows, generation) if learn_at_solve else None
            return apply_program(list(prog), query_input), \
                SolveResult(prog, True, evals, "discover")

        return query_input, SolveResult(None, False, evals, "none")

    def _abstract(self, program, examples, generation):
        skill = WrangleSkill(program=tuple(program), creator=self.id,
                             generation=generation, examples=list(examples[:3]))
        self.learn(skill)
        return skill
