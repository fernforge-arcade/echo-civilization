"""Skills as composable programs over strings.

A *primitive* is an atomic string->string transform (copy, reverse, caesar shift,
count, ...). A *program* is a left-to-right composition of primitives. A `Skill`
wraps a program with the metadata the brief asks for (name, creator, generation,
preconditions, examples, success rate, usage count) and is the unit that is
learned, taught, copied, modified, combined and inherited.

Crucially, primitives are the *building blocks of culture*: an agent that has
inherited `reverse` and `inc1` as known skills can discover the composite
`reverse then inc1` with a tiny search, whereas an agent starting from nothing
must search the full primitive space — usually beyond its budget. This is the
mechanism by which capability accumulates across generations.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

# A small, fixed alphabet keeps tabular learning + program search tractable.
ALPHABET = string.ascii_lowercase[:8]  # a..h


def _shift(s: str, k: int) -> str:
    out = []
    for ch in s:
        if ch in ALPHABET:
            out.append(ALPHABET[(ALPHABET.index(ch) + k) % len(ALPHABET)])
        else:
            out.append(ch)
    return "".join(out)


# Each primitive: name -> callable(str)->str
PRIMITIVES = {
    "copy": lambda s: s,
    "reverse": lambda s: s[::-1],
    "inc1": lambda s: _shift(s, 1),
    "inc2": lambda s: _shift(s, 2),
    "dec1": lambda s: _shift(s, -1),
    "count": lambda s: str(len(s)),
    "first": lambda s: s[0] if s else "",
    "last": lambda s: s[-1] if s else "",
    "double": lambda s: s + s,
    "dedup": lambda s: "".join(ch for i, ch in enumerate(s) if i == 0 or ch != s[i - 1]),
}

# Primitives that an agent could plausibly *discover from reward* via per-character
# tabular Q-learning (character-substitution maps). Structural primitives below
# are discovered instead by program search / composition.
SUBSTITUTION_PRIMITIVES = ["copy", "inc1", "inc2", "dec1"]


def run_program(program: tuple[str, ...], s: str) -> str:
    out = s
    for op in program:
        out = PRIMITIVES[op](out)
    return out


@dataclass
class Skill:
    """A learned, shareable capability."""

    name: str
    program: tuple[str, ...]          # behaviour pattern: ops applied in order
    creator: str                       # agent id that discovered it
    generation: int                    # generation it was discovered in
    preconditions: list[str] = field(default_factory=list)  # required known skills
    examples: list[tuple[str, str]] = field(default_factory=list)
    success_count: int = 0
    usage_count: int = 0
    reputation: float = 0.0            # culture-level standing
    adoption: int = 0                  # number of agents that adopted it

    @property
    def success_rate(self) -> float:
        return self.success_count / self.usage_count if self.usage_count else 0.0

    def apply(self, s: str) -> str:
        return run_program(self.program, s)

    def record_use(self, success: bool):
        self.usage_count += 1
        if success:
            self.success_count += 1

    def key(self) -> tuple[str, ...]:
        """Identity of a skill is its program (so duplicates merge in culture)."""
        return self.program

    def complexity(self) -> int:
        return len(self.program)

    def to_row(self) -> dict:
        return {
            "name": self.name,
            "program": "+".join(self.program),
            "creator": self.creator,
            "generation": self.generation,
            "preconditions": ",".join(self.preconditions),
            "success_count": self.success_count,
            "usage_count": self.usage_count,
            "success_rate": round(self.success_rate, 3),
            "reputation": round(self.reputation, 3),
            "adoption": self.adoption,
            "complexity": self.complexity(),
        }


def program_name(program: tuple[str, ...]) -> str:
    return " then ".join(program) if program else "noop"
