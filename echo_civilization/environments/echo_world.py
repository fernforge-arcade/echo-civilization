"""Environment 0 — Echo World (the simplest possible world).

The agent must reproduce a target string. The rule is always `copy`. This is the
ground where agents first learn the most basic competence: mapping input to
output / copying / remembering a sequence. Solved primarily by the tabular
Q-learning substitution loop, which yields the foundational `copy` skill that all
later composite skills build on.
"""

from __future__ import annotations

from .base import StringEnvironment, Task


class EchoWorld(StringEnvironment):
    name = "echo"

    def sample_task(self) -> Task:
        return self.make_task(("copy",), "echo")
