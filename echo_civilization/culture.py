"""Cultural memory — the shared civilization-level skill repository.

Agents contribute discovered skills here. Each skill accrues reputation, adoption
and an aggregate success rate. Successful skills spread (they are offered to new
agents and to peers); failed skills decay in reputation and are eventually
forgotten. This is the substrate of cultural evolution: it persists *between*
agents and *between* generations, unlike any individual's memory.
"""

from __future__ import annotations

from .skills import Skill, program_name


class CulturalMemory:
    def __init__(self, forget_threshold: float = -2.0):
        # keyed by program identity so duplicate discoveries merge
        self.skills: dict[tuple, Skill] = {}
        self.forget_threshold = forget_threshold
        # propagation events: (skill_program, from_agent, to_agent, generation)
        self.propagation_log: list[tuple] = []

    def contribute(self, skill: Skill) -> Skill:
        """Add or merge a skill. Returns the canonical stored skill."""
        key = skill.key()
        if key in self.skills:
            existing = self.skills[key]
            existing.usage_count += skill.usage_count
            existing.success_count += skill.success_count
            existing.reputation += 1.0
            for ex in skill.examples:
                if ex not in existing.examples:
                    existing.examples.append(ex)
            return existing
        skill.reputation += 1.0
        self.skills[key] = skill
        return skill

    def record_propagation(self, program, from_agent, to_agent, generation):
        self.propagation_log.append((program_name(program), from_agent, to_agent, generation))

    def reward_skill(self, key, amount: float):
        if key in self.skills:
            self.skills[key].reputation += amount

    def known_primitive_set(self) -> set[str]:
        """All primitive ops that appear in any cultural skill — the building
        blocks later generations can compose with cheaply."""
        prims: set[str] = set()
        for sk in self.skills.values():
            prims.update(sk.program)
        return prims

    def top_skills(self, n: int = 10):
        return sorted(self.skills.values(), key=lambda s: s.reputation, reverse=True)[:n]

    def decay(self, amount: float = 0.05):
        """Reputation decays; skills below threshold are culturally forgotten."""
        dead = []
        for key, sk in self.skills.items():
            sk.reputation -= amount
            if sk.reputation < self.forget_threshold:
                dead.append(key)
        for key in dead:
            del self.skills[key]
        return len(dead)

    def adoption_curve(self):
        return {program_name(k): s.adoption for k, s in self.skills.items()}

    def size(self) -> int:
        return len(self.skills)
