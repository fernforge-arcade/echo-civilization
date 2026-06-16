"""Teaching — horizontal (within-generation) skill transfer.

A teacher observes its own successful behaviour, extracts the underlying skill,
and offers it to a student. The student tries the skill on the demonstrated
examples and *keeps it only if it works for them* (and if their imitation
preference is high enough). This is distinct from inheritance (which is vertical,
parent->child) and from cultural contribution (which is to the shared repository).
"""

from __future__ import annotations

from .agent import Agent
from .culture import CulturalMemory
from .skills import Skill


def teach(teacher: Agent, student: Agent, skill: Skill,
          culture: CulturalMemory, generation: int) -> bool:
    """Teacher transfers `skill` to `student`. Returns True if adopted."""
    if student.preferences["imitation"] < student.rng.uniform(0.0, 1.0):
        return False  # student not in a receptive mood

    # student verifies the skill against the teacher's demonstrated examples
    works = all(skill.apply(i) == o for i, o in skill.examples) if skill.examples else True
    if not works:
        return False

    adopted = student.learn_skill(Skill(
        name=skill.name, program=skill.program, creator=skill.creator,
        generation=skill.generation, preconditions=list(skill.preconditions),
        examples=list(skill.examples),
    ))
    if adopted:
        skill.adoption += 1
        teacher.taught.append(skill.name)
        teacher.reputation += 0.5
        teacher.befriend(student.id, 0.2)
        student.befriend(teacher.id, 0.1)
        culture.record_propagation(skill.program, teacher.id, student.id, generation)
        culture.reward_skill(skill.key(), 0.3)
    return adopted


def share_round(agents, culture: CulturalMemory, generation: int, rng) -> int:
    """One round of peer teaching across a population. Each agent may teach its
    best skill to a random peer. Returns number of successful transfers."""
    transfers = 0
    ids = list(range(len(agents)))
    for i in ids:
        teacher = agents[i]
        if not teacher.known_skills:
            continue
        # teach the highest-reputation skill the teacher knows
        skill = max(teacher.known_skills.values(),
                    key=lambda s: (s.reputation, s.success_rate))
        j = int(rng.integers(0, len(agents)))
        if j == i:
            continue
        if teach(teacher, agents[j], skill, culture, generation):
            transfers += 1
    return transfers
