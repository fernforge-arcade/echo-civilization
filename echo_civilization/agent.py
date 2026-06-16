"""The Agent: a simple learning entity (NOT a language model).

Each agent has identity (id, generation, parents), internal state (a learner, a
known-skill library, memory, preferences), goals (maximise reward / explore /
learn), and social attributes (reputation, relationships, teaching &
contribution history).

The agent's `solve_task` method is the heart of capability accumulation. To solve
"map input string -> output string" it:
  1. tries skills it already knows (cultural inheritance + own discoveries),
  2. tries cheap compositions of its *known* skills,
  3. falls back to discovery — tabular Q-learning for character-substitution maps
     and bounded program search over the full primitive space —
all under a fixed evaluation BUDGET. Known skills make step 2 tiny; without them
the agent is forced into expensive step 3 and usually runs out of budget on
composite tasks. That asymmetry is what makes generation 100 outperform
generation 1 on hard tasks.
"""

from __future__ import annotations

import itertools

import numpy as np

from .learning import QLearner
from .memory import LongTermMemory, ShortTermMemory
from .skills import (ALPHABET, PRIMITIVES, SUBSTITUTION_PRIMITIVES, Skill,
                     program_name, run_program)


class Agent:
    _counter = 0

    def __init__(self, generation: int, rng: np.random.Generator,
                 parents=None, specialization: str | None = None):
        Agent._counter += 1
        self.id = f"A{Agent._counter:05d}"
        self.generation = generation
        self.parents = parents or []
        self.rng = rng

        # internal state
        self.short_term = ShortTermMemory(capacity=8)
        self.long_term = LongTermMemory()
        self.known_skills: dict[tuple, Skill] = {}   # string-domain program -> Skill
        self.computer_skills: dict[tuple, Skill] = {}  # computer-domain macro -> Skill
        self.preferences = {
            "exploration": float(rng.uniform(0.1, 0.5)),   # search appetite
            "imitation": float(rng.uniform(0.3, 0.9)),     # tendency to adopt culture
            "risk": float(rng.uniform(0.0, 1.0)),
        }
        self.specialization = specialization  # e.g. which task family it favours

        # goals / performance bookkeeping
        self.lifetime_reward = 0.0
        self.tasks_attempted = 0
        self.tasks_solved = 0

        # social
        self.reputation = 0.0
        self.relationships: dict[str, float] = {}     # other_id -> affinity
        self.taught: list[str] = []                    # skills taught to others
        self.contributions: list[tuple] = []           # skills contributed to culture

    # ------------------------------------------------------------------ skills
    def learn_skill(self, skill: Skill) -> bool:
        """Adopt a skill into the personal library. Returns True if newly added."""
        if skill.key() in self.known_skills:
            return False
        # store a per-agent copy so usage stats are individual
        self.known_skills[skill.key()] = skill
        return True

    def known_programs(self):
        return list(self.known_skills.keys())

    # ----------------------------------------------------- computer domain
    def learn_computer_skill(self, skill: Skill) -> bool:
        if skill.key() in self.computer_skills:
            return False
        self.computer_skills[skill.key()] = skill
        return True

    def _priority_computer_programs(self):
        progs = sorted(self.computer_skills.values(),
                       key=lambda s: (-s.reputation, s.complexity()))
        return [s.program for s in progs]

    def solve_computer_task(self, task, budget: int = 80, generation: int = 0,
                            allow_discovery: bool = True):
        """Operate the simulated computer to satisfy a task's goal.

        Reuses the domain-agnostic staged synthesiser: recall known macros,
        recombine them, then (optionally) discover from scratch. A newly solved
        program is abstracted into a reusable macro skill. Returns the SynthResult.
        """
        from .environments.computer_world import (COMPUTER_PRIMITIVES,
                                                   run_computer_program)
        from .synthesis import synthesize

        def evaluate(program):
            final = run_computer_program(program, task.machine, task.ctx)
            return task.grade(final)

        known = self._priority_computer_programs()
        primitives = list(COMPUTER_PRIMITIVES.keys())
        res = synthesize(
            known, primitives, evaluate, budget, self.rng,
            max_depth=(2 if not allow_discovery else 4))
        # abstract a freshly solved program into a macro skill
        disc = None
        if res.solved and res.program and (res.discovered or res.via_composition):
            disc = self._abstract_computer_skill(res.program, task, generation,
                                                  derived=res.via_composition)
        elif res.solved and res.program and res.program not in self.computer_skills:
            disc = self._abstract_computer_skill(res.program, task, generation)
        return res, disc

    def _abstract_computer_skill(self, program, task, generation, derived=False):
        from .skills import Skill, program_name
        skill = Skill(
            name=task.name if task.name else program_name(program),
            program=tuple(program), creator=self.id, generation=generation,
            preconditions=list(dict.fromkeys(program)) if derived else [],
            examples=[(task.ctx.input_file, task.expected_output[:24])],
        )
        self.learn_computer_skill(skill)
        return skill

    # --------------------------------------------------------------- solving
    def solve_task(self, examples, query_input, budget: int = 60,
                   generation: int = 0, allow_discovery: bool = True,
                   learn_at_solve: bool = True):
        """Attempt to produce the correct output for `query_input`.

        `examples` is a list of (inp, out) demonstrating the hidden rule. Returns
        (prediction, solved_program_or_None, evaluations_used, discovered_skill).

        With `allow_discovery=False` the agent may only recall and recombine
        skills it already knows (no blind search / no new learning). This is used
        by the evaluation framework to measure *accumulated* capability, isolating
        cultural inheritance from raw per-trial brute force.

        With `learn_at_solve=False` the agent does NOT abstract/store any program
        it finds (no mutation of its skill library). This is essential for a clean
        held-out evaluation: otherwise an agent could learn a held-out depth-2
        composite while solving one eval task and reuse it to solve a depth-3 eval
        task — test-time leakage. Frozen evaluation measures only knowledge that
        accumulated during *training*.
        """
        evals = 0
        train = examples

        def consistent(program) -> bool:
            return all(run_program(program, i) == o for i, o in train)

        # 1) KNOWN skills: recall own + inherited/cultural skills first. This is
        #    cheap because an agent's library is small — the payoff of culture.
        for prog in self._priority_known_programs():
            evals += 1
            if consistent(prog):
                self.known_skills[prog].record_use(True)
                return run_program(prog, query_input), prog, evals, None
            if evals >= budget:
                return self._give_up(query_input, evals)

        # 2) COMPOSE known programs (knowledge recombination). Composing two known
        #    multi-op skills can reach deep programs in a handful of checks — e.g.
        #    (reverse,inc1) + (count) solves a length-3 task. From-scratch search
        #    would need to find all three ops blindly.
        known = self._priority_known_programs()
        for a, b in itertools.product(known, known):
            combo = tuple(a) + tuple(b)
            evals += 1
            if consistent(combo):
                disc = (self._abstract_skill(combo, train, generation, derived=True)
                        if learn_at_solve else None)
                return run_program(combo, query_input), combo, evals, disc
            if evals >= budget:
                return self._give_up(query_input, evals)

        if not allow_discovery:
            return self._give_up(query_input, evals)

        # 3a) DISCOVERY shortcut: tabular Q-learning for character-substitution
        #     maps (the genuine experience->reward->policy loop). Cheap.
        sub = self._qlearn_substitution(train)
        if sub is not None:
            evals += 8  # cost of the learning trials
            prog = (sub,)
            if consistent(prog) and evals <= budget:
                disc = (self._abstract_skill(prog, train, generation)
                        if learn_at_solve else None)
                return run_program(prog, query_input), prog, evals, disc

        # 3b) DISCOVERY from scratch: blind search over the full primitive space.
        #     Candidates (singles + all length-2 + sampled length-3) are SHUFFLED,
        #     so with a limited budget an agent that lacks the building blocks only
        #     covers a fraction of the space => low success on composite tasks.
        #     This is the asymmetry that makes accumulated culture decisive.
        for combo in self._discovery_candidates():
            if evals >= budget:
                break
            evals += 1
            if consistent(combo):
                disc = (self._abstract_skill(combo, train, generation)
                        if learn_at_solve else None)
                return run_program(combo, query_input), combo, evals, disc

        return self._give_up(query_input, evals)

    def _discovery_candidates(self):
        """Blind-search pool, ordered by complexity so simple rules are reliably
        found but composite rules are not:
          - all single primitives first (cheap => primitives are learnable),
          - then SHUFFLED length-2 compositions (only a budget-limited fraction
            gets tried => composites are unreliable from scratch),
          - then a SHUFFLED sample of length-3 compositions (rarely reached).
        Culture short-circuits all of this via the known-composition stage."""
        all_prims = list(PRIMITIVES.keys())
        singles = [(op,) for op in all_prims]
        pairs = [(a, b) for a in all_prims for b in all_prims]
        self.rng.shuffle(pairs)
        triples = [(a, b, c) for a in all_prims for b in all_prims
                   for c in all_prims]
        self.rng.shuffle(triples)
        return singles + pairs + triples[:150]

    def _give_up(self, query_input, evals):
        # best guess: copy (a reasonable prior) or a random known skill output
        return query_input, None, evals, None

    def _priority_known_programs(self):
        """Order known programs by reputation/success so good culture is tried
        first (and short programs before long ones)."""
        progs = list(self.known_skills.values())
        progs.sort(key=lambda s: (-s.reputation, s.complexity()))
        return [s.program for s in progs]

    def _qlearn_substitution(self, train):
        """Use tabular Q-learning to discover a single-character substitution rule
        consistent with the examples. Returns a primitive name if one of the known
        substitution primitives matches the learned char map, else None.

        This is a genuine experience->reward->policy loop over characters."""
        # build char->char target map from examples; must be a function
        cmap = {}
        for inp, out in train:
            if len(inp) != len(out):
                return None
            for ci, co in zip(inp, out):
                if ci in cmap and cmap[ci] != co:
                    return None
                cmap[ci] = co
        if not cmap:
            return None
        ql = QLearner(n_actions=len(ALPHABET), rng=self.rng, epsilon=0.4)
        chars = list(cmap.keys())
        for _ in range(40):  # learning trials
            for ch in chars:
                if ch not in ALPHABET:
                    continue
                state = ALPHABET.index(ch)
                action = ql.act(state)
                guess = ALPHABET[action]
                reward = 1.0 if guess == cmap[ch] else -0.1
                ql.observe(state, action, reward, state, True)
            ql.decay_epsilon()
        learned = {ch: ALPHABET[int(np.argmax(ql.q[ALPHABET.index(ch)]))]
                   for ch in chars if ch in ALPHABET}
        # match learned map against known substitution primitives
        for op in SUBSTITUTION_PRIMITIVES:
            if all(PRIMITIVES[op](ch) == learned.get(ch, None) for ch in chars
                   if ch in ALPHABET):
                self.long_term.store_pattern(f"submap_{op}", {"map": learned})
                return op
        return None

    def _abstract_skill(self, program, examples, generation, derived=False):
        """Turn a discovered solution into a Skill object and adopt it."""
        name = program_name(program)
        preconds = list(dict.fromkeys(program)) if derived else []
        skill = Skill(
            name=name, program=tuple(program), creator=self.id,
            generation=generation, preconditions=preconds,
            examples=list(examples[:3]),
        )
        self.learn_skill(skill)
        return skill

    # ------------------------------------------------------------ bookkeeping
    def record_attempt(self, reward: float, solved: bool):
        self.tasks_attempted += 1
        self.lifetime_reward += reward
        if solved:
            self.tasks_solved += 1
        self.short_term.push(None, None, reward)

    @property
    def fitness(self) -> float:
        base = self.lifetime_reward
        # reward agents that contribute to and use culture (civilisation pressure)
        social = 0.2 * len(self.contributions) + 0.1 * len(self.taught)
        return base + social

    def befriend(self, other_id: str, amount: float = 0.1):
        self.relationships[other_id] = self.relationships.get(other_id, 0.0) + amount

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "generation": self.generation,
            "parents": ",".join(self.parents),
            "specialization": self.specialization or "",
            "lifetime_reward": round(self.lifetime_reward, 3),
            "tasks_attempted": self.tasks_attempted,
            "tasks_solved": self.tasks_solved,
            "reputation": round(self.reputation, 3),
            "n_known_skills": len(self.known_skills),
            "n_contributions": len(self.contributions),
            "n_taught": len(self.taught),
            "exploration": round(self.preferences["exploration"], 3),
            "imitation": round(self.preferences["imitation"], 3),
        }
