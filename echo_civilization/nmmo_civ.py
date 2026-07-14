"""
Experiment O — the civilization loop that runs Echo agents inside Neural MMO 2.0.

Same thesis as the rest of Echo, now on a real MMO: does *capability accumulate
across generations because of culture*, or does isolated learning get there too?

Capability is the cartography/foraging skill DAG (nmmo_primitives): a depth-4 chain
move -> explore -> catalog -> forage -> harvest, measured every generation by a live
nmmo rollout. An agent's genome is the SET of rungs it has DISCOVERED plus a
PROFICIENCY in each. The two-stage structure is what makes culture matter:

  DISCOVERY   a rung is discovered by individual trial-and-error. Rare, and rarer
              the higher the rung (p = P0 * DECAY**tier). A rung can only be
              discovered once its prerequisite is *mastered* (you can't invent
              foraging before you can reliably catalog).
  PRACTICE    a freshly discovered rung is fitness-NEUTRAL: it does not drive
              behavior and earns no capability until PRACTICED to mastery
              (proficiency >= 1). Mastery takes several generations of practice.

That practice phase is a valley. Because an in-progress rung gives no fitness,
selection can't protect it, so an isolated lineage tends to lose partial progress
to drift before it ever masters the rung. Culture crosses the valley three ways:

  INHERITANCE a child copies the parent's discovered rungs and proficiency.
  CULTURE     (C/D) a shared pool banks the BEST proficiency any top agent has
              reached in each rung — a monotone ratchet. Newborns start from the
              civilization's accumulated mastery, so the whole population's
              practice adds up instead of each lineage starting over.
  TEACHING    (D) reputation-directed teachers transfer proficiency to living
              students mid-generation, crossing the valley in parallel.

Four conditions isolate what culture buys:
  A  single agent, no memory   — one agent wiped to {move} every generation.
  B  population, no sharing     — inheritance only; each lineage must discover AND
                                  practice every rung by itself.
  C  population + culture        — B plus the proficiency ratchet.
  D  full civilization           — C plus reputation-directed teaching.
"""
from dataclasses import dataclass
import numpy as np

from . import nmmo_primitives as P
from .nmmo_agent import NMMOAgent, rollout


# discovery is steep: each rung is half as likely as the one below it.
DISCOVER_P0 = 0.25
DISCOVER_DECAY = 0.5
# a rung takes ~1/PRACTICE_STEP generations of solo practice to master. Slow, so an
# isolated lineage usually loses partial progress to drift before it masters a rung.
PRACTICE_STEP = 0.1
MASTER = 1.0
# social learning is far faster than solo practice — that's the point of teaching.
TEACH_STEP = 0.5


@dataclass
class CivConfig:
    name: str = "D"
    use_inheritance: bool = True      # children copy parent skills+proficiency (B/C/D)
    use_culture: bool = True          # shared proficiency ratchet seeds newborns (C/D)
    use_teaching: bool = True          # reputation-directed teaching of the living (D)
    pop: int = 12
    generations: int = 40
    rollout_steps: int = 300
    parent_frac: float = 0.5          # top fraction that may reproduce
    max_children: int = 3             # cap per parent -> keeps lineage diversity
    cultural_top_frac: float = 0.5    # top fraction whose mastery enters the pool
    n_teach: int = 3                  # teachers acting per generation (D)


def _discover_prob(skill):
    return DISCOVER_P0 * (DISCOVER_DECAY ** P.tier(skill))


def _discoverable(agent):
    """Lowest chain rung not yet discovered whose prerequisites are all MASTERED."""
    mastered = agent.mastered()
    for skill in P.CHAIN:
        if skill not in agent.known and all(p in mastered for p in P.prerequisites(skill)):
            return skill
    return None


def _lowest_unmastered(agent):
    """The rung the agent is currently practicing: lowest discovered-not-mastered."""
    for skill in P.CHAIN:
        if skill in agent.known and agent.prof.get(skill, 0.0) < MASTER:
            return skill
    return None


class NMMOCivilization:
    """One condition, one seed. Call run() -> list of per-generation stat dicts."""

    def __init__(self, cfg: CivConfig, seed: int):
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.pool_prof = {"move": 1.0}     # cultural ratchet: skill -> best proficiency
        self._next_id = 0
        self.pop = [self._new_agent(gen=0, known={"move"}) for _ in range(cfg.pop)]

    def _new_agent(self, gen, known, parents=()):
        a = NMMOAgent(self._next_id, generation=gen, parents=parents,
                      known=known, rng=self.rng)
        self._next_id += 1
        return a

    # -- per-generation mechanisms -------------------------------------------
    def _discovery(self):
        events = 0
        for a in self.pop:
            nxt = _discoverable(a)
            if nxt is not None and self.rng.random() < _discover_prob(nxt):
                a.known.add(nxt)
                a.prof.setdefault(nxt, 0.0)
                events += 1
        return events

    def _practice(self):
        """Each agent practices its lowest unmastered rung a little."""
        for a in self.pop:
            s = _lowest_unmastered(a)
            if s is not None:
                a.prof[s] = min(MASTER, a.prof.get(s, 0.0) + PRACTICE_STEP)

    def _teaching(self):
        """D only: highest-reputation agents transfer proficiency in their mastered
        frontier rung to a random student whose prerequisites are mastered. Social
        learning crosses the practice valley far faster than solo practice."""
        if not self.cfg.use_teaching:
            return 0
        teachers = sorted(self.pop, key=lambda a: (a.reputation, a.mastered_frontier()),
                          reverse=True)[: self.cfg.n_teach]
        taught = 0
        for t in teachers:
            mastered = t.mastered()
            skill = max(mastered, key=P.tier)
            if P.tier(skill) == 0:
                continue
            students = [s for s in self.pop if s is not t
                        and s.prof.get(skill, 0.0) < MASTER
                        and all(p in s.mastered() for p in P.prerequisites(skill))]
            if not students:
                continue
            s = students[int(self.rng.randint(len(students)))]
            s.known.add(skill)
            s.prof[skill] = min(MASTER, s.prof.get(skill, 0.0) + TEACH_STEP)
            t.taught.append(s.id)
            t.reputation += 1.0
            taught += 1
        return taught

    def _evaluate(self):
        agents = {i: a for i, a in enumerate(self.pop)}
        ep_seed = (self.seed * 2654435761 + self.pop[0].generation * 40503) & 0x3FFFFFFF
        scores, _ = rollout(agents, seed=ep_seed, steps=self.cfg.rollout_steps)
        for a in self.pop:
            a.fitness = float(scores.get(a.id, 0.0))

    def _update_culture(self):
        """C/D: ratchet the pool to the best proficiency any top agent has reached
        in each rung. Monotone — the civilization never forgets its best mastery."""
        if not self.cfg.use_culture:
            return
        ranked = sorted(self.pop, key=lambda a: a.fitness, reverse=True)
        k = max(1, int(round(self.cfg.cultural_top_frac * len(ranked))))
        for a in ranked[:k]:
            for s, pr in a.prof.items():
                if pr > self.pool_prof.get(s, 0.0):
                    if s not in self.pool_prof or pr >= MASTER > self.pool_prof.get(s, 0.0):
                        a.contributions += 1
                        a.reputation += 2.0
                    self.pool_prof[s] = pr

    def _seed_from_pool(self, known, prof):
        for s, pr in self.pool_prof.items():
            known.add(s)
            prof[s] = max(prof.get(s, 0.0), pr)

    def _reproduce(self):
        ranked = sorted(self.pop, key=lambda a: a.fitness, reverse=True)
        n_par = max(1, int(round(self.cfg.parent_frac * len(ranked))))
        parents = ranked[:n_par]
        fits = np.array([max(p.fitness, 1e-6) for p in parents])
        gen = self.pop[0].generation + 1
        children, quota = [], {id(p): 0 for p in parents}
        while len(children) < self.cfg.pop:
            probs = fits / fits.sum()
            parent = parents[int(self.rng.choice(len(parents), p=probs))]
            if quota[id(parent)] >= self.cfg.max_children:
                continue
            quota[id(parent)] += 1
            if self.cfg.use_inheritance:
                known, prof = set(parent.known), dict(parent.prof)
            else:
                known, prof = {"move"}, {"move": 1.0}
            if self.cfg.use_culture:
                self._seed_from_pool(known, prof)
            child = self._new_agent(gen, known, parents=(parent.id,))
            child.prof = prof
            child.prof.setdefault("move", 1.0)
            child.reputation = 0.25 * parent.reputation
            children.append(child)
        self.pop = children

    # -- driver ---------------------------------------------------------------
    def _stats(self, gen, disc, taught):
        frontiers = [a.mastered_frontier() for a in self.pop]
        known_frontiers = [max(P.tier(s) for s in a.known) for a in self.pop]
        adoption = {s: sum(1 for a in self.pop if s in a.mastered()) for s in P.CHAIN}
        return {
            "condition": self.cfg.name,
            "seed": self.seed,
            "gen": gen,
            "mean_frontier": float(np.mean(frontiers)),
            "max_frontier": int(np.max(frontiers)),
            "mean_known_frontier": float(np.mean(known_frontiers)),
            "mean_capability": float(np.mean([a.fitness for a in self.pop])),
            "max_capability": float(np.max([a.fitness for a in self.pop])),
            "discoveries": disc,
            "taught": taught,
            "culture_frontier": max((P.tier(s) for s, pr in self.pool_prof.items()
                                     if pr >= MASTER), default=0),
            "adoption": adoption,
        }

    def run(self):
        history = []
        for gen in range(self.cfg.generations):
            if self.cfg.name == "A":
                self.pop = [self._new_agent(gen, {"move"})]
            disc = self._discovery()
            self._practice()
            taught = self._teaching()
            self._evaluate()
            self._update_culture()
            history.append(self._stats(gen, disc, taught))
            if gen < self.cfg.generations - 1 and self.cfg.name != "A":
                self._reproduce()
        return history


CONDITIONS = {
    "A": CivConfig(name="A", use_inheritance=False, use_culture=False, use_teaching=False, pop=1),
    "B": CivConfig(name="B", use_inheritance=True,  use_culture=False, use_teaching=False),
    "C": CivConfig(name="C", use_inheritance=True,  use_culture=True,  use_teaching=False),
    "D": CivConfig(name="D", use_inheritance=True,  use_culture=True,  use_teaching=True),
}


def run_condition(name, seeds, generations=40, pop=12, rollout_steps=300):
    """Run one condition across seeds; returns a flat list of per-gen stat dicts."""
    base = CONDITIONS[name]
    rows = []
    for seed in seeds:
        cfg = CivConfig(**{**base.__dict__})
        cfg.generations = generations
        if name != "A":
            cfg.pop = pop
        cfg.rollout_steps = rollout_steps
        civ = NMMOCivilization(cfg, seed)
        rows.extend(civ.run())
    return rows
