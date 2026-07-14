"""
Experiment O — Echo agents that PLAY Neural MMO 2.0.

An agent is NOT a language model and NOT a big neural net. Its "brain" is a
GENOME = the SET OF SKILLS it knows (a subset of the cartography DAG in
nmmo_primitives) plus the priority order it runs them in. That is deliberately
minimal: the research question is whether *culture* (teaching + inheritance of
skills) makes later generations more capable than isolated learning could, so the
per-agent controller is kept simple and the interesting state lives in which
skills have accumulated in the population.

Each generation, agents are dropped into the real nmmo survival MMO for a fixed
rollout; their capability is the weighted cartography/foraging score they earn
(nmmo_primitives.capability_score). Skills enter an agent's genome only via
DISCOVERY (rare), TEACHING, or INHERITANCE — never for free.
"""
import numpy as np
import nmmo
from . import nmmo_world as W
from . import nmmo_primitives as P


class NMMOAgent:
    def __init__(self, aid, generation=0, parents=(), known=None, rng=None):
        self.id = aid
        self.generation = generation
        self.parents = tuple(parents)
        # genome: the set of skills this agent has DISCOVERED. Everyone can move.
        self.known = set(known) if known else {"move"}
        self.known.add("move")
        # proficiency: a discovered rung is useless (0) until PRACTICED to mastery
        # (>=1.0). Only mastered rungs drive behavior in the real rollout, so a
        # freshly discovered rung is fitness-neutral until its practice pays off.
        self.prof = {"move": 1.0}
        for s in self.known:
            self.prof.setdefault(s, 0.0)
        # social / civilization state
        self.reputation = 0.0
        self.taught = []          # ids this agent taught
        self.contributions = 0    # skills contributed to culture
        self.fitness = 0.0
        self.rng_seed = int(rng.randint(1 << 30)) if rng is not None else 0
        self._reset_episode()

    def mastered(self):
        """Rungs practiced to mastery — the ones that actually drive behavior."""
        return {s for s in self.known if self.prof.get(s, 0.0) >= 1.0}

    def mastered_frontier(self):
        return max((P.tier(s) for s in self.mastered()), default=0)

    def _reset_episode(self):
        self.spatial = {}
        self._heading = None
        self._heading_ttl = 0
        self.cap = P.new_cap()

    @property
    def priority(self):
        """Behavior runs only MASTERED rungs, in standard tier order (high first).
        Discovered-but-unmastered rungs don't fire — practice hasn't paid off yet."""
        mastered = self.mastered()
        return [s for s in P.DEFAULT_PRIORITY if s in mastered]

    @property
    def frontier(self):
        """Highest tier rung this agent has MASTERED — its real capability ceiling."""
        return self.mastered_frontier()

    def act(self, ob, stats, tiles, ents, rng):
        for skill in self.priority:
            fn = P.LIBRARY[skill][2]
            a = fn(self, ob, stats, tiles, ents, rng)
            if a is not None:
                return a
        return W.noop()


def rollout(agents, seed, steps=400, npc_n=0, depl=3, base=120):
    """Run ONE nmmo episode. `agents` maps nmmo player-id -> NMMOAgent (<=PLAYER_N
    of them). Fills each agent's .cap and returns {aid: capability_score}. Agents
    beyond PLAYER_N or missing an env slot simply don't play this episode."""
    W.MiniNMMO.RESOURCE_DEPLETION_RATE = depl
    W.MiniNMMO.RESOURCE_BASE = base
    env = nmmo.Env(W.make_config(player_n=len(agents), npc_n=npc_n, horizon=steps))
    obs, _ = env.reset(seed=seed)
    # map env ids -> our agents (deterministic order)
    env_ids = sorted(obs.keys())
    our = list(agents.values())
    slot = {eid: our[i] for i, eid in enumerate(env_ids) if i < len(our)}
    for ag in our:
        ag._reset_episode()
    rng = np.random.RandomState(seed ^ 0x5f3759df)
    alive = set(slot)
    for t in range(steps):
        actions = {}
        for eid in list(alive):
            if eid not in obs:
                continue
            ag = slot[eid]
            ob = obs[eid]
            st = W.self_stats(ob)
            tiles = W.scan_tiles(ob)
            ents = W.visible_entities(ob, st["id"])
            ag.cap["cover"].add(st["pos"])
            ag.cap["life"] = st["time_alive"]
            actions[eid] = ag.act(ob, st, tiles, ents, rng)
        obs, rew, term, trunc, info = env.step(actions)
        alive = {eid for eid in obs if not term.get(eid) and not trunc.get(eid) and eid in slot}
        if not alive:
            break
    return {ag.id: P.capability_score(ag.cap) for ag in slot.values()}, slot
