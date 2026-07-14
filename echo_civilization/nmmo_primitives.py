"""
Experiment O — tiered behavior primitives (the "skill library") for Neural MMO 2.0.

Each primitive is a pure-ish function:
    primitive(agent, ob, stats, tiles, ents, rng) -> action | None
Returning None means "not applicable right now", letting the agent fall through
to the next primitive in its priority list.

WHY A CARTOGRAPHY DAG (not raw survival)
----------------------------------------
We measured the real nmmo economy exhaustively: because foilage is single-use +
respawns slowly and water refills automatically when adjacent, a LOCAL RANDOM WALK
is a near-optimal survival policy (it loiters where resources respawn). No simple
directed heuristic beats random on lifetime. So raw survival lifetime cannot
discriminate skill and is useless as a capability signal.

What random provably CANNOT do is anything requiring a *deliberate* skill:
charting the map, navigating back to a remembered resource, or visiting profession
sites. So capability here is CARTOGRAPHY + FORAGING COMPETENCE, a depth-4 skill DAG
where each rung unlocks a capability term that lower rungs score exactly zero on:

    S0 move      (everyone) ........ random legal step (coverage only)
    S1 explore   (needs move) ...... straight-line travel -> more map coverage
    S2 catalog   (needs explore) ... log resource sites seen -> `charted` term
    S3 forage    (needs catalog) ... navigate to charted/visible water|food when in
                                     need -> `provision` term
    S4 harvest   (needs forage) .... when provisioned, visit ore/tree/herb/fish
                                     profession sites -> `profvisit` term

Each skill increments a counter on the agent (agent.cap) only when it actually
fires. A skill can only fire if the agent KNOWS it, and higher skills are useless
without their prerequisites (forage does nothing without a catalog to navigate to).
Blind discovery of the depth-4 stack ~ p^4 (unreachable by isolated mutation), but
teaching + inheritance hand whole rungs over -> cultured populations accumulate.
"""
import numpy as np
from . import nmmo_world as W

# need thresholds: water/food deplete a few points per tick
NEED = 45          # forage when a resource dips to/below this
SURPLUS = 60       # only train professions when both resources exceed this
RES_MATS = (W.WATER, W.FOILAGE, W.ORE, W.TREE, W.HERB, W.FISH)


def new_cap():
    """Fresh capability accumulator for one rollout."""
    return {"cover": set(), "charted": set(), "provision": 0, "profvisit": 0, "life": 0}


# --- helpers ----------------------------------------------------------------
def _blocked_set(tiles):
    b = set()
    for mat in W.WALKABLE_BLOCK:
        for p in tiles.get(mat, ()):
            b.add(p)
    return b


def _nearest_charted(agent, mat, pos):
    cand = [p for (m, p) in agent.cap["charted"] if m == mat]
    if not cand:
        return None
    return min(cand, key=lambda p: W.manhattan(p, pos))


def _explore(agent, stats, tiles, rng):
    """Directed straight-line travel: keep a heading for several ticks so the
    agent covers new ground. Measured to chart ~40% more of the map than a local
    random walk. Bounces off obstacles."""
    pos = stats["pos"]
    blocked = _blocked_set(tiles)
    heading = getattr(agent, "_heading", None)
    ttl = getattr(agent, "_heading_ttl", 0)
    if heading is None or ttl <= 0:
        heading = int(rng.randint(4))
        agent._heading_ttl = int(rng.randint(5, 10))
    agent._heading = heading
    agent._heading_ttl -= 1
    ddr, ddc = W._DELTA[heading]
    if (pos[0] + ddr, pos[1] + ddc) in blocked:
        agent._heading_ttl = 0
        for d in rng.permutation(4):
            ddr, ddc = W._DELTA[int(d)]
            if (pos[0] + ddr, pos[1] + ddc) not in blocked:
                agent._heading = int(d)
                return W.move_action(int(d))
        return W.move_action(W.STAY)
    return W.move_action(heading)


# --- primitives (S0..S4) ----------------------------------------------------
def p_move(agent, ob, stats, tiles, ents, rng):
    """S0: random legal step. The floor behavior; never returns None."""
    blocked = _blocked_set(tiles)
    pos = stats["pos"]
    for d in rng.permutation(4):
        ddr, ddc = W._DELTA[int(d)]
        if (pos[0] + ddr, pos[1] + ddc) not in blocked:
            return W.move_action(int(d))
    return W.move_action(W.STAY)


def p_explore(agent, ob, stats, tiles, ents, rng):
    """S1: directed exploration -> map coverage. Always returns an action."""
    return _explore(agent, stats, tiles, rng)


def p_catalog(agent, ob, stats, tiles, ents, rng):
    """S2: side effect — record every visible resource site, then fall through.
    Feeds the `charted` capability term (unique resource sites known)."""
    for mat, poss in tiles.items():
        if mat in RES_MATS:
            for p in poss:
                agent.cap["charted"].add((mat, p))
    return None


def _go_to_resource(agent, stats, tiles, mat, pos, blocked):
    """Return an action moving toward the nearest visible-or-charted `mat`.
    For water, target a tile ADJACENT to it (drink by standing beside it);
    for food/profession tiles, target the tile itself. Returns (action, arrived)."""
    walk, _ = W.walkable_set(tiles, pos)
    vis = tiles.get(mat)
    if vis:
        if mat == W.WATER:
            goals = _adjacent_walkable(vis, walk)
        else:
            goals = {p for p in vis if p in walk}
        if pos in goals:
            return W.move_action(W.STAY), True
        step = W.bfs_step(pos, walk, goals)
        if step is not None:
            return W.move_action(step), False
    tgt = _nearest_charted(agent, mat, pos)
    if tgt is None:
        return None, False
    reach = 1 if mat == W.WATER else 0
    if W.manhattan(tgt, pos) <= reach + 1 and (reach == 1 and W.adjacent(pos, tgt) or reach == 0 and pos == tgt):
        return W.move_action(W.STAY), True
    return W.move_action(W.move_toward(tgt, pos, blocked)), False


def _adjacent_walkable(pos_list, walk):
    goals = set()
    for (r, c) in pos_list:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = (r + dr, c + dc)
                if p in walk:
                    goals.add(p)
    return goals


def p_forage(agent, ob, stats, tiles, ents, rng):
    """S3: when water or food runs low, deliberately navigate to a charted/visible
    source. Increments `provision` on arrival. Needs a catalog to aim at."""
    pos = stats["pos"]
    blocked = _blocked_set(tiles)
    needs = []
    if stats["water"] <= NEED:
        needs.append((stats["water"], W.WATER))
    if stats["food"] <= NEED:
        needs.append((stats["food"], W.FOILAGE))
    if not needs:
        return None
    needs.sort()
    for _, mat in needs:
        act, arrived = _go_to_resource(agent, stats, tiles, mat, pos, blocked)
        if act is not None:
            if arrived:
                agent.cap["provision"] += 1
            return act
    return None


def p_harvest(agent, ob, stats, tiles, ents, rng):
    """S4: when comfortably provisioned, visit a profession resource (ore/tree/
    herb/fish) to train a skill. Increments `profvisit` on arrival. Needs forage
    (survival) handled first so it never competes with staying alive."""
    if stats["water"] <= SURPLUS or stats["food"] <= SURPLUS:
        return None
    pos = stats["pos"]
    blocked = _blocked_set(tiles)
    # cycle professions by least-trained to spread xp
    best, best_key = None, None
    for mat, prof in W.PROFESSION_TILE.items():
        vis = tiles.get(mat)
        tgt = min(vis, key=lambda p: W.manhattan(p, pos)) if vis else _nearest_charted(agent, mat, pos)
        if tgt is None:
            continue
        key = (stats["levels"].get(prof, 1), W.manhattan(tgt, pos))
        if best_key is None or key < best_key:
            best_key, best = key, tgt
    if best is None:
        return None
    if W.adjacent(pos, best):
        agent.cap["profvisit"] += 1
        return W.move_action(W.STAY)
    return W.move_action(W.move_toward(best, pos, blocked))


# skill registry: name -> (tier, prerequisites, function)
LIBRARY = {
    "move":    (0, (),          p_move),
    "explore": (1, ("move",),   p_explore),
    "catalog": (2, ("explore",), p_catalog),
    "forage":  (3, ("catalog",), p_forage),
    "harvest": (4, ("forage",), p_harvest),
}

# execution priority: conditional high-tier skills first, floor behavior last.
DEFAULT_PRIORITY = ["harvest", "forage", "catalog", "explore", "move"]

ALL_SKILLS = list(LIBRARY.keys())
MAX_TIER = max(t for t, _, _ in LIBRARY.values())
# linear chain of the DAG, tier order (used for discovery / "next rung")
CHAIN = ["move", "explore", "catalog", "forage", "harvest"]


def prerequisites(skill):
    return LIBRARY[skill][1]


def tier(skill):
    return LIBRARY[skill][0]


def can_learn(skill, known):
    """Learnable only once all prerequisites are known."""
    return all(p in known for p in prerequisites(skill))


def next_rung(known):
    """The single next skill whose prerequisites are all satisfied but which is
    not yet known. Returns None if the whole chain is known."""
    for skill in CHAIN:
        if skill not in known and can_learn(skill, known):
            return skill
    return None


# capability weights: higher tiers dominate so fitness rewards climbing the DAG.
CAP_WEIGHTS = {"cover": 0.05, "charted": 0.10, "provision": 1.0, "profvisit": 3.0, "life": 0.01}


def capability_score(cap):
    """Scalar capability from a rollout's accumulator."""
    return (CAP_WEIGHTS["cover"] * len(cap["cover"])
            + CAP_WEIGHTS["charted"] * len(cap["charted"])
            + CAP_WEIGHTS["provision"] * cap["provision"]
            + CAP_WEIGHTS["profvisit"] * cap["profvisit"]
            + CAP_WEIGHTS["life"] * cap["life"])
