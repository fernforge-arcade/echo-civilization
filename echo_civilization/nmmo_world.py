"""
Experiment O — Neural MMO 2.0 world adapter for Echo Civilization.

Wraps the REAL nmmo 2.1.1 survival MMO with:
  - a small, fast config (Mini) that keeps every game system on,
  - observation parsers that turn raw obs tensors into plain dicts,
  - action encoders that build the nested action dict nmmo expects.

Runs ONLY under /home/node/nmmoenv (numpy 1.23 + nmmo). Do NOT import from
Echo's numpy-2 venv modules here.

Material ids (verified): Water=1 Grass=2 Foilage=4(food) Stone=5 Ore=7 Tree=9
Herb=13 Fish=15 Ocean=14. 0 = void / out-of-vision.

Move Direction Discrete(5): 0=North(-1,0) 1=South(+1,0) 2=East(0,+1) 3=West(0,-1) 4=Stay.

nmmo auto-forages: standing ON a Foilage tile restores food; standing ADJACENT
to a Water tile restores water. So survival is a *navigation* problem, which is
exactly what makes it skill/memory dependent.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import nmmo
from nmmo.core.config import Small, AllGameSystems

# ---- material ids -----------------------------------------------------------
WATER = 1
GRASS = 2
FOILAGE = 4      # walk onto it to eat
STONE = 5
ORE = 7
TREE = 9
HERB = 13
FISH = 15
OCEAN = 14
VOID = 0

FOOD_TILES = {FOILAGE}
WALKABLE_BLOCK = {WATER, STONE, OCEAN, VOID}  # cannot step onto these

# profession resource tiles: material -> skill column name it trains
PROFESSION_TILE = {ORE: "prospecting", TREE: "carving", HERB: "herbalism", FISH: "fishing"}

# ---- move encoding ----------------------------------------------------------
NORTH, SOUTH, EAST, WEST, STAY = 0, 1, 2, 3, 4
_DELTA = {NORTH: (-1, 0), SOUTH: (1, 0), EAST: (0, 1), WEST: (0, -1), STAY: (0, 0)}


class MiniNMMO(AllGameSystems, Small):
    """Small map (64x64, center 32) with every game system enabled. Fast."""
    PROVIDE_ACTION_TARGETS = True
    PLAYER_N = 16
    NPC_N = 24
    HORIZON = 300
    MAP_FORCE_GENERATION = True
    # Slower resource decay + bigger reserve so survival isn't pure spawn luck;
    # gives skilled agents headroom to develop professions. (defaults: 5 / 100)
    RESOURCE_DEPLETION_RATE = 2
    RESOURCE_BASE = 150


def make_config(player_n=16, npc_n=24, horizon=300):
    class _C(MiniNMMO):
        pass
    _C.PLAYER_N = player_n
    _C.NPC_N = npc_n
    _C.HORIZON = horizon
    return _C()


def make_env(player_n=16, npc_n=24, horizon=300):
    return nmmo.Env(make_config(player_n, npc_n, horizon))


# ---- observation parsing ----------------------------------------------------
# Entity attribute columns (verified against EntityState.State.attr_name_to_col)
_E = dict(id=0, npc_type=1, row=2, col=3, damage=4, time_alive=5, freeze=6,
          item_level=7, attacker_id=8, latest_combat_tick=9, message=10, gold=11,
          health=12, food=13, water=14, melee_level=15, range_level=17,
          mage_level=19, fishing_level=21, herbalism_level=23, prospecting_level=25,
          carving_level=27, alchemy_level=29)

PROF_COLS = ["fishing", "herbalism", "prospecting", "carving"]


def self_stats(ob):
    """Row 0 of Entity is always the observing agent."""
    e = ob["Entity"][0]
    return {
        "id": int(e[_E["id"]]),
        "pos": (int(e[_E["row"]]), int(e[_E["col"]])),
        "health": int(e[_E["health"]]),
        "food": int(e[_E["food"]]),
        "water": int(e[_E["water"]]),
        "gold": int(e[_E["gold"]]),
        "time_alive": int(e[_E["time_alive"]]),
        "levels": {p: int(e[_E[p + "_level"]]) for p in PROF_COLS},
        "combat": {c: int(e[_E[c + "_level"]]) for c in ("melee", "range", "mage")},
    }


def scan_tiles(ob):
    """material_id -> list of absolute (row, col) currently visible."""
    tiles = ob["Tile"]
    out = {}
    for row, col, mat in tiles:
        m = int(mat)
        if m == VOID:
            continue
        out.setdefault(m, []).append((int(row), int(col)))
    return out


def visible_entities(ob, exclude_self_id):
    """Other visible entities: list of dict(id, npc_type, pos, health).
    npc_type: 0=player, <0 hostile NPC (typically), >0 passive/friendly."""
    out = []
    for e in ob["Entity"]:
        eid = int(e[_E["id"]])
        if eid == 0 or eid == exclude_self_id:
            continue
        out.append({
            "id": eid,
            "npc_type": int(e[_E["npc_type"]]),
            "pos": (int(e[_E["row"]]), int(e[_E["col"]])),
            "health": int(e[_E["health"]]),
        })
    return out


# ---- action encoding --------------------------------------------------------
def _sign(x):
    return (x > 0) - (x < 0)


def move_toward(target, pos, blocked=None):
    """Return a Move-Direction int stepping from pos toward target.
    Prefers the axis with larger gap; falls back to the other axis if the
    preferred step is blocked. blocked: set of abs (r,c) we must not step onto."""
    if target is None:
        return STAY
    tr, tc = target
    r, c = pos
    dr, dc = tr - r, tc - c
    if dr == 0 and dc == 0:
        return STAY
    blocked = blocked or set()
    # candidate directions ordered by which axis has the bigger gap
    cands = []
    if abs(dr) >= abs(dc):
        if dr != 0:
            cands.append(SOUTH if dr > 0 else NORTH)
        if dc != 0:
            cands.append(EAST if dc > 0 else WEST)
    else:
        if dc != 0:
            cands.append(EAST if dc > 0 else WEST)
        if dr != 0:
            cands.append(SOUTH if dr > 0 else NORTH)
    for d in cands:
        ddr, ddc = _DELTA[d]
        if (r + ddr, c + ddc) not in blocked:
            return d
    return cands[0] if cands else STAY


def bfs_step(pos, walkable, goals):
    """Breadth-first search over the visible window for the shortest walkable
    path from pos to any tile in `goals`; return the first Move direction.

    walkable: set of abs (r,c) the agent may stand on (includes pos).
    goals: set of abs (r,c) destinations (must themselves be walkable).
    Returns a Direction int, or None if no goal is reachable. This robustly
    routes around concave obstacles that trip up greedy stepping.
    """
    if not goals:
        return None
    from collections import deque
    start = pos
    if start in goals:
        return STAY
    q = deque([start])
    came = {start: None}
    found = None
    while q:
        cur = q.popleft()
        if cur in goals:
            found = cur
            break
        r, c = cur
        for d in (NORTH, SOUTH, EAST, WEST):
            dr, dc = _DELTA[d]
            nxt = (r + dr, c + dc)
            if nxt in came or nxt not in walkable:
                continue
            came[nxt] = cur
            q.append(nxt)
    if found is None:
        return None
    # walk back to the first step out of start
    node = found
    while came[node] is not None and came[node] != start:
        node = came[node]
    dr, dc = node[0] - start[0], node[1] - start[1]
    for d, (ddr, ddc) in _DELTA.items():
        if (ddr, ddc) == (dr, dc):
            return d
    return STAY


def walkable_set(tiles, pos):
    """Absolute tiles the agent can stand on within the visible window."""
    blocked = set()
    for mat in WALKABLE_BLOCK:
        for p in tiles.get(mat, ()):
            blocked.add(p)
    walk = set()
    for mat, poss in tiles.items():
        if mat in WALKABLE_BLOCK:
            continue
        for p in poss:
            walk.add(p)
    walk.add(pos)
    return walk, blocked


def move_action(direction):
    return {"Move": {"Direction": int(direction)}}


def attack_action(target_slot, style=0):
    """target_slot is the row index in the Entity obs (1..100)."""
    return {"Attack": {"Style": int(style), "Target": int(target_slot)}}


def noop():
    return {}


def adjacent(a, b):
    return abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
