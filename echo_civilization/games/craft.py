"""Rung 4 — EchoCraft: an open-ended, procedurally-generated survival/crafting world.

This is the headline of the ladder. Where Tic-Tac-Toe is a solved tree that a lone agent
masters by itself, EchoCraft has a *tech tree*: a chain of achievements whose deep rungs can
only be reached after the shallow ones, because each recipe needs materials that an earlier
recipe unlocked. A brand-new map every life is the noise.

The civilization mechanism lives in one place: **recipes**. A recipe (make a wood pickaxe,
build a furnace, smelt iron, ...) has to be *discovered* by tinkering — each attempt only
succeeds with probability ``P_DISCOVER``. Crucially, you cannot even attempt a deep recipe
until you have gathered its materials, and gathering those materials needs the shallower
recipes. So discovery is a *serial chain*.

- SOLO / POP agents each have to climb the whole chain alone: discover recipe 1, use it to
  gather the next tier, discover recipe 2, and so on. Within a realistic life/episode budget
  they stall part-way up.
- A CIV pools discoveries across the population every generation and passes the whole recipe
  set to its children. If agent A stumbles on the furnace recipe and agent B on the stone
  pickaxe, next generation *everyone* has both and can push the frontier one rung deeper.
  Culture parallelises a serial chain — so generation N reaches tech depths generation 1
  provably could not.

The learning is a tabular Q over an abstract inventory state; the actions are macro-options
(go to the nearest X and interact / craft at the nearest station), each costing its BFS travel
distance out of a fixed step budget. No pretrained anything — numpy + stdlib.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

# --- terrain -------------------------------------------------------------------------------
GRASS, TREE, STONE, COAL, IRON, WATER, DIAMOND, BUSH, TABLE, FURNACE = range(10)
WALKABLE = {GRASS, TABLE, FURNACE}

GRID = 9
STEP_BUDGET = 90     # total BFS travel steps per life
ACTION_CAP = 22      # hard cap on macro-options per life
P_DISCOVER = 0.10    # per-attempt chance of inventing an unknown recipe while tinkering

# --- the tech tree -------------------------------------------------------------------------
# Each achievement fires the first time its option succeeds in a life. `depth` is its position
# in the dependency chain (used for the tech-depth metric and the tree figure). `recipe` marks
# the craft options whose knowledge is the cultural unit — these must be discovered.
ACHIEVEMENTS = [
    # name                 depth  recipe
    ("collect_wood",        0,    False),
    ("collect_drink",       0,    False),
    ("eat_plant",           0,    False),
    ("place_table",         1,    False),
    ("make_wood_pickaxe",   2,    True),
    ("collect_stone",       3,    False),
    ("place_furnace",       4,    True),
    ("make_stone_pickaxe",  4,    True),
    ("collect_coal",        5,    False),
    ("collect_iron",        5,    False),
    ("smelt_iron",          6,    True),
    ("make_iron_pickaxe",   7,    True),
    ("collect_diamond",     8,    False),
]
ACH_NAMES = [a[0] for a in ACHIEVEMENTS]
ACH_DEPTH = {a[0]: a[1] for a in ACHIEVEMENTS}
RECIPE_NAMES = [a[0] for a in ACHIEVEMENTS if a[2]]
MAX_DEPTH = max(ACH_DEPTH.values())


def generate_map(rng):
    """A fresh procedural world. Guarantees at least one of every base resource so the tree is
    in principle completable; scatter counts vary run to run (the stochastic part)."""
    grid = np.full((GRID, GRID), GRASS, dtype=np.int8)
    counts = {
        TREE: rng.integers(5, 9),
        STONE: rng.integers(4, 8),
        COAL: rng.integers(2, 5),
        IRON: rng.integers(2, 5),
        WATER: rng.integers(2, 5),
        DIAMOND: rng.integers(1, 3),
        BUSH: rng.integers(2, 5),
    }
    cells = [(r, c) for r in range(GRID) for c in range(GRID)]
    rng.shuffle(cells)
    it = iter(cells)
    for terrain, n in counts.items():
        for _ in range(int(n)):
            r, c = next(it)
            grid[r, c] = terrain
    start = next(it)  # agent starts on a guaranteed-grass cell
    return grid, start


def _neighbors(r, c):
    if r > 0:
        yield r - 1, c
    if r < GRID - 1:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c < GRID - 1:
        yield r, c + 1


def bfs(grid, start, goal_test):
    """Shortest walk over WALKABLE cells from `start` to any cell satisfying goal_test(r,c).
    Returns (distance, cell) or (None, None). The start cell itself is always allowed."""
    if goal_test(*start):
        return 0, start
    seen = {start}
    q = deque([(start, 0)])
    while q:
        (r, c), d = q.popleft()
        for nr, nc in _neighbors(r, c):
            if (nr, nc) in seen:
                continue
            if grid[nr, nc] not in WALKABLE:
                continue
            if goal_test(nr, nc):
                return d + 1, (nr, nc)
            seen.add((nr, nc))
            q.append(((nr, nc), d + 1))
    return None, None


def _adjacent_to(grid, terrain):
    """goal_test: a walkable cell orthogonally adjacent to a `terrain` cell."""
    def test(r, c):
        return any(grid[nr, nc] == terrain for nr, nc in _neighbors(r, c))
    return test


class World:
    """One life: a map, the agent's position, inventory, and the achievements unlocked so far.

    `known_recipes` is *not* part of the world — it belongs to the agent and persists across
    lives — but the world consults it to decide whether a craft succeeds or merely tinkers.
    """

    def __init__(self, rng):
        self.rng = rng
        self.grid, self.pos = generate_map(rng)
        self.inv = {"wood": 0, "stone": 0, "coal": 0, "iron_ore": 0, "iron_ingot": 0}
        self.tools = {"wood_pickaxe": False, "stone_pickaxe": False, "iron_pickaxe": False}
        self.has_table = False
        self.has_furnace = False
        self.steps = 0
        self.actions = 0
        self.unlocked = set()

    # -- abstract state key the Q-table is indexed by -------------------------------------
    def state_key(self):
        i = self.inv
        return (
            i["wood"] > 0, i["stone"] > 0, i["coal"] > 0,
            i["iron_ore"] > 0, i["iron_ingot"] > 0,
            self.has_table, self.has_furnace,
            self.tools["wood_pickaxe"], self.tools["stone_pickaxe"],
            self.tools["iron_pickaxe"],
        )

    def any_pickaxe(self):
        return any(self.tools.values())

    # -- which macro-options are physically available right now ---------------------------
    def available(self):
        g, opts = self.grid, []

        def can_reach(goal_test):
            d, _ = bfs(g, self.pos, goal_test)
            return d is not None and self.steps + d <= STEP_BUDGET

        # gathering
        if can_reach(_adjacent_to(g, TREE)):
            opts.append("collect_wood")
        if can_reach(_adjacent_to(g, WATER)):
            opts.append("collect_drink")
        if can_reach(_adjacent_to(g, BUSH)):
            opts.append("eat_plant")
        if self.tools["wood_pickaxe"] and can_reach(_adjacent_to(g, STONE)):
            opts.append("collect_stone")
        if self.any_pickaxe() and can_reach(_adjacent_to(g, COAL)):
            opts.append("collect_coal")
        if self.tools["stone_pickaxe"] and can_reach(_adjacent_to(g, IRON)):
            opts.append("collect_iron")
        if self.tools["iron_pickaxe"] and can_reach(_adjacent_to(g, DIAMOND)):
            opts.append("collect_diamond")

        # placing / crafting: needs materials (+ a station for the deep ones)
        if self.inv["wood"] >= 1:
            opts.append("place_table")
        if self.has_table and self.inv["wood"] >= 1:
            opts.append("make_wood_pickaxe")
        if self.inv["stone"] >= 1:
            opts.append("place_furnace")
        if self.has_table and self.inv["wood"] >= 1 and self.inv["stone"] >= 1:
            opts.append("make_stone_pickaxe")
        if self.has_furnace and self.inv["coal"] >= 1 and self.inv["iron_ore"] >= 1:
            opts.append("smelt_iron")
        if (self.has_table and self.has_furnace and self.inv["wood"] >= 1
                and self.inv["coal"] >= 1 and self.inv["iron_ingot"] >= 1):
            opts.append("make_iron_pickaxe")
        return opts

    def _walk(self, goal_test):
        d, cell = bfs(self.grid, self.pos, goal_test)
        if d is None or self.steps + d > STEP_BUDGET:
            return False
        self.steps += d
        self.pos = cell
        return True

    def _harvest(self, terrain, gain_key, ach):
        if not self._walk(_adjacent_to(self.grid, terrain)):
            return None
        # consume the adjacent resource cell (turns to grass)
        for nr, nc in _neighbors(*self.pos):
            if self.grid[nr, nc] == terrain:
                self.grid[nr, nc] = GRASS
                break
        if gain_key:
            self.inv[gain_key] += 1
        return ach

    def _place(self, cost_wood_or_stone, station_attr, terrain, ach):
        # place a station on any grass cell adjacent to the agent's current spot
        for nr, nc in _neighbors(*self.pos):
            if self.grid[nr, nc] == GRASS:
                self.grid[nr, nc] = terrain
                setattr(self, station_attr, True)
                self.inv[cost_wood_or_stone] -= 1
                self.steps += 1
                return ach
        return None

    def _craft(self, name, known_recipes, consume, produce_tool=None):
        """Attempt a recipe-gated craft. If the recipe is known it succeeds; otherwise the
        agent tinkers and only discovers (and then applies) it with probability P_DISCOVER.
        Returns (achievement_or_None, discovered_bool)."""
        # walk to the required station
        if name in ("make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe"):
            if not self._walk(_adjacent_to(self.grid, TABLE)):
                return None, False
        if name in ("smelt_iron", "make_iron_pickaxe"):
            if not self._walk(_adjacent_to(self.grid, FURNACE)):
                return None, False
        discovered = False
        if name not in known_recipes:
            if self.rng.random() >= P_DISCOVER:
                self.steps += 1  # a failed tinker still costs a little time
                return None, False
            known_recipes.add(name)
            discovered = True
        for k, n in consume.items():
            self.inv[k] -= n
        if produce_tool:
            self.tools[produce_tool] = True
        if name == "smelt_iron":
            self.inv["iron_ingot"] += 1
        self.steps += 1
        return name, discovered

    def step(self, option, known_recipes):
        """Execute one macro-option. Returns (reward, discovered_recipe_or_None). Reward is the
        number of *new* achievements unlocked (0 or 1 here)."""
        self.actions += 1
        ach, discovered = None, None
        if option == "collect_wood":
            ach = self._harvest(TREE, "wood", "collect_wood")
        elif option == "collect_drink":
            ach = self._harvest(WATER, None, "collect_drink")
        elif option == "eat_plant":
            ach = self._harvest(BUSH, None, "eat_plant")
        elif option == "collect_stone":
            ach = self._harvest(STONE, "stone", "collect_stone")
        elif option == "collect_coal":
            ach = self._harvest(COAL, "coal", "collect_coal")
        elif option == "collect_iron":
            ach = self._harvest(IRON, "iron_ore", "collect_iron")
        elif option == "collect_diamond":
            ach = self._harvest(DIAMOND, None, "collect_diamond")
        elif option == "place_table":
            ach = self._place("wood", "has_table", TABLE, "place_table")
        elif option == "place_furnace":
            ach = self._place("stone", "has_furnace", FURNACE, "place_furnace")
        elif option == "make_wood_pickaxe":
            ach, disc = self._craft("make_wood_pickaxe", known_recipes,
                                    {"wood": 1}, "wood_pickaxe")
            discovered = "make_wood_pickaxe" if disc else None
        elif option == "make_stone_pickaxe":
            ach, disc = self._craft("make_stone_pickaxe", known_recipes,
                                    {"wood": 1, "stone": 1}, "stone_pickaxe")
            discovered = "make_stone_pickaxe" if disc else None
        elif option == "smelt_iron":
            ach, disc = self._craft("smelt_iron", known_recipes,
                                    {"coal": 1, "iron_ore": 1})
            discovered = "smelt_iron" if disc else None
        elif option == "make_iron_pickaxe":
            ach, disc = self._craft("make_iron_pickaxe", known_recipes,
                                    {"wood": 1, "coal": 1, "iron_ingot": 1}, "iron_pickaxe")
            discovered = "make_iron_pickaxe" if disc else None

        reward = 0.0
        if ach is not None and ach not in self.unlocked:
            self.unlocked.add(ach)
            reward = 1.0
        return reward, discovered

    def done(self):
        return self.steps >= STEP_BUDGET or self.actions >= ACTION_CAP or not self.available()


class CraftAgent:
    """Tabular Q-learner over the abstract inventory state, choosing among available macro-
    options. `known_recipes` persists across the agent's whole lifetime (all episodes and
    generations it survives) — that persistent, shareable knowledge is what culture pools."""

    def __init__(self, rng, alpha=0.4, gamma=0.95, epsilon=0.25):
        self.rng = rng
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q = defaultdict(lambda: defaultdict(float))
        self.known_recipes = set()

    def act(self, state, options, greedy=False):
        if not options:
            return None
        if (not greedy) and self.rng.random() < self.epsilon:
            return options[int(self.rng.integers(len(options)))]
        qs = self.q[state]
        best = max(qs.get(o, 0.0) for o in options)
        top = [o for o in options if qs.get(o, 0.0) >= best - 1e-9]
        return top[int(self.rng.integers(len(top)))]

    def update(self, s, a, r, s2, opts2, done):
        best_next = 0.0 if (done or not opts2) else max(self.q[s2].get(o, 0.0) for o in opts2)
        cur = self.q[s][a]
        self.q[s][a] = cur + self.alpha * (r + self.gamma * best_next - cur)


def run_life(agent, rng, learn=True, greedy=False):
    """Play one life. Returns the set of achievements unlocked and the recipes discovered."""
    world = World(rng)
    discovered = set()
    while not world.done():
        opts = world.available()
        s = world.state_key()
        a = agent.act(s, opts, greedy=greedy)
        if a is None:
            break
        r, disc = world.step(a, agent.known_recipes)
        if disc:
            discovered.add(disc)
        if learn:
            agent.update(s, a, r, world.state_key(), world.available(), world.done())
    return world.unlocked, discovered


def evaluate(agent, rng, n_lives=40):
    """Grade capability by running greedy lives on fresh maps. Reports per-achievement unlock
    rates, the Crafter-style score (geometric mean of unlock rates, %), the deepest tech rung
    reached, and the mean achievement count."""
    counts = {name: 0 for name in ACH_NAMES}
    depths, totals = [], []
    for _ in range(n_lives):
        unlocked, _ = run_life(agent, rng, learn=False, greedy=True)
        for name in unlocked:
            counts[name] += 1
        depths.append(max((ACH_DEPTH[n] for n in unlocked), default=0))
        totals.append(len(unlocked))
    rates = {name: counts[name] / n_lives for name in ACH_NAMES}
    # Crafter score: geometric mean of (1 + rate%) - 1, in percent
    logs = [np.log(1.0 + 100.0 * rates[name]) for name in ACH_NAMES]
    crafter = float(np.exp(np.mean(logs)) - 1.0)
    max_depth = max(depths) if depths else 0
    return {
        "score": float(np.mean(totals)) / len(ACH_NAMES),  # harness score: mean fraction unlocked
        "crafter": crafter,
        "max_depth": int(max_depth),
        "mean_depth": float(np.mean(depths)),
        "n_achievements": float(np.mean(totals)),
        "n_recipes_known": len(agent.known_recipes),
        "rates": rates,
    }


class CraftCulture:
    """Shared civilization store. The primary cultural unit is the *recipe set*: the union of
    every recipe any contributing agent has discovered. It also keeps a pooled Q-table so
    newcomers inherit not just what can be built but how to sequence the options to build it."""

    def __init__(self):
        self.recipes = set()
        self.q = defaultdict(lambda: defaultdict(float))

    def contribute(self, agent):
        self.recipes |= agent.known_recipes
        for s, qs in agent.q.items():
            for a, v in qs.items():
                if v > self.q[s][a]:
                    self.q[s][a] = v  # keep the most optimistic value discovered so far

    def imbue(self, agent):
        agent.known_recipes |= set(self.recipes)
        for s, qs in self.q.items():
            for a, v in qs.items():
                if v > agent.q[s][a]:
                    agent.q[s][a] = v

    def size(self):
        return len(self.recipes)


class CraftRung:
    """Adapter binding EchoCraft to the generational harness."""

    name = "echocraft"
    complexity = 100  # open-ended; placed past chess (123 tree cx) on the ladder's x-axis

    def new_culture(self):
        return CraftCulture()

    def new_agent(self, rng, culture=None, parent=None):
        ag = CraftAgent(rng)
        if culture is not None:
            culture.imbue(ag)  # born into the civilization's recipes + know-how
        return ag

    def train(self, agent, rng, episodes):
        for _ in range(episodes):
            run_life(agent, rng, learn=True)
            agent.epsilon = max(0.05, agent.epsilon * 0.997)

    def evaluate(self, agent, rng):
        return evaluate(agent, rng, n_lives=25)

    def extract(self, agent, culture):
        culture.contribute(agent)

    def transfer(self, agent, culture):
        culture.imbue(agent)

    def culture_size(self, culture):
        return culture.size() if culture else 0
