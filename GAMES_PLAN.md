# Game World — plan (Experiment L)

Operator steer: *"push this further up into the realm of actual usefulness. Software is
super open-ended, so start with games (tic-tac-toe, chess, progressively more complex up to
RPGs — research Minecraft), and see how it performs on increasingly open-ended/noisy
environments."*

## Thesis (what this experiment actually tests)

The whole project's claim is: **culture — shared, inheritable knowledge — makes a population's
capability accumulate across generations faster than isolated learning can.** Games give that
claim a clean, externally-graded difficulty axis. The prediction, and the interesting result
to look for, is:

> The more **open-ended and noisy** the environment, the **larger** culture's advantage.
> In a tiny solved game a lone agent eventually masters it by itself and culture only speeds
> things up. In a deep open-ended world the skill tree is too deep to rediscover from scratch
> within any realistic budget, so **only cultural accumulation across generations reaches the
> bottom of the tree** — generation N does what generation 1 provably could not.

## The ladder (grounded in the RL literature)

Game-tree complexity (log10) from the literature: TTT ≈ 5, Connect Four ≈ 21, Chess ≈ 123;
Crafter/Minecraft is *open-ended* (no fixed tree) with a 22-achievement dependency tree of
depth ≤ 8 (Hafner 2021; Craftax, Matthews 2024). Four rungs of rising complexity + noise:

1. **Tic-Tac-Toe** — deterministic, adversarial, ~10^5 states, *solved*.
   Tabular Q-learning by self-play. Culture = a shared opening/answer book (state→move votes).
   Capability = result vs a perfect minimax opponent (optimum = never lose) and vs random.

2. **Connect Four** (reduced 6×5, win = 4) — deterministic, adversarial, large.
   Too big for a full table → feature-based **linear TD** on afterstate values, self-play.
   Culture = averaged learned weight vector + discovered win-now/block-now tactics.
   Capability = win rate vs a 1-ply greedy heuristic and vs random.

3. **Los Alamos minichess** (6×6, no bishops — the historical first chess-like program, 1956)
   — deterministic, adversarial, huge. This is the "chess" rung, tractable without pretrained
   models. Linear eval (material + piece-square + mobility) learned by **TD-leaf self-play**
   with depth-1 alpha-beta. Culture = averaged eval weights. Capability vs a material-only
   baseline and vs random-legal.

4. **EchoCraft** — a Crafter-style open-ended survival/crafting world (the RPG/Minecraft
   endpoint). Procedurally generated grid (grass/tree/stone/coal/iron/water), health & hunger,
   a **tech tree** of ~14 achievements with dependency depth (wood→table→wood-pickaxe→stone→
   stone-pickaxe→coal/iron→furnace→smelt→iron-pickaxe→…). Noise: new procedural map every
   episode, stochastic spawns, resource decay. Agents act through learned **macro-options**
   (go-to-nearest-X-and-interact); a tabular Q over the abstract inventory/achievement state
   chains them. **Culture = the discovered recipe/option set** — this is the civilization
   thesis in its purest form: a discovered recipe, shared, lets later agents skip rediscovery
   and unlock deeper tech. Capability = Crafter score (geo-mean of achievement rates) + tech
   depth reached.

## Conditions (per rung)

- **SOLO** — one agent, learns alone, no sharing (Experiment-A analogue).
- **POP** — a population, each learns alone, no knowledge sharing (Experiment-B analogue).
- **CIV** — population + cultural contribution + inheritance across generations (Exp-D analogue).

Measure per generation: capability (rung-specific), and for EchoCraft the tech-tree depth and
per-achievement unlock rates. Headline plot: culture's advantage (CIV − POP) vs rung
complexity — expected to grow left→right across the ladder.

## Deliverables

- `echo_civilization/games/` subpackage: `tictactoe.py`, `connect4.py`, `minichess.py`,
  `craft.py`, `harness.py` (generation/culture loop, rung-agnostic).
- `run_games.py` runner (`--seeds`, `--rung`, `--quick`), writes `results/games.json`.
- Figures: per-rung learning curves, the tech tree, and the headline "culture advantage vs
  complexity" plot.
- `GAMES_FINDINGS.md` flagship + a `## 11. Game World` section in `REPORT.md` + README section.

## Design rules kept from the rest of the project

No pretrained models; pure numpy + stdlib. Learning is swappable behind the existing `Learner`
idea (tabular Q / linear TD). Culture is a shared store contributed-to by strong agents and
inherited by new ones, exactly as `culture.py` does for programs. Keep it self-contained under
`games/` so the delicate tuning of Experiments A–K is untouched.
