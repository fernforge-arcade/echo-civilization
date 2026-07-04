"""Rung-agnostic generation / culture loop shared by every game in the ladder.

Three conditions, matched on total training budget, isolate the contribution of culture:

- SOLO : one lifelong learner. No population, no sharing. (Experiment-A analogue.)
- POP  : a population of independent lifelong learners. Each keeps its own knowledge; nothing
         is shared. Best-of-population is reported. (Experiment-B analogue.)
- CIV  : a population plus a shared cultural store. Each generation the strongest agents
         contribute what they know to culture; every agent then absorbs culture (horizontal
         transfer) and the weakest agent is replaced by a fresh child born into culture
         (vertical inheritance). (Experiment-D analogue — the full civilization.)

A `Rung` is any object exposing the small adapter interface used below (new_culture,
new_agent, train, evaluate, extract, transfer, replace_weakest). This keeps the four games
decoupled from the orchestration.
"""

from __future__ import annotations

import numpy as np

SOLO, POP, CIV = "SOLO", "POP", "CIV"


def run_condition(rung, condition, seed, gens, pop, episodes_per_gen,
                  cull_per_gen=1, top_k=3, renew_pop=False):
    """Run one condition on one rung for one seed. Returns a per-generation record list.

    ``renew_pop`` (default off, so the deterministic-game rungs are unchanged) turns on full
    generational turnover for the population conditions: at the end of every generation the
    whole population is replaced by a new generation. This is the sharp test of *cultural
    inheritance* — under POP the children are born naive, so any knowledge a parent found dies
    with it and the population cannot accumulate across generations; under CIV the children are
    born into the shared culture, so knowledge ratchets forward generation on generation. SOLO
    is always a single lifelong learner (the no-population, no-culture control).
    """
    rng = np.random.default_rng(seed)
    n = 1 if condition == SOLO else pop
    culture = rung.new_culture() if condition == CIV else None
    agents = [rung.new_agent(rng, culture) for _ in range(n)]

    records = []
    for gen in range(gens):
        for ag in agents:
            rung.train(ag, rng, episodes_per_gen)

        evals = [rung.evaluate(ag, rng) for ag in agents]
        scores = [e["score"] for e in evals]
        order = sorted(range(len(agents)), key=lambda i: scores[i], reverse=True)
        best_i = order[0]

        rec = {
            "gen": gen,
            "mean_score": float(np.mean(scores)),
            "best_score": float(scores[best_i]),
            "culture_size": rung.culture_size(culture) if culture else 0,
        }
        # carry rung-specific extra metrics from the best agent's eval
        for k, v in evals[best_i].items():
            if k != "score":
                rec[f"best_{k}"] = v
        records.append(rec)

        if condition == CIV:
            top = [agents[i] for i in order[:top_k]]
            for ag in top:
                rung.extract(ag, culture)
            for ag in agents:
                rung.transfer(ag, culture)
            if renew_pop:
                # full generational turnover: a whole new generation, each born into culture
                # (vertical inheritance of the civilization's accumulated knowledge)
                agents = [rung.new_agent(rng, culture, parent=agents[best_i])
                          for _ in range(n)]
            else:
                # partial turnover: replace only the weakest with a culture-born child
                for i in order[-cull_per_gen:]:
                    if i != best_i:
                        agents[i] = rung.new_agent(rng, culture, parent=agents[best_i])
        elif condition == POP and renew_pop:
            # same mortality, but children are born naive — knowledge cannot persist
            agents = [rung.new_agent(rng, None) for _ in range(n)]

    return records
