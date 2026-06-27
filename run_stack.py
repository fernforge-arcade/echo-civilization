"""Experiment K — Stack World runner.

Drives the resilient full-stack builder (echo_civilization/stack.py) across
generations under three conditions and writes results/stack.json + emits a few
real, bootable apps to output_apps/.

The question this run answers: does a *civilization* of resilient builder agents
accumulate enough know-how to climb from a 5-endpoint task API to a 20-endpoint
platform backend — and does repair (debugging near-misses) plus cultural transfer
of endpoint TYPES across resources make that climb happen, where neither alone does?

Conditions
  BRITTLE           : no repair, no cultural transfer. Every project from scratch,
                      blind preset search per endpoint. The floor.
  RESILIENT         : repair (single-flag hill-climb on near-misses), still no
                      cultural transfer. Each project rediscovers everything, but
                      reliably, and degrades gracefully.
  RESILIENT+CULTURE : repair AND a shared culture of proven endpoint-type configs
                      that accumulates across generations. A `create` handler
                      debugged on `tasks` is inherited by `posts`, `users`, ... so
                      after the five types are mastered once, arbitrarily many
                      resources become near-free and the frontier climbs.

Every grade is a real Node execution; a process-wide memo cache keeps the run
tractable without faking anything (identical (project, bindings, tests) triples
only ever run once).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from echo_civilization import stack as S


# ----------------------------------------------------------------------
# Memoising cache around the real Node grader. Keyed on the exact project +
# bound handlers + test set, so a cache hit is a genuinely identical execution.
# ----------------------------------------------------------------------
_CACHE = {}
_STATS = {"calls": 0, "node_runs": 0}


def _cached_grade(spec, bindings, test_map, timeout=10):
    bkey = json.dumps({f"{k[0]}.{k[1]}": v for k, v in sorted(bindings.items())},
                      sort_keys=True)
    tkey = json.dumps(test_map, sort_keys=True)
    key = (spec.name, bkey, tkey)
    _STATS["calls"] += 1
    if key in _CACHE:
        return _CACHE[key]
    _STATS["node_runs"] += 1
    out = S.grade(spec, bindings, test_map, timeout=timeout)
    _CACHE[key] = out
    return out


S.GRADE = _cached_grade


# ----------------------------------------------------------------------
# Generational simulation.
# ----------------------------------------------------------------------

def run_condition(name, *, resilient, share_culture, specs, gens, pop, budget,
                  seeds, repair_budget):
    """Run one condition across seeds; return per-generation aggregate metrics."""
    per_gen = [dict(frontier=[], completion=[], endpoint_rate=[], recovery=[],
                    culture_size=[], built_by_size={}) for _ in range(gens)]

    for seed in seeds:
        rng = random.Random(seed)
        culture = {}                      # op -> proven config (accumulates if shared)
        for g in range(gens):
            frozen = dict(culture)        # culture is frozen within a generation
            gen_discovered = {}
            for _ in range(pop):
                # each agent attempts every spec (small -> large)
                agent_max_built = 0
                for spec in specs:
                    r = S.build_project(spec, budget, rng, resilient=resilient,
                                        culture=(frozen if share_culture else None),
                                        repair_budget=repair_budget)
                    solved = len(r.solved)
                    attempted = spec.endpoint_count
                    per_gen[g]["endpoint_rate"].append(solved / attempted)
                    per_gen[g]["completion"].append(1.0 if r.built else 0.0)
                    if r.built:
                        agent_max_built = max(agent_max_built, attempted)
                        d = per_gen[g]["built_by_size"]
                        d[attempted] = d.get(attempted, 0) + 1
                    if solved:
                        per_gen[g]["recovery"].append(r.via_repair / solved)
                    # collect proven configs to (maybe) seed culture next gen
                    for op, cfg in r.discovered.items():
                        gen_discovered[op] = cfg
                per_gen[g]["frontier"].append(agent_max_built)
            if share_culture:
                culture.update(gen_discovered)        # accumulate across generations
            per_gen[g]["culture_size"].append(len(culture))

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    out = []
    for g in range(gens):
        d = per_gen[g]
        out.append(dict(
            gen=g,
            frontier=mean(d["frontier"]),
            completion_rate=mean(d["completion"]),
            endpoint_rate=mean(d["endpoint_rate"]),
            recovery_rate=mean(d["recovery"]),
            culture_size=mean(d["culture_size"]),
            built_by_size=d["built_by_size"],
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--pop", type=int, default=5)
    ap.add_argument("--budget", type=int, default=90)
    ap.add_argument("--repair-budget", type=int, default=45)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/stack.json")
    ap.add_argument("--emit", action="store_true",
                    help="also emit + boot-probe real apps to output_apps/")
    args = ap.parse_args()

    specs = S.build_specs()
    t0 = time.time()
    conditions = {
        "BRITTLE":           dict(resilient=False, share_culture=False),
        "RESILIENT":         dict(resilient=True,  share_culture=False),
        "BRITTLE+CULTURE":   dict(resilient=False, share_culture=True),
        "RESILIENT+CULTURE": dict(resilient=True,  share_culture=True),
    }
    results = {}
    for name, cfg in conditions.items():
        print(f"[stack] running {name} ...", flush=True)
        results[name] = run_condition(
            name, specs=specs, gens=args.gens, pop=args.pop, budget=args.budget,
            seeds=args.seeds, repair_budget=args.repair_budget, **cfg)
        last = results[name][-1]
        print(f"    final gen: frontier={last['frontier']:.1f} "
              f"completion={last['completion_rate']:.2f} "
              f"endpoint_rate={last['endpoint_rate']:.2f} "
              f"culture={last['culture_size']:.1f}", flush=True)

    payload = dict(
        meta=dict(gens=args.gens, pop=args.pop, budget=args.budget,
                  repair_budget=args.repair_budget, seeds=args.seeds,
                  specs=[dict(name=s.name, endpoints=s.endpoint_count,
                              prompt=s.prompt) for s in specs],
                  node_runs=_STATS["node_runs"], grade_calls=_STATS["calls"],
                  wall_seconds=round(time.time() - t0, 1)),
        conditions=results,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[stack] wrote {args.out}  "
          f"(node_runs={_STATS['node_runs']} of {_STATS['calls']} grades, "
          f"{payload['meta']['wall_seconds']}s)")

    if args.emit:
        emit_apps(specs, args)


def emit_apps(specs, args):
    """Build each spec for real with a fully-cultured resilient agent, write it to
    output_apps/, then actually boot the server and round-trip over HTTP."""
    print("[stack] emitting real apps + boot-probing ...", flush=True)
    out_dir = "output_apps"
    # master all five endpoint types once on the smallest spec, then reuse.
    culture = {}
    rng = random.Random(0)
    seed = S.build_project(specs[0], 300, rng, resilient=True, culture=None,
                           repair_budget=60)
    culture.update(seed.discovered)
    probes = {}
    port = 3201
    for spec in specs:
        r = S.build_project(spec, 300, rng, resilient=True, culture=culture,
                            repair_budget=60)
        culture.update(r.discovered)
        app_dir = S.emit_project(spec, r.bindings, out_dir)
        probe = S.boot_and_probe(app_dir, spec, port=port)
        port += 1
        probes[spec.name] = dict(built=r.built, solved=len(r.solved),
                                 endpoints=spec.endpoint_count, boot=probe)
        print(f"    {spec.name:14s} built={r.built} solved={len(r.solved)}/"
              f"{spec.endpoint_count} boot_ok={probe.get('ok')}", flush=True)
    with open(os.path.join(out_dir, "stack_probes.json"), "w") as fh:
        json.dump(probes, fh, indent=2)


if __name__ == "__main__":
    main()
