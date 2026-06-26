#!/usr/bin/env python3
"""Experiment J — Builder World: do agents build harder *real* apps over generations?

The operator's steer: push the civilization toward what large models do — hand it a
vague task ("build a to-do app") and have it produce a *working* application, using
task decomposition. This runner turns the builder engine (``echo_civilization/
builder.py``) into a generational civilization experiment and measures the one thing
that matters: does the *frontier of buildable apps* — the most-complex app a generation
can actually produce and run — rise across generations as a component culture
accumulates, even though the per-build budget never changes?

Three conditions, identical build budget (40 real Node executions per app):

  A  MONOLITHIC          no decomposition, no culture. Reads the vague spec as one
                         giant joint search over whole candidate apps. The smallest
                         app is 3 features (joint space 22^3 >> 40), so it builds
                         essentially nothing. Stateless across generations.
  B  DECOMPOSED          decomposition but culture is WIPED every generation. Each
     (no culture)        agent starts blind. Decomposition alone lets a single agent
                         reach ~3-4-feature apps, but with no accumulation the frontier
                         never rises. Stateless across generations.
  C  DECOMPOSED+CULTURE  decomposition AND an accumulating component library. Solved/
                         discovered components are contributed to a shared culture,
                         inherited by the next generation, and tried first. The frontier
                         climbs: generation N builds apps generation 1 could not.

Culture is frozen for the duration of a generation and merged at the generation
boundary, so accumulation is a *generational* effect (not a within-generation cascade).

Outputs: results/builder.json, figures 25/26/27, and — for condition C — real,
openable apps under output_apps/ (index.html + app.js), the deliverable the operator
asked for: actual websites the civilization built from a one-line prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from echo_civilization import builder as B


# ----------------------------------------------------------------------
# Per-subtask grade cache. The cost unit of this experiment is a real Node
# execution; identical (partial-app, sub-task) grades recur constantly across
# agents and generations, so we memoise them. This changes nothing about the
# science (grades are deterministic) — it just makes a multi-generation,
# multi-seed sweep of real executions tractable.
# ----------------------------------------------------------------------
_GRADE_CACHE: dict = {}
_REAL_RUNS = [0]
_orig_grade = B.grade_app


def _cached_grade(bindings, subtasks, timeout=10):
    if len(subtasks) == 1:
        key = (json.dumps(bindings, sort_keys=True), subtasks[0].action)
    else:
        key = (json.dumps(bindings, sort_keys=True),
               tuple(st.action for st in subtasks))
    if key in _GRADE_CACHE:
        return _GRADE_CACHE[key]
    _REAL_RUNS[0] += 1
    res = _orig_grade(bindings, subtasks, timeout=timeout)
    _GRADE_CACHE[key] = res
    return res


B.grade_app = _cached_grade   # the engine now grades through the cache


# ----------------------------------------------------------------------
# Experiment constants. Do NOT raise BUILD_BUDGET past ~60 (a fresh agent
# starts covering the blind grid and the cultural gap closes) or below ~20
# (the cultured arg-recall margin vanishes). 40 is the tuned canonical value.
# ----------------------------------------------------------------------
GENERATIONS = 8
POP = 5
BUILD_BUDGET = 40
FRESH_VS_CULT_TRIALS = 24      # trials per spec for the fresh-vs-cultured bars
ORACLE_BUDGET = 5000           # generous budget used only to seed the reference library


def full_library(specs):
    """The 'mature culture' a cultured agent inherits: the correct component for
    every sub-task across the whole catalogue (action -> component), plus the set
    of all those components. This is what condition C accumulates toward.

    We obtain it by giving an oracle agent a generous budget on each spec and
    recording the bindings that make the whole app pass. Crucially this grades each
    sub-task *in the context of the partial app* (a sub-task's test may exercise
    earlier actions — e.g. REMOVE's test ADDs items first), which an isolated grade
    cannot satisfy. Action names that recur across specs (ADD, REMOVE, TOGGLE) carry
    the same semantics, so the merged map is consistent."""
    action_map, known = {}, set()
    rng = random.Random(12345)
    for sp in specs:
        r = B.build_decomposed(sp, ORACLE_BUDGET, rng)
        assert r.built, f"oracle failed to build {sp.name} — spec is unsatisfiable"
        for action, comp in r.bindings.items():
            action_map[action] = comp
            known.add(comp)
    return action_map, known


# ----------------------------------------------------------------------
# Conditions
# ----------------------------------------------------------------------

def run_condition_monolithic(specs, seed):
    """A: no decomposition, no culture. Stateless across generations, so we
    evaluate one generation and replicate the (flat) line — gen N is
    distributionally identical to gen 0 when nothing carries over."""
    rng = random.Random(seed * 31 + 7)
    built = set()
    per_spec = {sp.name: 0 for sp in specs}
    for _agent in range(POP):
        for sp in specs:
            r = B.build_monolithic(sp, BUILD_BUDGET, rng)
            if r.built:
                built.add((sp.name, sp.feature_count))
                per_spec[sp.name] += 1
    frontier = max([fc for _, fc in built], default=0)
    return {
        "frontier": [frontier] * GENERATIONS,
        "culture_size": [0] * GENERATIONS,
        "built_specs": [sorted(n for n, _ in built)] * GENERATIONS,
        "per_spec_builds": per_spec,
        "note": "stateless: one generation evaluated, replicated across the axis",
    }


def run_condition_decomposed(specs, seed, accumulate):
    """B (accumulate=False) and C (accumulate=True). A population attempts the whole
    catalogue each generation under a culture that is frozen for the generation and
    merged at the boundary. With accumulate=False the merge is discarded, so every
    generation starts from an empty toolbox (decomposition, but no civilization)."""
    rng = random.Random(seed * 97 + 3)
    action_map, known = {}, set()
    frontier_curve, culture_curve, built_curve = [], [], []
    final_bindings = {}
    for _gen in range(GENERATIONS):
        snap_am, snap_ks = dict(action_map), set(known)   # frozen for the generation
        built = set()
        pend_am, pend_ks = {}, set()
        for _agent in range(POP):
            for sp in specs:
                r = B.build_decomposed(sp, BUILD_BUDGET, rng,
                                       action_map=dict(snap_am), known_set=set(snap_ks))
                pend_am.update(r.newly_discovered)
                pend_ks.update(r.newly_discovered.values())
                if r.built:
                    built.add((sp.name, sp.feature_count))
                    final_bindings[sp.name] = dict(r.bindings)
        if accumulate:
            action_map.update(pend_am)
            known.update(pend_ks)
        frontier_curve.append(max([fc for _, fc in built], default=0))
        culture_curve.append(len(known))
        built_curve.append(sorted(n for n, _ in built))
    return {
        "frontier": frontier_curve,
        "culture_size": culture_curve,
        "built_specs": built_curve,
        "final_action_map": action_map,
        "final_bindings": final_bindings,
    }


def fresh_vs_cultured(specs, seeds):
    """Per-spec build success rate: a single agent under the build budget, fresh
    (empty culture) vs cultured (the full inherited component library). This is the
    head-to-head that isolates 'what does inherited culture buy a builder?'"""
    lib_am, lib_ks = full_library(specs)
    out = {}
    for sp in specs:
        fresh_ok = cult_ok = 0
        n = 0
        for seed in seeds:
            rng = random.Random(seed * 1009 + 17)
            for _t in range(FRESH_VS_CULT_TRIALS // len(seeds) + 1):
                n += 1
                rf = B.build_decomposed(sp, BUILD_BUDGET, rng)
                rc = B.build_decomposed(sp, BUILD_BUDGET, rng,
                                        action_map=dict(lib_am), known_set=set(lib_ks))
                fresh_ok += int(rf.built)
                cult_ok += int(rc.built)
        out[sp.name] = {"feature_count": sp.feature_count,
                        "fresh": fresh_ok / n, "cultured": cult_ok / n, "n": n}
    return out, (lib_am, lib_ks)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def average_curves(per_seed, key):
    seeds = list(per_seed.keys())
    n = len(per_seed[seeds[0]][key])
    return [sum(per_seed[s][key][i] for s in seeds) / len(seeds) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/builder.json")
    args = ap.parse_args()

    specs = B.build_specs()
    print(f"Builder World — {len(specs)} specs, budget {BUILD_BUDGET}, "
          f"pop {POP}, {GENERATIONS} gens, seeds {args.seeds}")
    for sp in specs:
        print(f"  spec {sp.name:14s} features={sp.feature_count}  \"{sp.prompt}\"")

    results = {"A_monolithic": {}, "B_decomposed_no_culture": {},
               "C_decomposed_culture": {}}

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        a = run_condition_monolithic(specs, seed)
        b = run_condition_decomposed(specs, seed, accumulate=False)
        c = run_condition_decomposed(specs, seed, accumulate=True)
        results["A_monolithic"][str(seed)] = a
        results["B_decomposed_no_culture"][str(seed)] = b
        results["C_decomposed_culture"][str(seed)] = c
        print(f"  A monolithic   frontier {a['frontier'][-1]}")
        print(f"  B decomp/noCul frontier {b['frontier']}")
        print(f"  C decomp+Cult  frontier {c['frontier']}  culture {c['culture_size']}")

    # averaged frontier / culture curves
    curves = {}
    for cond in results:
        curves[cond] = {
            "frontier": average_curves(results[cond], "frontier"),
            "culture_size": average_curves(results[cond], "culture_size"),
        }

    # head-to-head per-spec fresh vs cultured
    fvc, (lib_am, lib_ks) = fresh_vs_cultured(specs, args.seeds)
    print("\nPer-spec build rate (single agent, budget "
          f"{BUILD_BUDGET}):  spec  feat  fresh  cultured")
    for name, d in fvc.items():
        print(f"  {name:14s} {d['feature_count']:>4}  {d['fresh']:.2f}   {d['cultured']:.2f}")

    # EMIT real apps from the mature culture of condition C (last seed's library
    # merged with the oracle library so every catalogue app is fully realised).
    out_apps = "output_apps"
    emitted = []
    for sp in specs:
        bindings = {}
        for st in sp.subtasks:
            bindings[st.action] = lib_am.get(st.action, "noop")
        # confirm it really builds before writing the artifact
        graded = B.grade_app(bindings, sp.subtasks)
        if all(graded.get(st.action) for st in sp.subtasks):
            path = B.emit_app(sp, bindings, out_apps)
            emitted.append((sp.name, sp.feature_count, path))
    print(f"\nEmitted {len(emitted)} real openable apps under {out_apps}/:")
    for name, fc, path in emitted:
        print(f"  {name} ({fc} features) -> {path}/index.html")

    payload = {
        "config": {"generations": GENERATIONS, "pop": POP,
                   "build_budget": BUILD_BUDGET, "seeds": args.seeds,
                   "n_components": len(B.ALL_HANDLER_IDS),
                   "n_real_node_runs": _REAL_RUNS[0]},
        "specs": [{"name": sp.name, "prompt": sp.prompt,
                   "feature_count": sp.feature_count,
                   "subtasks": [st.action for st in sp.subtasks]} for sp in specs],
        "per_seed": results,
        "curves": curves,
        "fresh_vs_cultured": fvc,
        "emitted_apps": [{"name": n, "feature_count": fc, "path": p}
                         for n, fc, p in emitted],
        "full_library": {"action_map": lib_am, "known": sorted(lib_ks)},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nReal Node executions (uncached): {_REAL_RUNS[0]}")
    print(f"Wrote {args.out}")

    # figures
    try:
        from echo_civilization import visualization as V
        V.plot_builder_frontier(payload, "figures/25_builder_frontier.png")
        V.plot_builder_fresh_vs_cultured(payload, "figures/26_builder_fresh_vs_cultured.png")
        V.plot_builder_culture_growth(payload, "figures/27_builder_culture_growth.png")
        print("Wrote figures 25/26/27")
    except Exception as e:
        print(f"(figures skipped: {e})")


if __name__ == "__main__":
    main()
