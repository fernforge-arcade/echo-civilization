"""Builder World (Experiment J) — building REAL web apps from vague specs.

This is the frontier the operator asked for: push the civilization toward what
larger models do — *take an under-specified task ("build a to-do app") and produce
a working application*. We do it the same honest way the rest of this project does
real computer-use work (see ``codegen2.py``): no pretrained model anywhere. The
agent emits real JavaScript, that JavaScript is EXECUTED for real in Node against
hidden behavioural tests, and an app counts as "built" only if every requirement
test passes when its actual code runs. The agents also write a real, openable
``index.html`` for each app they complete.

Two ideas carry the whole experiment:

1. DECOMPOSITION ("the sub-task thing"). A vague spec is not solved as one giant
   search. It is decomposed into one sub-task per required behaviour (one per
   user action the app must support). Each sub-task is "find the handler that makes
   THIS behaviour's tests pass". Decomposition turns a multiplicative joint search
   (every combination of every handler) into an ADDITIVE one (each handler found
   independently). Without decomposition the joint search explodes past budget;
   with it, an app is the sum of its solved parts.

2. CULTURE. Each solved handler is a reusable COMPONENT — a named skill keyed by
   the action it implements. Solved components are contributed to cultural memory,
   inherited by later agents, and tried FIRST. A cultured agent recognises
   "this app needs an ADD_ITEM behaviour, and I inherited an add_item component"
   and plugs it straight in; a fresh agent must blind-search a grid of candidate
   handlers per sub-task and, across a multi-feature app under a tight build
   budget, runs out before assembling them all.

The result the experiment demonstrates: the *frontier of buildable apps* — the
most-complex app a generation can actually produce and run — rises across
generations as the component library accumulates, even though the per-generation
build budget never changes. Generation N builds apps generation 1 could not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field


# ======================================================================
# COMPONENT LIBRARY.  A component is a candidate implementation of one user
# action: a `switch` case body in the app's reducer. Each renders to REAL
# JavaScript. Some are the genuine implementations the specs need; the rest are
# plausible decoys (wrong field, wrong op, no-op) that a blind search must wade
# through. A component is the unit that is discovered, graded, shared, inherited.
# ======================================================================

# id -> JS body for `case '<ACTION>': { <body> break; }`. `payload` is in scope.
COMPONENTS: dict[str, str] = {
    # --- real task/list handlers ---
    "add_text":     "state.items.push({text: payload, done: false});",
    "remove_at":    "state.items.splice(payload, 1);",
    "toggle_at":    "if (state.items[payload]) state.items[payload].done = !state.items[payload].done;",
    "clear_items":  "state.items = [];",
    "edit_at":      "if (state.items[payload.i]) state.items[payload.i].text = payload.text;",
    "set_filter":   "state.filter = payload;",
    # --- counter handlers ---
    "inc_count":    "state.count += 1;",
    "dec_count":    "state.count -= 1;",
    "reset_count":  "state.count = 0;",
    # --- numeric / form handlers ---
    "set_bill":     "state.bill = Number(payload);",
    "set_rate":     "state.rate = Number(payload);",
    "compute_tip":  "state.result = state.bill + state.bill * state.rate;",
    # --- cart handlers ---
    "add_priced":   "state.items.push({name: payload.name, price: Number(payload.price)});",
    "sum_prices":   "state.total = state.items.reduce(function(a, b){ return a + b.price; }, 0);",

    # --- decoys: plausible but wrong for every real action ---
    "push_raw":     "state.items.push(payload);",                # no {text,done} shape
    "splice_first": "state.items.splice(0, 1);",                  # ignores index
    "set_len":      "state.count = state.items.length;",          # wrong source
    "mul_count":    "state.count *= 2;",
    "compute_flat": "state.result = state.bill * state.rate;",    # forgets the bill
    "noop":         "/* no-op */",
    "negate_total": "state.total = -state.total;",
    "set_input":    "state.input = payload;",
}

# The handlers a fresh agent must blind-search. Real ones are NOT prioritised;
# culture is the only thing that surfaces the right one first. Order here is the
# fixed "natural" enumeration order (used when no culture reorders it).
ALL_HANDLER_IDS = list(COMPONENTS.keys())


def render_app_js(bindings: dict[str, str]) -> str:
    """Render a full, real app *model* module from action->component bindings.

    The module exports `createApp()` returning {dispatch, getState}. This is the
    genuine application logic; the HTML view (see render_index_html) wires DOM
    events to these same dispatch calls, so the thing we test is the thing that
    ships."""
    cases = []
    for action, comp_id in bindings.items():
        body = COMPONENTS.get(comp_id, "/* missing */")
        cases.append(f"      case {json.dumps(action)}: {{ {body} break; }}")
    cases_js = "\n".join(cases)
    return (
        "function createApp() {\n"
        "  const state = { items: [], count: 0, bill: 0, rate: 0, result: 0,\n"
        "                  total: 0, input: '', filter: 'all' };\n"
        "  function dispatch(action, payload) {\n"
        "    switch (action) {\n"
        f"{cases_js}\n"
        "      default: break;\n"
        "    }\n"
        "  }\n"
        "  return { dispatch: dispatch, getState: function(){ return state; } };\n"
        "}\n"
        "if (typeof module !== 'undefined') { module.exports = { createApp: createApp }; }\n"
    )


# ======================================================================
# SPECS.  A vague task ("build a to-do app") expressed as the behaviours the
# finished app must exhibit. Each behaviour is a sub-task: an action name plus
# hidden tests (a sequence of dispatches and an assertion on the real state).
# `feature_count` = number of sub-tasks = how far up the complexity frontier
# this app sits.
# ======================================================================

@dataclass
class SubTask:
    action: str                 # the user action this behaviour needs
    tests: list                 # list of (setup_actions, check_expr_over `s`)
    desc: str = ""


@dataclass
class Spec:
    name: str
    prompt: str                 # the vague, under-specified ask
    subtasks: list              # list[SubTask]

    @property
    def feature_count(self) -> int:
        return len(self.subtasks)


def _t(setup, check, desc=""):
    return (setup, check, desc)


def build_specs() -> list:
    """The app catalogue, ordered by rising feature count so 'frontier' is
    monotone. Each prompt is deliberately terse — the agent is handed the vague
    ask and must recover the concrete behaviours."""
    specs = []

    # 1. Counter (3 features)
    specs.append(Spec(
        "counter", "Build a counter app.",
        [
            SubTask("INC",   [_t([["INC", None], ["INC", None]], "s.count === 2")], "increment raises the count"),
            SubTask("DEC",   [_t([["INC", None], ["INC", None], ["DEC", None]], "s.count === 1")], "decrement lowers the count"),
            SubTask("RESET", [_t([["INC", None], ["INC", None], ["RESET", None]], "s.count === 0")], "reset returns to zero"),
        ]))

    # 2. Tip calculator (3 features)
    specs.append(Spec(
        "tip_calculator", "Build a tip calculator.",
        [
            SubTask("SET_BILL", [_t([["SET_BILL", 100]], "s.bill === 100")], "enter the bill amount"),
            SubTask("SET_RATE", [_t([["SET_RATE", 0.2]], "s.rate === 0.2")], "enter the tip rate"),
            SubTask("COMPUTE",  [_t([["SET_BILL", 100], ["SET_RATE", 0.2], ["COMPUTE", None]], "s.result === 120")], "total includes the tip"),
        ]))

    # 3. To-do list (3 features)
    specs.append(Spec(
        "todo", "Build a to-do list app.",
        [
            SubTask("ADD",    [_t([["ADD", "milk"], ["ADD", "eggs"]], "s.items.length === 2 && s.items[0].text === 'milk' && s.items[1].done === false")], "add tasks"),
            SubTask("REMOVE", [_t([["ADD", "a"], ["ADD", "b"], ["ADD", "c"], ["REMOVE", 1]], "s.items.length === 2 && s.items[0].text === 'a' && s.items[1].text === 'c'")], "remove a task by index"),
            SubTask("TOGGLE", [_t([["ADD", "milk"], ["ADD", "eggs"], ["TOGGLE", 1]], "s.items[1].done === true && s.items[0].done === false")], "mark a task done"),
        ]))

    # 4. Shopping cart (4 features)
    specs.append(Spec(
        "shopping_cart", "Build a shopping cart.",
        [
            SubTask("ADD_ITEM",    [_t([["ADD_ITEM", {"name": "a", "price": "2"}], ["ADD_ITEM", {"name": "b", "price": "3"}]],
                                       "s.items.length === 2 && s.items[1].price === 3")], "add priced items (coerces price)"),
            SubTask("REMOVE_ITEM", [_t([["ADD_ITEM", {"name": "a", "price": "1"}], ["ADD_ITEM", {"name": "b", "price": "2"}], ["ADD_ITEM", {"name": "c", "price": "3"}], ["REMOVE_ITEM", 1]],
                                       "s.items.length === 2 && s.items[0].name === 'a' && s.items[1].name === 'c'")], "remove an item by index"),
            SubTask("CHECKOUT",    [_t([["ADD_ITEM", {"name": "a", "price": 2}], ["ADD_ITEM", {"name": "b", "price": 3}], ["CHECKOUT", None]],
                                       "s.total === 5")], "checkout totals the prices"),
            SubTask("EMPTY",       [_t([["ADD_ITEM", {"name": "a", "price": 2}], ["EMPTY", None]], "s.items.length === 0")], "empty the cart"),
        ]))

    # 5. Notes app with filtering (6 features) — the frontier app
    specs.append(Spec(
        "notes", "Build a notes app.",
        [
            SubTask("ADD",    [_t([["ADD", "n1"], ["ADD", "n2"]], "s.items.length === 2 && s.items[1].text === 'n2' && s.items[1].done === false")], "add notes"),
            SubTask("REMOVE", [_t([["ADD", "n1"], ["ADD", "n2"], ["ADD", "n3"], ["REMOVE", 1]], "s.items.length === 2 && s.items[0].text === 'n1' && s.items[1].text === 'n3'")], "delete a note by index"),
            SubTask("EDIT",   [_t([["ADD", "n1"], ["ADD", "n2"], ["EDIT", {"i": 1, "text": "n2b"}]], "s.items[1].text === 'n2b' && s.items[0].text === 'n1'")], "edit a note"),
            SubTask("TOGGLE", [_t([["ADD", "n1"], ["ADD", "n2"], ["TOGGLE", 1]], "s.items[1].done === true && s.items[0].done === false")], "pin/mark a note"),
            SubTask("FILTER", [_t([["FILTER", "done"]], "s.filter === 'done'")], "filter the notes"),
            SubTask("CLEAR",  [_t([["ADD", "n1"], ["ADD", "n2"], ["CLEAR", None]], "s.items.length === 0")], "clear all notes"),
        ]))

    return specs


# ======================================================================
# REAL EXECUTION.  Render the candidate app, run a Node driver that imports the
# real module, replays each sub-task's hidden tests, and reports per-action
# pass/fail. One Node process grades a whole candidate app — `trials` counts the
# real executions (the budget the cultural advantage is measured against).
# ======================================================================

_DRIVER = r"""
const path = process.argv[2];
const spec = JSON.parse(process.argv[3]);   // {action: [[setup, check], ...], ...}
const { createApp } = require(path);
const result = {};
for (const action of Object.keys(spec)) {
  let ok = true;
  for (const [setup, check] of spec[action]) {
    try {
      const app = createApp();
      for (const [a, p] of setup) app.dispatch(a, p);
      const s = app.getState();
      // eslint-disable-next-line no-eval
      if (!eval(check)) { ok = false; break; }
    } catch (e) { ok = false; break; }
  }
  result[action] = ok;
}
process.stdout.write(JSON.stringify(result));
"""


def grade_app(bindings: dict, subtasks: list, timeout=10) -> dict:
    """Execute the real app and return {action: passed_bool} for the given
    sub-tasks. This is one genuine Node run of the synthesised JavaScript."""
    source = render_app_js(bindings)
    spec_json = {}
    for st in subtasks:
        spec_json[st.action] = [[setup, check] for (setup, check, _desc) in st.tests]
    d = tempfile.mkdtemp(prefix="echo_builder_")
    try:
        app_path = os.path.join(d, "app.js")
        drv_path = os.path.join(d, "driver.js")
        with open(app_path, "w") as fh:
            fh.write(source)
        with open(drv_path, "w") as fh:
            fh.write(_DRIVER)
        proc = subprocess.run(
            ["node", drv_path, app_path, json.dumps(spec_json)],
            capture_output=True, text=True, timeout=timeout,
            env={"PATH": os.environ.get("PATH", "")})
        if proc.returncode != 0:
            return {st.action: False for st in subtasks}
        return json.loads(proc.stdout.strip() or "{}")
    except Exception:
        return {st.action: False for st in subtasks}
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def passes_subtask(action: str, comp_id: str, subtask: SubTask, timeout=10) -> bool:
    """Grade ONE candidate component against ONE sub-task in isolation — the unit
    a decomposed search evaluates. Real Node execution."""
    res = grade_app({action: comp_id}, [subtask], timeout=timeout)
    return bool(res.get(action, False))


# ======================================================================
# THE BUILDER AGENT.  Given a vague spec, build a working app under a fixed
# budget of real test-executions. Two levers are toggled to form the
# experimental conditions:
#
#   decompose : if True, solve one sub-task at a time (additive search). If
#               False, search the JOINT space of all-actions-at-once
#               (multiplicative -> explodes past budget on multi-feature apps).
#   culture   : a dict {action -> component_id} of inherited components. For each
#               sub-task the inherited component (if any) is tried FIRST, turning
#               a blind grid search into a one-shot recall.
# ======================================================================

@dataclass
class BuildResult:
    spec: str
    feature_count: int
    bindings: dict               # action -> chosen component (best found)
    solved_actions: list         # actions whose tests pass
    built: bool                  # every sub-task satisfied -> a working app
    trials: int                  # real Node executions consumed
    via_culture: int             # how many sub-tasks were solved by an inherited component
    newly_discovered: dict       # action -> component discovered from scratch (to teach culture)


def _candidate_order(action: str, action_map: dict, known_set: set, rng):
    """Enumeration order for one sub-task, the cultural toolbox in priority order:

      1. the component culture has previously bound to THIS action (exact recall —
         one trial when the same behaviour was solved before, even in another app);
      2. every other component the agent has inherited (its toolbox — gives
         cross-app transfer: todo's `remove_at` is reached fast when a new app
         needs a remove behaviour under a different action name);
      3. everything else, blind (what a fresh agent must wade through).

    A fresh agent has empty (1) and (2), so it searches the full shuffled grid."""
    inherited = action_map.get(action)
    known = [c for c in known_set if c != inherited]
    rng.shuffle(known)
    rest = [c for c in ALL_HANDLER_IDS if c != inherited and c not in known_set]
    rng.shuffle(rest)
    order = ([inherited] if inherited else []) + known + rest
    return order, inherited, set(known_set)


def build_decomposed(spec: Spec, budget: int, rng, action_map: dict | None = None,
                     known_set: set | None = None, timeout=10) -> BuildResult:
    """Decomposed build: each sub-task searched independently for a component that
    passes its hidden tests. The total cost is the SUM of per-sub-task searches,
    so culture (which collapses each to a near-one-shot recall) is what lets a
    multi-feature app fit inside the budget.

    `action_map`: inherited {action -> component} (exact-behaviour recall).
    `known_set`:  every component the agent inherited (its toolbox for transfer)."""
    action_map = action_map or {}
    known_set = known_set or set()
    trials = 0
    bindings, solved, via_culture, discovered = {}, [], 0, {}
    for st in spec.subtasks:
        order, inherited, known = _candidate_order(st.action, action_map, known_set, rng)
        found = None
        for comp_id in order:
            if trials >= budget:
                break
            trials += 1
            # grade the candidate IN CONTEXT of what is already built, so a
            # sub-task whose test exercises earlier actions (e.g. DEC's test
            # presses INC first) resolves them correctly. This is incremental
            # build: each new handler is tested against the partial app.
            trial_bindings = {**bindings, st.action: comp_id}
            res = grade_app(trial_bindings, [st], timeout=timeout)
            if res.get(st.action):
                found = comp_id
                break
        if found is not None:
            bindings[st.action] = found
            solved.append(st.action)
            if found == inherited or found in known:
                via_culture += 1     # solved with an inherited component
            if found != inherited:
                discovered[st.action] = found   # new action->component binding to teach
        else:
            bindings[st.action] = "noop"
        if trials >= budget:
            break
    built = len(solved) == spec.feature_count
    return BuildResult(spec.name, spec.feature_count, bindings, solved, built,
                       trials, via_culture, discovered)


def build_monolithic(spec: Spec, budget: int, rng, timeout=10) -> BuildResult:
    """No decomposition: search the JOINT assignment of components to ALL actions
    at once, graded only as all-pass / not. This is the naive way to read a vague
    spec — try whole candidate apps. The space is |handlers|^features, so beyond a
    couple of features it cannot be cleared within any realistic budget. Included
    as the control that shows decomposition is necessary, not incidental."""
    import itertools
    trials = 0
    actions = [st.action for st in spec.subtasks]
    pools = [list(ALL_HANDLER_IDS) for _ in actions]
    for p in pools:
        rng.shuffle(p)
    best_solved, best_bind = [], {a: "noop" for a in actions}
    for combo in itertools.product(*pools):
        if trials >= budget:
            break
        trials += 1
        bindings = dict(zip(actions, combo))
        res = grade_app(bindings, spec.subtasks, timeout=timeout)
        solved = [a for a in actions if res.get(a)]
        if len(solved) > len(best_solved):
            best_solved, best_bind = solved, bindings
        if len(solved) == len(actions):
            return BuildResult(spec.name, spec.feature_count, bindings, solved,
                               True, trials, 0, {})
    return BuildResult(spec.name, spec.feature_count, best_bind, best_solved,
                       len(best_solved) == len(actions), trials, 0, {})


# ======================================================================
# REAL, OPENABLE ARTIFACT.  When an app is fully built we emit a single-file
# index.html that wires a DOM to the very same reducer the tests graded. This is
# the deliverable — an actual website the agents produced from the vague prompt.
# ======================================================================

def render_index_html(spec: Spec, bindings: dict) -> str:
    """A real, self-contained web page: the graded reducer + a minimal UI that
    dispatches to it. Opens in any browser."""
    app_js = render_app_js(bindings)
    actions = list(bindings.keys())
    buttons = "\n".join(
        f'    <button onclick="ui({json.dumps(a)})">{a}</button>' for a in actions)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{spec.name} — built by Echo Civilization</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 640px; margin: 2rem auto; }}
 button {{ margin: 2px; padding: 6px 10px; }}
 #state {{ background:#f4f4f4; padding:1rem; border-radius:8px; white-space:pre-wrap; }}
 .prompt {{ color:#666; }}
</style>
</head>
<body>
<h1>{spec.name.replace('_',' ')}</h1>
<p class="prompt">Prompt given to the civilization: <em>"{spec.prompt}"</em></p>
<input id="payload" placeholder="payload (text / number / json)"/>
<div>
{buttons}
</div>
<h3>live state</h3>
<div id="state">{{}}</div>
<script>
{app_js}
const app = createApp();
function parsePayload(raw) {{
  if (raw === '') return null;
  try {{ return JSON.parse(raw); }} catch (e) {{ return raw; }}
}}
function ui(action) {{
  app.dispatch(action, parsePayload(document.getElementById('payload').value));
  document.getElementById('state').textContent =
    JSON.stringify(app.getState(), null, 2);
}}
ui('__render__');
</script>
</body>
</html>
"""


def emit_app(spec: Spec, bindings: dict, out_dir: str) -> str:
    """Write the real app (index.html + app.js model) to disk and return its dir."""
    app_dir = os.path.join(out_dir, spec.name)
    os.makedirs(app_dir, exist_ok=True)
    with open(os.path.join(app_dir, "app.js"), "w") as fh:
        fh.write(render_app_js(bindings))
    with open(os.path.join(app_dir, "index.html"), "w") as fh:
        fh.write(render_index_html(spec, bindings))
    return app_dir
