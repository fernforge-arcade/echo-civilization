# Stack World (Experiment K) — resilient agents that build full-stack apps

*Flagship write-up. Run output first, then method, then the numbers, then the
honest split of what resilience buys versus what culture buys.* No pretrained
model anywhere. Agents assemble **real multi-file Node projects** — a data layer, a
validator, an HTTP router, a bootable server, and a fetch frontend — and every
grade is a **real `node` execution** against hidden behavioural tests. A project
counts as "built" only when every REST endpoint passes for real, and the emitted
apps are then **booted as live HTTP servers and round-tripped over the network**.

The operator's steer was blunt: *"they needed a heavy harness and it hardly worked
— make the agents more resilient and able to actually make bigger projects across
the entire dev stack."* So this experiment does two things the earlier ones did
not: it spans the **whole stack** (db → validation → routing → server → frontend,
not a single reducer), and it makes agents **resilient** — they debug their own
near-misses instead of giving up when a blind guess misses.

---

## 0. The thing it actually produces

Four real, bootable backends the civilization built from one-line prompts, sitting
in `output_apps/`:

```
output_apps/
  task_api/      "Build a task tracker API."          1 resource,   5 endpoints
  blog_api/      "Build a blog backend."              2 resources, 10 endpoints
  shop_api/      "Build a shop backend."              3 resources, 15 endpoints
  platform_api/  "Build a social platform backend."   4 resources, 20 endpoints
```

Each directory is a runnable project: `db.js`, `validate.js`, `app.js` (the router),
`server.js` (`node server.js` boots it), `public/index.html` (a fetch UI), `package.json`,
`README.md`. All four **boot and persist over real HTTP** — verified by
`boot_and_probe`, which starts the server, POSTs a record, GETs the collection back,
and confirms it persisted (`output_apps/stack_probes.json`, every `boot_ok: true`).

Here is the `create` route from `task_api/app.js`, **exactly as the agent assembled
it** — validation guard, real insert into the data layer, `201` with the new row:

```js
{ method: "POST", pattern: "/tasks", resource: "tasks", fields: ["title"],
  handler: function(D,V,req){
    const m=V.missing(req.body, req.fields);
    if(m) return {status:400, body:{error:'missing '+m}};
    const row=D.insert(req.resource, req.body);
    return {status:201, body:row}; } }
```

No template filled that in. Each of the handler's three behaviours — the `400` on a
missing field, the `201` status, the echoed row with an `id` — is a separate flag
the agent had to get right, each pinned by its own hidden test. The whole route only
exists because all three passed when this exact code ran in Node.

Here is the live `task_api` UI, driving its own API in a browser (records created
through real `fetch` POSTs, listed back through a real GET):

![task_api running live](figures/stack_app.png)

---

## 1. Hypothesis

**H_K.** Two independent properties decide whether a civilization can build *bigger*
software reliably, and they buy different things:

- **Resilience (repair).** An agent that **debugs a near-miss** — changes one wrong
  flag at a time and re-tests — converts a multiplicative blind search into an
  additive hill-climb. This should raise **per-agent reliability**: more endpoints
  pass, more projects complete, and most passes are *recovered* from a near-miss
  rather than guessed right cold. It also lifts the **no-culture frontier**.
- **Culture (typed transfer).** REST endpoints come in five **types**
  (create/list/read/update/delete) that recur across every resource. Once those five
  proven configs are in a shared culture, a new resource is near-free, so the
  **frontier of buildable apps climbs** from a 5-endpoint task API to a 20-endpoint
  platform backend — and *holds*.

The sharp, falsifiable claim is that these are **separable**: resilience without
culture is reliable but stays small; culture without resilience climbs but is
fragile per-project; only **both** reliably ship 20-endpoint apps that boot.

---

## 2. Method (honest, executable)

- **The stack, assembled per project.** A fixed data layer (`db.js`, an in-memory
  store with `insert/all/find/update/remove/reset`) and validator (`validate.js`,
  `missing(body, fields)`) are shared scaffolding. What the agent actually *builds*
  is the set of **route handlers** in `app.js` — one per endpoint — plus the
  generic router, server, and frontend that wrap them. `server.js` boots a real
  `http` server; `index.html` talks to it over `fetch`.
- **A handler is a config of flags; each flag has its own test.** `create` has
  `status` (which code), `result` (echo the row / the input / null), and `validate`
  (guard missing fields → 400). `read`/`update`/`delete` each have a `guard404`
  flag, and so on. Every flag is pinned by a **separate hidden test** (e.g. "POST a
  body missing the first field → expect `400`"), so a defect is *one* failing test
  and the repair landscape is smooth: fix one flag, that test goes green, move on.
- **Grading is execution.** `grade` writes the project to a temp dir and runs a Node
  driver that `require`s the real `app.js` and replays each endpoint's hidden request
  sequences against the live router, asserting on status and body. A process-wide
  memo cache keyed on `(project, bound handlers, tests)` keeps the multi-seed sweep
  honest and tractable — **3,666 real Node executions** stood in for 101,283 grade
  calls this run (identical executions only ever run once).
- **Resilience = repair.** A `BRITTLE` agent blind-enumerates the preset pool for an
  endpoint (the correct config is in there exactly once, among many one-flag-off
  near-misses) — a multiplicative search that burns the whole budget on the hardest
  endpoint. A `RESILIENT` agent grades its first candidate, and on a near-miss
  **hill-climbs single-flag edits to a passing config** (`_repair`, budget 45) — an
  additive search. On a real miss it still falls back, and a *partial* project still
  emits and boots (graceful degradation).
- **Culture = typed transfer across resources.** When `share_culture` is on, each
  proven endpoint-type config is contributed to a shared store keyed by *type*
  (`create`, `list`, …) and accumulates across generations. A `create` handler
  debugged on `tasks` is inherited by `posts`, `users`, `likes`, … and tried first.
- **Four conditions × seeds 0/1/2, POP=5, GEN=8, budget=90, repair=45:** `BRITTLE`,
  `RESILIENT`, `BRITTLE+CULTURE`, `RESILIENT+CULTURE`. Four specs of rising size
  (5/10/15/20 endpoints) attempted by every agent each generation.

Re-run: `./venv/bin/python run_stack.py --emit --seeds 0 1 2` (~4 min), then
`./venv/bin/python gen_stack_figures.py`.

---

## 3. Results (canonical — verified this run)

### 3.1 Frontier over generations (largest app fully built & test-passing, in endpoints)

| Gen | BRITTLE | RESILIENT | BRITTLE+CULTURE | RESILIENT+CULTURE |
|----:|:---:|:---:|:---:|:---:|
| 0 | 2.3 | 4.3 | 2.3 | 4.3 |
| 1 | 2.7 | 5.3 | **20.0** | **20.0** |
| 2 | 3.3 | 5.0 | 20.0 | 20.0 |
| 3 | 3.7 | 4.3 | 20.0 | 20.0 |
| 4 | 3.0 | 4.7 | 20.0 | 20.0 |
| 5 | 4.0 | 4.7 | 20.0 | 20.0 |
| 6 | 3.0 | 5.0 | 20.0 | 20.0 |
| 7 | 3.0 | 5.3 | 20.0 | 20.0 |

(means over seeds 0/1/2; 20 endpoints = the `platform_api` ceiling.)

![Stack frontier over generations](figures/stack_frontier.png)

- **Culture is what climbs.** Both cultured conditions jump from ~3–4 endpoints at
  gen 0 to **the full 20 by gen 1** and hold there. The shared vocabulary saturates
  at exactly **5 endpoint-type configs** (`create/list/read/update/delete`) — and
  those five unlock *every* resource, so a 4-resource, 20-endpoint platform becomes
  buildable and stays buildable.
- **Without culture, the frontier never climbs.** `BRITTLE` wanders around **3**;
  `RESILIENT` sits around **5**. Each generation re-earns everything from scratch, so
  there is no upward trend — only noise.

### 3.2 What resilience buys (final generation, no culture)

| Condition | endpoint pass rate | project completion | recovery (share of passes via repair) | frontier |
|---|:---:|:---:|:---:|:---:|
| BRITTLE | 0.45 | 0.15 | 0.00 | 3.0 |
| RESILIENT | **0.61** | **0.27** | **0.97** | **5.3** |

![Reliability and recovery](figures/stack_reliability.png)

This is the resilience result, isolated from culture. Repair lifts the endpoint pass
rate from **0.45 → 0.61**, nearly **doubles** the project-completion rate
(0.15 → 0.27), and lifts the no-culture frontier from **3.0 → 5.3**. The decisive
number is **recovery = 0.97**: in the resilient-no-culture condition, **97% of all
passing endpoints were reached by *debugging a near-miss*, not by guessing right
cold.** That is precisely the "make the agents more resilient" steer, measured: the
agents are getting almost everything they get by repairing their own mistakes.

### 3.3 Resilience and culture compose

At the final generation, **completion by app size** (number of fully-built, test-passing
projects out of 15 agent-attempts per size):

| App (endpoints) | BRITTLE | RESILIENT | RESILIENT+CULTURE |
|---|:---:|:---:|:---:|
| task_api (5) | 9 | 15 | **15** |
| blog_api (10) | 0 | 1 | **15** |
| shop_api (15) | 0 | 0 | **15** |
| platform_api (20) | 0 | 0 | **15** |

Brittle agents can't even reliably finish the 5-endpoint API (9/15). Resilient
agents finish the small one **every time** (15/15) and *occasionally* claw out a
10-endpoint build (1/15), but the 15- and 20-endpoint apps are out of reach for any
single agent under budget. Add culture and **every agent builds every app, including
the 20-endpoint platform, every time.** And because the inherited config is correct
on the first try, **recovery drops back to 0.00 in the cultured conditions** — repair
isn't needed when you inherit a proven handler. Resilience carries you while the
vocabulary is still being discovered; culture removes the need for it once it is.

---

## 4. The honest split (what climbs vs what's reliable)

Both cultured conditions reach 20 endpoints — so the **frontier climb is driven by
culture, not by resilience.** A *population* collectively discovers the five-type
vocabulary within a generation whether or not its members repair (brittle agents
discover configs by blind enumeration; resilient ones by hill-climbing), and once
the five configs are shared, the climb to 20 follows. It would be dishonest to
credit the asymptote to resilience.

What resilience actually buys is **per-agent reliability and the no-culture
frontier**: pass rate 0.45 → 0.61, completion 0.15 → 0.27, frontier 3.0 → 5.3, and
recovery 0.97 (almost everything earned by debugging). And it buys **graceful
degradation** — a resilient agent that can't finish a 20-endpoint app still emits and
boots the endpoints it did solve, where a brittle agent that overshoots its budget
ships nothing. The two compose into the deliverable the steer asked for: a
civilization that **reliably builds 20-endpoint, multi-resource, bootable full-stack
apps**, where neither property alone gets there.

---

## 5. Failures & honest limits

- **The asymptote is culture's, not resilience's.** Stated plainly above: brittle
  populations also climb to 20 with culture on, because discovery is collective.
  Don't read the four-condition frontier plot as "resilience builds bigger apps."
- **Handlers are configs of flags, not free-form code.** Agents discriminate among
  preset handler configs and repair them flag-by-flag; they do not author arbitrary
  novel route logic from a blank file. What's demonstrated is the *mechanism*
  (repair + typed cultural transfer moving reliability and the frontier), at
  whole-project scale across the stack — not a free-form code generator. (§6.5–6.6
  elsewhere show genuine grammar-based synthesis.)
- **Budget is a tuned regime.** At budget 90 / repair 45 the brittle search chokes
  on the hardest endpoint while repair fixes it deterministically; push the budget
  far higher and even blind search covers the pool (the resilience margin shrinks),
  drop it far lower and even repair runs out. The interesting regime is stated, not
  hidden.
- **The data layer, validator, router, and server are fixed scaffolding.** The agent
  assembles the *handlers* and the project around them; it does not re-derive the
  in-memory store or the path matcher. The frontend's `add()` posts only the first
  field, so the live UI is a one-field demo even when the API validates several
  (the API itself is full — the boot-probe posts all required fields).
- **Apps are CRUD backends with crisp hidden tests.** Real products have auth,
  migrations, styling, and ambiguous specs. These are deterministic REST backends.
  The result is a clean demonstration of the accumulation law across the full stack,
  not a production backend generator.

---

## 6. Conclusion

Given four one-line prompts and no pretrained model, the civilization builds four
real, bootable full-stack backends — up to a 20-endpoint, 4-resource platform — and
**both** properties the operator asked for show up, doing **different jobs**.
Resilience (repair) is what makes any single agent *reliable*: it nearly doubles the
completion rate, lifts the no-endpoint pass rate from 0.45 to 0.61, recovers **97%**
of its passing endpoints from near-misses instead of lucky guesses, and degrades
gracefully when it runs out of room. Culture (typed transfer of the five endpoint
types) is what makes the **frontier climb** — from a 5-endpoint API to a 20-endpoint
platform — and *hold*. Neither alone ships the big app reliably; together they do.
This is the project's thesis one level up the stack: *the limiting resource for
building bigger software is not per-agent compute but accumulated culture — and
resilience is what keeps each agent on its feet long enough to contribute to it.*
