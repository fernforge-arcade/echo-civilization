#!/usr/bin/env python3
"""Stack World CLI — prompt the civilization to build a real app.

Experiment K (echo_civilization/stack.py) showed that a population of resilient,
cultured builder agents can assemble multi-file Node backends from a one-line
prompt. This CLI is the front door to that builder: you type what you want, a
fully-cultured resilient agent assembles a real, bootable project, and you can
boot it and poke the live API.

No pretrained model is involved. The prompt is parsed deterministically into a
set of REST resources; the agent then synthesises every endpoint by debugging
near-misses (repair) and reusing endpoint types it has already mastered
(culture), exactly as in the experiment. Every endpoint is graded by real Node
execution before the project is written to disk.

Examples
  # build the four built-in demo apps by name
  ./venv/bin/python stack_cli.py build task_api
  ./venv/bin/python stack_cli.py build "a blog with posts and comments"

  # describe resources and fields yourself
  ./venv/bin/python stack_cli.py build "a store with products(name, price), orders(item), customers(name)"

  # build AND boot it, round-tripping a record over real HTTP
  ./venv/bin/python stack_cli.py build "a recipe box with recipes(title, ingredients)" --boot

  # build it and leave a server running so you can open it in a browser
  ./venv/bin/python stack_cli.py serve "a guestbook with entries(name, message)"

  # see what's understood from a prompt without building anything
  ./venv/bin/python stack_cli.py parse "a forum with threads(title), replies(body), users"
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys

from echo_civilization import stack as S


# ----------------------------------------------------------------------
# Prompt -> spec.  A deterministic parser (no model). It pulls REST resources
# out of the prompt two ways: explicit `name(field, field)` groups, and bare
# nouns matched against a small vocabulary of common resources with sensible
# default fields. Whatever it can't recognise still becomes a resource with a
# single `name` field, so any prompt yields something buildable.
# ----------------------------------------------------------------------

# Common resource nouns -> their default required fields. Keys are singular;
# the resource is exposed under its plural (so paths read `/posts`, `/users`).
KNOWN_RESOURCES = {
    "task": ["title"], "todo": ["title"], "note": ["text"], "item": ["name"],
    "post": ["title", "body"], "article": ["title", "body"], "comment": ["text"],
    "reply": ["body"], "thread": ["title"], "message": ["text"], "entry": ["name", "message"],
    "user": ["name"], "customer": ["name"], "member": ["name"], "author": ["name"],
    "account": ["name"], "profile": ["name"],
    "product": ["name", "price"], "order": ["item"], "cart": ["item"],
    "review": ["text"], "rating": ["score"], "like": ["target"], "follow": ["target"],
    "event": ["name"], "booking": ["name"], "ticket": ["title"], "issue": ["title"],
    "project": ["name"], "team": ["name"], "recipe": ["title", "ingredients"],
    "book": ["title", "author"], "movie": ["title"], "song": ["title", "artist"],
    "photo": ["caption"], "file": ["name"], "document": ["title"],
    "contact": ["name"], "customer_note": ["text"], "tag": ["name"],
    "category": ["name"], "page": ["title"], "guest": ["name"], "rsvp": ["name"],
}

# Tokens that are never resources, even though they're nouns.
_STOPWORDS = {
    "a", "an", "the", "with", "and", "or", "for", "app", "api", "backend",
    "build", "make", "create", "service", "site", "system", "that", "of",
    "to", "manage", "track", "tracker", "store", "stores", "have", "has",
    "where", "which", "can", "some", "my", "our", "their", "this", "it",
    "simple", "small", "basic", "little", "web", "website", "server",
}

# Plural forms of irregular nouns we care about.
_IRREGULAR_PLURALS = {
    "category": "categories", "reply": "replies", "story": "stories",
    "company": "companies", "entry": "entries", "library": "libraries",
}


def _singular(word: str) -> str:
    """Best-effort singular form of a lowercase noun (only what the parser needs)."""
    for sing, plur in _IRREGULAR_PLURALS.items():
        if word == plur:
            return sing
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") or word.endswith("xes") or word.endswith("ches") \
            or word.endswith("shes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _plural(word: str) -> str:
    """Plural form used as the resource (and URL) name."""
    if word in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[word]
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    return word + "s"


def _clean_fields(raw: str) -> list:
    """Parse `title, body` -> ['title','body']; sanitise to JS-safe identifiers."""
    fields = []
    for part in re.split(r"[,/]", raw):
        name = re.sub(r"[^a-z0-9_]", "_", part.strip().lower()).strip("_")
        if name and name not in fields:
            fields.append(name)
    return fields


def parse_prompt(prompt: str):
    """Turn a freeform prompt into a (name, list[Resource]) pair.

    Returns (spec_name, resources). Resources preserve first-seen order. The
    parser is deliberately forgiving: explicit `noun(fields)` wins, known nouns
    contribute their default fields, and any other plausible noun becomes a
    `name`-only resource.
    """
    text = prompt.strip()
    resources = []          # list of (plural_name, fields)
    seen = set()

    def add(name_singular, fields):
        plural = _plural(name_singular)
        if plural in seen:
            # merge any new fields into the existing resource
            for r in resources:
                if r[0] == plural:
                    for f in fields:
                        if f not in r[1]:
                            r[1].append(f)
            return
        seen.add(plural)
        resources.append((plural, list(fields)))

    # 1) explicit `noun(field, field)` groups — highest priority, exact control.
    consumed_spans = []
    for m in re.finditer(r"([A-Za-z][A-Za-z0-9_]*)\s*\(([^)]*)\)", text):
        noun = _singular(m.group(1).lower())
        fields = _clean_fields(m.group(2))
        if not fields:
            fields = KNOWN_RESOURCES.get(noun, ["name"])
        add(noun, fields)
        consumed_spans.append((m.start(), m.end()))

    # blank out consumed spans so their inner words aren't re-parsed as resources
    masked = list(text)
    for s, e in consumed_spans:
        for i in range(s, e):
            masked[i] = " "
    remaining = "".join(masked)

    # 2) bare nouns matched against the known vocabulary, then any other noun.
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", remaining):
        low = token.lower()
        if low in _STOPWORDS:
            continue
        sing = _singular(low)
        if sing in KNOWN_RESOURCES:
            add(sing, KNOWN_RESOURCES[sing])
        elif low != sing and low.endswith("s"):
            # an unrecognised but clearly-plural noun -> a generic resource
            add(sing, ["name"])

    # 3) nothing recognised at all -> a single generic `items` resource.
    if not resources:
        resources.append(("items", ["name"]))

    name = _spec_name(prompt, resources)
    res_objs = [S._resource(n, f) for n, f in resources]
    return name, res_objs


def _spec_name(prompt: str, resources: list) -> str:
    """A filesystem-safe project name derived from the resources (the most stable,
    recognisable handle), e.g. posts+comments -> `posts_comments_api`."""
    base = "_".join(r[0] for r in resources[:3])
    return (base or "app") + "_api"


# ----------------------------------------------------------------------
# Building.  Master the five endpoint types once on a tiny scratch spec, then
# build the requested spec with that culture in hand (resilient agent, repair on).
# Mirrors run_stack.emit_apps so the CLI builds exactly what the experiment does.
# ----------------------------------------------------------------------

def _cultured_resilient_build(spec, *, budget, repair_budget, seed):
    """Return a StackResult for `spec` from a fully-cultured resilient agent."""
    rng = random.Random(seed)
    culture = {}
    # bootstrap: master all five endpoint types on the smallest possible spec.
    scratch = S.StackSpec("scratch", "scratch", [S._resource("widgets", ["name"])])
    boot = S.build_project(scratch, 300, rng, resilient=True, culture=None,
                           repair_budget=60)
    culture.update(boot.discovered)
    return S.build_project(spec, budget, rng, resilient=True, culture=culture,
                           repair_budget=repair_budget)


def _resolve_spec(prompt: str):
    """A built-in spec name resolves to the canonical demo spec; anything else is
    parsed as a freeform prompt."""
    builtins = {s.name: s for s in S.build_specs()}
    key = prompt.strip().lower()
    if key in builtins:
        return builtins[key], True
    name, resources = parse_prompt(prompt)
    return S.StackSpec(name, prompt.strip(), resources), False


def _describe_spec(spec) -> str:
    lines = [f"project: {spec.name}   ({spec.endpoint_count} endpoints)",
             f"prompt : \"{spec.prompt}\"", "resources:"]
    for r in spec.resources:
        lines.append(f"  - {r.name}  fields: {', '.join(r.fields)}  "
                     f"({len(r.endpoints)} endpoints: "
                     f"{', '.join(e.op for e in r.endpoints)})")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Subcommands.
# ----------------------------------------------------------------------

def cmd_parse(args):
    spec, builtin = _resolve_spec(args.prompt)
    tag = " (built-in demo spec)" if builtin else ""
    print(_describe_spec(spec) + tag)
    return 0


def cmd_specs(args):
    print("Built-in demo specs (build by name):\n")
    for s in S.build_specs():
        print(f"  {s.name:14s} {s.endpoint_count:2d} endpoints   "
              f"{', '.join(r.name for r in s.resources)}")
        print(f"  {'':14s} \"{s.prompt}\"")
    print("\nOr describe your own, e.g.:")
    print('  build "a blog with posts(title, body) and comments(text)"')
    return 0


def cmd_build(args, *, do_serve=False):
    spec, builtin = _resolve_spec(args.prompt)
    print(_describe_spec(spec))
    print()
    print(f"[stack] building {spec.name} with a resilient, cultured agent "
          f"(budget={args.budget}, repair={args.repair_budget}) ...", flush=True)

    r = _cultured_resilient_build(spec, budget=args.budget,
                                  repair_budget=args.repair_budget, seed=args.seed)
    pct = 100.0 * len(r.solved) / max(1, spec.endpoint_count)
    print(f"[stack] solved {len(r.solved)}/{spec.endpoint_count} endpoints "
          f"({pct:.0f}%)   built={r.built}   "
          f"via_culture={r.via_culture} via_repair={r.via_repair} "
          f"node_grades={r.trials}", flush=True)
    if not r.built:
        missing = [(res.name, ep.op) for res in spec.resources for ep in res.endpoints
                   if (res.name, ep.op) not in set(r.solved)]
        print(f"[stack] note: {len(missing)} endpoint(s) left as stubs (the app still "
              f"boots and serves the rest): {missing}", flush=True)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    app_dir = S.emit_project(spec, r.bindings, out_dir)
    print(f"[stack] wrote real Node project -> {app_dir}/", flush=True)
    print(f"        files: db.js validate.js app.js server.js public/index.html "
          f"package.json README.md", flush=True)

    if do_serve:
        return _serve(app_dir, spec, args.port)

    if args.boot:
        print(f"[stack] boot-probing the live server on port {args.port} ...", flush=True)
        probe = S.boot_and_probe(app_dir, spec, port=args.port)
        if probe.get("ok"):
            print(f"[stack] boot OK — round-tripped a record over real HTTP "
                  f"(POST then GET /{probe['resource']}).", flush=True)
            for step in probe.get("steps", []):
                print(f"           {step['req']:18s} -> {step['status']}", flush=True)
        else:
            print(f"[stack] boot probe failed: {probe.get('reason', probe)}", flush=True)

    print()
    print(f"To run it yourself:")
    print(f"  cd {app_dir} && node server.js")
    print(f"  # then open http://localhost:3000")
    return 0


def _serve(app_dir, spec, port):
    """Boot the emitted server in the foreground until Ctrl-C."""
    import subprocess
    res0 = spec.resources[0].name
    print(f"\n[stack] starting {spec.name} server on http://localhost:{port}", flush=True)
    print(f"        open it in a browser, or try:", flush=True)
    print(f"          curl -s localhost:{port}/{res0}", flush=True)
    print(f"          curl -s -XPOST localhost:{port}/{res0} "
          f"-d '{{\"{spec.resources[0].fields[0]}\":\"hello\"}}'", flush=True)
    print(f"        Ctrl-C to stop.\n", flush=True)
    env = dict(os.environ)
    env["PORT"] = str(port)
    try:
        proc = subprocess.Popen(["node", "server.js"], cwd=app_dir, env=env)
        proc.wait()
    except KeyboardInterrupt:
        print("\n[stack] stopping server.", flush=True)
        proc.terminate()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="stack_cli.py",
        description="Prompt the Echo Civilization to build a real, bootable Node app.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    def add_build_args(p):
        p.add_argument("prompt", help="what to build — a built-in spec name or a "
                       "freeform description like 'a blog with posts and comments'")
        p.add_argument("--out", default="output_apps",
                       help="directory to write the project into (default: output_apps)")
        p.add_argument("--budget", type=int, default=300,
                       help="per-project Node-grade budget (default: 300)")
        p.add_argument("--repair-budget", type=int, default=60,
                       help="per-endpoint repair hill-climb budget (default: 60)")
        p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
        p.add_argument("--port", type=int, default=3000, help="port for --boot/serve")

    pb = sub.add_parser("build", help="parse a prompt, build the app, write it to disk")
    add_build_args(pb)
    pb.add_argument("--boot", action="store_true",
                    help="also boot the server and round-trip a record over HTTP")
    pb.set_defaults(func=lambda a: cmd_build(a, do_serve=False))

    ps = sub.add_parser("serve", help="build the app and run the server in the foreground")
    add_build_args(ps)
    ps.set_defaults(func=lambda a: cmd_build(a, do_serve=True))

    pp = sub.add_parser("parse", help="show what a prompt resolves to without building")
    pp.add_argument("prompt")
    pp.set_defaults(func=cmd_parse)

    pl = sub.add_parser("specs", help="list the built-in demo specs")
    pl.set_defaults(func=cmd_specs)

    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
