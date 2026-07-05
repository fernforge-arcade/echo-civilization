#!/usr/bin/env python3
"""echofill — learn a column transform from a few examples, apply it to a file.

The civilization's wrangling engine as a command-line tool. Show it two or three
input=>output examples; it infers a deterministic rule, prints it so you can check
it, and applies it to a column (from --csv/--col, or stdin, one value per line).

Examples:
  echofill_cli.py --demo
  echofill_cli.py --ex "john@acme.com=>acme" --ex "sue@globex.io=>globex" \\
                  --csv contacts.csv --col email
  cat emails.txt | echofill_cli.py --ex "john@x.com=>x.com"
"""

import argparse
import csv
import sys

from echo_civilization.echofill import learn


def parse_ex(s):
    if "=>" not in s:
        sys.exit(f"bad --ex (need 'in=>out'): {s!r}")
    i, o = s.split("=>", 1)
    return (i, o)


def main():
    ap = argparse.ArgumentParser(description="Learn a column transform by example.")
    ap.add_argument("--ex", action="append", default=[],
                    help="an example 'input=>output' (repeatable)")
    ap.add_argument("--csv", help="input CSV file")
    ap.add_argument("--col", help="column name (with --csv)")
    ap.add_argument("--demo", action="store_true", help="run a built-in demo")
    args = ap.parse_args()

    if args.demo:
        exs = [("john_smith@x.com", "John Smith"), ("mary_jane@z.io", "Mary Jane")]
        # this one needs culture (two parametric ops) — show the honest fallback
        for label, exs in [
            ("slug -> Title Case", [("hello-world", "Hello World"),
                                    ("my-post", "My Post")]),
            ("email -> domain", [("john@acme.com", "acme.com"),
                                 ("a@x.io", "x.io")]),
            ("'Last, First' -> 'First Last'", [("Smith, John", "John Smith"),
                                               ("Doe, Jane", "Jane Doe")]),
        ]:
            rule = learn(exs)
            print(f"\n# {label}")
            print(f"  examples: {exs}")
            print(f"  inferred rule: {rule.describe()}")
            if rule.solved:
                for i, _ in exs:
                    print(f"    {i!r} -> {rule.apply(i)!r}")
        print("\n(For composite rules an LLM would do per row — e.g. email -> "
              "company name — see run_echofill.py, where the cultured agents "
              "recombine inherited pieces.)")
        return

    if not args.ex:
        sys.exit("give at least one --ex 'input=>output' (or --demo)")
    examples = [parse_ex(e) for e in args.ex]
    rule = learn(examples)
    print(f"inferred rule: {rule.describe()}", file=sys.stderr)
    if not rule.solved:
        sys.exit("no deterministic rule reproduces all examples "
                 "(try adding an example, or the transform is out of scope)")

    # gather the column
    if args.csv:
        if not args.col:
            sys.exit("--col is required with --csv")
        with open(args.csv, newline="") as f:
            rows = list(csv.DictReader(f))
        if args.col not in (rows[0].keys() if rows else []):
            sys.exit(f"column {args.col!r} not found")
        values = [r[args.col] for r in rows]
    else:
        values = [ln.rstrip("\n") for ln in sys.stdin if ln.strip() != ""]

    for v in rule.apply_column(values):
        print(v)


if __name__ == "__main__":
    main()
