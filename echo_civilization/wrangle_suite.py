"""The wrangling benchmark — realistic per-row transforms a company today pays an
LLM to do, split into TRAINING tasks (the culture builds from these) and HELD-OUT
tasks (all arms are scored on these).

Every task: train example rows (2-3, the "few-shot demo") + held-out rows (never
seen, used only to score). Held-out COMPOSITE tasks are built so their solution is
a concatenation of two TRAINING-task pieces — the exact recombination a cultured
agent can do and a from-scratch agent cannot.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Task:
    name: str
    kind: str                  # "single" | "composite"
    train: list                # [(inp, out), ...] few-shot demo
    test: list                 # [(inp, out), ...] held-out scoring rows
    note: str = ""


# --------------------------------------------------------------------------- #
# TRAINING tasks — each is discoverable from scratch (single op, or a covered
# param+nonparam depth-2). Agents solve these; the winning grounded pieces become
# culture.
# --------------------------------------------------------------------------- #

TRAIN = [
    Task("email_local", "single",
         [("john@acme.com", "john"), ("a.b@x.io", "a.b")],
         [("sue@globex.com", "sue"), ("k.lee@z.net", "k.lee")]),
    Task("email_domain", "single",
         [("john@acme.com", "acme.com"), ("a@x.io", "x.io")],
         [("sue@globex.com", "globex.com"), ("k@z.net", "z.net")]),
    Task("domain_no_tld", "single",
         [("acme.com", "acme"), ("x.io", "x"), ("foo.co.uk", "foo")],
         [("globex.com", "globex"), ("mail.z.net", "mail")]),
    Task("extract_lastname", "single",
         [("John Smith", "Smith"), ("A B C", "C")],
         [("Mary Jane Watson", "Watson"), ("Ann Cole", "Cole")]),
    Task("phone_digits", "single",
         [("(415) 555-1234", "4155551234"), ("+1 202.555.9999", "12025559999")],
         [("(650) 111-2222", "6501112222"), ("212-333-4444", "2123334444")]),
    Task("underscore_title", "single",
         [("john_smith", "John Smith"), ("a_b", "A B")],
         [("mary_jane_watson", "Mary Jane Watson"), ("k_lee", "K Lee")]),
    Task("last_first", "single",
         [("Smith, John", "John Smith"), ("Doe, Jane", "Jane Doe")],
         [("Watson, Mary", "Mary Watson"), ("Cole, Ann", "Ann Cole")]),
    Task("iso_to_us_date", "single",
         [("2024-01-05", "01/05/2024"), ("2023-12-31", "12/31/2023")],
         [("2022-07-04", "07/04/2022"), ("2020-02-29", "02/29/2020")]),
    Task("upper", "single",
         [("acme", "ACME"), ("x", "X")],
         [("globex", "GLOBEX"), ("z", "Z")]),
    Task("titlecase", "single",
         [("john smith", "John Smith"), ("mary jane", "Mary Jane")],
         [("ann cole", "Ann Cole"), ("k lee", "K Lee")]),
    Task("dotted_last", "single",
         [("a.b.c", "c"), ("x.y", "y")],
         [("one.two.three", "three"), ("p.q", "q")]),
]


# --------------------------------------------------------------------------- #
# HELD-OUT tasks. Composites are concatenations of two TRAIN pieces; their
# ground-truth programs use two parametric ops, which from-scratch search cannot
# compose. A few single held-outs are included as controls that BOTH arms solve
# (honesty: culture isn't magic, and naive isn't uniformly broken).
# --------------------------------------------------------------------------- #

HELDOUT = [
    # --- composites requiring recombination of two parametric pieces ---
    Task("email_to_company", "composite",
         [("john@acme.com", "acme"), ("sue@globex.io", "globex")],
         [("k.lee@foo.co.uk", "foo"), ("z@mail.z.net", "mail"),
          ("a@bar.com", "bar")],
         "email_domain + domain_no_tld  (split '@', then split '.')"),
    Task("email_to_fullname", "composite",
         [("john_smith@x.com", "John Smith"), ("mary_jane@z.io", "Mary Jane")],
         [("k_lee@acme.com", "K Lee"), ("ann_cole@z.net", "Ann Cole")],
         "email_local + underscore_title  (split '@', then replace '_' & title)"),
    Task("email_to_lastname", "composite",
         [("john.smith@x.com", "smith"), ("mary.watson@z.io", "watson")],
         [("k.lee@acme.com", "lee"), ("ann.cole@z.net", "cole")],
         "email_local + dotted_last  (split '@', then last dotted field)"),
    # --- single-piece held-outs: controls both arms should solve ---
    Task("lastname_upper", "composite",
         [("John Smith", "SMITH"), ("Ann B Cole", "COLE")],
         [("Mary Jane Watson", "WATSON"), ("K Lee", "LEE")],
         "extract_lastname + upper  (param+nonparam: naive CAN reach this)"),
    Task("plain_titlecase", "single",
         [("john smith", "John Smith")],
         [("ann cole", "Ann Cole"), ("k lee", "K Lee")],
         "control: a single op, both arms solve"),
    Task("plain_phone_digits", "single",
         [("(415) 555-1234", "4155551234")],
         [("(650) 111-2222", "6501112222"), ("212-333-4444", "2123334444")],
         "control: a single op, both arms solve"),
]
