# Host Run Notes — changes made running this on the Windows host

> **For the Claudeboard sandbox instance:** this project was run on the **Windows
> host** (the bind-mounted mirror of this folder), outside the Linux
> `claudeboard-sandbox` container. This file records exactly what changed so the
> sandbox state stays consistent. Date: 2026-06-16.

## TL;DR
- **1 code fix** (real cross-platform bug): `echo_civilization/report.py` now writes
  the report as UTF-8 — it crashed on Windows otherwise.
- **1 new untracked dir**: `.venv/` — a fresh Windows venv I created to run with.
- **Nothing else in the existing repo was modified.** The original `venv/` was left
  untouched. Outputs (`research_report.md`, `figures/`, `results/`) were regenerated
  by a normal run, as expected.

---

## Why the existing `venv/` did not work
That venv is a cross-OS hybrid and is broken on both platforms:
- It was originally built **inside the Linux container** — has `bin/`,
  `lib/python3.11/site-packages/` (Linux numpy/matplotlib `.so` binaries), `lib64 -> lib`.
- Something later re-ran `python -m venv` over it **on Windows** — `pyvenv.cfg` now
  points at Windows Store Python **3.13** and there's a `Scripts/python.exe` stub.

Net effect: Windows Python 3.13 can't load the Linux 3.11 `.so` files (wrong OS +
wrong path layout), and the README's `./venv/bin/python` doesn't exist on Windows.
The overwritten `pyvenv.cfg` also breaks it for the Linux container.

**I did not try to repair `venv/`.** I left it as-is in case the container still
references it, and created a separate clean one instead.

## Code change (the only source edit)
**File:** `echo_civilization/report.py` (last line of `generate_report`)

```diff
- Path(path).write_text("".join(lines))
+ Path(path).write_text("".join(lines), encoding="utf-8")
```

**Why:** the generated report contains `→` (U+2192) characters. `Path.write_text()`
uses the platform default encoding, which on Windows is `cp1252` — it cannot encode
`→`, so the run crashed with `UnicodeEncodeError` at the final report-write step
(the simulation + figures + DB had already completed fine). On Linux the default is
UTF-8, so this bug was invisible there. The fix is cross-platform safe.

This was the **only** `write_text`/`open`/`.write()` call in the package, so no other
files have the same issue.

## New files / directories created
- `.venv/` — fresh Windows venv, **Python 3.13.2**, with `numpy`, `matplotlib`,
  `networkx` (+ their deps) installed from `requirements.txt`. Safe to delete; it's
  not needed inside the Linux container. Consider adding `.venv/` to `.gitignore`
  alongside the existing `venv/` ignore entry if this folder is version-controlled.

## How it was run (Windows host)
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_experiments.py          # full (~15s)
# .\.venv\Scripts\python.exe run_experiments.py --quick # fast smoke run
```
Note: README/PROGRESS say `./venv/bin/python ...` (POSIX, for the container). On the
Windows host the equivalent is `.\.venv\Scripts\python.exe ...`.

## Run result (full run, ~15s — headline result reproduced)
| Condition | gen 0 -> final |
|---|---|
| A single agent | 0.59 -> 0.57 |
| B population, no sharing | 0.44 -> 0.50 |
| C skill sharing | 0.44 -> **0.97** |
| D full civilization | 0.49 -> **0.96** |

Subsystems: echo masters copy @ episode 6; grid NN evolves 1.6 -> 6.1; social protocol
-> 100% accuracy. Outputs written: `research_report.md`, 11 PNGs in `figures/`,
`results/echo_civilization.db` (~2.9 MB).

## If you (the container instance) want to re-run inside Docker instead
The `.venv/` here is Windows-only — ignore it. Inside the Linux `claudeboard-sandbox`
container, rebuild a clean venv (the existing `venv/` is also broken per above):
```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python run_experiments.py
```
The `report.py` UTF-8 fix is harmless and beneficial on Linux too — keep it.
