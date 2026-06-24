"""Environment 6 — Real Computer World (genuine sandboxed shell execution).

This is the next rung the operator asked for: agents stop operating a *simulated*
VM and instead drive a **real operating system**. Every primitive op maps to an
actual coreutils command (`cat`, `grep`, `sort`, `uniq`, `wc`, `tr`, `tac`, `cp`,
…) executed by `bash` inside a throwaway temp directory. A solution is still a
program of op-names — so an agent's *inherited macros transfer unchanged from the
simulated world to the real shell* — but it now reads and writes real files on
disk, and is graded on the real command output.

Safety: the op set is a fixed whitelist of read-only/file-local coreutils; every
argument is `shlex.quote`-d and comes from a controlled word list; commands run
with a minimal `PATH`-only environment, a short timeout, and `cwd` pinned to a
per-task temp sandbox. No network-capable command is in the set. This is the same
posture a CI sandbox uses to run untrusted-but-bounded shell.

Because real subprocess calls are ~1000× slower than the simulated ops, this world
is used as a *validation + demonstration* layer (does inherited skill execute for
real? does culture cut the number of real executions needed?) rather than for full
generational evolution.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np

from .computer_world import CURRICULUM, MAX_LEVEL, WORDS

REG = "._reg"          # the working register, as a real file
TMP = "._tmp"
OUT = "output.txt"

# Each op -> a bash fragment operating on the register file REG (and OUT). Args
# (keyword KEY, input file IN) are substituted already-quoted by build_command().
_OP_SHELL = {
    "read_input":    "cat -- {IN} > {REG}",
    "find":          "grep -lF -- {KEY} *.txt 2>/dev/null | head -n1 > {REG} || true",
    "read_found":    "f=$(cat {REG}); cat -- \"$f\" > {TMP} 2>/dev/null; mv {TMP} {REG}",
    "grep":          "grep -F -- {KEY} {REG} > {TMP} || true; mv {TMP} {REG}",
    "grep_v":        "grep -vF -- {KEY} {REG} > {TMP} || true; mv {TMP} {REG}",
    "upper":         "tr '[:lower:]' '[:upper:]' < {REG} > {TMP}; mv {TMP} {REG}",
    "lower":         "tr '[:upper:]' '[:lower:]' < {REG} > {TMP}; mv {TMP} {REG}",
    "sort":          "sort {REG} -o {REG}",
    "reverse_lines": "tac {REG} > {TMP}; mv {TMP} {REG}",
    "uniq":          "uniq {REG} > {TMP}; mv {TMP} {REG}",
    "count_lines":   "grep -c . {REG} > {TMP} || true; mv {TMP} {REG}",
    "head":          "head -n1 {REG} > {TMP}; mv {TMP} {REG}",
    "tail":          "tail -n1 {REG} > {TMP}; mv {TMP} {REG}",
    "concat_all":    "cat -- *.txt > {TMP} 2>/dev/null; mv {TMP} {REG}",
    "write_output":  "cp {REG} {OUT}",
    "append_output": "cat {REG} >> {OUT}",
    # genuine whole-file operations (the operator's literal "move this file")
    "move_file":     "mv -- {IN} {OUT}",
    "copy_file":     "cp -- {IN} {OUT}",
}

ALLOWED_OPS = set(_OP_SHELL)
# the actual binaries we permit (documentation + a guard)
WHITELIST_BINS = {"cat", "grep", "head", "tail", "sort", "uniq", "wc", "tr",
                  "tac", "cp", "mv", "echo"}


@dataclass
class RealTask:
    name: str
    level: int
    sandbox: str
    input_file: str
    keyword: str
    canonical: tuple
    expected_output: str = ""

    def cleanup(self):
        shutil.rmtree(self.sandbox, ignore_errors=True)


def build_command(op: str, keyword: str, input_file: str) -> str | None:
    if op not in _OP_SHELL:
        return None
    return _OP_SHELL[op].format(
        REG=shlex.quote(REG), TMP=shlex.quote(TMP), OUT=shlex.quote(OUT),
        KEY=shlex.quote(keyword), IN=shlex.quote(input_file))


class RealComputerWorld:
    name = "real_computer"

    def __init__(self, rng: np.random.Generator, timeout: float = 5.0):
        self.rng = rng
        self.timeout = timeout
        self._env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}

    # ---------------------------------------------------------- sandboxing
    def _write_files(self, path: str, keyword: str):
        names = self.rng.choice(WORDS, size=int(self.rng.integers(3, 6)),
                                replace=False)
        files = []
        for nm in names:
            lines = []
            for _ in range(int(self.rng.integers(3, 7))):
                w = list(self.rng.choice(WORDS, size=int(self.rng.integers(2, 4))))
                if self.rng.random() < 0.5:
                    w.append(keyword)
                lines.append(" ".join(w))
            fn = f"{nm}.txt"
            with open(os.path.join(path, fn), "w") as fh:
                fh.write("\n".join(lines) + "\n")
            files.append(fn)
        return files

    def make_task(self, level: int) -> RealTask:
        if level in CURRICULUM:
            name, canonical = CURRICULUM[level][int(self.rng.integers(0, len(CURRICULUM[level])))]
        else:
            name, canonical = CURRICULUM[MAX_LEVEL][0]
        keyword = str(self.rng.choice(WORDS))
        sandbox = tempfile.mkdtemp(prefix="echo_realvm_")
        files = self._write_files(sandbox, keyword)
        input_file = files[int(self.rng.integers(0, len(files)))]
        task = RealTask(name=name, level=level, sandbox=sandbox,
                        input_file=input_file, keyword=keyword,
                        canonical=tuple(canonical))
        # define the expected output by REALLY running the canonical pipeline
        task.expected_output, _, _ = self.execute(canonical, task)
        return task

    # ---------------------------------------------------------- execution
    def execute(self, program, task: RealTask):
        """Run a program of op-names as real shell commands in the sandbox.
        Returns (output_text, n_shell_calls, command_log)."""
        # reset register and output
        for f in (REG, TMP, OUT):
            p = os.path.join(task.sandbox, f)
            if os.path.exists(p):
                os.remove(p)
        log, calls = [], 0
        for op in program:
            cmd = build_command(op, task.keyword, task.input_file)
            if cmd is None:
                continue
            calls += 1
            log.append(cmd)
            try:
                subprocess.run(["bash", "-c", cmd], cwd=task.sandbox,
                               env=self._env, timeout=self.timeout,
                               capture_output=True)
            except subprocess.TimeoutExpired:
                break
        out_path = os.path.join(task.sandbox, OUT)
        output = ""
        if os.path.exists(out_path):
            with open(out_path) as fh:
                output = fh.read().rstrip("\n")
        return output, calls, log

    def grade(self, program, task: RealTask):
        output, calls, log = self.execute(program, task)
        expected = task.expected_output
        if output == expected:
            return True, 1.0, output, calls, log
        n = max(len(output), len(expected), 1)
        matches = sum(1 for i in range(min(len(output), len(expected)))
                      if output[i] == expected[i])
        return False, matches / n, output, calls, log
