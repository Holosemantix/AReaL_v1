"""Subprocess sandbox for executing untrusted Python code.

Isolation layers:
1. Separate subprocess (python3) with its own address space.
2. ``resource.setrlimit`` caps on address space, CPU time, and core dumps.
3. New process group (``os.setsid``) so a hung interpreter and any forks it
   spawned get cleaned up together on timeout.
4. Ephemeral tempdir as CWD so the program cannot pollute the training directory.
"""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import tempfile
from dataclasses import dataclass

from areal.utils import logging

logger = logging.getLogger("CodeSandbox")


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: int
    timeout: bool
    error: str | None = None


def _set_rlimits(memory_mb: int, cpu_seconds: int) -> None:
    # Virtual address space (RSS + swap proxy on Linux)
    mem_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    # CPU time — defense in depth vs. a fork() storm escaping wall-clock timeout
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    # Suppress core dumps (disk I/O + noise)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_python(
    code: str,
    stdin: str = "",
    timeout: float = 5.0,
    memory_mb: int = 2048,
) -> ExecutionResult:
    """Run ``code`` under ``python3`` with ``stdin`` piped in.

    :param code: Full Python source.
    :param stdin: Standard input fed to the program.
    :param timeout: Wall-clock seconds before the subprocess is killed.
    :param memory_mb: Address-space ceiling for the subprocess.
    """
    if not code.strip():
        return ExecutionResult(
            stdout="", stderr="", returncode=-1, timeout=False, error="empty code"
        )

    cpu_limit = max(int(timeout) + 2, 2)

    def _preexec() -> None:
        _set_rlimits(memory_mb, cpu_limit)
        os.setsid()

    with tempfile.TemporaryDirectory(prefix="arealcode_") as tmpdir:
        code_path = os.path.join(tmpdir, "solution.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(
                ["python3", "-u", code_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdir,
                preexec_fn=_preexec,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(input=stdin, timeout=timeout)
                return ExecutionResult(
                    stdout=stdout or "",
                    stderr=stderr or "",
                    returncode=proc.returncode,
                    timeout=False,
                )
            except subprocess.TimeoutExpired:
                # Kill the whole process group — catches child forks.
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                # Drain pipes so they don't dangle.
                try:
                    stdout, stderr = proc.communicate(timeout=1.0)
                except Exception:
                    stdout, stderr = "", ""
                return ExecutionResult(
                    stdout=stdout or "",
                    stderr=stderr or "",
                    returncode=-1,
                    timeout=True,
                )
        except Exception as e:
            if proc is not None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            return ExecutionResult(
                stdout="", stderr="", returncode=-1, timeout=False, error=str(e)
            )


def normalize_output(text: str) -> str:
    """Canonical form for stdout comparison: LF line endings, right-trim each
    line, drop trailing blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.rstrip("\n")


def outputs_match(produced: str, expected: str) -> bool:
    return normalize_output(produced) == normalize_output(expected)
