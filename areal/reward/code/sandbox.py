"""Subprocess sandbox for executing untrusted code in multiple languages.

Supports Python (interpreted), C++/Java (compiled), and JavaScript/Node
(interpreted). For compiled languages the compile step happens once per
submission and the resulting artifact is reused across all unit tests —
critical for GRPO with ``n_samples * max_tests`` executions per problem.

Isolation layers (per execution):
1. Separate subprocess with its own address space.
2. ``resource.setrlimit`` caps on address space, CPU time, and core dumps.
   Java skips RLIMIT_AS because the JVM reserves the entire heap up-front;
   we cap the JVM heap with ``-Xmx`` instead.
3. New process group (``os.setsid``) so a hung interpreter and any forks it
   spawned get cleaned up together on timeout.
4. Ephemeral tempdir as CWD.
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field

from areal.utils import logging

logger = logging.getLogger("CodeSandbox")


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: int
    timeout: bool
    error: str | None = None


@dataclass
class CompileResult:
    """Outcome of the compile (or stage-only) step.

    ``ok`` is False when the compiler returned non-zero or the toolchain is
    missing. The reward fn treats a failed compile as 0 reward.
    """

    ok: bool
    workdir: str
    run_cmd: list[str]
    stderr: str = ""
    error: str | None = None
    needs_jvm_xmx: bool = False
    skip_address_space_limit: bool = False


@dataclass
class LanguageSpec:
    """Toolchain configuration for one language.

    ``compile_cmd`` is None for interpreted languages; the source file is
    just staged on disk.

    Format placeholders (substituted at compile/run time):
        {src}      absolute path to the source file
        {exe}      absolute path to the compiled executable (compiled langs only)
        {workdir}  the per-submission working directory
    """

    name: str
    source_filename: str
    run_cmd: list[str]
    compile_cmd: list[str] | None = None
    exe_name: str | None = None  # only for compiled languages
    needs_jvm_xmx: bool = False
    # JVMs (Java) and V8 (Node) reserve large virtual-memory segments at
    # startup that exceed any reasonable RSS-based cap. Setting RLIMIT_AS
    # tight enough to bound RSS would kill them before main runs, so we
    # skip RLIMIT_AS for these and rely on per-runtime heap caps instead.
    skip_address_space_limit: bool = False
    extra_run_env: dict[str, str] = field(default_factory=dict)


LANGUAGE_SPECS: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        name="python",
        source_filename="solution.py",
        run_cmd=["python3", "-u", "{src}"],
    ),
    "cpp": LanguageSpec(
        name="cpp",
        source_filename="solution.cpp",
        compile_cmd=["g++", "-O2", "-std=c++17", "-pipe", "{src}", "-o", "{exe}"],
        exe_name="solution",
        run_cmd=["{exe}"],
    ),
    "java": LanguageSpec(
        name="java",
        # Class must match the filename for `java Main` to find it.
        source_filename="Main.java",
        compile_cmd=["javac", "{src}"],
        # JVM heap cap is filled in at run time from java_xmx_mb.
        run_cmd=["java", "-Xmx{xmx}m", "-cp", "{workdir}", "Main"],
        needs_jvm_xmx=True,
        skip_address_space_limit=True,
    ),
    "javascript": LanguageSpec(
        name="javascript",
        source_filename="solution.js",
        # `--max-old-space-size` is the per-process JS heap cap V8 honors.
        run_cmd=["node", "--max-old-space-size={node_old_mb}", "{src}"],
        skip_address_space_limit=True,
    ),
}


_LANG_ALIASES = {
    "python": "python",
    "py": "python",
    "cpp": "cpp",
    "c++": "cpp",
    "cxx": "cpp",
    "java": "java",
    "javascript": "javascript",
    "js": "javascript",
    "node": "javascript",
}


def canonical_language(lang: str) -> str | None:
    """Normalize a language alias to one of LANGUAGE_SPECS keys, or None."""
    if not lang:
        return None
    return _LANG_ALIASES.get(lang.strip().lower())


def _set_rlimits(memory_mb: int | None, cpu_seconds: int) -> None:
    if memory_mb is not None:
        mem_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def compile_code(
    language: str,
    code: str,
    compile_timeout: float = 30.0,
) -> CompileResult:
    """Stage source for interpreted languages or compile for compiled ones.

    The returned ``CompileResult`` owns a tempdir; pass it to ``run_artifact``
    one or more times, then call :func:`cleanup_compile_result` (or wrap with
    a context manager) to delete the tempdir.

    On compile failure the workdir is cleaned up immediately and ``ok=False``
    is returned.
    """
    canon = canonical_language(language)
    if canon is None:
        return CompileResult(
            ok=False, workdir="", run_cmd=[], error=f"unknown language: {language!r}"
        )
    spec = LANGUAGE_SPECS[canon]

    if not code.strip():
        return CompileResult(ok=False, workdir="", run_cmd=[], error="empty code")

    workdir = tempfile.mkdtemp(prefix=f"arealcode_{canon}_")
    try:
        src_path = os.path.join(workdir, spec.source_filename)
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(code)

        exe_path = os.path.join(workdir, spec.exe_name) if spec.exe_name else ""

        if spec.compile_cmd is not None:
            cmd = [
                arg.format(src=src_path, exe=exe_path, workdir=workdir)
                for arg in spec.compile_cmd
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=compile_timeout,
                )
            except FileNotFoundError as e:
                shutil.rmtree(workdir, ignore_errors=True)
                return CompileResult(
                    ok=False,
                    workdir="",
                    run_cmd=[],
                    error=f"toolchain missing for {canon}: {e}",
                )
            except subprocess.TimeoutExpired:
                shutil.rmtree(workdir, ignore_errors=True)
                return CompileResult(
                    ok=False,
                    workdir="",
                    run_cmd=[],
                    error=f"compile timeout after {compile_timeout}s",
                )

            if proc.returncode != 0:
                stderr = proc.stderr or ""
                shutil.rmtree(workdir, ignore_errors=True)
                return CompileResult(
                    ok=False,
                    workdir="",
                    run_cmd=[],
                    stderr=stderr,
                    error="compile failed",
                )

        run_cmd_template = list(spec.run_cmd)
        return CompileResult(
            ok=True,
            workdir=workdir,
            run_cmd=[
                arg.replace("{src}", src_path)
                .replace("{exe}", exe_path)
                .replace("{workdir}", workdir)
                for arg in run_cmd_template
            ],
            needs_jvm_xmx=spec.needs_jvm_xmx,
            skip_address_space_limit=spec.skip_address_space_limit,
        )
    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        return CompileResult(
            ok=False, workdir="", run_cmd=[], error=f"compile exception: {e}"
        )


def cleanup_compile_result(result: CompileResult) -> None:
    if result.workdir and os.path.isdir(result.workdir):
        shutil.rmtree(result.workdir, ignore_errors=True)


def run_artifact(
    artifact: CompileResult,
    stdin: str = "",
    timeout: float = 5.0,
    memory_mb: int = 2048,
    java_xmx_mb: int = 2048,
) -> ExecutionResult:
    """Execute a compiled/staged artifact against one stdin payload.

    For Java the run command is finalized here using ``java_xmx_mb``; callers
    pass a single value for both the address-space cap (skipped for Java) and
    the JVM heap cap.
    """
    if not artifact.ok:
        return ExecutionResult(
            stdout="",
            stderr=artifact.stderr,
            returncode=-1,
            timeout=False,
            error=artifact.error or "compile failed",
        )

    cpu_limit = max(int(timeout) + 2, 2)
    # JVM and V8 reserve large virtual-memory ranges at startup; RLIMIT_AS
    # would kill them before main runs. Their per-runtime heap caps
    # (-Xmx, --max-old-space-size) bound the actual JS/JVM heap.
    address_space_cap = None if artifact.skip_address_space_limit else memory_mb

    cmd = [
        arg.replace("{xmx}", str(java_xmx_mb)).replace(
            "{node_old_mb}", str(max(memory_mb - 64, 64))
        )
        for arg in artifact.run_cmd
    ]

    def _preexec() -> None:
        _set_rlimits(address_space_cap, cpu_limit)
        os.setsid()

    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=artifact.workdir,
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
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
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
    except FileNotFoundError as e:
        return ExecutionResult(
            stdout="", stderr="", returncode=-1, timeout=False, error=str(e)
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


def run_python(
    code: str,
    stdin: str = "",
    timeout: float = 5.0,
    memory_mb: int = 2048,
) -> ExecutionResult:
    """Backward-compat shim: stage + run a Python program in one shot."""
    artifact = compile_code("python", code)
    try:
        return run_artifact(artifact, stdin=stdin, timeout=timeout, memory_mb=memory_mb)
    finally:
        cleanup_compile_result(artifact)


def normalize_output(text: str) -> str:
    """Canonical form for stdout comparison: LF line endings, right-trim each
    line, drop trailing blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.rstrip("\n")


def outputs_match(produced: str, expected: str) -> bool:
    return normalize_output(produced) == normalize_output(expected)
