"""Reward for Nemotron-RL-coding-competitive_coding (and future multi-lang data).

Extracts the last fenced code block from the model completion (Python by
default, configurable per sample), compiles once, runs against stdin/stdout
unit tests, returns pass rate ∈ [0, 1].

Test subsampling is deterministic per ``prompt`` so all rollouts of the same
question in a GRPO group see the same tests — otherwise the group-level
advantage normalization gets noisy.
"""

from __future__ import annotations

import random
import re

from areal.reward.code.sandbox import (
    canonical_language,
    cleanup_compile_result,
    compile_code,
    outputs_match,
    run_artifact,
)
from areal.utils import logging

logger = logging.getLogger("CompetitiveCodeReward")

# Group 1: language tag (optional). Group 2: code body.
_CODE_BLOCK_RE = re.compile(
    r"```(python|py|cpp|c\+\+|cxx|java|javascript|js|node)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_code(
    text: str, expected_language: str = "python"
) -> tuple[str, str] | None:
    """Return ``(canonical_language, code)`` of the last useful fenced block.

    Selection:
      1. Prefer the LAST fence whose tag matches ``expected_language``.
      2. Else fall back to the LAST fence with any recognized tag.
      3. Else fall back to the LAST untagged fence, treated as
         ``expected_language``.
      4. Return None if no fenced block is present.
    """
    expected_canon = canonical_language(expected_language) or "python"

    matches = _CODE_BLOCK_RE.findall(text)
    if not matches:
        return None

    matched_expected: tuple[str, str] | None = None
    matched_any_tag: tuple[str, str] | None = None
    matched_untagged: tuple[str, str] | None = None

    for tag, code in matches:
        body = code.strip()
        if not body:
            continue
        canon = canonical_language(tag) if tag else None
        if canon == expected_canon:
            matched_expected = (canon, body)
        elif canon is not None:
            matched_any_tag = (canon, body)
        else:
            matched_untagged = (expected_canon, body)

    return matched_expected or matched_any_tag or matched_untagged


def nemotron_competitive_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    test_inputs: list[str] | None = None,
    test_outputs: list[str] | None = None,
    language: str = "python",
    per_test_timeout: float = 5.0,
    max_tests: int = 15,
    memory_mb: int = 2048,
    compile_timeout: float = 30.0,
    java_xmx_mb: int = 2048,
    **kwargs,
) -> float:
    """Pass rate over (a sample of) unit tests.

    :param language: Source language for the model output. May be overridden
        by ``task_data["language"]`` injected by the dataset loader. Defaults
        to ``"python"`` for Nemotron-RL-coding-competitive_coding.
    :param per_test_timeout: Wall-clock seconds for each test case.
    :param max_tests: Upper bound on tests executed per reward call; tests
        beyond this cap are sampled deterministically per ``prompt``.
    :param memory_mb: Address-space cap for each test subprocess (Python/C++/
        Node). Java uses ``java_xmx_mb`` instead.
    :param compile_timeout: Wall-clock seconds for the compile step (compiled
        languages only). Compile happens once per submission.
    :param java_xmx_mb: JVM heap cap for Java submissions (``-Xmx``).
    """
    if not test_inputs or not test_outputs:
        return 0.0
    if len(test_inputs) != len(test_outputs):
        logger.warning(
            f"Mismatched test inputs/outputs: {len(test_inputs)} vs {len(test_outputs)}"
        )
        return 0.0

    extracted = extract_code(completions, expected_language=language)
    if extracted is None:
        return 0.0
    detected_lang, code = extracted

    artifact = compile_code(detected_lang, code, compile_timeout=compile_timeout)
    if not artifact.ok:
        # Compile failure or unsupported language — caller already logged in
        # CompileResult.error/stderr; treat as zero reward.
        cleanup_compile_result(artifact)
        return 0.0

    try:
        n = len(test_inputs)
        if n <= max_tests:
            idx = list(range(n))
        else:
            # Seeding with the prompt keeps the sampled subset identical
            # across all rollouts of the same problem in a GRPO group.
            rng = random.Random(prompt)
            idx = rng.sample(range(n), max_tests)

        passed = 0
        for i in idx:
            result = run_artifact(
                artifact,
                stdin=test_inputs[i],
                timeout=per_test_timeout,
                memory_mb=memory_mb,
                java_xmx_mb=java_xmx_mb,
            )
            if result.timeout or result.error is not None or result.returncode != 0:
                continue
            if outputs_match(result.stdout, test_outputs[i]):
                passed += 1

        return passed / len(idx)
    finally:
        cleanup_compile_result(artifact)
