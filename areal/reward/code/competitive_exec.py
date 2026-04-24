"""Reward for Nemotron-RL-coding-competitive_coding.

Extracts the last ```python code block from the model completion, runs it
against stdin/stdout unit tests in a sandbox, returns pass rate ∈ [0, 1].

Test subsampling is deterministic per ``prompt`` so that all rollouts of the
same question in a GRPO group see the same tests — otherwise the group-level
advantage normalization gets noisy.
"""

from __future__ import annotations

import random
import re

from areal.reward.code.sandbox import outputs_match, run_python
from areal.utils import logging

logger = logging.getLogger("CompetitiveCodeReward")

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    """Return the last fenced Python block; empty string if none is present."""
    matches = _CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return ""


def nemotron_competitive_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    test_inputs: list[str] | None = None,
    test_outputs: list[str] | None = None,
    per_test_timeout: float = 5.0,
    max_tests: int = 15,
    memory_mb: int = 2048,
    **kwargs,
) -> float:
    """Pass rate over (a sample of) unit tests.

    :param per_test_timeout: Wall-clock seconds for each test case.
    :param max_tests: Upper bound on tests executed per reward call; tests
        beyond this cap are sampled deterministically per ``prompt``.
    :param memory_mb: Address-space cap for each test subprocess.
    """
    if not test_inputs or not test_outputs:
        return 0.0
    if len(test_inputs) != len(test_outputs):
        logger.warning(
            f"Mismatched test inputs/outputs: {len(test_inputs)} vs {len(test_outputs)}"
        )
        return 0.0

    code = extract_python_code(completions)
    if not code:
        return 0.0

    n = len(test_inputs)
    if n <= max_tests:
        idx = list(range(n))
    else:
        # Seeding with the prompt keeps the sampled subset identical across
        # all rollouts of the same problem in a GRPO group.
        rng = random.Random(prompt)
        idx = rng.sample(range(n), max_tests)

    passed = 0
    for i in idx:
        result = run_python(
            code=code,
            stdin=test_inputs[i],
            timeout=per_test_timeout,
            memory_mb=memory_mb,
        )
        if result.timeout or result.error is not None or result.returncode != 0:
            continue
        if outputs_match(result.stdout, test_outputs[i]):
            passed += 1

    return passed / len(idx)
