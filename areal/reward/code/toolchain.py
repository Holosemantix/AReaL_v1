"""Toolchain availability probe for multi-language code execution.

Run :func:`assert_toolchain_or_warn` once at trainer startup so deployment
issues (missing g++, java, node) surface as warnings instead of as silent
zero rewards during training.
"""

from __future__ import annotations

import shutil

from areal.utils import logging

logger = logging.getLogger("CodeSandbox")

# Maps each canonical language to the executable(s) that must be on PATH.
# We probe the first command of compile_cmd (or run_cmd for interpreted langs).
_LANGUAGE_BINARIES: dict[str, list[str]] = {
    "python": ["python3"],
    "cpp": ["g++"],
    "java": ["javac", "java"],
    "javascript": ["node"],
}


def check_toolchain(languages: list[str]) -> dict[str, bool]:
    """Return ``{language: True}`` if every required binary is on PATH."""
    result: dict[str, bool] = {}
    for lang in languages:
        bins = _LANGUAGE_BINARIES.get(lang)
        if bins is None:
            result[lang] = False
            continue
        result[lang] = all(shutil.which(b) is not None for b in bins)
    return result


def assert_toolchain_or_warn(languages: list[str]) -> None:
    """Log a warning for each language whose toolchain is missing.

    Never raises — the caller may still want to train on a Python-only subset
    even if g++/java/node are absent.
    """
    status = check_toolchain(languages)
    available = [lang for lang, ok in status.items() if ok]
    missing = [lang for lang, ok in status.items() if not ok]
    if available:
        logger.info(f"Code sandbox toolchain available: {available}")
    if missing:
        missing_bins = {lang: _LANGUAGE_BINARIES.get(lang, []) for lang in missing}
        logger.warning(
            f"Code sandbox toolchain MISSING for {missing}; required binaries: "
            f"{missing_bins}. Samples in those languages will receive 0 reward."
        )
