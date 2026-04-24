"""Loader for ``nvidia/Nemotron-RL-coding-competitive_coding``.

Source schema (per row):
    input:              list[{"role": "user", "content": str}]  — problem prompt
    verifier_metadata:  {"unit_tests": {"inputs": list[str], "outputs": list[str]}}
    ... (hash_id, dataset, source, responses_create_params, ...)

Supports three ways to specify ``path``:
    * HuggingFace Hub name, e.g. ``nvidia/Nemotron-RL-coding-competitive_coding``
    * Local directory with the HF dataset layout
    * Local ``.jsonl`` / ``.parquet`` single-file snapshot
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from datasets import load_dataset

if TYPE_CHECKING:
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def _load_raw(path: str, split: str):
    if os.path.isfile(path) and (path.endswith(".jsonl") or path.endswith(".parquet")):
        parent = os.path.dirname(path)
        fname = os.path.basename(path)
        return load_dataset(path=parent, data_files=fname)["train"]
    if os.path.isdir(path):
        return load_dataset(path=path, split=split)
    # HF Hub fallback
    return load_dataset(path=path, split=split)


def get_nemotron_competitive_rl_dataset(
    path: str,
    split: str,
    tokenizer: "PreTrainedTokenizerFast",
    max_length: int | None = None,
):
    dataset = _load_raw(path, split)

    def process(sample):
        messages = sample["input"]
        verifier = sample.get("verifier_metadata") or {}
        unit_tests = verifier.get("unit_tests") or {}
        return {
            "messages": messages,
            "test_inputs": list(unit_tests.get("inputs") or []),
            "test_outputs": list(unit_tests.get("outputs") or []),
        }

    dataset = dataset.map(process, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["test_inputs"]) > 0)

    if max_length is not None:

        def fits(sample):
            content = sample["messages"][0]["content"] if sample["messages"] else ""
            return len(tokenizer.encode(content)) <= max_length

        dataset = dataset.filter(fits)

    return dataset
