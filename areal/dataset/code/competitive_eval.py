"""Load stdin/stdout code evaluation datasets into the RLVR task format."""

from __future__ import annotations

import base64
import json
import os
import pickle
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import load_dataset

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


_DATA_FILE_SUFFIXES = (".parquet", ".jsonl", ".json")


def _matches_split(path: Path, root: Path, split: str | None) -> bool:
    if split is None:
        return True
    relative = path.relative_to(root)
    return (
        split in relative.parts[:-1]
        or path.name == f"{split}{path.suffix}"
        or path.name.startswith(f"{split}-")
        or path.name.startswith(f"{split}.")
    )


def _find_data_files(path: str, split: str | None) -> tuple[str, list[str]] | None:
    root = Path(path)
    files = [
        file
        for file in root.rglob("*")
        if file.is_file()
        and file.suffix in _DATA_FILE_SUFFIXES
        and _matches_split(file, root, split)
    ]
    if not files:
        return None

    for suffix, dataset_type in (
        (".parquet", "parquet"),
        (".jsonl", "json"),
        (".json", "json"),
    ):
        selected = sorted(str(file) for file in files if file.suffix == suffix)
        if selected:
            return dataset_type, selected
    return None


def _load_raw(path: str, split: str | None) -> Dataset:
    if os.path.isfile(path):
        if path.endswith((".json", ".jsonl")):
            return load_dataset("json", data_files=path, split="train")
        if path.endswith(".parquet"):
            return load_dataset("parquet", data_files=path, split="train")
    if os.path.isdir(path):
        data_files = _find_data_files(path, split)
        if data_files is not None:
            dataset_type, files = data_files
            dataset_split = split or "train"
            return load_dataset(
                dataset_type,
                data_files={dataset_split: files},
                split=dataset_split,
            )
        return load_dataset(path=path, split=split)
    return load_dataset(path=path, split=split)


def _loads_jsonish(value: Any, default: Any):
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def _build_messages(
    problem: str,
    *,
    title: str | None = None,
    starter_code: str | None = None,
) -> list[dict[str, str]]:
    parts = [
        "Write a complete Python program that reads from stdin and writes to stdout.",
        "Return only one fenced Python code block.",
    ]
    if title:
        parts.append(f"Title:\n{title}")
    parts.append(f"Problem:\n{problem}")
    if starter_code:
        parts.append(f"Starter code:\n{starter_code}")
    return [{"role": "user", "content": "\n\n".join(parts)}]


def _filter_by_prompt_length(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int | None,
) -> Dataset:
    if max_length is None:
        return dataset

    def fits(sample):
        content = sample["messages"][0]["content"] if sample["messages"] else ""
        return len(tokenizer.encode(content)) <= max_length

    return dataset.filter(fits)


def _extract_apps_style_tests(input_output: Any) -> tuple[list[str], list[str]]:
    data = _loads_jsonish(input_output, {})
    if data.get("fn_name"):
        return [], []
    inputs = list(data.get("inputs") or [])
    outputs = list(data.get("outputs") or [])
    if len(inputs) != len(outputs):
        return [], []
    return inputs, outputs


def _decode_livecodebench_private_tests(value: Any) -> list[dict[str, Any]]:
    try:
        return list(_loads_jsonish(value, []))
    except Exception:
        decoded = pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8"))))
        return list(json.loads(decoded))


def _extract_livecodebench_tests(
    public_test_cases: Any, private_test_cases: Any
) -> tuple[list[str], list[str]]:
    public_cases = list(_loads_jsonish(public_test_cases, []))
    private_cases = _decode_livecodebench_private_tests(private_test_cases)
    inputs = []
    outputs = []
    for case in public_cases + private_cases:
        if case.get("testtype") != "stdin":
            continue
        inputs.append(case.get("input", ""))
        outputs.append(case.get("output", ""))
    if len(inputs) != len(outputs):
        return [], []
    return inputs, outputs


def get_apps_code_rl_dataset(
    path: str,
    split: str | None,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int | None = None,
):
    dataset = _load_raw(path, split)

    def process(sample):
        test_inputs, test_outputs = _extract_apps_style_tests(
            sample.get("input_output")
        )
        return {
            "messages": _build_messages(
                sample.get("question", ""),
                starter_code=sample.get("starter_code") or None,
            ),
            "test_inputs": test_inputs,
            "test_outputs": test_outputs,
            "language": "python",
        }

    dataset = dataset.map(process, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["test_inputs"]) > 0)
    return _filter_by_prompt_length(dataset, tokenizer, max_length)


def get_taco_code_rl_dataset(
    path: str,
    split: str | None,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int | None = None,
):
    dataset = _load_raw(path, split)

    def process(sample):
        test_inputs, test_outputs = _extract_apps_style_tests(
            sample.get("input_output")
        )
        return {
            "messages": _build_messages(
                sample.get("question", ""),
                title=sample.get("name") or None,
                starter_code=sample.get("starter_code") or None,
            ),
            "test_inputs": test_inputs,
            "test_outputs": test_outputs,
            "language": "python",
        }

    dataset = dataset.map(process, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["test_inputs"]) > 0)
    return _filter_by_prompt_length(dataset, tokenizer, max_length)


def get_livecodebench_code_rl_dataset(
    path: str,
    split: str | None,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int | None = None,
):
    dataset = _load_raw(path, split)

    def process(sample):
        test_inputs, test_outputs = _extract_livecodebench_tests(
            sample.get("public_test_cases"), sample.get("private_test_cases")
        )
        return {
            "messages": _build_messages(
                sample.get("question_content", ""),
                title=sample.get("question_title") or None,
                starter_code=sample.get("starter_code") or None,
            ),
            "test_inputs": test_inputs,
            "test_outputs": test_outputs,
            "language": "python",
        }

    dataset = dataset.map(process, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["test_inputs"]) > 0)
    return _filter_by_prompt_length(dataset, tokenizer, max_length)
