from __future__ import annotations

import base64
import json
import pickle
import zlib

from areal.dataset.code.competitive_eval import (
    _build_messages,
    _extract_apps_style_tests,
    _extract_livecodebench_tests,
    _find_data_files,
)


def test_extract_apps_style_tests_reads_stdin_cases():
    input_output = json.dumps(
        {
            "inputs": ["1\n", "2\n"],
            "outputs": ["2\n", "4\n"],
        }
    )

    inputs, outputs = _extract_apps_style_tests(input_output)

    assert inputs == ["1\n", "2\n"]
    assert outputs == ["2\n", "4\n"]


def test_extract_apps_style_tests_skips_function_cases():
    input_output = json.dumps(
        {
            "fn_name": "solve",
            "inputs": ["[1]"],
            "outputs": ["[2]"],
        }
    )

    inputs, outputs = _extract_apps_style_tests(input_output)

    assert inputs == []
    assert outputs == []


def test_extract_livecodebench_tests_reads_json_and_compressed_private_cases():
    public_cases = json.dumps(
        [
            {
                "input": "1\n",
                "output": "2\n",
                "testtype": "stdin",
            }
        ]
    )
    private_payload = json.dumps(
        [
            {
                "input": "2\n",
                "output": "4\n",
                "testtype": "stdin",
            },
            {
                "input": "[3]",
                "output": "[6]",
                "testtype": "functional",
            },
        ]
    )
    compressed_private_cases = base64.b64encode(
        zlib.compress(pickle.dumps(private_payload))
    ).decode("utf-8")

    inputs, outputs = _extract_livecodebench_tests(
        public_cases, compressed_private_cases
    )

    assert inputs == ["1\n", "2\n"]
    assert outputs == ["2\n", "4\n"]


def test_build_messages_instructs_complete_stdin_program():
    messages = _build_messages(
        "Double the input.",
        title="Double",
        starter_code="# optional starter",
    )

    assert messages == [
        {
            "role": "user",
            "content": (
                "Write a complete Python program that reads from stdin and writes to stdout.\n\n"
                "Return only one fenced Python code block.\n\n"
                "Title:\nDouble\n\n"
                "Problem:\nDouble the input.\n\n"
                "Starter code:\n# optional starter"
            ),
        }
    ]


def test_find_data_files_ignores_dataset_script_and_selects_split(tmp_path):
    (tmp_path / "code_generation_lite.py").write_text("raise RuntimeError\n")
    (tmp_path / "README.md").write_text("dataset card\n")
    (tmp_path / "train.jsonl").write_text("{}\n")
    (tmp_path / "test.jsonl").write_text("{}\n")

    data_files = _find_data_files(str(tmp_path), split="test")

    assert data_files == ("json", [str(tmp_path / "test.jsonl")])


def test_find_data_files_prefers_parquet_split_shards(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test-00000-of-00002.parquet").write_bytes(b"")
    (data_dir / "test-00001-of-00002.parquet").write_bytes(b"")
    (data_dir / "train-00000-of-00001.parquet").write_bytes(b"")

    data_files = _find_data_files(str(tmp_path), split="test")

    assert data_files == (
        "parquet",
        [
            str(data_dir / "test-00000-of-00002.parquet"),
            str(data_dir / "test-00001-of-00002.parquet"),
        ],
    )
