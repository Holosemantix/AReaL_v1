"""Smoke tests for the multi-language code sandbox.

Each test compiles + runs a tiny "double the input integer" program against
two stdin/stdout test cases. Languages whose toolchain is absent are skipped
(via ``check_toolchain``) so CI runners without g++/javac/node still pass.
"""

from __future__ import annotations

import pytest

from areal.reward.code import (
    check_toolchain,
    cleanup_compile_result,
    compile_code,
    extract_code,
    nemotron_competitive_reward_fn,
    outputs_match,
    run_artifact,
    run_python,
)

TOOLCHAIN = check_toolchain(["python", "cpp", "java", "javascript"])


# (language, source, expected canonical name)
LANGUAGE_SOURCES = [
    (
        "python",
        "x = int(input())\nprint(x * 2)\n",
        "python",
    ),
    (
        "cpp",
        "#include <iostream>\nint main(){int x; std::cin >> x; std::cout << x * 2 << '\\n'; return 0;}\n",
        "cpp",
    ),
    (
        "java",
        (
            "import java.util.Scanner;\n"
            "public class Main {\n"
            "  public static void main(String[] args) {\n"
            "    Scanner s = new Scanner(System.in);\n"
            "    System.out.println(s.nextInt() * 2);\n"
            "  }\n"
            "}\n"
        ),
        "java",
    ),
    (
        "javascript",
        (
            "let buf='';\n"
            "process.stdin.on('data',c=>buf+=c);\n"
            "process.stdin.on('end',()=>{console.log(parseInt(buf,10)*2);});\n"
        ),
        "javascript",
    ),
]


@pytest.mark.parametrize("language,source,canon", LANGUAGE_SOURCES)
def test_compile_and_run_round_trip(language: str, source: str, canon: str):
    if not TOOLCHAIN.get(language):
        pytest.skip(f"{language} toolchain not available")

    artifact = compile_code(language, source, compile_timeout=30.0)
    try:
        assert artifact.ok, f"compile failed: {artifact.error}\n{artifact.stderr}"
        result = run_artifact(artifact, stdin="21\n", timeout=10.0, memory_mb=512)
        assert result.error is None, f"run error: {result.error}"
        assert not result.timeout, "should not timeout"
        assert result.returncode == 0, f"non-zero rc: {result.stderr}"
        assert outputs_match(result.stdout, "42")
    finally:
        cleanup_compile_result(artifact)


def test_run_python_backward_compat():
    if not TOOLCHAIN.get("python"):
        pytest.skip("python toolchain not available")
    res = run_python("print(int(input()) + 1)", stdin="5\n", timeout=5.0)
    assert res.returncode == 0
    assert outputs_match(res.stdout, "6")


def test_compile_failure_returns_zero_reward_artifact():
    if not TOOLCHAIN.get("cpp"):
        pytest.skip("cpp toolchain not available")
    # Missing semicolon — compile error.
    artifact = compile_code("cpp", "int main(){ return 0 }", compile_timeout=10.0)
    try:
        assert not artifact.ok
        assert artifact.error == "compile failed"
        assert (
            "expected" in artifact.stderr.lower() or artifact.stderr
        )  # diagnostic present
    finally:
        cleanup_compile_result(artifact)


def test_unknown_language_rejected():
    artifact = compile_code("ruby", "puts 42", compile_timeout=5.0)
    try:
        assert not artifact.ok
        assert "unknown language" in (artifact.error or "")
    finally:
        cleanup_compile_result(artifact)


def test_empty_code_rejected():
    artifact = compile_code("python", "   \n  ", compile_timeout=5.0)
    try:
        assert not artifact.ok
        assert "empty" in (artifact.error or "")
    finally:
        cleanup_compile_result(artifact)


def test_runtime_timeout_kills_process():
    if not TOOLCHAIN.get("python"):
        pytest.skip("python toolchain not available")
    # Infinite loop — must be killed by wall-clock timeout, not block the test.
    res = run_python("while True: pass", stdin="", timeout=1.0)
    assert res.timeout is True
    assert res.returncode == -1


# extract_code coverage --------------------------------------------------------


def test_extract_code_prefers_expected_language():
    text = (
        "Here's a Java attempt:\n```java\nclass Main {}\n```\n"
        "But the real one is:\n```python\nprint(1)\n```\n"
    )
    got = extract_code(text, expected_language="python")
    assert got is not None
    assert got[0] == "python"
    assert got[1].strip() == "print(1)"


def test_extract_code_falls_back_to_any_tag():
    text = "```cpp\nint main(){}\n```"
    got = extract_code(text, expected_language="python")
    assert got == ("cpp", "int main(){}")


def test_extract_code_treats_untagged_as_expected_language():
    text = "```\nprint(1)\n```"
    got = extract_code(text, expected_language="python")
    assert got == ("python", "print(1)")


def test_extract_code_returns_none_when_no_fence():
    assert extract_code("no fences here", expected_language="python") is None


def test_extract_code_picks_last_fence():
    text = (
        "First try:\n```python\nprint('first')\n```\n"
        "Second try:\n```python\nprint('second')\n```\n"
    )
    got = extract_code(text, expected_language="python")
    assert got is not None
    assert "second" in got[1]


# end-to-end reward fn ---------------------------------------------------------


def test_reward_fn_python_pass_rate():
    if not TOOLCHAIN.get("python"):
        pytest.skip("python toolchain not available")
    completion = "Here's the answer:\n```python\nx = int(input())\nprint(x * 2)\n```\n"
    result = nemotron_competitive_reward_fn(
        prompt="dummy",
        completions=completion,
        prompt_ids=[],
        completion_ids=[],
        test_inputs=["1\n", "10\n", "100\n"],
        test_outputs=["2", "20", "200"],
        language="python",
        per_test_timeout=5.0,
        max_tests=15,
        memory_mb=512,
    )
    assert result["reward"] == 1.0


def test_reward_fn_reports_test_outcomes():
    if not TOOLCHAIN.get("python"):
        pytest.skip("python toolchain not available")
    completion = "```python\nx = int(input())\nprint(x * 2)\n```"
    result = nemotron_competitive_reward_fn(
        prompt="dummy",
        completions=completion,
        prompt_ids=[],
        completion_ids=[],
        test_inputs=["1\n", "10\n"],
        test_outputs=["2", "21"],
        language="python",
        per_test_timeout=5.0,
        max_tests=15,
        memory_mb=512,
    )
    assert result["reward"] == 0.5
    assert result["sampled_tests"] == 2.0
    assert result["passed_tests"] == 1.0
    assert result["wrong_answer_tests"] == 1.0
    assert result["no_code"] == 0.0


def test_reward_fn_cpp_pass_rate():
    if not TOOLCHAIN.get("cpp"):
        pytest.skip("cpp toolchain not available")
    completion = (
        "```cpp\n"
        "#include <iostream>\nint main(){int x; std::cin >> x; std::cout << x * 2 << '\\n';}\n"
        "```\n"
    )
    result = nemotron_competitive_reward_fn(
        prompt="dummy",
        completions=completion,
        prompt_ids=[],
        completion_ids=[],
        test_inputs=["3\n", "7\n"],
        test_outputs=["6", "14"],
        language="cpp",
        per_test_timeout=10.0,
        max_tests=15,
        memory_mb=1024,
        compile_timeout=30.0,
    )
    assert result["reward"] == 1.0


def test_reward_fn_no_fence_returns_zero():
    result = nemotron_competitive_reward_fn(
        prompt="dummy",
        completions="I would solve this by ...",  # no code block
        prompt_ids=[],
        completion_ids=[],
        test_inputs=["1\n"],
        test_outputs=["2"],
        language="python",
    )
    assert result["reward"] == 0.0


def test_reward_fn_compile_failure_returns_zero():
    if not TOOLCHAIN.get("cpp"):
        pytest.skip("cpp toolchain not available")
    completion = "```cpp\nint main(){ return 0 }\n```"
    result = nemotron_competitive_reward_fn(
        prompt="dummy",
        completions=completion,
        prompt_ids=[],
        completion_ids=[],
        test_inputs=["1\n"],
        test_outputs=["2"],
        language="cpp",
        compile_timeout=10.0,
    )
    assert result["reward"] == 0.0
