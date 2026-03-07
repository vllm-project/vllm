# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the --eval feature in vllm bench serve.

Uses an existing server (managed by the server fixture) so the test
does not start a second vllm process. Keeps --eval-limit very small
so the test completes quickly.
"""

import json
import subprocess

import pytest

from ..utils import RemoteOpenAIServer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

_SERVER_ARGS = [
    "--max-model-len",
    "4096",
    "--enforce-eager",
    "--load-format",
    "dummy",
]


@pytest.fixture(scope="function")
def server():
    with RemoteOpenAIServer(MODEL_NAME, _SERVER_ARGS) as remote_server:
        yield remote_server


def _bench_serve_eval_cmd(server, output_path, extra_args=None):
    """Build a vllm bench serve --eval command list."""
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        MODEL_NAME,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--eval",
        "--eval-tasks",
        "gsm8k",
        "--eval-limit",
        "3",
        "--eval-num-fewshot",
        "8",
        "--eval-output",
        str(output_path),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


@pytest.mark.benchmark
def test_bench_serve_eval(server, tmp_path):
    """vllm bench serve --eval runs end-to-end and writes valid JSONL."""
    out = tmp_path / "results.jsonl"
    cmd = _bench_serve_eval_cmd(server, out)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"bench serve --eval failed:\n{result.stderr}"
    assert out.exists(), "No JSONL output file was created"

    lines = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 1, f"Expected 1 JSONL record, got {len(lines)}"

    record = json.loads(lines[0])
    expected_keys = {"metadata", "accuracy", "performance", "environment"}
    assert expected_keys <= record.keys(), (
        f"Missing top-level keys. Got: {list(record.keys())}"
    )

    meta = record["metadata"]
    assert meta["model"] == MODEL_NAME
    assert meta["tasks"] == "gsm8k"
    assert meta["bench_type"] == "serve"
    assert "run_id" in meta
    assert "timestamp" in meta
    assert "base_url" in meta

    assert isinstance(record["accuracy"], dict)
    assert isinstance(record["performance"], dict)

    env = record["environment"]
    assert "vllm_info" in env
    assert "system_info" in env


@pytest.mark.benchmark
def test_bench_serve_eval_output_appends(server, tmp_path):
    """Running --eval twice appends a second record rather than overwriting."""
    out = tmp_path / "results.jsonl"
    cmd = _bench_serve_eval_cmd(server, out)
    for _ in range(2):
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    lines = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 2, f"Expected 2 appended records, got {len(lines)}"

    ids = [json.loads(line)["metadata"]["run_id"] for line in lines]
    assert ids[0] != ids[1], "run_ids should be unique across runs"


@pytest.mark.benchmark
def test_bench_serve_eval_requires_tasks(server):
    """--eval without --eval-tasks should fail."""
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        MODEL_NAME,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--eval",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0, "Should fail without --eval-tasks"


@pytest.mark.benchmark
def test_bench_serve_without_eval(server):
    """Plain bench serve (no --eval) still works and does not produce
    eval-specific output."""
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        MODEL_NAME,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "3",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Plain bench serve failed:\n{result.stderr}"
    assert "[eval]" not in result.stdout, "Eval output should not appear without --eval"


@pytest.mark.benchmark
def test_eval_not_on_bench_throughput():
    """--eval flag should not be recognized by bench throughput."""
    cmd = [
        "vllm",
        "bench",
        "throughput",
        "--help",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--eval" not in result.stdout, (
        "--eval should not appear in bench throughput help"
    )


@pytest.mark.benchmark
def test_eval_not_on_bench_latency():
    """--eval flag should not be recognized by bench latency."""
    cmd = [
        "vllm",
        "bench",
        "latency",
        "--help",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--eval" not in result.stdout, (
        "--eval should not appear in bench latency help"
    )
