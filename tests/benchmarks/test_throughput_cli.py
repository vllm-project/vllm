# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import subprocess
import sys

import pytest

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.mark.benchmark
def test_bench_throughput(tmp_path):
    output_json = tmp_path / "throughput.json"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "throughput",
        "--model",
        MODEL_NAME,
        "--input-len",
        "32",
        "--output-len",
        "1",
        "--num-prompts",
        "2",
        "--enforce-eager",
        "--load-format",
        "dummy",
        "--output-json",
        str(output_json),
        "--metadata",
        "test_key=test_value",
        "source=pytest",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
    results = json.loads(output_json.read_text())
    assert "requests_per_second" in results
    assert results["test_key"] == "test_value"
    assert results["source"] == "pytest"
