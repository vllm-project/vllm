# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import subprocess

import pytest

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.mark.benchmark
def test_bench_latency(tmp_path):
    output_json = tmp_path / "latency_output.json"
    command = [
        "vllm",
        "bench",
        "latency",
        "--model",
        MODEL_NAME,
        "--input-len",
        "32",
        "--output-len",
        "1",
        "--enforce-eager",
        "--load-format",
        "dummy",
        "--output-json",
        str(output_json),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
    # https://github.com/vllm-project/vllm/issues/58100: --output-json must
    # record model_id, matching `vllm bench serve`.
    assert json.loads(output_json.read_text())["model_id"] == MODEL_NAME
