# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for the ``vllm bench mm-processor`` CLI.

Most other bench subcommands have a subprocess smoke test in this directory.
This file covers mm-processor, including the per-stage stats its JSON output
reports.
"""

import json
import math
import subprocess

import pytest

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
NUM_PROMPTS = 3


@pytest.mark.benchmark
def test_bench_mm_processor(tmp_path):
    output_json = tmp_path / "mm_processor.json"
    command = [
        "vllm",
        "bench",
        "mm-processor",
        "--model",
        MODEL_NAME,
        "--dataset-name",
        "random-mm",
        "--num-prompts",
        str(NUM_PROMPTS),
        "--random-input-len",
        "32",
        "--random-output-len",
        "1",
        "--random-mm-base-items-per-request",
        "1",
        "--random-mm-num-mm-items-range-ratio",
        "0",
        "--random-mm-bucket-config",
        "{(224, 224, 1): 1.0}",
        "--random-mm-limit-mm-per-prompt",
        '{"image": 1}',
        "--load-format",
        "dummy",
        "--enforce-eager",
        "--disable-tqdm",
        "--output-json",
        str(output_json),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as exc:
        print(exc.stdout or "")
        print(exc.stderr or "")
        raise
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, (
        f"Benchmark failed with exit code {result.returncode}: {result.stderr}"
    )

    payload = json.loads(output_json.read_text())
    assert payload["config"]["model"] == MODEL_NAME
    assert payload["completed"] == NUM_PROMPTS
    assert payload["failed"] == 0

    # The benchmark exists to report per-stage processor timing. A text-only
    # request would complete without producing any of these stats.
    stats = payload["mm_processor_stats"]
    assert "preprocessor_total_ms" in stats
    assert stats["preprocessor_total_ms"]["mean"] > 0
    assert all(
        math.isfinite(value) for stage in stats.values() for value in stage.values()
    )

    # The vision encoder must actually have run for these image requests.
    encoder_summary = payload["encoder_summary"]
    assert encoder_summary, "no encoder timing recorded"
    assert encoder_summary["num_requests_with_encoder_calls"] == NUM_PROMPTS
