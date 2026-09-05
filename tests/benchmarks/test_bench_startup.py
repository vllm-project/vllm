# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess

import pytest

from vllm.benchmarks.startup import _collect_phase_metrics


@pytest.mark.benchmark
def test_bench_startup():
    command = [
        "vllm",
        "bench",
        "startup",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


def test_empty_startup_phase_has_no_metrics():
    assert _collect_phase_metrics("cold", [], has_encoder=False) == []
