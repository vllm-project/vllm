# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
import sys

import pytest


@pytest.mark.benchmark
def test_bench_startup():
    command = [
        "vllm",
        "bench",
        "startup",
    ]
    result = subprocess.run(
        command,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, f"Benchmark failed with exit code {result.returncode}"
