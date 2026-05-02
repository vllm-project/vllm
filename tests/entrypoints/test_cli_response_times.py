# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[2]
MEASURED_RUNS = 4
MAX_AVG_MS = 1000

BAD_ARG = "--bad-speling"  # typos:disable-line

COMMANDS = (
    (["vllm", "serve", "--help"], 0),
    ([sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", "--help"], 0),
    (["vllm", "serve", BAD_ARG], 2),
    ([sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", BAD_ARG], 2),
)


def test_vllm_serve_response_time():
    command, expected_returncode = COMMANDS[0]
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == expected_returncode, result.stderr

    for command, expected_returncode in COMMANDS:
        times_ms: list[float] = []
        for _ in range(MEASURED_RUNS):
            start = time.perf_counter()
            result = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, timeout=30
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert result.returncode == expected_returncode, result.stderr
            times_ms.append(elapsed_ms)

        avg_ms = sum(times_ms) / MEASURED_RUNS
        assert avg_ms < MAX_AVG_MS, (
            f"`{command}`: avg={avg_ms:.1f}ms max={MAX_AVG_MS}ms "
            f"runs={[round(t, 1) for t in times_ms]}"
        )
