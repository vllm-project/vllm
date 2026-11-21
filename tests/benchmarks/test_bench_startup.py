# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import socket
import subprocess

import pytest

from vllm.utils.network_utils import get_open_port


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


@pytest.mark.benchmark
def test_bench_startup_serves_matching_completion_after_restart(tmp_path):
    port = get_open_port()
    output = tmp_path / "startup.json"
    cmd = [
        "vllm",
        "bench",
        "startup",
        "--serve-cmd",
        f"vllm serve facebook/opt-125m --enforce-eager --port {port}",
        "--num-iters-cold",
        "1",
        "--num-iters-warmup",
        "0",
        "--num-iters-warm",
        "1",
        "--output-json",
        str(output),
    ]
    subprocess.run(cmd, check=True, timeout=900)
    results = json.loads(output.read_text())
    for phase in ("cold", "warm"):
        assert results[f"{phase}_verified_starts"] == 1
        (ready,) = results[f"{phase}_ready_times"]
        (first_token,) = results[f"{phase}_first_token_times"]
        assert 0 < ready < first_token
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
