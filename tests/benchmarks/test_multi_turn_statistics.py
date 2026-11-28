# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path


def test_multi_turn_runtime_ends_at_last_retained_completion(monkeypatch, capsys):
    script_dir = Path(__file__).resolve().parents[2] / "benchmarks" / "multi_turn"
    monkeypatch.syspath_prepend(str(script_dir))
    from benchmark_serving_multi_turn import RequestStats, process_statistics

    slow = RequestStats(
        ttft_ms=10,
        tpot_ms=10,
        latency_ms=5000,
        start_time_ms=1000,
        input_num_turns=1,
        input_num_tokens=4,
        output_num_tokens=2,
        output_num_chunks=2,
        output_num_first_chunk_tokens=1,
        approx_cached_percent=0,
        conversation_id="slow",
        client_id=0,
    )
    fast = slow._replace(
        start_time_ms=2000, latency_ms=1000, conversation_id="fast", client_id=1
    )
    process_statistics([slow, fast], [0.0, 0.5], {}, False)

    lines = capsys.readouterr().out.splitlines()
    assert [line for line in lines if line.startswith("runtime_sec =")] == [
        "runtime_sec = 5.000",
        "runtime_sec = 1.000",
    ]
    assert [line for line in lines if line.startswith("requests_per_sec =")] == [
        "requests_per_sec = 0.400",
        "requests_per_sec = 1.000",
    ]
