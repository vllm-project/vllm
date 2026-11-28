# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace

import pytest

from vllm.benchmarks import serve
from vllm.benchmarks.datasets import SampleRequest


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("wall_clock_jump", [-5.0, 5.0], ids=["backward", "forward"])
@pytest.mark.parametrize("self_timed", [False, True], ids=["fixed-rate", "trace"])
def test_request_pacing_ignores_wall_clock_jumps(
    monkeypatch, wall_clock_jump, self_timed
):
    offsets = [0.0, 0.25, 0.5]
    requests = [
        SampleRequest(prompt="hello", prompt_len=1, expected_output_len=1, timestamp=t)
        for t in offsets
    ]
    elapsed = 0.0
    wall_offset = 0.0

    async def sleep(delay: float) -> None:
        nonlocal elapsed
        elapsed += delay

    monkeypatch.setattr(
        serve,
        "time",
        SimpleNamespace(
            time=lambda: 100.0 + elapsed + wall_offset, perf_counter=lambda: elapsed
        ),
    )
    monkeypatch.setattr(serve, "asyncio", SimpleNamespace(sleep=sleep))

    async def arrivals() -> list[float]:
        nonlocal wall_offset
        send_times = []
        async for _ in serve.get_request(
            requests, request_rate=4.0, burstiness=float("inf"), self_timed=self_timed
        ):
            send_times.append(elapsed)
            wall_offset = wall_clock_jump
        return send_times

    first_arrival = 0.0 if self_timed else 0.25
    assert asyncio.run(arrivals()) == [first_arrival + offset for offset in offsets]
