# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.platforms import current_platform


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_rocm()),
    reason="The FP8 tuning script requires CUDA or ROCm.",
)
@pytest.mark.parametrize("num_iters", [1, 3, 10])
def test_benchmark_config_returns_mean_latency_in_microseconds(monkeypatch, num_iters):
    from benchmarks.kernels import benchmark_w8a8_block_fp8 as benchmark

    start_event, end_event = Mock(), Mock()
    start_event.elapsed_time.side_effect = [0.25 * (i + 1) for i in range(num_iters)]
    monkeypatch.setattr(
        benchmark.torch, "Event", Mock(side_effect=[start_event, end_event])
    )
    monkeypatch.setattr(benchmark.torch.accelerator, "synchronize", Mock())
    matmul = Mock()
    monkeypatch.setattr(benchmark, "w8a8_block_matmul", matmul)

    latency_us = benchmark.benchmark_config(
        None, None, None, None, [128, 128], {}, num_iters=num_iters
    )

    assert latency_us == pytest.approx(125 * (num_iters + 1))
    assert matmul.call_count == 5 + num_iters  # Warmup is outside the event intervals.
    assert start_event.record.call_count == num_iters
    assert end_event.record.call_count == num_iters
    assert start_event.elapsed_time.call_count == num_iters
