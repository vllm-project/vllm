# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.benchmarks.lib.utils import calculate_concurrency


def _concurrency(
    intervals: list[tuple[float, float]],
) -> tuple[int, list[float], list[int]]:
    min_start = min((start for start, _ in intervals), default=0.0)
    return calculate_concurrency(intervals, min_start)


def test_concurrency_uses_half_open_intervals() -> None:
    peak, _, _ = _concurrency([(0.0, 1.0), (1.0, 2.0)])

    assert peak == 1


def test_concurrency_counts_overlapping_requests() -> None:
    peak, _, _ = _concurrency([(0.0, 1.5), (1.0, 2.0)])

    assert peak == 2


def test_concurrency_does_not_bucket_sequential_subsecond_requests() -> None:
    peak, _, _ = _concurrency(
        [(0.0, 0.1), (0.2, 0.3), (0.4, 0.5), (0.6, 0.7), (0.8, 0.9)]
    )

    assert peak == 1


def test_concurrency_groups_simultaneous_departures_and_arrivals() -> None:
    peak, times, levels = _concurrency(
        [(0.0, 1.0), (0.0, 1.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0)]
    )

    assert peak == 3
    at_boundary = [level for time, level in zip(times, levels) if time == 1.0]
    assert at_boundary == [2, 3]


def test_concurrency_treats_zero_latency_as_empty_interval() -> None:
    peak, times, levels = _concurrency([(0.0, 0.0)])

    assert peak == 0
    assert times == []
    assert levels == []
