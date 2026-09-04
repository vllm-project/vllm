# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression tests for logging-interval accounting in the periodic stat loggers.

``SpecDecodingLogging.log()`` and ``PerfMetricsLogging.log()`` both return early
when nothing was observed during the interval. That early return used to leave
``last_log_time`` pointing at the previous *non-empty* interval, so the first
interval with traffic after an idle stretch divided its token/flop counts by the
whole idle gap and under-reported throughput by the ratio of the idle gap to the
real interval.

These tests drive both loggers with a deterministic fake clock and assert that an
idle interval does not contaminate the interval that follows it.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.metrics import perf as perf_module
from vllm.v1.metrics.perf import DebugPerfStats, PerfMetricsLogging, PerfStats
from vllm.v1.spec_decode import metrics as spec_decode_module
from vllm.v1.spec_decode.metrics import SpecDecodingLogging, SpecDecodingStats

# A long idle stretch followed by a short busy one. If the idle gap leaks into
# the next interval the reported rates are wrong by ~101x, which is far larger
# than any plausible timing jitter.
IDLE_SECONDS = 100.0
ACTIVE_SECONDS = 1.0


class FakeClock:
    """Stands in for the ``time`` module inside the logger under test."""

    def __init__(self, now: float = 0.0):
        self._now = now

    def monotonic(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _fail_if_called(*args, **kwargs):
    raise AssertionError("log_fn must not be called for an empty interval")


def _make_spec_decoding_stats(
    num_spec_tokens: int,
    num_drafts: int,
    draft_tokens_per_draft: int,
    accepted_tokens_per_draft: int,
) -> SpecDecodingStats:
    stats = SpecDecodingStats.new(num_spec_tokens=num_spec_tokens)
    for _ in range(num_drafts):
        stats.observe_draft(
            num_draft_tokens=draft_tokens_per_draft,
            num_accepted_tokens=accepted_tokens_per_draft,
        )
    return stats


def test_spec_decoding_empty_interval_does_not_skew_next_throughput(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(spec_decode_module, "time", clock)

    spec_logging = SpecDecodingLogging()

    # A long stretch with no spec-decode traffic. log() has nothing to report.
    clock.advance(IDLE_SECONDS)
    spec_logging.log(log_fn=_fail_if_called)

    # A short, busy interval: 100 drafts x 4 draft tokens, 3 accepted each.
    spec_logging.observe(
        _make_spec_decoding_stats(
            num_spec_tokens=4,
            num_drafts=100,
            draft_tokens_per_draft=4,
            accepted_tokens_per_draft=3,
        )
    )
    clock.advance(ACTIVE_SECONDS)

    records = []
    spec_logging.log(log_fn=lambda *args: records.append(args))

    assert len(records) == 1
    _, _mean_acceptance_length, accepted_throughput, draft_throughput = records[0][:4]

    # 300 accepted / 400 drafted tokens over ACTIVE_SECONDS, not over
    # IDLE_SECONDS + ACTIVE_SECONDS.
    assert accepted_throughput == pytest.approx(300 / ACTIVE_SECONDS)
    assert draft_throughput == pytest.approx(400 / ACTIVE_SECONDS)


def test_spec_decoding_busy_interval_unaffected(monkeypatch):
    """Control: the ordinary back-to-back case must keep working."""
    clock = FakeClock()
    monkeypatch.setattr(spec_decode_module, "time", clock)

    spec_logging = SpecDecodingLogging()
    spec_logging.observe(
        _make_spec_decoding_stats(
            num_spec_tokens=4,
            num_drafts=100,
            draft_tokens_per_draft=4,
            accepted_tokens_per_draft=3,
        )
    )
    clock.advance(ACTIVE_SECONDS)

    records = []
    spec_logging.log(log_fn=lambda *args: records.append(args))

    assert len(records) == 1
    accepted_throughput, draft_throughput = records[0][2:4]
    assert accepted_throughput == pytest.approx(300 / ACTIVE_SECONDS)
    assert draft_throughput == pytest.approx(400 / ACTIVE_SECONDS)


def _make_perf_logging(monkeypatch, debug: bool) -> PerfMetricsLogging:
    monkeypatch.setattr(perf_module.envs, "VLLM_DEBUG_MFU_METRICS", debug)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1)
    )
    return PerfMetricsLogging(vllm_config)


def test_perf_metrics_empty_interval_does_not_skew_next_rates(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(perf_module, "time", clock)

    perf_logging = _make_perf_logging(monkeypatch, debug=False)

    # A long stretch where the engine did no work.
    clock.advance(IDLE_SECONDS)
    perf_logging.log(log_fn=_fail_if_called)

    perf_logging.observe(
        PerfStats(
            num_flops_per_gpu=2 * 10**12,
            num_read_bytes_per_gpu=2 * 10**9,
            num_write_bytes_per_gpu=1 * 10**9,
        )
    )
    clock.advance(ACTIVE_SECONDS)

    records = []
    perf_logging.log(log_fn=lambda *args: records.append(args))

    assert len(records) == 1
    _, _prefix, avg_tflops_per_gpu, avg_gbps_per_gpu = records[0][:4]

    assert avg_tflops_per_gpu == pytest.approx(2.0 / ACTIVE_SECONDS)
    assert avg_gbps_per_gpu == pytest.approx(3.0 / ACTIVE_SECONDS)


def test_perf_metrics_empty_interval_keeps_debug_accumulators(monkeypatch):
    """The empty-interval branch must restart the clock *only*.

    ``PerfMetricsLogging.reset()`` also zeroes the debug accumulators, which can
    be non-empty here: ``observe()`` bumps ``total_num_batches`` and friends for
    every batch, including batches that contribute no flops or bytes. Those
    observations belong to the window that is still accumulating, so the
    empty-interval branch must not discard them.
    """
    clock = FakeClock()
    monkeypatch.setattr(perf_module, "time", clock)

    perf_logging = _make_perf_logging(monkeypatch, debug=True)
    assert perf_logging.debug_logging is not None

    # A batch that did no flops but still cost wall-clock time to evaluate.
    perf_logging.observe(
        PerfStats(
            num_flops_per_gpu=0,
            num_read_bytes_per_gpu=0,
            num_write_bytes_per_gpu=0,
            debug_stats=DebugPerfStats(
                calc_duration=0.25,
                num_prefill_requests=1,
                num_decode_requests=2,
                context_breakdown={},
                num_flops_per_gpu_breakdown={},
                num_read_bytes_per_gpu_breakdown={},
                num_write_bytes_per_gpu_breakdown={},
            ),
        )
    )
    assert perf_logging.debug_logging.total_num_batches == 1

    clock.advance(IDLE_SECONDS)
    perf_logging.log(log_fn=_fail_if_called)

    # Clock restarted...
    assert perf_logging.last_log_time == pytest.approx(IDLE_SECONDS)
    # ...but the debug window was not thrown away.
    assert perf_logging.debug_logging.total_num_batches == 1
    assert perf_logging.debug_logging.total_num_prefill_requests == 1
    assert perf_logging.debug_logging.total_num_decode_requests == 2
    assert perf_logging.debug_logging.total_calc_duration == pytest.approx(0.25)
