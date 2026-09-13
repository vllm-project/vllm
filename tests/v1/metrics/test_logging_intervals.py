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


def test_perf_metrics_empty_interval_resets_the_debug_window_too(monkeypatch):
    """The clock and the debug accumulators have to restart together.

    ``PerfMetricsDebugLogging.log()`` is handed ``delta_time`` and reports
    ``total_calc_duration / delta_time`` as ``mfu_calc_overhead``, plus
    ``delta_time`` itself as ``duration``. ``observe()`` bumps the debug
    counters for every batch, including batches that produce no flops or bytes,
    so the empty-interval branch can be reached with a non-empty debug window.

    Restarting only the clock would leave those totals to be divided by a later,
    shorter interval, which inflates ``mfu_calc_overhead`` and understates
    ``duration`` - the same failure this module is fixing, one level down.
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
    # ...and the debug window restarted with it, so nothing is carried into an
    # interval it did not happen in.
    assert perf_logging.debug_logging.total_num_batches == 0
    assert perf_logging.debug_logging.total_num_prefill_requests == 0
    assert perf_logging.debug_logging.total_num_decode_requests == 0
    assert perf_logging.debug_logging.total_calc_duration == pytest.approx(0.0)


def test_perf_metrics_debug_overhead_not_inflated_after_idle(monkeypatch):
    """End to end: mfu_calc_overhead must describe the interval it is reported in.

    PerfMetricsDebugLogging.log() writes through the module logger rather than
    the caller's log_fn, so capture logger.debug.
    """
    clock = FakeClock()
    monkeypatch.setattr(perf_module, "time", clock)

    captured: list[str] = []

    class _CapturingLogger:
        def debug(self, fmt, *args):
            captured.append(fmt % args if args else fmt)

        def __getattr__(self, _name):
            return lambda *a, **k: None

    monkeypatch.setattr(perf_module, "logger", _CapturingLogger())

    perf_logging = _make_perf_logging(monkeypatch, debug=True)

    def observe(calc_duration, flops):
        perf_logging.observe(
            PerfStats(
                num_flops_per_gpu=flops,
                num_read_bytes_per_gpu=0,
                num_write_bytes_per_gpu=0,
                debug_stats=DebugPerfStats(
                    calc_duration=calc_duration,
                    num_prefill_requests=0,
                    num_decode_requests=0,
                    context_breakdown={},
                    num_flops_per_gpu_breakdown={},
                    num_read_bytes_per_gpu_breakdown={},
                    num_write_bytes_per_gpu_breakdown={},
                ),
            )
        )

    # An interval of zero-flop batches, so log() takes the empty branch.
    observe(calc_duration=9.0, flops=0)
    clock.advance(IDLE_SECONDS)
    perf_logging.log(log_fn=_fail_if_called)

    # A short busy interval: 0.1s of calc over ACTIVE_SECONDS is 10%.
    observe(calc_duration=0.1, flops=10**12)
    clock.advance(ACTIVE_SECONDS)
    perf_logging.log(log_fn=lambda *args: None)

    assert captured, "the debug logger was never called"
    payload = captured[-1]
    assert f'"duration": "{ACTIVE_SECONDS:.1f}s"' in payload
    assert '"mfu_calc_overhead": "10.0%"' in payload
