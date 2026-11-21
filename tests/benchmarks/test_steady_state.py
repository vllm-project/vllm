# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steady-state metrics for `vllm bench serve`; same cases as the Rust
benchmark's `metrics/steady_state.rs` tests, plus the closed-loop gate."""

import json
import math
from dataclasses import asdict

import numpy as np
import pytest

from vllm.benchmarks.lib import steady_state
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput


def mk(start, latency, ttft=0.0, itl=None, output_tokens=0, prompt_len=10, ok=True):
    return RequestFuncOutput(
        success=ok,
        start_time=start,
        latency=latency,
        ttft=ttft,
        itl=itl or [],
        output_tokens=output_tokens,
        prompt_len=prompt_len,
    )


def detect(outputs, target, threshold, min_window_s, duration_s):
    w = steady_state.detect_window(outputs, target, threshold, min_window_s, duration_s)
    assert w is not None
    return w


def test_window_opens_at_up_crossing_and_closes_at_last_down_crossing():
    # 10 requests start at t=0..9 and last 30s; one more runs 15-20s.
    outputs = [mk(float(i), 30.0) for i in range(10)] + [mk(15.0, 5.0)]
    w = detect(outputs, 10, 1.0, 5.0, 30.0)
    assert (w.threshold_abs, w.start_s, w.end_s, w.observed_peak) == (10, 9.0, 30.0, 11)
    assert w.warning is None


def test_window_none_when_threshold_never_reached():
    outputs = [mk(i * 10.0, 1.0) for i in range(5)]
    assert steady_state.detect_window(outputs, 5, 0.95, 1.0, 100.0) is None


def test_window_none_when_all_requests_failed():
    assert (
        steady_state.detect_window([mk(0.0, 1.0, ok=False)], 1, 1.0, 0.1, 10.0) is None
    )


def test_window_ties_count_ends_first():
    # Two requests end at t=5 as a third starts: concurrency never reaches 3.
    outputs = [mk(0.0, 5.0), mk(2.0, 3.0), mk(5.0, 2.0)]
    assert detect(outputs, 2, 1.0, 0.1, 20.0).observed_peak == 2


def test_failed_requests_are_ignored():
    outputs = [mk(0.0, 10.0), mk(0.0, 10.0), mk(1.0, 1.0, ok=False)]
    w = detect(outputs, 2, 1.0, 0.1, 10.0)
    assert (w.observed_peak, w.requests_total) == (2, 2)


def test_window_warns_when_shorter_than_min_window():
    outputs = [mk(i * 0.1, 1.0) for i in range(10)]
    assert detect(outputs, 10, 1.0, 5.0, 10.0).warning


def test_constant_concurrency_counts_tokens_in_half_open_window():
    # Tokens at t=2,4,6,8,10; the one at t=10 is outside [0, 10).
    outputs = [mk(0.0, 10.0, 2.0, [2.0] * 4, 5) for _ in range(3)]
    w = detect(outputs, 3, 1.0, 0.1, 10.0)
    m = steady_state.compute(outputs, w, [99.0])
    assert (w.start_s, w.end_s, w.requests_started_in_window) == (0.0, 10.0, 3)
    assert m.request_throughput == 0.0  # completions at t=10 are outside too
    assert m.output_throughput == pytest.approx(1.2)
    assert m.input_throughput == pytest.approx(3.0)
    assert m.mean_tpot_ms == pytest.approx(2000.0)
    assert m.p99_tpot_ms == pytest.approx(2000.0)


def test_ttft_and_tpot_stats_match_numpy():
    ttfts = [0.1, 0.4, 0.2, 0.9, 0.3]
    outputs = [mk(0.0, 10.0, t, [1.0] * 4, 5) for t in ttfts]
    w = detect(outputs, 5, 1.0, 0.1, 10.0)
    m = steady_state.compute(outputs, w, [50.0, 90.0, 99.0])
    ms = np.array(ttfts) * 1000
    assert m.mean_ttft_ms == pytest.approx(ms.mean())
    assert m.median_ttft_ms == pytest.approx(np.median(ms))
    assert [p for p, _ in m.percentiles_ttft_ms] == [50.0, 90.0, 99.0]
    assert [v for _, v in m.percentiles_ttft_ms] == pytest.approx(
        np.percentile(ms, [50.0, 90.0, 99.0])
    )
    tpots = [(10.0 - t) / 4 * 1000 for t in ttfts]
    assert m.p90_tpot_ms == pytest.approx(np.percentile(tpots, 90))


def test_multi_token_events_count_all_tokens():
    # 10 tokens in 5 stream events (2 per event, e.g. speculative decoding).
    outputs = [mk(0.0, 20.0, 1.0, [1.0] * 4, 10) for _ in range(2)]
    outputs += [mk(0.0, 20.0, 1.0, [1.0] * 4, None)]  # no usage reported
    w = detect(outputs, 3, 1.0, 0.1, 20.0)
    m = steady_state.compute(outputs, w, [99.0])
    assert m.output_throughput == pytest.approx((10 + 10 + 5) / 20.0)


def test_tokens_emitted_after_window_are_not_counted():
    # A request admitted at 9.9s emits most of its 1000 tokens after the window
    # closes at 10s.
    outputs = [
        mk(0.0, 10.0, 0.0, [1.0] * 9, 10),
        mk(9.9, 30.0, 0.05, [0.02] * 999, 1000),
    ]
    w = detect(outputs, 2, 1.0, 0.1, 40.0)
    m = steady_state.compute(outputs, w, [99.0])
    assert m.output_throughput <= (10 + 6) / w.duration_s


def test_pooling_reports_no_ttft_or_tpot():
    outputs = [mk(0.0, 5.0, 5.0), mk(0.0, 5.0, 5.0)]
    w = detect(outputs, 2, 1.0, 0.1, 10.0)
    m = steady_state.compute(outputs, w, [99.0], is_pooling=True)
    assert (m.mean_ttft_ms, m.percentiles_ttft_ms, m.mean_tpot_ms, m.p99_tpot_ms) == (
        0.0,
        [],
        0.0,
        0.0,
    )
    assert m.input_throughput > 0


def test_trapezoid_run_json_matches_rust_layout():
    outputs = [mk(float(i), 20.0, 1.0, [1.0] * 9, 10) for i in range(5)]
    w = detect(outputs, 5, 1.0, 1.0, 25.0)
    m = steady_state.compute(outputs, w, [50.0, 99.0])
    assert (w.start_s, w.end_s) == (4.0, 20.0)
    d = json.loads(json.dumps(asdict(m)))
    assert d["window"]["target_concurrency"] == 5
    assert [p for p, _ in d["percentiles_ttft_ms"]] == [50.0, 99.0]
    assert set(d) == {
        "window",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "percentiles_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p90_tpot_ms",
        "p99_tpot_ms",
    }


@pytest.mark.parametrize(
    "max_concurrency, request_rate, ramp_up, self_timed, enabled, expected",
    [
        (5, math.inf, False, False, True, True),
        (None, math.inf, False, False, True, False),  # open loop
        (5, 10.0, False, False, True, False),  # finite request rate
        (5, math.inf, True, False, True, False),  # --ramp-up-strategy
        (5, math.inf, False, True, True, False),  # replayed trace
        (5, math.inf, False, False, False, False),  # --no-steady-state
    ],
)
def test_for_run_only_in_closed_loop(
    max_concurrency, request_rate, ramp_up, self_timed, enabled, expected
):
    outputs = [mk(float(i), 20.0, 1.0, [1.0] * 9, 10) for i in range(5)]
    ss = steady_state.for_run(
        outputs,
        max_concurrency=max_concurrency,
        request_rate=request_rate,
        ramp_up=ramp_up,
        self_timed=self_timed,
        enabled=enabled,
        threshold=1.0,
        min_window_s=None,
        duration_s=25.0,
        percentiles=[99.0],
        is_pooling=False,
    )
    assert (ss is not None) == expected
    if ss is not None:
        # 16s window, above the default min window of max(10, 0.1 * 25) s
        assert ss.window.warning is None
