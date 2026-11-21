# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steady-state metrics for closed-loop runs.

Throughput, TTFT and TPOT over the window from when client-side in-flight
requests first reach a fraction of `--max-concurrency` to when they last drop
below it, so the startup burst and the drain, whose share depends on
`--num-prompts`, are left out. Port of the Rust benchmark's
`metrics/steady_state.rs`, with the same JSON layout.
"""

import math
from dataclasses import dataclass

import numpy as np

from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput


@dataclass
class SteadyStateWindow:
    start_s: float
    end_s: float
    duration_s: float
    target_concurrency: int
    threshold: float
    threshold_abs: int
    observed_peak: int
    requests_started_in_window: int
    requests_completed_in_window: int
    requests_total: int
    warning: str | None


@dataclass
class SteadyStateMetrics:
    window: SteadyStateWindow
    request_throughput: float
    input_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float


def detect_window(
    outputs: list[RequestFuncOutput],
    target_concurrency: int,
    threshold: float,
    min_window_s: float,
    total_run_duration_s: float,
) -> SteadyStateWindow | None:
    """Find the window in which in-flight requests stay at or above
    `ceil(threshold * target_concurrency)`.

    The window opens at the first up-crossing of that level and closes at the
    last down-crossing. Times are seconds from the earliest successful start.
    Returns None if concurrency never reaches the level or no request starts
    or completes inside the window.
    """
    threshold_abs = max(math.ceil(threshold * target_concurrency), 1)
    succeeded = [o for o in outputs if o.success]
    if not succeeded:
        return None
    starts = sorted(o.start_time for o in succeeded)
    ends = sorted(o.start_time + o.latency for o in succeeded)
    min_start = starts[0]

    # Merge start and end events in time order; at equal times, ends go first.
    i = j = concurrency = observed_peak = 0
    up_crossing: float | None = None
    last_down_crossing: float | None = None
    while i < len(starts) or j < len(ends):
        if i == len(starts) or (j < len(ends) and ends[j] <= starts[i]):
            before = concurrency
            concurrency -= 1
            if before >= threshold_abs > concurrency:
                last_down_crossing = ends[j]
            j += 1
        else:
            before = concurrency
            concurrency += 1
            observed_peak = max(observed_peak, concurrency)
            if up_crossing is None and before < threshold_abs <= concurrency:
                up_crossing = starts[i]
            i += 1

    if up_crossing is None:
        return None
    start_abs = up_crossing
    end_abs = last_down_crossing if last_down_crossing is not None else ends[-1]

    started = sum(start_abs <= o.start_time < end_abs for o in succeeded)
    completed = sum(start_abs <= o.start_time + o.latency < end_abs for o in succeeded)
    if started == 0 and completed == 0:
        return None

    duration_s = end_abs - start_abs
    warning = None
    if duration_s < min_window_s:
        warning = (
            f"steady-state window is {duration_s:.1f}s "
            f"({100.0 * duration_s / max(total_run_duration_s, 1e-9):.1f}% of run); "
            "may not be representative"
        )
    return SteadyStateWindow(
        start_s=start_abs - min_start,
        end_s=end_abs - min_start,
        duration_s=duration_s,
        target_concurrency=target_concurrency,
        threshold=threshold,
        threshold_abs=threshold_abs,
        observed_peak=observed_peak,
        requests_started_in_window=started,
        requests_completed_in_window=completed,
        requests_total=len(succeeded),
        warning=warning,
    )


def _dist_stats(
    values: list[float], percentiles: list[float]
) -> tuple[float, float, list[tuple[float, float]]]:
    """Mean, median and percentiles in ms (seconds in); zeros when empty."""
    if not values:
        return 0.0, 0.0, [(p, 0.0) for p in percentiles]
    ms = np.asarray(values) * 1000.0
    pct = [(p, float(np.percentile(ms, p))) for p in percentiles]
    return float(np.mean(ms)), float(np.median(ms)), pct


def compute(
    outputs: list[RequestFuncOutput],
    window: SteadyStateWindow,
    percentiles: list[float],
    is_pooling: bool = False,
) -> SteadyStateMetrics:
    """Metrics over `window`.

    Requests completed in the window count toward request throughput; requests
    started in it contribute input tokens, TTFT and TPOT; output tokens count
    when their stream event lands in the window, with a request's tokens split
    evenly over its events, since one event can carry several tokens. Pooling
    backends report TTFT equal to latency, so TTFT and TPOT are zero for them.
    """
    succeeded = [o for o in outputs if o.success]
    min_start = min(o.start_time for o in succeeded)
    start_abs = window.start_s + min_start
    end_abs = window.end_s + min_start
    duration = max(window.duration_s, 1e-9)

    completed = input_tokens = 0
    output_tokens = 0.0
    ttfts: list[float] = []
    tpots: list[float] = []
    for o in succeeded:
        if start_abs <= o.start_time + o.latency < end_abs:
            completed += 1
        if start_abs <= o.start_time < end_abs:
            input_tokens += o.prompt_len
            if not is_pooling:
                ttfts.append(o.ttft)
                if (o.output_tokens or 0) > 1:
                    tpots.append((o.latency - o.ttft) / (o.output_tokens - 1))
        num_events = 1 + len(o.itl)
        tokens_per_event = (o.output_tokens or num_events) / num_events
        # First event at start + ttft, then one per inter-token gap.
        t = o.start_time + o.ttft
        events_in_window = int(start_abs <= t < end_abs)
        for dt in o.itl:
            t += dt
            if t >= end_abs:
                break
            events_in_window += t >= start_abs
        output_tokens += events_in_window * tokens_per_event

    pct_ttft: list[tuple[float, float]] = []
    pct_tpot: list[tuple[float, float]] = []
    if is_pooling:
        mean_ttft = median_ttft = mean_tpot = median_tpot = 0.0
    else:
        mean_ttft, median_ttft, pct_ttft = _dist_stats(ttfts, percentiles)
        mean_tpot, median_tpot, pct_tpot = _dist_stats(tpots, [90.0, 99.0])
    input_throughput = input_tokens / duration
    output_throughput = output_tokens / duration
    return SteadyStateMetrics(
        window=window,
        request_throughput=completed / duration,
        input_throughput=input_throughput,
        output_throughput=output_throughput,
        total_token_throughput=input_throughput + output_throughput,
        mean_ttft_ms=mean_ttft,
        median_ttft_ms=median_ttft,
        percentiles_ttft_ms=pct_ttft,
        mean_tpot_ms=mean_tpot,
        median_tpot_ms=median_tpot,
        p90_tpot_ms=pct_tpot[0][1] if pct_tpot else 0.0,
        p99_tpot_ms=pct_tpot[1][1] if pct_tpot else 0.0,
    )


def for_run(
    outputs: list[RequestFuncOutput],
    *,
    max_concurrency: int | None,
    request_rate: float,
    ramp_up: bool,
    self_timed: bool,
    enabled: bool,
    threshold: float,
    min_window_s: float | None,
    duration_s: float,
    percentiles: list[float],
    is_pooling: bool,
) -> SteadyStateMetrics | None:
    """Steady-state metrics for a finished run, or None outside closed-loop runs.

    With a finite or ramping request rate or a replayed trace, an in-flight
    plateau reflects the arrival pattern, not a saturated server, so only runs
    with `--max-concurrency` and a constant `--request-rate inf` qualify.
    """
    if (
        not enabled
        or self_timed
        or ramp_up
        or max_concurrency is None
        or not math.isinf(request_rate)
    ):
        return None
    if min_window_s is None:
        min_window_s = max(10.0, 0.1 * duration_s)
    window = detect_window(
        outputs, max_concurrency, threshold, min_window_s, duration_s
    )
    if window is None:
        return None
    return compute(outputs, window, percentiles, is_pooling)


def print_metrics(ss: SteadyStateMetrics) -> None:
    w = ss.window
    if w.warning:
        print(f"Warning: {w.warning}")
    print("{s:{c}^{n}}".format(s=" Steady-State Metrics ", n=50, c="="))
    print(
        "{:<40} >= {:.2f} * {} = {}".format(
            "Concurrency threshold:", w.threshold, w.target_concurrency, w.threshold_abs
        )
    )
    print(
        "{:<40} {:.1f}s -> {:.1f}s ({:.1f}s)".format(
            "Window:", w.start_s, w.end_s, w.duration_s
        )
    )
    print("{:<40} {:<10}".format("Observed peak concurrency:", w.observed_peak))
    print(
        "{:<40} {} / {} ({:.1f}%)".format(
            "Requests started in window:",
            w.requests_started_in_window,
            w.requests_total,
            100.0 * w.requests_started_in_window / max(w.requests_total, 1),
        )
    )
    print(
        "{:<40} {:<10}".format(
            "Requests completed in window:", w.requests_completed_in_window
        )
    )
    print("-" * 50)
    print(
        "{:<40} {:<10.2f}".format("Request throughput (req/s):", ss.request_throughput)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", ss.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", ss.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", ss.total_token_throughput
        )
    )
    if ss.percentiles_ttft_ms or ss.mean_ttft_ms > 0:
        print("-" * 50)
        print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", ss.mean_ttft_ms))
        print("{:<40} {:<10.2f}".format("Median TTFT (ms):", ss.median_ttft_ms))
        for p, v in ss.percentiles_ttft_ms:
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} TTFT (ms):", v))
    if ss.mean_tpot_ms > 0 or ss.median_tpot_ms > 0:
        print("-" * 50)
        print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", ss.mean_tpot_ms))
        print("{:<40} {:<10.2f}".format("Median TPOT (ms):", ss.median_tpot_ms))
        print("{:<40} {:<10.2f}".format("P90 TPOT (ms):", ss.p90_tpot_ms))
        print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", ss.p99_tpot_ms))
    print("=" * 50)
