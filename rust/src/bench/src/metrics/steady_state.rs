// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Steady-state metrics: throughput and TTFT measured over the window
//! during which client-side in-flight concurrency is at or above a
//! configurable fraction of `--max-concurrency`. Excludes ramp-up and
//! drain phases to reduce run-to-run variance at very high concurrency.

use serde::{Deserialize, Serialize};

/// Bounds and metadata of the detected steady-state window.
///
/// `observed_peak` is event-exact and may differ by 1 from the
/// bucket-approximated `max_concurrent_requests` on `BenchmarkMetrics`;
/// this is expected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteadyStateWindow {
    pub start_s: f64,
    pub end_s: f64,
    pub duration_s: f64,
    pub target_concurrency: usize,
    pub threshold: f64,
    pub threshold_abs: usize,
    pub observed_peak: usize,
    pub requests_started_in_window: usize,
    pub requests_completed_in_window: usize,
    pub requests_total: usize,
    pub warning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteadyStateMetrics {
    pub window: SteadyStateWindow,
    pub request_throughput: f64,
    pub input_throughput: f64,
    pub output_throughput: f64,
    pub total_token_throughput: f64,
    pub mean_ttft_ms: f64,
    pub median_ttft_ms: f64,
    pub percentiles_ttft_ms: Vec<(f64, f64)>,
    pub mean_tpot_ms: f64,
    pub median_tpot_ms: f64,
    pub p90_tpot_ms: f64,
    pub p99_tpot_ms: f64,
}

use crate::backends::RequestFuncOutput;

/// Detect the steady-state window over `outputs` under the given target concurrency.
///
/// Returns `None` when closed-loop gate is not met (no target), when concurrency
/// never crosses `threshold_abs`, or when both started and completed in-window
/// request counts are zero.
///
/// Timestamps in the returned window are normalized so `start_s` / `end_s` are
/// "seconds from the earliest successful request's start_time."
///
/// `user_min_window_s`: if `Some`, sets the minimum window duration below which
/// a `warning` is attached. If `None`, the caller is expected to have pre-resolved
/// the default `max(10.0, 0.1 * total_run_duration_s)`.
pub fn detect_window(
    outputs: &[RequestFuncOutput],
    target_concurrency: Option<usize>,
    threshold: f64,
    min_window_s: f64,
    total_run_duration_s: f64,
) -> Option<SteadyStateWindow> {
    let target = target_concurrency?;
    let threshold_abs = ((threshold * target as f64).ceil() as usize).max(1);

    // Collect sorted start / end timestamps from successful requests only.
    let mut starts: Vec<f64> = outputs.iter().filter(|o| o.success).map(|o| o.start_time).collect();
    let mut ends: Vec<f64> =
        outputs.iter().filter(|o| o.success).map(|o| o.start_time + o.latency).collect();
    if starts.is_empty() {
        return None;
    }
    starts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ends.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min_start = starts[0];
    let requests_total = outputs.iter().filter(|o| o.success).count();

    // Two-pointer merge. Tie rule: process `end` before `start` at equal timestamps.
    let mut i = 0usize;
    let mut j = 0usize;
    let mut concurrency: usize = 0;
    let mut observed_peak: usize = 0;
    let mut up_crossing: Option<f64> = None;
    let mut last_down_crossing: Option<f64> = None;

    while i < starts.len() || j < ends.len() {
        let next_is_end = match (starts.get(i), ends.get(j)) {
            (Some(&s), Some(&e)) => e <= s, // tie -> end first
            (None, Some(_)) => true,
            (Some(_), None) => false,
            (None, None) => break,
        };

        if next_is_end {
            let t = ends[j];
            j += 1;
            let before = concurrency;
            concurrency = concurrency.saturating_sub(1);
            if before >= threshold_abs && concurrency < threshold_abs {
                last_down_crossing = Some(t);
            }
        } else {
            let t = starts[i];
            i += 1;
            let before = concurrency;
            concurrency += 1;
            if concurrency > observed_peak {
                observed_peak = concurrency;
            }
            if before < threshold_abs && concurrency >= threshold_abs && up_crossing.is_none() {
                up_crossing = Some(t);
            }
        }
    }

    let start_abs = up_crossing?;
    // A down-crossing always exists once an up-crossing exists — the final ends
    // drain concurrency to 0. Fall back defensively to the last end timestamp.
    let end_abs = last_down_crossing.or_else(|| ends.last().copied())?;

    let start_s = start_abs - min_start;
    let end_s = end_abs - min_start;
    let duration_s = end_s - start_s;

    // Count requests whose start / completion events land in [start_abs, end_abs).
    let mut requests_started_in_window = 0usize;
    let mut requests_completed_in_window = 0usize;
    for o in outputs.iter().filter(|o| o.success) {
        if o.start_time >= start_abs && o.start_time < end_abs {
            requests_started_in_window += 1;
        }
        let end_t = o.start_time + o.latency;
        if end_t >= start_abs && end_t < end_abs {
            requests_completed_in_window += 1;
        }
    }

    if requests_started_in_window == 0 && requests_completed_in_window == 0 {
        return None;
    }

    let warning = if duration_s < min_window_s {
        Some(format!(
            "steady-state window is {:.1}s ({:.1}% of run); may not be representative",
            duration_s,
            100.0 * duration_s / total_run_duration_s.max(1e-9)
        ))
    } else {
        None
    };

    Some(SteadyStateWindow {
        start_s,
        end_s,
        duration_s,
        target_concurrency: target,
        threshold,
        threshold_abs,
        observed_peak,
        requests_started_in_window,
        requests_completed_in_window,
        requests_total,
        warning,
    })
}

use crate::datasets::SampleRequest;

/// Compute steady-state metrics from per-request outputs, the detected window,
/// and the original request list (for input-token attribution).
///
/// `is_pooling` gates off TTFT because pooling backends write `ttft = latency`
/// as a placeholder; reporting TTFT over that would silently be E2EL.
pub fn compute(
    outputs: &[RequestFuncOutput],
    input_requests: &[SampleRequest],
    window: &SteadyStateWindow,
    percentiles: &[f64],
    is_pooling: bool,
) -> SteadyStateMetrics {
    // Re-derive min_start to convert normalized window bounds back to absolute times.
    let min_start = outputs
        .iter()
        .filter(|o| o.success)
        .map(|o| o.start_time)
        .fold(f64::INFINITY, f64::min);
    let start_abs = window.start_s + min_start;
    let end_abs = window.end_s + min_start;
    let duration = window.duration_s.max(1e-9);

    let mut completed_in_window = 0usize;
    let mut input_tokens_in_window: usize = 0;
    let mut output_tokens_in_window: usize = 0;
    let mut ttfts_in_window: Vec<f64> = Vec::new();
    let mut tpots_in_window: Vec<f64> = Vec::new();

    for (idx, o) in outputs.iter().enumerate() {
        if !o.success {
            continue;
        }
        let started_in = o.start_time >= start_abs && o.start_time < end_abs;
        let end_t = o.start_time + o.latency;
        let completed_in = end_t >= start_abs && end_t < end_abs;

        if completed_in {
            completed_in_window += 1;
        }

        if started_in {
            input_tokens_in_window += input_requests[idx].prompt_len;
            if !is_pooling {
                ttfts_in_window.push(o.ttft);
                // Per-request TPOT: (latency - ttft) / (output_tokens - 1).
                // Matches calculator.rs:46-50; skip requests with <= 1 token.
                if o.output_tokens > 1 {
                    let tpot = (o.latency - o.ttft) / (o.output_tokens as f64 - 1.0);
                    tpots_in_window.push(tpot);
                }
            }
        }

        // Per-token emission attribution (same approach as calculator.rs:98-113).
        // Reconstruct: first token at start_time + ttft, then cumulative itl.
        if o.output_tokens == 0 && o.itl.is_empty() {
            continue;
        }
        let first_token_t = o.start_time + o.ttft;
        if first_token_t >= start_abs && first_token_t < end_abs {
            output_tokens_in_window += 1;
        }
        let mut t = first_token_t;
        for &dt in &o.itl {
            t += dt;
            if t >= start_abs && t < end_abs {
                output_tokens_in_window += 1;
            }
            if t >= end_abs {
                break; // tokens only move forward in time
            }
        }
    }

    let request_throughput = completed_in_window as f64 / duration;
    let input_throughput = input_tokens_in_window as f64 / duration;
    let output_throughput = output_tokens_in_window as f64 / duration;
    let total_token_throughput = input_throughput + output_throughput;

    let (mean_ttft, median_ttft, pct_ttft) = if is_pooling {
        (0.0, 0.0, Vec::new())
    } else {
        dist_stats(&ttfts_in_window, percentiles)
    };

    let (mean_tpot, median_tpot, pct_tpot) = if is_pooling {
        (0.0, 0.0, Vec::new())
    } else {
        dist_stats(&tpots_in_window, &[90.0, 99.0])
    };
    let p90_tpot = pct_tpot.first().map(|(_, v)| *v).unwrap_or(0.0);
    let p99_tpot = pct_tpot.get(1).map(|(_, v)| *v).unwrap_or(0.0);

    SteadyStateMetrics {
        window: window.clone(),
        request_throughput,
        input_throughput,
        output_throughput,
        total_token_throughput,
        mean_ttft_ms: mean_ttft,
        median_ttft_ms: median_ttft,
        percentiles_ttft_ms: pct_ttft,
        mean_tpot_ms: mean_tpot,
        median_tpot_ms: median_tpot,
        p90_tpot_ms: p90_tpot,
        p99_tpot_ms: p99_tpot,
    }
}

/// Returns (mean_ms, median_ms, percentiles_ms).  Values are seconds in; output
/// is milliseconds.  Empty input yields zeros with a zero-filled percentile vec
/// aligned to the requested percentile list.
fn dist_stats(values: &[f64], percentiles: &[f64]) -> (f64, f64, Vec<(f64, f64)>) {
    if values.is_empty() {
        let pct = percentiles.iter().map(|&p| (p, 0.0)).collect();
        return (0.0, 0.0, pct);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let median = if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    let pct = percentiles
        .iter()
        .map(|&p| {
            let idx = (p / 100.0) * (n - 1) as f64;
            let lo = idx.floor() as usize;
            let hi = idx.ceil() as usize;
            let frac = idx - lo as f64;
            let v = if lo == hi {
                sorted[lo]
            } else {
                sorted[lo] * (1.0 - frac) + sorted[hi] * frac
            };
            (p, v * 1000.0)
        })
        .collect();
    (mean * 1000.0, median * 1000.0, pct)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::RequestFuncOutput;

    /// Build a successful synthetic output with given `start_time` and `latency`.
    fn mk(start: f64, latency: f64) -> RequestFuncOutput {
        RequestFuncOutput {
            success: true,
            start_time: start,
            latency,
            ttft: 0.0,
            itl: Vec::new(),
            output_tokens: 0,
            ..Default::default()
        }
    }

    #[test]
    fn window_plateau_trapezoid() {
        // 10 requests ramp in at t=0..10 each lasting 30s.
        // Concurrency reaches 10 at t=10 and stays until t=30 when requests start ending.
        // threshold = 1.0 means exact-target; threshold_abs = 10.
        let mut outputs: Vec<RequestFuncOutput> = (0..10).map(|i| mk(i as f64, 30.0)).collect();
        // Add one request fully inside the plateau so it can pick up as "in window"
        outputs.push(mk(15.0, 5.0));

        let w = detect_window(&outputs, Some(10), 1.0, 5.0, 30.0).unwrap();
        assert_eq!(w.target_concurrency, 10);
        assert_eq!(w.threshold_abs, 10);
        // Up-crossing at t=9 (the 10th start) — concurrency transitions 9→10.
        // Window start reported in normalized time (min_start = 0.0).
        assert!((w.start_s - 9.0).abs() < 1e-9, "start_s = {}", w.start_s);
        // Last down-crossing: the first end at t=30 (concurrency 10→9, falling below 10).
        assert!((w.end_s - 30.0).abs() < 1e-9, "end_s = {}", w.end_s);
        assert!(w.duration_s > 0.0);
        assert_eq!(w.observed_peak, 11); // 10 ramp-in + 1 extra = 11 peak briefly
        assert!(w.warning.is_none());
    }

    #[test]
    fn window_none_when_target_none() {
        let outputs = vec![mk(0.0, 1.0)];
        assert!(detect_window(&outputs, None, 0.95, 1.0, 10.0).is_none());
    }

    #[test]
    fn window_none_when_never_reaches_threshold() {
        // Serial requests, concurrency never exceeds 1.
        let outputs: Vec<RequestFuncOutput> = (0..5).map(|i| mk(i as f64 * 10.0, 1.0)).collect();
        assert!(detect_window(&outputs, Some(5), 0.95, 1.0, 100.0).is_none());
    }

    #[test]
    fn window_none_when_all_fail() {
        let mut o = mk(0.0, 1.0);
        o.success = false;
        assert!(detect_window(&[o], Some(1), 1.0, 0.1, 10.0).is_none());
    }

    #[test]
    fn window_threshold_one() {
        // threshold=1.0, target=2 -> threshold_abs=2; needs both requests in flight.
        let outputs = vec![mk(0.0, 5.0), mk(1.0, 5.0)];
        let w = detect_window(&outputs, Some(2), 1.0, 0.1, 10.0).unwrap();
        assert_eq!(w.threshold_abs, 2);
        assert!((w.start_s - 1.0).abs() < 1e-9);
        assert!((w.end_s - 5.0).abs() < 1e-9);
    }

    #[test]
    fn window_ties_net_to_zero() {
        // A request finishes and another starts at exactly the same instant;
        // tie rule (end before start) prevents concurrency > 2 blip.
        let outputs = vec![
            mk(0.0, 5.0), // ends at 5.0
            mk(2.0, 3.0), // ends at 5.0
            mk(5.0, 2.0), // starts at 5.0 — ties with the two ends
        ];
        let w = detect_window(&outputs, Some(2), 1.0, 0.1, 20.0).unwrap();
        assert_eq!(w.observed_peak, 2);
    }

    #[test]
    fn window_warning_when_short() {
        let outputs: Vec<RequestFuncOutput> = (0..10).map(|i| mk(i as f64 * 0.1, 1.0)).collect();
        // Plateau is ~0.1s; min_window_s = 5s => warning.
        let w = detect_window(&outputs, Some(10), 1.0, 5.0, 10.0).unwrap();
        assert!(w.warning.is_some(), "expected warning");
    }

    #[test]
    fn window_constant_concurrency_equals_full_run() {
        // Concurrency is at target throughout — steady-state window covers the
        // entire middle of the run where all 3 requests overlap.
        let outputs = vec![mk(0.0, 10.0), mk(0.0, 10.0), mk(0.0, 10.0)];
        let w = detect_window(&outputs, Some(3), 1.0, 0.1, 10.0).unwrap();
        assert!((w.start_s - 0.0).abs() < 1e-9);
        assert!((w.end_s - 10.0).abs() < 1e-9);
    }

    use std::sync::Arc;

    use crate::datasets::SampleRequest;

    fn mk_full(
        start: f64,
        latency: f64,
        ttft: f64,
        itl: Vec<f64>,
        output_tokens: usize,
    ) -> RequestFuncOutput {
        RequestFuncOutput {
            success: true,
            start_time: start,
            latency,
            ttft,
            itl,
            output_tokens,
            prompt_len: 10,
            ..Default::default()
        }
    }

    fn mk_sample(prompt_len: usize) -> SampleRequest {
        SampleRequest {
            prompt: Arc::from(""),
            prompt_len,
            expected_output_len: 0,
            request_id: None,
            ..Default::default()
        }
    }

    #[test]
    fn compute_constant_concurrency_matches_full_run() {
        // 3 identical requests, each 10s, each emitting 5 tokens uniformly via itl=2s.
        let itl = vec![2.0, 2.0, 2.0, 2.0]; // 5 tokens: first at t=start+ttft, then 4 more
        let outputs = vec![
            mk_full(0.0, 10.0, 2.0, itl.clone(), 5),
            mk_full(0.0, 10.0, 2.0, itl.clone(), 5),
            mk_full(0.0, 10.0, 2.0, itl.clone(), 5),
        ];
        let requests: Vec<SampleRequest> =
            outputs.iter().map(|o| mk_sample(o.prompt_len)).collect();

        let w = detect_window(&outputs, Some(3), 1.0, 0.1, 10.0).unwrap();
        let m = compute(
            &outputs,
            &requests,
            &w,
            &[99.0],
            false, // is_pooling
        );

        // Window is [0.0, 10.0). All 3 requests start in window (3).
        // Tokens per request: first token at ttft=2s, then 4 more at itl=2s each
        //   -> t = 2, 4, 6, 8, 10. The 10.0 token is AT end_s and is half-open excluded.
        //   -> 4 tokens per request land in window.
        assert_eq!(w.requests_started_in_window, 3);
        // Half-open window [0.0, 10.0) excludes the end_t=10.0 completions, so
        // request_throughput is exactly 0.0 here — tightens the half-open invariant.
        assert!((m.request_throughput - 0.0).abs() < 1e-9);
        // output tokens in window = 4 per request * 3 = 12, over 10s = 1.2 tok/s
        assert!(
            (m.output_throughput - 1.2).abs() < 1e-6,
            "got {}",
            m.output_throughput
        );
        // input throughput = 30 input tokens / 10s = 3.0
        assert!(
            (m.input_throughput - 3.0).abs() < 1e-9,
            "got {}",
            m.input_throughput
        );
    }

    #[test]
    fn compute_drain_only_tokens_not_counted() {
        // 1 "steady" request defining the window, + 1 late-admitted request whose
        // 1000 output tokens land after end_s — must not pump output_throughput.
        let mut outputs = vec![mk_full(0.0, 10.0, 0.0, vec![1.0; 9], 10)];
        // Late request: starts at 9.9 (within window), long output that drains past end_s.
        let late_itl = vec![0.02; 999]; // 1000 tokens; emission starts after end_s
        let late = mk_full(9.9, 30.0, 0.05, late_itl, 1000);
        outputs.push(late);
        let requests: Vec<SampleRequest> =
            outputs.iter().map(|o| mk_sample(o.prompt_len)).collect();

        // Target = 2 so threshold_abs = 2; window opens when both are in flight.
        let w = detect_window(&outputs, Some(2), 1.0, 0.1, 40.0).unwrap();
        let m = compute(&outputs, &requests, &w, &[99.0], false);

        // Late request started in window but most of its tokens are emitted AFTER end_s.
        // Only a handful (≤3) should count.
        // Window end is 10.0 (the first request's completion).
        // Late request first token at 9.9 + 0.05 = 9.95 (in window).
        // Next tokens at 9.97, 9.99, 10.01... -> only ~3 fit before end_s=10.0.
        let expected_max_late_tokens = 6.0; // generous bound
        let window_dur = w.duration_s;
        // Upper bound on output throughput: 10 (from req 1) + ≤6 (from late) / window_dur.
        let upper_bound = (10.0 + expected_max_late_tokens) / window_dur;
        assert!(
            m.output_throughput <= upper_bound,
            "output_throughput {} exceeds upper bound {} — drain tokens are being counted",
            m.output_throughput,
            upper_bound
        );
    }

    #[test]
    fn compute_pooling_skips_ttft_and_tpot() {
        // For pooling backends, ttft == latency is a placeholder; we must NOT emit TTFT stats.
        // Pooling has no output tokens so TPOT is also zeroed.
        let outputs = vec![
            mk_full(0.0, 5.0, 5.0, Vec::new(), 0),
            mk_full(0.0, 5.0, 5.0, Vec::new(), 0),
        ];
        let requests: Vec<SampleRequest> =
            outputs.iter().map(|o| mk_sample(o.prompt_len)).collect();
        let w = detect_window(&outputs, Some(2), 1.0, 0.1, 10.0).unwrap();
        let m = compute(&outputs, &requests, &w, &[99.0], /* is_pooling */ true);
        assert_eq!(m.mean_ttft_ms, 0.0);
        assert_eq!(m.median_ttft_ms, 0.0);
        assert!(m.percentiles_ttft_ms.is_empty());
        assert_eq!(m.mean_tpot_ms, 0.0);
        assert_eq!(m.median_tpot_ms, 0.0);
        assert_eq!(m.p90_tpot_ms, 0.0);
        assert_eq!(m.p99_tpot_ms, 0.0);
        // Throughput fields still populated.
        assert!(m.input_throughput > 0.0);
    }

    #[test]
    fn compute_tpot_matches_per_request_definition() {
        // 3 requests, each 10s latency with ttft=2s, 5 output tokens.
        // Per-request TPOT = (10 - 2) / (5 - 1) = 2s = 2000ms.
        // All requests start at t=0 so all are in-window; window duration = 10s.
        let itl = vec![2.0; 4];
        let outputs = vec![
            mk_full(0.0, 10.0, 2.0, itl.clone(), 5),
            mk_full(0.0, 10.0, 2.0, itl.clone(), 5),
            mk_full(0.0, 10.0, 2.0, itl.clone(), 5),
        ];
        let requests: Vec<SampleRequest> =
            outputs.iter().map(|o| mk_sample(o.prompt_len)).collect();
        let w = detect_window(&outputs, Some(3), 1.0, 0.1, 10.0).unwrap();
        let m = compute(&outputs, &requests, &w, &[99.0], false);

        assert!((m.mean_tpot_ms - 2000.0).abs() < 1e-6);
        assert!((m.median_tpot_ms - 2000.0).abs() < 1e-6);
        assert!((m.p90_tpot_ms - 2000.0).abs() < 1e-6);
        assert!((m.p99_tpot_ms - 2000.0).abs() < 1e-6);
    }

    #[test]
    fn json_roundtrip_some() {
        let outputs = vec![mk_full(0.0, 10.0, 2.0, vec![2.0; 4], 5); 3];
        let requests: Vec<SampleRequest> =
            outputs.iter().map(|o| mk_sample(o.prompt_len)).collect();
        let w = detect_window(&outputs, Some(3), 1.0, 0.1, 10.0).unwrap();
        let m = compute(&outputs, &requests, &w, &[99.0], false);
        let s = serde_json::to_string(&m).unwrap();
        let parsed: SteadyStateMetrics = serde_json::from_str(&s).unwrap();
        assert!((parsed.output_throughput - m.output_throughput).abs() < 1e-12);
        assert_eq!(parsed.window.target_concurrency, 3);
    }

    #[test]
    fn json_deserialize_missing_key_as_none() {
        // Pre-feature JSON lacks `steady_state` — deserializing a wrapper that
        // contains an Option field with #[serde(default)] should yield None.
        #[derive(serde::Deserialize)]
        struct Wrap {
            #[serde(default)]
            steady_state: Option<SteadyStateMetrics>,
        }
        let old_json = r#"{"other":"value"}"#;
        let w: Wrap = serde_json::from_str(old_json).unwrap();
        assert!(w.steady_state.is_none());
    }

    #[test]
    fn end_to_end_trapezoid_run() {
        // 5 requests ramp in at t=0..5, each lasts 20s, each emits 10 tokens
        // at 1s intervals starting at ttft=1s.
        let itl = vec![1.0; 9];
        let outputs: Vec<RequestFuncOutput> =
            (0..5).map(|i| mk_full(i as f64, 20.0, 1.0, itl.clone(), 10)).collect();
        let requests: Vec<SampleRequest> = (0..5).map(|_| mk_sample(100)).collect();

        let w = detect_window(&outputs, Some(5), 1.0, 1.0, 25.0).unwrap();
        assert_eq!(w.threshold_abs, 5);
        assert!((w.start_s - 4.0).abs() < 1e-9);
        assert!((w.end_s - 20.0).abs() < 1e-9);

        let m = compute(&outputs, &requests, &w, &[50.0, 99.0], false);

        // Round-trip through JSON.
        let s = serde_json::to_string(&m).unwrap();
        let back: SteadyStateMetrics = serde_json::from_str(&s).unwrap();
        assert!((back.request_throughput - m.request_throughput).abs() < 1e-12);
        assert!((back.output_throughput - m.output_throughput).abs() < 1e-12);
        assert_eq!(back.window.target_concurrency, 5);

        // TTFT percentiles should have 2 entries matching the input percentiles.
        assert_eq!(m.percentiles_ttft_ms.len(), 2);
        assert!((m.percentiles_ttft_ms[0].0 - 50.0).abs() < 1e-9);
        assert!((m.percentiles_ttft_ms[1].0 - 99.0).abs() < 1e-9);
    }
}
