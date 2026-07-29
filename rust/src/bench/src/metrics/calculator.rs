// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use crate::backends::RequestFuncOutput;
use crate::config::GoodputConfig;
use crate::datasets::SampleRequest;
use crate::metrics::{BenchmarkMetrics, MultiTurnMetrics};
use crate::multi_turn::ConversationOutput;

fn log_failed_requests(outputs: &[RequestFuncOutput]) {
    let failed_outputs: Vec<_> = outputs.iter().filter(|output| !output.success).collect();
    if failed_outputs.is_empty() {
        return;
    }

    tracing::warn!(
        failed_requests = failed_outputs.len(),
        displayed_errors = failed_outputs.len().min(10),
        "benchmark requests failed"
    );
    for (index, output) in failed_outputs.into_iter().take(10).enumerate() {
        tracing::warn!(index, error = %output.error, "benchmark request failed");
    }
}

/// Calculate benchmark metrics from request outputs.
///
/// Mirrors Python's `calculate_metrics()` from serve.py:392-599.
pub fn calculate_metrics(
    input_requests: &[SampleRequest],
    outputs: &[RequestFuncOutput],
    dur_s: f64,
    selected_percentiles: &[f64],
    _has_tokenizer: bool,
    goodput_config: &GoodputConfig,
) -> (BenchmarkMetrics, Vec<usize>) {
    let mut actual_output_lens: Vec<usize> = Vec::with_capacity(outputs.len());
    let mut total_input: usize = 0;
    let mut completed: usize = 0;
    let mut itls: Vec<f64> = Vec::new();
    let mut tpots: Vec<f64> = Vec::new();
    // Per-request TPOT for goodput SLO checking (parallel to ttfts/e2els).
    let mut all_tpots: Vec<f64> = Vec::new();
    let mut ttfts: Vec<f64> = Vec::new();
    let mut e2els: Vec<f64> = Vec::new();

    for (i, output) in outputs.iter().enumerate() {
        if output.success {
            let output_len = if output.output_tokens > 0 {
                output.output_tokens
            } else {
                // Fallback: first token + ITL entries.
                // Python re-encodes generated_text when tokenizer is available,
                // but with vLLM's stream_options.include_usage=true,
                // output_tokens is always set so this path is rarely hit.
                1 + output.itl.len()
            };

            actual_output_lens.push(output_len);
            total_input += input_requests[i].prompt_len;

            if output_len > 1 {
                let latency_minus_ttft = output.latency - output.ttft;
                let tpot = latency_minus_ttft / (output_len as f64 - 1.0);
                tpots.push(tpot);
                all_tpots.push(tpot);
            } else {
                all_tpots.push(0.0);
            }

            itls.extend_from_slice(&output.itl);
            ttfts.push(output.ttft);
            e2els.push(output.latency);
            completed += 1;
        } else {
            actual_output_lens.push(0);
        }
    }

    let failed = outputs.len() - completed;

    log_failed_requests(outputs);

    // Calculate max output tokens per second and max concurrent requests
    let mut max_output_tokens_per_s = 0.0_f64;
    let mut max_concurrent_requests: usize = 0;

    let successful_outputs: Vec<&RequestFuncOutput> =
        outputs.iter().filter(|o| o.success).collect();

    if !successful_outputs.is_empty() {
        let min_start_time =
            successful_outputs.iter().map(|o| o.start_time).fold(f64::INFINITY, f64::min);
        let max_end_time = successful_outputs
            .iter()
            .map(|o| o.start_time + o.latency)
            .fold(f64::NEG_INFINITY, f64::max);

        let raw_duration = (max_end_time - min_start_time).ceil() as usize + 1;
        // Cap at 24 hours to prevent OOM from corrupted timing data
        let duration_seconds = raw_duration.min(86_400);
        let mut tokens_per_second = vec![0.0_f64; duration_seconds];
        let mut concurrent_per_second = vec![0usize; duration_seconds];

        for output in &successful_outputs {
            // Calculate token generation timestamps
            let mut token_times = vec![output.start_time + output.ttft];
            let mut current_time = token_times[0];
            for itl_value in &output.itl {
                current_time += itl_value;
                token_times.push(current_time);
            }

            // Add tokens to second buckets
            for token_time in &token_times {
                let bucket = (token_time - min_start_time) as usize;
                if bucket < duration_seconds {
                    tokens_per_second[bucket] += 1.0;
                }
            }

            // Track concurrent requests
            let start_second = (output.start_time - min_start_time) as usize;
            let end_second = ((output.start_time + output.latency) - min_start_time) as usize;
            for slot in concurrent_per_second
                .iter_mut()
                .take(end_second.min(duration_seconds - 1) + 1)
                .skip(start_second)
            {
                *slot += 1;
            }
        }

        max_output_tokens_per_s = tokens_per_second.iter().cloned().fold(0.0_f64, f64::max);
        max_concurrent_requests = *concurrent_per_second.iter().max().unwrap_or(&0);
    }

    let total_output: usize = actual_output_lens.iter().sum();

    // Compute goodput: count requests meeting ALL specified SLOs.
    // Mirrors Python serve.py:458-481.
    let good_completed = if !goodput_config.is_empty() {
        let mut good = 0usize;
        // ttfts, all_tpots, e2els are parallel (one per successful request)
        for i in 0..ttfts.len() {
            let mut is_good = true;
            if let Some(slo_ms) = goodput_config.ttft_ms
                && ttfts[i] * 1000.0 > slo_ms
            {
                is_good = false;
            }
            if let Some(slo_ms) = goodput_config.tpot_ms
                && all_tpots[i] * 1000.0 > slo_ms
            {
                is_good = false;
            }
            if let Some(slo_ms) = goodput_config.e2el_ms
                && e2els[i] * 1000.0 > slo_ms
            {
                is_good = false;
            }
            if is_good {
                good += 1;
            }
        }
        good
    } else {
        0
    };

    // Sort each metric array once, then compute median + percentiles from sorted data
    let sorted_ttfts = sort_clone(&ttfts);
    let sorted_tpots = sort_clone(&tpots);
    let sorted_itls = sort_clone(&itls);
    let sorted_e2els = sort_clone(&e2els);

    let request_goodput = if !goodput_config.is_empty() {
        good_completed as f64 / dur_s
    } else {
        0.0
    };

    let metrics = BenchmarkMetrics {
        completed,
        failed,
        total_input,
        total_output,
        request_throughput: completed as f64 / dur_s,
        request_goodput,
        input_throughput: total_input as f64 / dur_s,
        output_throughput: total_output as f64 / dur_s,
        total_token_throughput: (total_input + total_output) as f64 / dur_s,
        mean_ttft_ms: mean(&ttfts) * 1000.0,
        median_ttft_ms: median_sorted(&sorted_ttfts) * 1000.0,
        std_ttft_ms: std_dev(&ttfts) * 1000.0,
        percentiles_ttft_ms: percentiles_from_sorted(&sorted_ttfts, selected_percentiles),
        mean_tpot_ms: mean(&tpots) * 1000.0,
        median_tpot_ms: median_sorted(&sorted_tpots) * 1000.0,
        std_tpot_ms: std_dev(&tpots) * 1000.0,
        percentiles_tpot_ms: percentiles_from_sorted(&sorted_tpots, selected_percentiles),
        mean_itl_ms: mean(&itls) * 1000.0,
        median_itl_ms: median_sorted(&sorted_itls) * 1000.0,
        std_itl_ms: std_dev(&itls) * 1000.0,
        percentiles_itl_ms: percentiles_from_sorted(&sorted_itls, selected_percentiles),
        mean_e2el_ms: mean(&e2els) * 1000.0,
        median_e2el_ms: median_sorted(&sorted_e2els) * 1000.0,
        std_e2el_ms: std_dev(&e2els) * 1000.0,
        percentiles_e2el_ms: percentiles_from_sorted(&sorted_e2els, selected_percentiles),
        max_output_tokens_per_s,
        max_concurrent_requests,
        steady_state: None,
    };

    (metrics, actual_output_lens)
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn sort_clone(data: &[f64]) -> Vec<f64> {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

fn median_sorted(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn std_dev(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let m = mean(data);
    let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

fn percentile_sorted(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    if sorted_data.len() == 1 {
        return sorted_data[0];
    }
    // Use the same interpolation as numpy (linear)
    let idx = p / 100.0 * (sorted_data.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    if lo == hi {
        sorted_data[lo]
    } else {
        sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac
    }
}

fn percentiles_from_sorted(sorted: &[f64], percentiles: &[f64]) -> Vec<(f64, f64)> {
    if sorted.is_empty() {
        return percentiles.iter().map(|&p| (p, 0.0)).collect();
    }
    percentiles
        .iter()
        .map(|&p| (p, percentile_sorted(sorted, p) * 1000.0))
        .collect()
}

/// Calculate benchmark metrics for embedding/pooling requests.
///
/// Mirrors Python's `calculate_metrics_for_embeddings()` from serve.py.
/// Key differences from `calculate_metrics()`:
/// - Uses `output.prompt_len` (server-reported) instead of `input_requests[i].prompt_len`
/// - Only computes E2EL (no TTFT/TPOT/ITL since pooling is non-streaming)
/// - `total_output` is 0 (pooling produces no output tokens)
/// - `total_token_throughput` = `total_input / dur_s` (input tokens only)
pub fn calculate_embedding_metrics(
    outputs: &[RequestFuncOutput],
    dur_s: f64,
    selected_percentiles: &[f64],
) -> BenchmarkMetrics {
    let mut total_input: usize = 0;
    let mut completed: usize = 0;
    let mut e2els: Vec<f64> = Vec::new();

    for output in outputs {
        if output.success {
            e2els.push(output.latency);
            completed += 1;
            total_input += output.prompt_len;
        }
    }

    let failed = outputs.len() - completed;

    log_failed_requests(outputs);

    // Compute peak concurrent requests from start_time + latency windows
    let successful_outputs: Vec<&RequestFuncOutput> =
        outputs.iter().filter(|o| o.success).collect();
    let max_concurrent_requests = if !successful_outputs.is_empty() {
        let min_start =
            successful_outputs.iter().map(|o| o.start_time).fold(f64::INFINITY, f64::min);
        let max_end = successful_outputs
            .iter()
            .map(|o| o.start_time + o.latency)
            .fold(f64::NEG_INFINITY, f64::max);

        let raw_duration = (max_end - min_start).ceil() as usize + 1;
        let duration_seconds = raw_duration.min(86_400);
        let mut concurrent_per_second = vec![0usize; duration_seconds];

        for output in &successful_outputs {
            let start_second = (output.start_time - min_start) as usize;
            let end_second = ((output.start_time + output.latency) - min_start) as usize;
            for slot in concurrent_per_second
                .iter_mut()
                .take(end_second.min(duration_seconds - 1) + 1)
                .skip(start_second)
            {
                *slot += 1;
            }
        }

        *concurrent_per_second.iter().max().unwrap_or(&0)
    } else {
        0
    };

    let sorted_e2els = sort_clone(&e2els);

    BenchmarkMetrics {
        completed,
        failed,
        total_input,
        total_output: 0,
        request_throughput: completed as f64 / dur_s,
        request_goodput: 0.0,
        input_throughput: total_input as f64 / dur_s,
        output_throughput: 0.0,
        total_token_throughput: total_input as f64 / dur_s,
        mean_ttft_ms: 0.0,
        median_ttft_ms: 0.0,
        std_ttft_ms: 0.0,
        percentiles_ttft_ms: Vec::new(),
        mean_tpot_ms: 0.0,
        median_tpot_ms: 0.0,
        std_tpot_ms: 0.0,
        percentiles_tpot_ms: Vec::new(),
        mean_itl_ms: 0.0,
        median_itl_ms: 0.0,
        std_itl_ms: 0.0,
        percentiles_itl_ms: Vec::new(),
        mean_e2el_ms: mean(&e2els) * 1000.0,
        median_e2el_ms: median_sorted(&sorted_e2els) * 1000.0,
        std_e2el_ms: std_dev(&e2els) * 1000.0,
        percentiles_e2el_ms: percentiles_from_sorted(&sorted_e2els, selected_percentiles),
        max_output_tokens_per_s: 0.0,
        max_concurrent_requests,
        steady_state: None,
    }
}

/// Calculate multi-turn benchmark metrics with per-turn breakdown.
///
/// 1. Flatten all turns into SampleRequest/RequestFuncOutput pairs → overall metrics
/// 2. Group by turn_index → per-turn metrics
/// 3. Conversation-level stats
pub fn calculate_multi_turn_metrics(
    conversation_outputs: &[ConversationOutput],
    dur_s: f64,
    selected_percentiles: &[f64],
    goodput_config: &GoodputConfig,
    max_turn_count: usize,
) -> MultiTurnMetrics {
    // Flatten all turns for overall metrics
    let mut all_requests: Vec<SampleRequest> = Vec::new();
    let mut all_outputs: Vec<RequestFuncOutput> = Vec::new();

    for conv in conversation_outputs {
        for turn in &conv.turns {
            all_requests.push(SampleRequest {
                prompt: std::sync::Arc::from(""),
                prompt_len: turn.cumulative_input_tokens,
                expected_output_len: turn.request_output.output_tokens,
                request_id: None,
                ..Default::default()
            });
            all_outputs.push(turn.request_output.clone());
        }
    }

    let (overall, _) = calculate_metrics(
        &all_requests,
        &all_outputs,
        dur_s,
        selected_percentiles,
        true,
        goodput_config,
    );

    // Per-turn breakdown
    let mut per_turn: Vec<BenchmarkMetrics> = Vec::new();
    for turn_idx in 0..max_turn_count {
        let mut turn_requests: Vec<SampleRequest> = Vec::new();
        let mut turn_outputs: Vec<RequestFuncOutput> = Vec::new();

        for conv in conversation_outputs {
            if let Some(turn) = conv.turns.get(turn_idx) {
                turn_requests.push(SampleRequest {
                    prompt: std::sync::Arc::from(""),
                    prompt_len: turn.cumulative_input_tokens,
                    expected_output_len: turn.request_output.output_tokens,
                    request_id: None,
                    ..Default::default()
                });
                turn_outputs.push(turn.request_output.clone());
            }
        }

        if !turn_outputs.is_empty() {
            let (turn_metrics, _) = calculate_metrics(
                &turn_requests,
                &turn_outputs,
                dur_s,
                selected_percentiles,
                true,
                goodput_config,
            );
            per_turn.push(turn_metrics);
        }
    }

    // Conversation-level stats
    let conversations_completed = conversation_outputs.iter().filter(|c| c.all_success).count();
    let conversations_failed = conversation_outputs.len() - conversations_completed;

    let avg_turns_completed = if conversation_outputs.is_empty() {
        0.0
    } else {
        conversation_outputs.iter().map(|c| c.turns.len() as f64).sum::<f64>()
            / conversation_outputs.len() as f64
    };

    let avg_conversation_duration_ms = if conversation_outputs.is_empty() {
        0.0
    } else {
        conversation_outputs.iter().map(|c| c.total_duration_ms).sum::<f64>()
            / conversation_outputs.len() as f64
    };

    MultiTurnMetrics {
        overall,
        per_turn,
        conversations_completed,
        conversations_failed,
        avg_turns_completed,
        avg_conversation_duration_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_empty() {
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn test_mean_values() {
        assert!((mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_odd() {
        let sorted = sort_clone(&[3.0, 1.0, 2.0]);
        assert!((median_sorted(&sorted) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_even() {
        let sorted = sort_clone(&[1.0, 2.0, 3.0, 4.0]);
        assert!((median_sorted(&sorted) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_99() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let p99 = percentile_sorted(&data, 99.0);
        assert!((p99 - 98.01).abs() < 0.1);
    }
}
