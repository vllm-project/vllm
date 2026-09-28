// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use crate::benchmark::SpecDecodeStats;
use crate::config::BenchConfig;
use crate::metrics::{BenchmarkMetrics, MultiTurnMetrics};

/// Print benchmark results to console in the same format as Python.
///
/// Mirrors the output format from serve.py:918-1093.
pub fn print_results(
    metrics: &BenchmarkMetrics,
    benchmark_duration: f64,
    config: &BenchConfig,
    has_tokenizer: bool,
    spec_decode_stats: Option<&SpecDecodeStats>,
) {
    let is_pooling = config.backend.is_pooling();

    if is_pooling {
        println!("{:=^60}", " Embedding/Pooling Benchmark Result ");
    } else {
        println!("{:=^60}", " Serving Benchmark Result ");
    }
    println!("{:<40} {:<10}", "Successful requests:", metrics.completed);
    println!("{:<40} {:<10}", "Failed requests:", metrics.failed);

    if let Some(mc) = config.max_concurrency {
        println!("{:<40} {:<10}", "Maximum request concurrency:", mc);
    }
    if !config.request_rate.is_infinite() {
        println!(
            "{:<40} {:<10.2}",
            "Request rate configured (RPS):", config.request_rate
        );
    }
    println!(
        "{:<40} {:<10.2}",
        "Benchmark duration (s):", benchmark_duration
    );
    println!("{:<40} {:<10}", "Total input tokens:", metrics.total_input);

    if has_tokenizer && !is_pooling {
        println!(
            "{:<40} {:<10}",
            "Total generated tokens:", metrics.total_output
        );
    }

    println!(
        "{:<40} {:<10.2}",
        "Request throughput (req/s):", metrics.request_throughput
    );
    if metrics.request_goodput > 0.0 {
        println!(
            "{:<40} {:<10.2}",
            "Request goodput (req/s):", metrics.request_goodput
        );
    }

    if !is_pooling {
        if has_tokenizer {
            println!(
                "{:<40} {:<10.2}",
                "Output token throughput (tok/s):", metrics.output_throughput
            );
            println!(
                "{:<40} {:<10.2}",
                "Peak output token throughput (tok/s):", metrics.max_output_tokens_per_s
            );
        }

        println!(
            "{:<40} {:<10.2}",
            "Peak concurrent requests:", metrics.max_concurrent_requests as f64
        );

        if has_tokenizer {
            println!(
                "{:<40} {:<10.2}",
                "Total token throughput (tok/s):", metrics.total_token_throughput
            );
        }
    } else {
        println!(
            "{:<40} {:<10.2}",
            "Input token throughput (tok/s):", metrics.input_throughput
        );
        println!(
            "{:<40} {:<10.2}",
            "Peak concurrent requests:", metrics.max_concurrent_requests as f64
        );
    }

    // Print per-metric percentiles
    if has_tokenizer && !is_pooling {
        print_metric_section(
            "ttft",
            "TTFT",
            "Time to First Token",
            &config.selected_percentile_metrics,
            metrics.mean_ttft_ms,
            metrics.median_ttft_ms,
            &metrics.percentiles_ttft_ms,
        );
        print_metric_section(
            "tpot",
            "TPOT",
            "Time per Output Token (excl. 1st token)",
            &config.selected_percentile_metrics,
            metrics.mean_tpot_ms,
            metrics.median_tpot_ms,
            &metrics.percentiles_tpot_ms,
        );
        print_metric_section(
            "itl",
            "ITL",
            "Inter-token Latency",
            &config.selected_percentile_metrics,
            metrics.mean_itl_ms,
            metrics.median_itl_ms,
            &metrics.percentiles_itl_ms,
        );
    }
    print_metric_section(
        "e2el",
        "E2EL",
        "End-to-end Latency",
        &config.selected_percentile_metrics,
        metrics.mean_e2el_ms,
        metrics.median_e2el_ms,
        &metrics.percentiles_e2el_ms,
    );

    if let Some(stats) = spec_decode_stats {
        print_spec_decode_section(stats);
    }

    println!("{:=<60}", "");

    print_steady_state(metrics);
}

/// Print multi-turn benchmark results to console.
pub fn print_multi_turn_results(
    mt_metrics: &MultiTurnMetrics,
    benchmark_duration: f64,
    config: &BenchConfig,
    spec_decode_stats: Option<&SpecDecodeStats>,
) {
    println!("{:=^60}", " Multi-Turn Benchmark Result ");
    println!(
        "{:<45} {}/{}",
        "Conversations completed/total:",
        mt_metrics.conversations_completed,
        mt_metrics.conversations_completed + mt_metrics.conversations_failed,
    );
    println!(
        "{:<45} {}",
        "Turns per conversation (configured):", config.multi_turn_num_turns,
    );
    println!(
        "{:<45} {:.1}",
        "Avg turns completed:", mt_metrics.avg_turns_completed,
    );
    println!(
        "{:<45} {:.0}",
        "Avg conversation duration (ms):", mt_metrics.avg_conversation_duration_ms,
    );
    println!(
        "{:<45} {}",
        "Concurrency:",
        config
            .multi_turn_concurrency
            .unwrap_or(config.max_concurrency.unwrap_or(config.num_prompts)),
    );
    println!(
        "{:<45} {}",
        "Inter-turn delay (ms):", config.multi_turn_delay_ms,
    );
    println!(
        "{:<45} {:.2}",
        "Benchmark duration (s):", benchmark_duration,
    );

    // Overall metrics
    println!("{:-^60}", " Overall (All Turns) ");
    print_metrics_block(&mt_metrics.overall, config);

    // Per-turn breakdown
    for (i, turn_metrics) in mt_metrics.per_turn.iter().enumerate() {
        let samples = turn_metrics.completed + turn_metrics.failed;
        println!("{:-^60}", format!(" Turn {} ({} samples) ", i + 1, samples));
        print_metrics_block(turn_metrics, config);
    }

    if let Some(stats) = spec_decode_stats {
        print_spec_decode_section(stats);
    }

    println!("{:=<60}", "");
}

/// Print a metrics block (reused for overall and per-turn).
fn print_metrics_block(metrics: &BenchmarkMetrics, config: &BenchConfig) {
    println!("{:<45} {:<10}", "Successful requests:", metrics.completed);
    println!("{:<45} {:<10}", "Failed requests:", metrics.failed);
    println!("{:<45} {:<10}", "Total input tokens:", metrics.total_input);
    println!(
        "{:<45} {:<10}",
        "Total generated tokens:", metrics.total_output
    );
    println!(
        "{:<45} {:<10.2}",
        "Request throughput (req/s):", metrics.request_throughput
    );
    println!(
        "{:<45} {:<10.2}",
        "Input token throughput (tok/s):", metrics.input_throughput
    );
    println!(
        "{:<45} {:<10.2}",
        "Output token throughput (tok/s):", metrics.output_throughput
    );
    println!(
        "{:<45} {:<10.2}",
        "Total token throughput (tok/s):", metrics.total_token_throughput
    );

    print_metric_section(
        "ttft",
        "TTFT",
        "Time to First Token",
        &config.selected_percentile_metrics,
        metrics.mean_ttft_ms,
        metrics.median_ttft_ms,
        &metrics.percentiles_ttft_ms,
    );
    print_metric_section(
        "tpot",
        "TPOT",
        "Time per Output Token (excl. 1st token)",
        &config.selected_percentile_metrics,
        metrics.mean_tpot_ms,
        metrics.median_tpot_ms,
        &metrics.percentiles_tpot_ms,
    );
    print_metric_section(
        "itl",
        "ITL",
        "Inter-token Latency",
        &config.selected_percentile_metrics,
        metrics.mean_itl_ms,
        metrics.median_itl_ms,
        &metrics.percentiles_itl_ms,
    );
    print_metric_section(
        "e2el",
        "E2EL",
        "End-to-end Latency",
        &config.selected_percentile_metrics,
        metrics.mean_e2el_ms,
        metrics.median_e2el_ms,
        &metrics.percentiles_e2el_ms,
    );
}

fn print_metric_section(
    attr_name: &str,
    short_name: &str,
    header: &str,
    selected: &[String],
    mean_ms: f64,
    median_ms: f64,
    percentiles: &[(f64, f64)],
) {
    if !selected.iter().any(|s| s == attr_name) {
        return;
    }
    println!("{:-^60}", header);
    println!(
        "{:<40} {:<10.2}",
        format!("Mean {short_name} (ms):"),
        mean_ms
    );
    println!(
        "{:<40} {:<10.2}",
        format!("Median {short_name} (ms):"),
        median_ms
    );
    for (p, value) in percentiles {
        let p_str = if *p == p.floor() {
            format!("{}", *p as i64)
        } else {
            format!("{p}")
        };
        println!(
            "{:<40} {:<10.2}",
            format!("P{p_str} {short_name} (ms):"),
            value
        );
    }
}

/// Print speculative decoding metrics section.
fn print_spec_decode_section(stats: &SpecDecodeStats) {
    println!("{:-^50}", "Speculative Decoding");
    println!(
        "{:<40} {:<10.2}",
        "Acceptance rate (%):", stats.acceptance_rate
    );
    println!(
        "{:<40} {:<10.2}",
        "Acceptance length:", stats.acceptance_length
    );
    println!("{:<40} {:<10}", "Drafts:", stats.num_drafts);
    println!("{:<40} {:<10}", "Draft tokens:", stats.draft_tokens);
    println!("{:<40} {:<10}", "Accepted tokens:", stats.accepted_tokens);
    if !stats.per_position_acceptance_rates.is_empty() {
        println!("Per-position acceptance (%):");
        for (i, rate) in stats.per_position_acceptance_rates.iter().enumerate() {
            println!("{:<40} {:<10.2}", format!("  Position {i}:"), rate * 100.0);
        }
    }
}

fn print_steady_state(metrics: &crate::metrics::BenchmarkMetrics) {
    let Some(ss) = metrics.steady_state.as_ref() else {
        return;
    };

    if let Some(warning) = ss.window.warning.as_ref() {
        println!("Warning: {warning}");
    }

    let total = ss.window.requests_total.max(1);
    let started_pct = 100.0 * ss.window.requests_started_in_window as f64 / total as f64;

    println!("{:=^47}", " Steady-State Metrics ");
    println!(
        "{:<35} >= {:.2} * {} = {}",
        "Concurrency threshold:",
        ss.window.threshold,
        ss.window.target_concurrency,
        ss.window.threshold_abs,
    );
    println!(
        "{:<35} {:.1}s -> {:.1}s  ({:.1}s)",
        "Window:", ss.window.start_s, ss.window.end_s, ss.window.duration_s
    );
    println!(
        "{:<35} {}",
        "Observed peak concurrency:", ss.window.observed_peak
    );
    println!(
        "{:<35} {} / {} ({:.1}%)",
        "Requests started in window:",
        ss.window.requests_started_in_window,
        ss.window.requests_total,
        started_pct
    );
    println!(
        "{:<35} {}",
        "Requests completed in window:", ss.window.requests_completed_in_window
    );
    println!("{:-<47}", "");
    println!(
        "{:<35} {:.2}",
        "Request throughput (req/s):", ss.request_throughput
    );
    println!(
        "{:<35} {:.2}",
        "Output token throughput (tok/s):", ss.output_throughput
    );
    println!(
        "{:<35} {:.2}",
        "Input token throughput (tok/s):", ss.input_throughput
    );
    println!(
        "{:<35} {:.2}",
        "Total token throughput (tok/s):", ss.total_token_throughput
    );

    if !ss.percentiles_ttft_ms.is_empty() || ss.mean_ttft_ms > 0.0 {
        println!("{:-<47}", "");
        println!("{:<35} {:.2}", "Mean TTFT (ms):", ss.mean_ttft_ms);
        println!("{:<35} {:.2}", "Median TTFT (ms):", ss.median_ttft_ms);
        for (p, v) in &ss.percentiles_ttft_ms {
            println!("{:<35} {:.2}", format!("P{} TTFT (ms):", *p as u32), v);
        }
    }
    if ss.mean_tpot_ms > 0.0 || ss.median_tpot_ms > 0.0 {
        println!("{:-<47}", "");
        println!("{:<35} {:.2}", "Mean TPOT (ms):", ss.mean_tpot_ms);
        println!("{:<35} {:.2}", "Median TPOT (ms):", ss.median_tpot_ms);
        println!("{:<35} {:.2}", "P90 TPOT (ms):", ss.p90_tpot_ms);
        println!("{:<35} {:.2}", "P99 TPOT (ms):", ss.p99_tpot_ms);
    }
    println!("{:=^47}", "");
}
