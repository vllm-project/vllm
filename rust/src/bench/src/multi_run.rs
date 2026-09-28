// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use crate::config::BenchConfig;
use crate::error::Result;

/// Key metrics extracted from a single run's JSON result.
struct RunMetrics {
    request_throughput: f64,
    output_throughput: f64,
    total_token_throughput: f64,
    mean_ttft_ms: f64,
    median_ttft_ms: f64,
    p99_ttft_ms: f64,
    mean_tpot_ms: f64,
    median_tpot_ms: f64,
    p99_tpot_ms: f64,
    mean_itl_ms: f64,
    mean_e2el_ms: f64,
    p99_e2el_ms: f64,
    max_output_tokens_per_s: f64,
    completed: f64,
    failed: f64,
    duration: f64,
    ss_request_throughput: Option<f64>,
    ss_output_throughput: Option<f64>,
    ss_input_throughput: Option<f64>,
    ss_total_token_throughput: Option<f64>,
    ss_mean_ttft_ms: Option<f64>,
    ss_median_ttft_ms: Option<f64>,
    ss_mean_tpot_ms: Option<f64>,
    ss_median_tpot_ms: Option<f64>,
    ss_p90_tpot_ms: Option<f64>,
    ss_p99_tpot_ms: Option<f64>,
}

impl RunMetrics {
    fn from_json(json: &serde_json::Value) -> Self {
        Self {
            request_throughput: get(json, "request_throughput"),
            output_throughput: get(json, "output_throughput"),
            total_token_throughput: get(json, "total_token_throughput"),
            mean_ttft_ms: get(json, "mean_ttft_ms"),
            median_ttft_ms: get(json, "median_ttft_ms"),
            p99_ttft_ms: get(json, "p99_ttft_ms"),
            mean_tpot_ms: get(json, "mean_tpot_ms"),
            median_tpot_ms: get(json, "median_tpot_ms"),
            p99_tpot_ms: get(json, "p99_tpot_ms"),
            mean_itl_ms: get(json, "mean_itl_ms"),
            mean_e2el_ms: get(json, "mean_e2el_ms"),
            p99_e2el_ms: get(json, "p99_e2el_ms"),
            max_output_tokens_per_s: get(json, "max_output_tokens_per_s"),
            completed: get(json, "completed"),
            failed: get(json, "failed"),
            duration: get(json, "duration"),
            ss_request_throughput: get_ss_opt(json, "request_throughput"),
            ss_output_throughput: get_ss_opt(json, "output_throughput"),
            ss_input_throughput: get_ss_opt(json, "input_throughput"),
            ss_total_token_throughput: get_ss_opt(json, "total_token_throughput"),
            ss_mean_ttft_ms: get_ss_opt(json, "mean_ttft_ms"),
            ss_median_ttft_ms: get_ss_opt(json, "median_ttft_ms"),
            ss_mean_tpot_ms: get_ss_opt(json, "mean_tpot_ms"),
            ss_median_tpot_ms: get_ss_opt(json, "median_tpot_ms"),
            ss_p90_tpot_ms: get_ss_opt(json, "p90_tpot_ms"),
            ss_p99_tpot_ms: get_ss_opt(json, "p99_tpot_ms"),
        }
    }
}

fn get(json: &serde_json::Value, key: &str) -> f64 {
    json.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0)
}

fn get_ss_opt(json: &serde_json::Value, key: &str) -> Option<f64> {
    json.get("steady_state")
        .and_then(|ss| if ss.is_null() { None } else { Some(ss) })
        .and_then(|ss| ss.get(key))
        .and_then(|v| v.as_f64())
}

/// Aggregated statistics for a single metric across N runs.
struct MetricStats {
    label: &'static str,
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
}

/// Run the benchmark N times and report aggregated statistics.
pub async fn run_multi(config: &BenchConfig, num_runs: usize) -> Result<()> {
    println!(
        "{:=^70}",
        format!(" Multi-Run Benchmark ({num_runs} runs) ")
    );
    println!();

    let mut all_runs: Vec<RunMetrics> = Vec::with_capacity(num_runs);

    for i in 0..num_runs {
        println!("{:-^70}", format!(" Run {}/{} ", i + 1, num_runs));

        let mut run_config = config.clone();
        // Suppress per-run save
        run_config.save_result = false;
        run_config.append_result = false;

        let result = crate::benchmark::run_benchmark(&run_config).await?;
        all_runs.push(RunMetrics::from_json(&result));

        println!();
    }

    print_multi_run_summary(&all_runs);
    Ok(())
}

fn print_multi_run_summary(runs: &[RunMetrics]) {
    let n = runs.len();

    // Collect each metric into a series, compute stats
    let stats = vec![
        compute_stats("Request throughput (req/s)", runs, |r| r.request_throughput),
        compute_stats("Output throughput (tok/s)", runs, |r| r.output_throughput),
        compute_stats("Total token throughput (tok/s)", runs, |r| {
            r.total_token_throughput
        }),
        compute_stats("Peak output tokens/s", runs, |r| r.max_output_tokens_per_s),
        compute_stats("Mean TTFT (ms)", runs, |r| r.mean_ttft_ms),
        compute_stats("Median TTFT (ms)", runs, |r| r.median_ttft_ms),
        compute_stats("P99 TTFT (ms)", runs, |r| r.p99_ttft_ms),
        compute_stats("Mean TPOT (ms)", runs, |r| r.mean_tpot_ms),
        compute_stats("Median TPOT (ms)", runs, |r| r.median_tpot_ms),
        compute_stats("P99 TPOT (ms)", runs, |r| r.p99_tpot_ms),
        compute_stats("Mean ITL (ms)", runs, |r| r.mean_itl_ms),
        compute_stats("Mean E2EL (ms)", runs, |r| r.mean_e2el_ms),
        compute_stats("P99 E2EL (ms)", runs, |r| r.p99_e2el_ms),
        compute_stats("Completed requests", runs, |r| r.completed),
        compute_stats("Failed requests", runs, |r| r.failed),
        compute_stats("Duration (s)", runs, |r| r.duration),
    ];

    println!("{:=^80}", format!(" Multi-Run Summary ({n} runs) "));
    println!(
        "{:<35} {:>10} {:>10} {:>10} {:>10}",
        "Metric", "Mean", "Std", "Min", "Max"
    );
    println!(
        "{:-<35} {:->10} {:->10} {:->10} {:->10}",
        "", "", "", "", ""
    );

    for s in &stats {
        println!(
            "{:<35} {:>10} {:>10} {:>10} {:>10}",
            s.label,
            fmt(s.mean),
            fmt(s.std),
            fmt(s.min),
            fmt(s.max),
        );
    }

    println!("{:=<80}", "");

    // Print coefficient of variation for throughput
    let tp = &stats[0]; // request_throughput
    if tp.mean > 0.0 {
        let cv = (tp.std / tp.mean) * 100.0;
        println!(
            "Throughput CV: {:.1}% (lower = more stable across runs)",
            cv
        );
    }

    // Steady-state summary — aggregate only runs that have a window.
    let ss_stats = vec![
        compute_stats_opt("Request throughput (req/s)", runs, |r| {
            r.ss_request_throughput
        }),
        compute_stats_opt("Output throughput (tok/s)", runs, |r| {
            r.ss_output_throughput
        }),
        compute_stats_opt("Input throughput (tok/s)", runs, |r| r.ss_input_throughput),
        compute_stats_opt("Total token throughput (tok/s)", runs, |r| {
            r.ss_total_token_throughput
        }),
        compute_stats_opt("Mean TTFT (ms)", runs, |r| r.ss_mean_ttft_ms),
        compute_stats_opt("Median TTFT (ms)", runs, |r| r.ss_median_ttft_ms),
        compute_stats_opt("Mean TPOT (ms)", runs, |r| r.ss_mean_tpot_ms),
        compute_stats_opt("Median TPOT (ms)", runs, |r| r.ss_median_tpot_ms),
        compute_stats_opt("P90 TPOT (ms)", runs, |r| r.ss_p90_tpot_ms),
        compute_stats_opt("P99 TPOT (ms)", runs, |r| r.ss_p99_tpot_ms),
    ];

    let k_present = ss_stats[0].n_present;
    let n_total = ss_stats[0].n_total;

    if k_present == 0 {
        println!();
        println!("Steady-state: no runs had a valid window");
    } else {
        println!();
        println!(
            "{:=^80}",
            format!(" Steady-State Summary ({k_present}/{n_total} runs) ")
        );
        if k_present < n_total {
            println!(
                "Aggregated over {k_present}/{n_total} runs ({} runs had no valid window)",
                n_total - k_present
            );
        }
        println!(
            "{:<35} {:>10} {:>10} {:>10} {:>10}",
            "Metric", "Mean", "Std", "Min", "Max"
        );
        for s in &ss_stats {
            println!(
                "{:<35} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
                s.label, s.mean, s.std, s.min, s.max
            );
        }
    }
}

fn compute_stats<F>(label: &'static str, runs: &[RunMetrics], f: F) -> MetricStats
where
    F: Fn(&RunMetrics) -> f64,
{
    let values: Vec<f64> = runs.iter().map(&f).collect();
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    MetricStats {
        label,
        mean,
        std,
        min,
        max,
    }
}

fn fmt(v: f64) -> String {
    if v == 0.0 {
        "0".to_string()
    } else if v == v.floor() && v.abs() < 1e9 {
        format!("{}", v as i64)
    } else {
        format!("{:.2}", v)
    }
}

/// Aggregated statistics over only the Some-valued runs.
struct MetricStatsOpt {
    label: &'static str,
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    n_present: usize,
    n_total: usize,
}

fn compute_stats_opt<F>(label: &'static str, runs: &[RunMetrics], f: F) -> MetricStatsOpt
where
    F: Fn(&RunMetrics) -> Option<f64>,
{
    let values: Vec<f64> = runs.iter().filter_map(&f).collect();
    let n_present = values.len();
    let n_total = runs.len();
    if values.is_empty() {
        return MetricStatsOpt {
            label,
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            n_present,
            n_total,
        };
    }
    let mean = values.iter().sum::<f64>() / n_present as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_present as f64;
    let std = variance.sqrt();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    MetricStatsOpt {
        label,
        mean,
        std,
        min,
        max,
        n_present,
        n_total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_run(ss: Option<f64>) -> RunMetrics {
        RunMetrics {
            request_throughput: 0.0,
            output_throughput: 0.0,
            total_token_throughput: 0.0,
            mean_ttft_ms: 0.0,
            median_ttft_ms: 0.0,
            p99_ttft_ms: 0.0,
            mean_tpot_ms: 0.0,
            median_tpot_ms: 0.0,
            p99_tpot_ms: 0.0,
            mean_itl_ms: 0.0,
            mean_e2el_ms: 0.0,
            p99_e2el_ms: 0.0,
            max_output_tokens_per_s: 0.0,
            completed: 0.0,
            failed: 0.0,
            duration: 0.0,
            ss_request_throughput: ss,
            ss_output_throughput: None,
            ss_input_throughput: None,
            ss_total_token_throughput: None,
            ss_mean_ttft_ms: None,
            ss_median_ttft_ms: None,
            ss_mean_tpot_ms: None,
            ss_median_tpot_ms: None,
            ss_p90_tpot_ms: None,
            ss_p99_tpot_ms: None,
        }
    }

    #[test]
    fn compute_stats_opt_excludes_none() {
        let runs = vec![mk_run(Some(10.0)), mk_run(None), mk_run(Some(20.0))];
        let s = compute_stats_opt("x", &runs, |r| r.ss_request_throughput);
        assert_eq!(s.n_present, 2);
        assert_eq!(s.n_total, 3);
        assert!((s.mean - 15.0).abs() < 1e-9);
        assert_eq!(s.min, 10.0);
        assert_eq!(s.max, 20.0);
    }

    #[test]
    fn compute_stats_opt_all_none() {
        let runs = vec![mk_run(None), mk_run(None)];
        let s = compute_stats_opt("x", &runs, |r| r.ss_request_throughput);
        assert_eq!(s.n_present, 0);
        assert_eq!(s.n_total, 2);
        assert_eq!(s.mean, 0.0);
    }
}
