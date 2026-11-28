// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use crate::config::BenchConfig;
use crate::error::Result;
use crate::output::json::save_point_result;

/// Key metrics extracted from a single run's JSON result.
struct RunMetrics {
    request_throughput: f64,
    input_sequence_throughput: Option<f64>,
    output_throughput: f64,
    total_token_throughput: f64,
    mean_ttft_ms: Option<f64>,
    median_ttft_ms: Option<f64>,
    p99_ttft_ms: Option<f64>,
    mean_tpot_ms: Option<f64>,
    median_tpot_ms: Option<f64>,
    p99_tpot_ms: Option<f64>,
    mean_itl_ms: Option<f64>,
    mean_e2el_ms: Option<f64>,
    p99_e2el_ms: Option<f64>,
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
            input_sequence_throughput: get_opt(json, "input_sequence_throughput"),
            output_throughput: get(json, "output_throughput"),
            total_token_throughput: get(json, "total_token_throughput"),
            mean_ttft_ms: get_opt(json, "mean_ttft_ms"),
            median_ttft_ms: get_opt(json, "median_ttft_ms"),
            p99_ttft_ms: get_opt(json, "p99_ttft_ms"),
            mean_tpot_ms: get_opt(json, "mean_tpot_ms"),
            median_tpot_ms: get_opt(json, "median_tpot_ms"),
            p99_tpot_ms: get_opt(json, "p99_tpot_ms"),
            mean_itl_ms: get_opt(json, "mean_itl_ms"),
            mean_e2el_ms: get_opt(json, "mean_e2el_ms"),
            p99_e2el_ms: get_opt(json, "p99_e2el_ms"),
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

fn get_opt(json: &serde_json::Value, key: &str) -> Option<f64> {
    json.get(key).and_then(|v| v.as_f64())
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
    let current_dt = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();

    for i in 0..num_runs {
        println!("{:-^70}", format!(" Run {}/{} ", i + 1, num_runs));

        let result = crate::benchmark::run_benchmark(&config.without_save()).await?;
        let point = format!("run{}", i + 1);
        save_point_result(config, &result, &current_dt, Some(&point))?;
        all_runs.push(RunMetrics::from_json(&result));

        println!();
    }

    print_multi_run_summary(&all_runs);
    Ok(())
}

fn print_multi_run_summary(runs: &[RunMetrics]) {
    let n = runs.len();

    let stats = summary_stats(runs);

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

/// Rows of the multi-run summary table. Latency metrics only exist in the
/// result JSON when they were measured (`--percentile-metrics`, the chosen
/// `--metric-percentiles`, pooling backends), so a row is shown only when every
/// run has it, the same rule `compare` uses, instead of printing zeros.
fn summary_stats(runs: &[RunMetrics]) -> Vec<MetricStats> {
    type OptMetric = fn(&RunMetrics) -> Option<f64>;

    fn push_if_all(
        stats: &mut Vec<MetricStats>,
        label: &'static str,
        runs: &[RunMetrics],
        f: OptMetric,
    ) {
        if runs.iter().all(|r| f(r).is_some()) {
            stats.push(compute_stats(label, runs, |r| f(r).unwrap_or_default()));
        }
    }

    let mut stats = vec![compute_stats("Request throughput (req/s)", runs, |r| {
        r.request_throughput
    })];
    push_if_all(&mut stats, "Input throughput (inputs/s)", runs, |r| {
        r.input_sequence_throughput
    });
    stats.extend([
        compute_stats("Output throughput (tok/s)", runs, |r| r.output_throughput),
        compute_stats("Total token throughput (tok/s)", runs, |r| {
            r.total_token_throughput
        }),
        compute_stats("Peak output tokens/s", runs, |r| r.max_output_tokens_per_s),
    ]);
    let latency: [(&'static str, OptMetric); 9] = [
        ("Mean TTFT (ms)", |r| r.mean_ttft_ms),
        ("Median TTFT (ms)", |r| r.median_ttft_ms),
        ("P99 TTFT (ms)", |r| r.p99_ttft_ms),
        ("Mean TPOT (ms)", |r| r.mean_tpot_ms),
        ("Median TPOT (ms)", |r| r.median_tpot_ms),
        ("P99 TPOT (ms)", |r| r.p99_tpot_ms),
        ("Mean ITL (ms)", |r| r.mean_itl_ms),
        ("Mean E2EL (ms)", |r| r.mean_e2el_ms),
        ("P99 E2EL (ms)", |r| r.p99_e2el_ms),
    ];
    for (label, f) in latency {
        push_if_all(&mut stats, label, runs, f);
    }
    stats.extend([
        compute_stats("Completed requests", runs, |r| r.completed),
        compute_stats("Failed requests", runs, |r| r.failed),
        compute_stats("Duration (s)", runs, |r| r.duration),
    ]);
    stats
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
            input_sequence_throughput: None,
            output_throughput: 0.0,
            total_token_throughput: 0.0,
            mean_ttft_ms: None,
            median_ttft_ms: None,
            p99_ttft_ms: None,
            mean_tpot_ms: None,
            median_tpot_ms: None,
            p99_tpot_ms: None,
            mean_itl_ms: None,
            mean_e2el_ms: None,
            p99_e2el_ms: None,
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

    fn labels(stats: &[MetricStats]) -> Vec<&'static str> {
        stats.iter().map(|s| s.label).collect()
    }

    #[test]
    fn summary_skips_metrics_that_were_not_measured() {
        // `--percentile-metrics ttft,tpot --metric-percentiles 50,90`: no ITL,
        // E2EL or p99 keys in the result JSON.
        let json = serde_json::json!({
            "request_throughput": 4.0,
            "mean_ttft_ms": 10.0,
            "median_ttft_ms": 9.0,
            "p50_ttft_ms": 9.0,
            "p90_ttft_ms": 12.0,
            "mean_tpot_ms": 2.0,
            "median_tpot_ms": 2.0,
            "completed": 20,
        });
        let runs = vec![RunMetrics::from_json(&json), RunMetrics::from_json(&json)];
        let stats = summary_stats(&runs);
        let labels = labels(&stats);

        for shown in ["Mean TTFT (ms)", "Median TTFT (ms)", "Mean TPOT (ms)"] {
            assert!(labels.contains(&shown), "{shown} missing from {labels:?}");
        }
        for hidden in [
            "P99 TTFT (ms)",
            "P99 TPOT (ms)",
            "Mean ITL (ms)",
            "Mean E2EL (ms)",
            "P99 E2EL (ms)",
        ] {
            assert!(!labels.contains(&hidden), "{hidden} should not be shown");
        }
        let ttft = stats.iter().find(|s| s.label == "Mean TTFT (ms)").unwrap();
        assert_eq!(ttft.mean, 10.0);
    }

    #[test]
    fn summary_skips_metric_missing_from_some_runs() {
        let mut with_e2el = mk_run(None);
        with_e2el.mean_e2el_ms = Some(50.0);
        let runs = vec![with_e2el, mk_run(None)];
        let labels = labels(&summary_stats(&runs));
        assert!(!labels.contains(&"Mean E2EL (ms)"));
        assert!(labels.contains(&"Completed requests"));
    }
}
