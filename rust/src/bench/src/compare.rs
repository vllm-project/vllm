// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use crate::error::{BenchError, Result};

/// Metric definition for comparison: name, JSON key, and whether lower is better.
struct MetricDef {
    label: &'static str,
    key: &'static str,
    lower_is_better: bool,
}

const METRICS: &[MetricDef] = &[
    MetricDef {
        label: "Request throughput (req/s)",
        key: "request_throughput",
        lower_is_better: false,
    },
    MetricDef {
        label: "Output throughput (tok/s)",
        key: "output_throughput",
        lower_is_better: false,
    },
    MetricDef {
        label: "Total token throughput (tok/s)",
        key: "total_token_throughput",
        lower_is_better: false,
    },
    MetricDef {
        label: "Peak output tokens/s",
        key: "max_output_tokens_per_s",
        lower_is_better: false,
    },
    MetricDef {
        label: "Peak concurrent requests",
        key: "max_concurrent_requests",
        lower_is_better: false,
    },
    MetricDef {
        label: "Mean TTFT (ms)",
        key: "mean_ttft_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "Median TTFT (ms)",
        key: "median_ttft_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "P99 TTFT (ms)",
        key: "p99_ttft_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "Mean TPOT (ms)",
        key: "mean_tpot_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "Median TPOT (ms)",
        key: "median_tpot_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "P99 TPOT (ms)",
        key: "p99_tpot_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "Mean ITL (ms)",
        key: "mean_itl_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "Median ITL (ms)",
        key: "median_itl_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "P99 ITL (ms)",
        key: "p99_itl_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "Mean E2EL (ms)",
        key: "mean_e2el_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "Median E2EL (ms)",
        key: "median_e2el_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "P99 E2EL (ms)",
        key: "p99_e2el_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "Completed requests",
        key: "completed",
        lower_is_better: false,
    },
    MetricDef {
        label: "Failed requests",
        key: "failed",
        lower_is_better: true,
    },
    MetricDef {
        label: "Duration (s)",
        key: "duration",
        lower_is_better: true,
    },
];

const STEADY_STATE_METRICS: &[MetricDef] = &[
    MetricDef {
        label: "SS Request throughput (req/s)",
        key: "request_throughput",
        lower_is_better: false,
    },
    MetricDef {
        label: "SS Output throughput (tok/s)",
        key: "output_throughput",
        lower_is_better: false,
    },
    MetricDef {
        label: "SS Input throughput (tok/s)",
        key: "input_throughput",
        lower_is_better: false,
    },
    MetricDef {
        label: "SS Total token throughput (tok/s)",
        key: "total_token_throughput",
        lower_is_better: false,
    },
    MetricDef {
        label: "SS Mean TTFT (ms)",
        key: "mean_ttft_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "SS Median TTFT (ms)",
        key: "median_ttft_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "SS Mean TPOT (ms)",
        key: "mean_tpot_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "SS Median TPOT (ms)",
        key: "median_tpot_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "SS P90 TPOT (ms)",
        key: "p90_tpot_ms",
        lower_is_better: true,
    },
    MetricDef {
        label: "SS P99 TPOT (ms)",
        key: "p99_tpot_ms",
        lower_is_better: true,
    },
];

/// Compare two benchmark result JSON files and print a side-by-side table.
pub fn compare_results(file_a: &str, file_b: &str) -> Result<()> {
    let json_a = load_result_json(file_a)?;
    let json_b = load_result_json(file_b)?;

    // Print header with file context
    let model_a = json_a.get("model_id").and_then(|v| v.as_str()).unwrap_or("?");
    let model_b = json_b.get("model_id").and_then(|v| v.as_str()).unwrap_or("?");
    let date_a = json_a.get("date").and_then(|v| v.as_str()).unwrap_or("?");
    let date_b = json_b.get("date").and_then(|v| v.as_str()).unwrap_or("?");

    println!("{:=^90}", " Benchmark Comparison ");
    println!("  A: {} (model: {}, date: {})", file_a, model_a, date_a);
    println!("  B: {} (model: {}, date: {})", file_b, model_b, date_b);
    println!();

    // Print comparison table
    println!(
        "{:<35} {:>12} {:>12} {:>10} {:>8}",
        "Metric", "A", "B", "Delta", "Change"
    );
    println!("{:-<35} {:->12} {:->12} {:->10} {:->8}", "", "", "", "", "");

    for metric in METRICS {
        let val_a = get_f64(&json_a, metric.key);
        let val_b = get_f64(&json_b, metric.key);

        match (val_a, val_b) {
            (Some(a), Some(b)) => print_diff_row(metric, a, b),
            _ => {
                // One or both values missing — skip
            }
        }
    }

    // Steady-state section — both sides must have the block; otherwise render N/A.
    let ss_a = json_a.get("steady_state");
    let ss_b = json_b.get("steady_state");
    let both_present =
        matches!(ss_a, Some(v) if !v.is_null()) && matches!(ss_b, Some(v) if !v.is_null());

    println!();
    println!("{:=^70}", " Steady-State Comparison ");
    if !both_present {
        println!("N/A — one or both runs have no steady-state window");
    } else {
        let ss_a = ss_a.unwrap();
        let ss_b = ss_b.unwrap();
        for m in STEADY_STATE_METRICS {
            let a = ss_a.get(m.key).and_then(|v| v.as_f64());
            let b = ss_b.get(m.key).and_then(|v| v.as_f64());
            match (a, b) {
                (Some(a), Some(b)) => print_diff_row(m, a, b),
                _ => println!("{:<35} N/A", m.label),
            }
        }
    }

    println!("{:=<90}", "");
    println!();
    println!("Legend: + = improvement, - = regression (relative to A → B)");

    Ok(())
}

fn print_diff_row(metric: &MetricDef, a: f64, b: f64) {
    let delta = b - a;
    let pct = if a.abs() > 1e-10 {
        (delta / a) * 100.0
    } else if b.abs() > 1e-10 {
        f64::INFINITY
    } else {
        0.0
    };

    // Determine if change is good/bad/neutral
    let marker = if delta.abs() < 1e-10 {
        " "
    } else if metric.lower_is_better {
        if delta < 0.0 { "+" } else { "-" }
    } else if delta > 0.0 {
        "+"
    } else {
        "-"
    };

    let delta_str = format_delta(delta);
    let pct_str = if pct.is_infinite() {
        "inf%".to_string()
    } else {
        format!("{:+.1}%", pct)
    };

    println!(
        "{:<35} {:>12} {:>12} {:>10} {:>7}{}",
        metric.label,
        format_value(a),
        format_value(b),
        delta_str,
        pct_str,
        marker,
    );
}

fn load_result_json(path: &str) -> Result<serde_json::Value> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| BenchError::Config(format!("Cannot read result file '{path}': {e}")))?;

    // Support JSONL: take the last line (most recent run)
    let json_str = content.lines().rfind(|l| !l.trim().is_empty()).unwrap_or(&content);

    serde_json::from_str(json_str)
        .map_err(|e| BenchError::Config(format!("Cannot parse JSON from '{path}': {e}")))
}

fn get_f64(json: &serde_json::Value, key: &str) -> Option<f64> {
    json.get(key).and_then(|v| v.as_f64())
}

fn format_value(v: f64) -> String {
    if v == v.floor() && v.abs() < 1e12 {
        format!("{}", v as i64)
    } else {
        format!("{:.2}", v)
    }
}

fn format_delta(d: f64) -> String {
    if d == d.floor() && d.abs() < 1e12 {
        format!("{:+}", d as i64)
    } else {
        format!("{:+.2}", d)
    }
}
