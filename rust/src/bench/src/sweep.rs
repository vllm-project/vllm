// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use crate::config::BenchConfig;
use crate::error::{BenchError, Result};
use crate::output::json::{compute_result_filename, save_result};

/// Reset the server's prefix cache by calling POST /reset_prefix_cache.
/// Requires VLLM_SERVER_DEV_MODE=1 on the vLLM server.
async fn reset_prefix_cache(base_url: &str) -> Result<()> {
    let url = format!("{}/reset_prefix_cache", base_url.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .send()
        .await
        .map_err(|e| BenchError::Backend(format!("Failed to reset prefix cache: {e}")))?;
    if resp.status().is_success() {
        println!("Prefix cache reset successfully.");
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(BenchError::Backend(format!(
            "Failed to reset prefix cache: HTTP {status} — {body}. \
             Is VLLM_SERVER_DEV_MODE=1 set on the server?"
        )));
    }
    Ok(())
}

/// Result of a single sweep point.
struct SweepPoint {
    label: String,
    #[allow(dead_code)]
    value: f64,
    result_json: serde_json::Value,
    ss_request_throughput: Option<f64>,
    ss_output_throughput: Option<f64>,
}

/// Run a sweep over max-concurrency values.
pub async fn run_concurrency_sweep(
    base_config: &BenchConfig,
    values: &[usize],
    num_prompts_factor: Option<usize>,
) -> Result<()> {
    println!("{:=^70}", " Concurrency Sweep ");
    println!(
        "Sweeping --max-concurrency over {} values: {:?}",
        values.len(),
        values
    );
    println!();

    let mut points = Vec::with_capacity(values.len());

    let current_dt = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();

    for (i, &mc) in values.iter().enumerate() {
        println!(
            "{:-^70}",
            format!(" Run {}/{}: max_concurrency={} ", i + 1, values.len(), mc)
        );

        if base_config.reset_prefix_cache {
            reset_prefix_cache(&base_config.base_url).await?;
        }

        let mut config = base_config.clone();
        config.max_concurrency = Some(mc);
        if let Some(factor) = num_prompts_factor {
            config.num_prompts = mc * factor;
        }
        // Suppress per-run save (we save with concurrency suffix below)
        config.save_result = false;
        config.append_result = false;

        let result = crate::benchmark::run_benchmark(&config).await?;

        if base_config.save_result {
            let model_id = config.model.as_deref().unwrap_or("unknown");
            let file_name = compute_result_filename(&config, model_id, &current_dt);
            if let Some(ref dir) = config.result_dir {
                std::fs::create_dir_all(dir)?;
            }
            save_result(&result, &file_name)?;
        }

        points.push(SweepPoint {
            label: format!("concurrency={mc}"),
            value: mc as f64,
            ss_request_throughput: ss_f64(&result, "request_throughput"),
            ss_output_throughput: ss_f64(&result, "output_throughput"),
            result_json: result,
        });

        println!();
    }

    print_sweep_summary(
        "Max Concurrency",
        &points,
        &base_config.sweep_summary_percentiles,
    );
    Ok(())
}

/// Run a sweep over multi-turn concurrency values.
pub async fn run_multi_turn_concurrency_sweep(
    base_config: &BenchConfig,
    values: &[usize],
    num_prompts_factor: Option<usize>,
) -> Result<()> {
    println!("{:=^70}", " Multi-Turn Concurrency Sweep ");
    println!(
        "Sweeping --multi-turn-concurrency over {} values: {:?}",
        values.len(),
        values
    );
    println!();

    let mut points = Vec::with_capacity(values.len());
    let current_dt = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();

    for (i, &mc) in values.iter().enumerate() {
        println!(
            "{:-^70}",
            format!(
                " Run {}/{}: multi_turn_concurrency={} ",
                i + 1,
                values.len(),
                mc
            )
        );

        if base_config.reset_prefix_cache {
            reset_prefix_cache(&base_config.base_url).await?;
        }

        let mut config = base_config.clone();
        config.multi_turn_concurrency = Some(mc);
        // Set max_concurrency so compute_result_filename includes the
        // concurrency suffix — without this every sweep point overwrites
        // the same file.
        config.max_concurrency = Some(mc);
        if let Some(factor) = num_prompts_factor {
            config.num_prompts = mc * factor;
        }
        // Suppress per-run save (we save with concurrency suffix below)
        config.save_result = false;
        config.append_result = false;

        let result = crate::multi_turn::run_multi_turn_benchmark(&config).await?;

        if base_config.save_result {
            let model_id = config.model.as_deref().unwrap_or("unknown");
            let file_name = compute_result_filename(&config, model_id, &current_dt);
            if let Some(ref dir) = config.result_dir {
                std::fs::create_dir_all(dir)?;
            }
            save_result(&result, &file_name)?;
        }

        points.push(SweepPoint {
            label: format!("concurrency={mc}"),
            value: mc as f64,
            ss_request_throughput: ss_f64(&result, "request_throughput"),
            ss_output_throughput: ss_f64(&result, "output_throughput"),
            result_json: result,
        });

        println!();
    }

    print_sweep_summary(
        "MT Concurrency",
        &points,
        &base_config.sweep_summary_percentiles,
    );
    Ok(())
}

/// Run a sweep over request-rate values.
pub async fn run_rate_sweep(base_config: &BenchConfig, values: &[f64]) -> Result<()> {
    println!("{:=^70}", " Request Rate Sweep ");
    println!(
        "Sweeping --request-rate over {} values: {:?}",
        values.len(),
        values
    );
    println!();

    let mut points = Vec::with_capacity(values.len());
    let current_dt = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();

    for (i, &rate) in values.iter().enumerate() {
        let rate_str = if rate.is_infinite() {
            "inf".to_string()
        } else {
            format!("{rate}")
        };
        println!(
            "{:-^70}",
            format!(
                " Run {}/{}: request_rate={} ",
                i + 1,
                values.len(),
                rate_str
            )
        );

        if base_config.reset_prefix_cache {
            reset_prefix_cache(&base_config.base_url).await?;
        }

        let mut config = base_config.clone();
        config.request_rate = rate;
        config.save_result = false;
        config.append_result = false;

        let result = crate::benchmark::run_benchmark(&config).await?;

        if base_config.save_result {
            let model_id = config.model.as_deref().unwrap_or("unknown");
            let file_name = compute_result_filename(&config, model_id, &current_dt);
            if let Some(ref dir) = config.result_dir {
                std::fs::create_dir_all(dir)?;
            }
            save_result(&result, &file_name)?;
        }

        points.push(SweepPoint {
            label: format!("rate={rate_str}"),
            value: rate,
            ss_request_throughput: ss_f64(&result, "request_throughput"),
            ss_output_throughput: ss_f64(&result, "output_throughput"),
            result_json: result,
        });

        println!();
    }

    print_sweep_summary(
        "Request Rate",
        &points,
        &base_config.sweep_summary_percentiles,
    );
    Ok(())
}

/// Print a summary table after all sweep points complete.
fn print_sweep_summary(param_name: &str, points: &[SweepPoint], summary_percentiles: &[f64]) {
    let param_width = param_name.len().max(20);
    let columns = build_summary_columns(summary_percentiles);
    let total_width = param_width
        + columns.iter().map(SummaryColumn::render_width).sum::<usize>()
        + columns.len();

    println!("{:=^width$}", " Sweep Summary ", width = total_width);

    print!("{param_name:<param_width$}");
    for column in &columns {
        print!(" {:>width$}", column.header, width = column.render_width());
    }
    println!();

    print!("{:-<param_width$}", "");
    for column in &columns {
        print!(" {:-<width$}", "", width = column.render_width());
    }
    println!();

    for point in points {
        let row = render_summary_row(point, param_width, &columns);
        println!("{row}");
    }

    println!("{:=<width$}", "", width = total_width);

    // Find the best throughput point
    if let Some(best) = points
        .iter()
        .filter(|p| get_f64(&p.result_json, "request_throughput").is_some())
        .max_by(|a, b| {
            let ta = get_f64(&a.result_json, "request_throughput").unwrap_or(0.0);
            let tb = get_f64(&b.result_json, "request_throughput").unwrap_or(0.0);
            ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        let tp = get_f64(&best.result_json, "request_throughput").unwrap_or(0.0);
        println!("Best throughput: {:.2} req/s at {}", tp, best.label);
    }
}

fn build_summary_columns(summary_percentiles: &[f64]) -> Vec<SummaryColumn> {
    let mut columns = vec![
        SummaryColumn::new("Req/s", "request_throughput", 10),
        SummaryColumn::new("Tok/s", "output_throughput", 10),
        SummaryColumn::new("Total tok/s", "total_token_throughput", 12),
        SummaryColumn::new("SS req/s", SS_REQUEST_THROUGHPUT_KEY, 10),
        SummaryColumn::new("SS out tok/s", SS_OUTPUT_THROUGHPUT_KEY, 12),
        SummaryColumn::new("P50 TTFT(ms)", "median_ttft_ms", 12),
        SummaryColumn::new("P50 TPOT(ms)", "median_tpot_ms", 12),
        SummaryColumn::new("P90 TTFT(ms)", "p90_ttft_ms", 12),
        SummaryColumn::new("P90 TPOT(ms)", "p90_tpot_ms", 12),
    ];

    for &percentile in summary_percentiles {
        if percentile == 50.0 || percentile == 90.0 {
            continue;
        }
        let p_str = format_percentile(percentile);
        columns.push(SummaryColumn::new(
            &format!("P{p_str} TTFT(ms)"),
            &format!("p{p_str}_ttft_ms"),
            12,
        ));
        columns.push(SummaryColumn::new(
            &format!("P{p_str} TPOT(ms)"),
            &format!("p{p_str}_tpot_ms"),
            12,
        ));
    }

    columns
}

fn render_summary_row(point: &SweepPoint, param_width: usize, columns: &[SummaryColumn]) -> String {
    let mut row = format!("{:<param_width$}", point.label, param_width = param_width);
    for column in columns {
        let value = match column.key.as_str() {
            SS_REQUEST_THROUGHPUT_KEY => point.ss_request_throughput,
            SS_OUTPUT_THROUGHPUT_KEY => point.ss_output_throughput,
            key => get_f64(&point.result_json, key),
        };
        row.push(' ');
        row.push_str(&format!(
            "{:>width$}",
            fmt_f64(value),
            width = column.render_width()
        ));
    }
    row
}

fn format_percentile(percentile: f64) -> String {
    if percentile == percentile.floor() {
        format!("{}", percentile as i64)
    } else {
        format!("{percentile}")
    }
}

struct SummaryColumn {
    header: String,
    key: String,
    width: usize,
}

impl SummaryColumn {
    fn new(header: &str, key: &str, width: usize) -> Self {
        Self {
            header: header.to_string(),
            key: key.to_string(),
            width,
        }
    }

    fn render_width(&self) -> usize {
        self.width.max(self.header.len())
    }
}

/// Parse a comma-separated list of concurrency values.
pub fn parse_concurrency_values(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|v| {
            v.trim().parse::<usize>().map_err(|_| {
                BenchError::Config(format!("Invalid concurrency value: '{}'", v.trim()))
            })
        })
        .collect()
}

/// Parse a comma-separated list of request rate values (supports "inf").
pub fn parse_rate_values(s: &str) -> Result<Vec<f64>> {
    s.split(',')
        .map(|v| {
            let v = v.trim();
            if v == "inf" {
                Ok(f64::INFINITY)
            } else {
                v.parse::<f64>()
                    .map_err(|_| BenchError::Config(format!("Invalid request rate value: '{v}'")))
            }
        })
        .collect()
}

fn get_f64(json: &serde_json::Value, key: &str) -> Option<f64> {
    json.get(key).and_then(|v| v.as_f64())
}

/// Sentinel keys used on `SummaryColumn` to mark steady-state columns whose
/// values come from `SweepPoint::ss_request_throughput` / `ss_output_throughput`
/// rather than from the result JSON top level.
const SS_REQUEST_THROUGHPUT_KEY: &str = "__ss_request_throughput";
const SS_OUTPUT_THROUGHPUT_KEY: &str = "__ss_output_throughput";

/// Extract a numeric field from the `steady_state` sub-object of a result JSON.
/// Returns `None` when `steady_state` is missing or null, or when the field is
/// absent / non-numeric.
fn ss_f64(result: &serde_json::Value, key: &str) -> Option<f64> {
    result
        .get("steady_state")
        .and_then(|ss| if ss.is_null() { None } else { Some(ss) })
        .and_then(|ss| ss.get(key))
        .and_then(|v| v.as_f64())
}

fn fmt_f64(v: Option<f64>) -> String {
    match v {
        Some(f) => format!("{:.2}", f),
        None => "-".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn point(result_json: serde_json::Value) -> SweepPoint {
        let ss_request_throughput = ss_f64(&result_json, "request_throughput");
        let ss_output_throughput = ss_f64(&result_json, "output_throughput");
        SweepPoint {
            label: "point".to_string(),
            value: 1.0,
            result_json,
            ss_request_throughput,
            ss_output_throughput,
        }
    }

    #[test]
    fn summary_columns_follow_requested_percentile_order() {
        let columns = build_summary_columns(&[90.0, 95.0]);
        let headers: Vec<&str> = columns.iter().map(|column| column.header.as_str()).collect();

        assert_eq!(
            headers,
            vec![
                "Req/s",
                "Tok/s",
                "Total tok/s",
                "SS req/s",
                "SS out tok/s",
                "P50 TTFT(ms)",
                "P50 TPOT(ms)",
                "P90 TTFT(ms)",
                "P90 TPOT(ms)",
                "P95 TTFT(ms)",
                "P95 TPOT(ms)",
            ]
        );
    }

    #[test]
    fn summary_row_uses_dash_for_missing_requested_percentile_values() {
        let columns = build_summary_columns(&[90.0]);
        let row = render_summary_row(
            &point(json!({
                "request_throughput": 1.0,
                "output_throughput": 2.0,
                "total_token_throughput": 3.0,
                "median_ttft_ms": 10.0,
                "median_tpot_ms": 20.0,
                "p90_ttft_ms": 40.0,
            })),
            20,
            &columns,
        );

        assert!(row.contains("40.00"));
        assert!(row.contains(" -"));
    }

    #[test]
    fn summary_columns_skip_duplicate_p50_and_p90_percentiles() {
        let columns = build_summary_columns(&[50.0, 90.0]);
        let headers: Vec<&str> = columns.iter().map(|column| column.header.as_str()).collect();

        assert_eq!(
            headers,
            vec![
                "Req/s",
                "Tok/s",
                "Total tok/s",
                "SS req/s",
                "SS out tok/s",
                "P50 TTFT(ms)",
                "P50 TPOT(ms)",
                "P90 TTFT(ms)",
                "P90 TPOT(ms)",
            ]
        );
    }

    #[test]
    fn summary_row_renders_steady_state_columns() {
        let columns = build_summary_columns(&[]);
        let row = render_summary_row(
            &point(json!({
                "request_throughput": 1.0,
                "output_throughput": 2.0,
                "total_token_throughput": 3.0,
                "median_ttft_ms": 10.0,
                "median_tpot_ms": 20.0,
                "p90_ttft_ms": 40.0,
                "p90_tpot_ms": 50.0,
                "steady_state": {
                    "request_throughput": 0.77,
                    "output_throughput": 88.88,
                },
            })),
            20,
            &columns,
        );

        assert!(row.contains("0.77"));
        assert!(row.contains("88.88"));
    }

    #[test]
    fn summary_row_renders_dash_when_steady_state_missing() {
        let columns = build_summary_columns(&[]);
        let row = render_summary_row(
            &point(json!({
                "request_throughput": 1.0,
                "output_throughput": 2.0,
                "total_token_throughput": 3.0,
                "median_ttft_ms": 10.0,
                "median_tpot_ms": 20.0,
                "p90_ttft_ms": 40.0,
                "p90_tpot_ms": 50.0,
                "steady_state": null,
            })),
            20,
            &columns,
        );

        // Two dashes for the two SS columns (SS req/s, SS out tok/s). Other
        // columns have values so they won't emit dashes.
        let dash_count = row.matches(" -").count();
        assert!(
            dash_count >= 2,
            "expected at least 2 dashes, got {dash_count}: {row}"
        );
    }
}
