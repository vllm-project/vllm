// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Compare process-startup ("time-to-ready") latency across `vllm serve`-style
//! commands, e.g. plain Python `vllm serve` vs `VLLM_USE_RUST_FRONTEND=1
//! vllm serve`, or two arbitrary CLI invocations. Built to give concrete,
//! reproducible before/after numbers for changes on vLLM's startup path,
//! rather than relying on anecdotal impressions of "faster".
//!
//! See `README.md` for usage. Not a general HTTP load-testing tool — for
//! steady-state serving throughput, use `vllm-bench` instead.

mod cli;
mod process_group;
mod runner;
mod stats;
mod variant;

use std::time::Duration;

use anyhow::{Context, Result};
use serde::Serialize;
use tracing::info;

pub use cli::Args;
use runner::RunOutcome;
use stats::Summary;
use variant::Variant;

/// Per-variant timed samples and derived summary statistics.
#[derive(Debug, Serialize)]
struct VariantResult {
    name: String,
    command: String,
    runs: Vec<RunOutcome>,
    summary: Option<Summary>,
}

/// Full result set, suitable for `--save-result`.
#[derive(Debug, Serialize)]
struct Results {
    health_url: String,
    variants: Vec<VariantResult>,
}

/// Run the full benchmark: warmups, interleaved timed runs, then print (and
/// optionally save) a comparison report.
pub async fn run(args: Args) -> Result<()> {
    if let Some(dir) = &args.log_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create log dir {}", dir.display()))?;
    }

    let ready_timeout = Duration::from_secs(args.ready_timeout_secs);
    let poll_interval = Duration::from_millis(args.poll_interval_ms);
    let shutdown_timeout = Duration::from_secs(args.shutdown_timeout_secs);
    let cooldown = Duration::from_secs(args.cooldown_secs);

    let mut timed_runs: Vec<Vec<RunOutcome>> = vec![Vec::new(); args.variants.len()];

    let total_rounds = args.warmup_runs + args.runs;
    for round in 0..total_rounds {
        let is_warmup = round < args.warmup_runs;
        let kind = if is_warmup { "warmup" } else { "timed" };

        for (idx, variant) in args.variants.iter().enumerate() {
            info!(variant = %variant.name, round, kind, "starting run");

            let log_path = args.log_dir.as_ref().map(|dir| {
                let suffix = if is_warmup {
                    format!("warmup{round}")
                } else {
                    format!("run{round}")
                };
                dir.join(format!("{}-{suffix}.log", variant.name))
            });

            let outcome = runner::run_once(
                &variant.command,
                &args.health_url,
                ready_timeout,
                poll_interval,
                shutdown_timeout,
                log_path.as_deref(),
            )
            .await
            .with_context(|| format!("run failed for variant {:?}", variant.name))?;

            match outcome.ready_secs {
                Some(secs) => {
                    info!(variant = %variant.name, round, kind, ready_secs = secs, "ready")
                }
                None => {
                    tracing::warn!(variant = %variant.name, round, kind, "did not become ready")
                }
            }

            if !is_warmup {
                timed_runs[idx].push(outcome);
            }

            tokio::time::sleep(cooldown).await;
        }
    }

    let results = summarize(&args.variants, timed_runs, &args.health_url);
    print_report(&results);

    if let Some(path) = &args.save_result {
        let json = serde_json::to_string_pretty(&results).context("failed to serialize results")?;
        std::fs::write(path, json)
            .with_context(|| format!("failed to write results to {}", path.display()))?;
        println!("\nSaved results to {}", path.display());
    }

    Ok(())
}

fn summarize(variants: &[Variant], timed_runs: Vec<Vec<RunOutcome>>, health_url: &str) -> Results {
    let variants = variants
        .iter()
        .zip(timed_runs)
        .map(|(variant, runs)| {
            let (successes, failures): (Vec<_>, Vec<_>) =
                runs.iter().partition(|r| r.ready_secs.is_some());
            let samples: Vec<f64> = successes.iter().filter_map(|r| r.ready_secs).collect();
            let summary = Summary::from_samples(samples, failures.len());
            VariantResult {
                name: variant.name.clone(),
                command: variant.command.clone(),
                runs,
                summary,
            }
        })
        .collect();

    Results {
        health_url: health_url.to_string(),
        variants,
    }
}

fn print_report(results: &Results) {
    println!("\n=== Startup time-to-ready ({}) ===\n", results.health_url);
    println!(
        "{:<16} {:>7} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "variant", "runs", "failures", "mean(s)", "median(s)", "min(s)", "max(s)", "stddev(s)"
    );
    for variant in &results.variants {
        match &variant.summary {
            Some(s) => println!(
                "{:<16} {:>7} {:>9} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                variant.name,
                s.samples,
                s.failures,
                s.mean_secs,
                s.median_secs,
                s.min_secs,
                s.max_secs,
                s.stddev_secs
            ),
            None => println!(
                "{:<16} {:>7} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}",
                variant.name,
                0,
                variant.runs.len(),
                "-",
                "-",
                "-",
                "-",
                "-"
            ),
        }
    }

    let Some(baseline) = results.variants.first() else {
        return;
    };
    let Some(baseline_summary) = &baseline.summary else {
        println!(
            "\nBaseline variant {:?} never became ready; skipping comparison.",
            baseline.name
        );
        return;
    };

    println!(
        "\n=== Comparison vs. baseline {:?} (median) ===\n",
        baseline.name
    );
    for variant in &results.variants[1..] {
        let Some(summary) = &variant.summary else {
            println!("{:<16} never became ready", variant.name);
            continue;
        };
        let delta = summary.median_secs - baseline_summary.median_secs;
        let pct = (delta / baseline_summary.median_secs) * 100.0;
        if delta < 0.0 {
            println!(
                "{:<16} {:.2}x faster  ({:+.3}s, {:+.1}%)",
                variant.name,
                baseline_summary.median_secs / summary.median_secs,
                delta,
                pct
            );
        } else {
            println!(
                "{:<16} SLOWER by {:.2}x ({:+.3}s, {:+.1}%) -- regression vs. baseline",
                variant.name,
                summary.median_secs / baseline_summary.median_secs,
                delta,
                pct
            );
        }
    }
}
