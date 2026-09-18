// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod backends;
mod benchmark;
mod cli;
mod compare;
mod config;
mod datasets;
mod error;
mod hub;
mod metrics;
mod multi_run;
mod multi_turn;
mod output;
mod rate_control;
mod ready_checker;
mod sweep;
mod tiktoken;
mod tokenizer;

use anyhow::Context;

pub use cli::{
    BackendKind, BenchServeArgs, DatasetName, LoraAssignment, RampUpStrategy, SpeedBenchConfig,
};
use config::BenchConfig;

/// Prepare process-wide resources for a benchmark run.
pub fn prepare_process() {
    // Raise the open-file soft limit to the hard limit. High-concurrency
    // benchmarks (1024+ requests) easily exceed the default 1024 fd soft limit.
    if let Ok(new) = rlimit::increase_nofile_limit(u64::MAX)
        && new > 1024
    {
        tracing::info!(soft_limit = new, "raised open-file limit");
    }
}

/// Run the online serving benchmark.
pub async fn run(args: BenchServeArgs) -> anyhow::Result<()> {
    // --- Compare mode: no server needed, just diff two JSON files ---
    if let Some(ref files) = args.compare {
        return compare::compare_results(&files[0], &files[1]).context("Comparison failed");
    }

    let config = BenchConfig::from_args(&args).context("Configuration error")?;

    async {
        if config.multi_turn {
            if let Some(ref sweep_mc) = args.sweep_max_concurrency {
                // --- Sweep over concurrency in multi-turn mode ---
                let values = sweep::parse_concurrency_values(sweep_mc)
                    .context("Invalid --sweep-max-concurrency")?;
                sweep::run_multi_turn_concurrency_sweep(
                    &config,
                    &values,
                    args.sweep_num_prompts_factor,
                )
                .await?;
            } else {
                // --- Single multi-turn conversation benchmark ---
                multi_turn::run_multi_turn_benchmark(&config).await?;
            }
        } else if let Some(ref sweep_mc) = args.sweep_max_concurrency {
            // --- Sweep over max-concurrency ---
            let values = sweep::parse_concurrency_values(sweep_mc)
                .context("Invalid --sweep-max-concurrency")?;
            sweep::run_concurrency_sweep(&config, &values, args.sweep_num_prompts_factor).await?;
        } else if let Some(ref sweep_rate) = args.sweep_request_rate {
            // --- Sweep over request-rate ---
            let values =
                sweep::parse_rate_values(sweep_rate).context("Invalid --sweep-request-rate")?;
            sweep::run_rate_sweep(&config, &values).await?;
        } else if args.num_runs > 1 {
            // --- Multi-run with statistical aggregation ---
            multi_run::run_multi(&config, args.num_runs).await?;
        } else {
            // --- Normal single benchmark ---
            benchmark::run_benchmark(&config).await?;
        }
        anyhow::Ok(())
    }
    .await
    .context("Benchmark failed")
}
