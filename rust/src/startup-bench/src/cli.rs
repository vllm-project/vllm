// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! CLI argument definitions for `vllm-startup-bench`.

use std::path::PathBuf;

use clap::Parser;

use crate::variant::Variant;

/// Measure and compare process-startup ("time-to-ready") latency across one
/// or more `vllm serve`-style commands.
///
/// Each variant is spawned as a fresh child process (via `sh -c`), timed from
/// spawn until an HTTP readiness endpoint starts responding successfully,
/// then torn down before the next run. Runs are interleaved round-robin
/// across variants (run 1 of A, run 1 of B, run 2 of A, ...) to spread out
/// any systematic drift (thermal throttling, filesystem cache state, etc.)
/// evenly across variants instead of confounding it with run order.
#[derive(Debug, Parser)]
#[command(
    name = "vllm-startup-bench",
    about = "Compare vLLM process startup (time-to-ready) across variants",
    version
)]
pub struct Args {
    /// One variant to benchmark, as `NAME=COMMAND`. The command is executed
    /// with `sh -c`, so shell syntax (env var prefixes, `&&`, quoting) works.
    /// Repeat this flag once per variant; at least one is required, and the
    /// first one given is treated as the baseline for comparison.
    ///
    /// Example: --variant python="vllm serve Qwen/Qwen3-0.6B"
    ///          --variant rust="VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B"
    #[arg(long = "variant", required = true, value_parser = Variant::parse)]
    pub variants: Vec<Variant>,

    /// URL polled with HTTP GET until it returns a successful (2xx) status.
    #[arg(long, default_value = "http://127.0.0.1:8000/health")]
    pub health_url: String,

    /// Number of timed runs per variant (in addition to `--warmup-runs`).
    #[arg(long, default_value_t = 5)]
    pub runs: usize,

    /// Number of untimed warmup runs per variant, run before the timed runs.
    /// Useful to prime OS/filesystem caches so the timed runs aren't skewed
    /// by one-time cold-cache effects unrelated to the code under test.
    #[arg(long, default_value_t = 1)]
    pub warmup_runs: usize,

    /// Maximum time to wait for the readiness endpoint per run, in seconds.
    /// A run that times out is recorded as a failure and excluded from
    /// latency statistics.
    #[arg(long, default_value_t = 300)]
    pub ready_timeout_secs: u64,

    /// Interval between readiness poll attempts, in milliseconds.
    #[arg(long, default_value_t = 50)]
    pub poll_interval_ms: u64,

    /// Grace period after SIGTERM before escalating to SIGKILL, in seconds.
    #[arg(long, default_value_t = 15)]
    pub shutdown_timeout_secs: u64,

    /// Delay after a variant's process has fully exited before starting the
    /// next run, in seconds. Gives the OS time to release the listening port.
    #[arg(long, default_value_t = 2)]
    pub cooldown_secs: u64,

    /// Directory to write each run's stdout/stderr to, named
    /// `<variant>-run<N>.log`. If unset, child output is discarded.
    #[arg(long)]
    pub log_dir: Option<PathBuf>,

    /// Write the full result set (per-run samples and summary stats) as JSON
    /// to this file.
    #[arg(long)]
    pub save_result: Option<PathBuf>,
}
