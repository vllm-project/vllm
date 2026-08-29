// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Context;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "vllm-bench",
    about = "Benchmark online serving throughput and offline multimodal preprocessing",
    version = vllm_build_info::VERSION
)]
struct Cli {
    #[command(flatten)]
    args: vllm_bench::BenchServeArgs,

    /// Optional subcommand; absent by default for the online serving benchmark.
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Offline multimodal preprocessing latency benchmark.
    MmProcessor(vllm_bench::MmProcessorArgs),
}

fn main() -> anyhow::Result<()> {
    vllm_tracing::init_tracing("Bench");

    let cli = Cli::parse();
    vllm_bench::prepare_process();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Failed to build tokio runtime")?;

    match cli.command {
        Some(Command::MmProcessor(args)) => runtime.block_on(vllm_bench::run_mm_processor(args)),
        None => runtime.block_on(vllm_bench::run(cli.args)),
    }
}
