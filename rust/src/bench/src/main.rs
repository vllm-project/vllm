// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Context;
use clap::Parser;

#[derive(Parser)]
#[command(
    name = "vllm-bench",
    about = "Benchmark online serving throughput",
    version
)]
struct Cli {
    #[command(flatten)]
    args: vllm_bench::BenchServeArgs,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    vllm_bench::prepare_process();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Failed to build tokio runtime")?;

    runtime.block_on(vllm_bench::run(cli.args))
}
