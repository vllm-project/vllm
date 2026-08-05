// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use anyhow::{Context, Result};
use clap::Parser as _;
use vllm_startup_bench::Args;

fn main() -> Result<()> {
    vllm_tracing::init_tracing("StartupBench");

    let args = Args::parse();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build Tokio runtime")?;

    runtime.block_on(vllm_startup_bench::run(args))
}
