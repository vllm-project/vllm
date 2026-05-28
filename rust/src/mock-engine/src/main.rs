use anyhow::{Context, Result};
use clap::Parser as _;
use tokio_util::sync::CancellationToken;
use tracing::{Level, info};
use vllm_mock_engine::Opt;

fn init_tracing() {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
}

/// Create a cancellation token that is triggered by Ctrl-C.
fn shutdown_signal() -> CancellationToken {
    let token = CancellationToken::new();
    let shutdown = token.clone();

    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("failed to install Ctrl-C signal handler");
        info!("received shutdown signal (Ctrl-C), shutting down...");
        shutdown.cancel();
    });

    token
}

fn main() -> Result<()> {
    init_tracing();
    let opt = Opt::parse();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build Tokio runtime")?;

    runtime.block_on(async move {
        let shutdown = shutdown_signal();
        vllm_mock_engine::run(opt, shutdown).await
    })
}
