use anyhow::{Context, Result, bail};
use clap::Parser;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use vllm_engine_core_client::EngineId;
use vllm_engine_core_client::mock_engine::{
    MockEngineConfig, MockEngineSockets, connect_to_frontend,
};

pub mod engine;
pub mod io;

/// Standalone engine-core protocol emulator for frontend stress testing.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "vllm-mock-engine",
    about = "Run a mock vLLM headless engine for Rust frontend stress testing."
)]
pub struct Opt {
    /// Frontend-owned ZMQ handshake address.
    #[arg(long, default_value = "tcp://127.0.0.1:29550")]
    pub handshake_address: String,

    /// Number of mock engine identities to register with the frontend.
    #[arg(long, default_value_t = 1)]
    pub engine_count: usize,

    /// Number of accepted output tokens included in each EngineCoreOutput.
    #[arg(long, default_value_t = 1)]
    pub output_token_chunk_size: usize,

    /// Random token IDs are sampled uniformly from 0..vocab_size.
    #[arg(long, default_value_t = 32_000)]
    pub vocab_size: u32,

    /// Base seed for deterministic random token generation.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Log a summary line for each request.
    #[arg(long)]
    pub log_requests: bool,
}

/// Run one mock engine until shutdown or transport failure.
async fn run_engine(engine_index: u32, opt: Opt, shutdown: CancellationToken) -> Result<()> {
    let MockEngineSockets { data_sockets, .. } = connect_to_frontend(
        &opt.handshake_address,
        EngineId::from_engine_index(engine_index),
        MockEngineConfig::default(),
    )
    .await
    .with_context(|| format!("mock engine {engine_index} failed to connect to frontend"))?;

    info!(engine_index, "mock engine connected to frontend");

    let (input_tx, input_rx) = mpsc::unbounded_channel();
    let (output_tx, output_rx) = mpsc::channel(64);

    // IO loop: dealer -> input_tx, output_rx -> push
    let mut io_loop = tokio::spawn(io::run_io_loop(
        data_sockets,
        input_tx,
        output_rx,
        shutdown.clone(),
    ));
    // Engine loop: input_rx -> engine logic -> output_tx
    let mut engine_loop = tokio::spawn(engine::run_engine_loop(
        engine_index,
        opt,
        input_rx,
        output_tx,
        shutdown.clone(),
    ));

    tokio::select! {
        biased;
        _ = shutdown.cancelled() => {
            io_loop.abort();
            engine_loop.abort();
            io_loop.await.ok();
            engine_loop.await.ok();
        }

        result = &mut io_loop => {
            error!(engine_index, "mock engine IO loop exited unexpectedly");
            engine_loop.abort();
            engine_loop.await.ok();
            result??;
        }
        result = &mut engine_loop => {
            error!(engine_index, "mock engine loop exited unexpectedly");
            io_loop.abort();
            io_loop.await.ok();
            result??;
        }
    }

    info!(engine_index, "mock engine shut down");
    Ok(())
}

/// Run all requested mock engines until cancellation or one engine task fails.
pub async fn run(opt: Opt, shutdown: CancellationToken) -> Result<()> {
    info!(?opt, "starting mock engine");

    let mut engines = JoinSet::new();
    for engine_index in 0..opt.engine_count {
        engines.spawn(run_engine(
            engine_index as u32,
            opt.clone(),
            shutdown.clone(),
        ));
    }

    tokio::select! {
        biased;
        _ = shutdown.cancelled() => {
            engines.abort_all();
            while engines.join_next().await.is_some() {}
            Ok(())
        }

        joined = engines.join_next() => {
            match joined {
                Some(Ok(Ok(()))) => bail!("mock engine exited unexpectedly"),
                Some(Ok(Err(error))) => Err(error),
                Some(Err(error)) => Err(error).context("mock engine task join failed"),
                None => Ok(()),
            }
        }
    }
}

#[cfg(test)]
mod tests;
