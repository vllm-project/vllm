mod cli;
mod logging;
mod managed_engine;

use std::process::ExitStatus;

use anyhow::{Context, Result, anyhow};
use futures::FutureExt as _;
use tokio::sync::oneshot;
use tracing::{info, warn};

use crate::cli::{Cli, Command};
use crate::managed_engine::{ManagedEngineHandle, allocate_handshake_port};

/// Reason that caused a managed `serve` session to stop.
#[derive(Debug)]
enum ShutdownReason {
    Signal,
    EngineExited(ExitStatus),
}

/// Shutdown signal from Ctrl-C.
async fn ctrl_c() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install Ctrl-C signal handler");

    info!("received shutdown signal (Ctrl-C), shutting down...");
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    logging::init_tracing();

    match cli.command {
        Command::Frontend(args) => vllm_openai_server::serve(args.into_config(), ctrl_c()).await,
        Command::Serve(args) => {
            let handshake_port = match args.handshake_port {
                Some(port) => port,
                None => allocate_handshake_port(&args.handshake_host)?,
            };
            let engine_config = args.clone().into_managed_engine_config(handshake_port);
            let handshake_address = engine_config.handshake_address();

            let engine = ManagedEngineHandle::spawn(engine_config)
                .await
                .context("failed to start managed Python headless engine")?;

            let (shutdown_signal_tx, shutdown_signal_rx) = oneshot::channel();
            let shutdown_engine = engine.clone();
            let shutdown_task = tokio::spawn(async move {
                // Drive the frontend shutdown from either Ctrl-C or unexpected
                // Python-engine termination, whichever happens first.
                let reason = tokio::select! {
                    _ = ctrl_c() => ShutdownReason::Signal,

                    status = shutdown_engine.wait_for_exit() => {
                        warn!(%status, "managed Python headless engine exited");
                        ShutdownReason::EngineExited(status)
                    },
                };
                let _ = shutdown_signal_tx.send(());
                reason
            });

            let serve_result = if args.headless {
                info!("running managed Python headless engine without Rust frontend");
                let _ = shutdown_signal_rx.await;
                Ok(())
            } else {
                let config = args.to_frontend_config(handshake_address);
                vllm_openai_server::serve(config, shutdown_signal_rx.map(|_| ()))
                    .await
                    .inspect(|_| info!("OpenAI server shut down gracefully"))
            };

            let shutdown_result = engine.shutdown().await;
            let shutdown_reason = shutdown_task.await.context("shutdown task join failed")?;

            serve_result?;
            shutdown_result?;
            info!("managed Python headless engine shut down gracefully");

            match shutdown_reason {
                ShutdownReason::Signal => Ok(()),
                ShutdownReason::EngineExited(status) => Err(anyhow!(
                    "managed Python headless engine exited unexpectedly with status {status}"
                )),
            }
        }
    }
}
