mod cli;
mod logging;
mod managed_engine;

use std::process::ExitStatus;

use anyhow::{Context, Result, anyhow, bail};
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

/// Shutdown signal from Ctrl-C or SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C signal handler");
    };

    let sigterm = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => info!("received shutdown signal (Ctrl-C), shutting down..."),
        _ = sigterm => info!("received shutdown signal (SIGTERM), shutting down..."),
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    logging::init_tracing();
    let cli = Cli::parse();

    match cli.command {
        Command::Frontend(args) => vllm_server::serve(args.into_config(), shutdown_signal()).await,
        Command::Serve(args) => {
            let handshake_port = match args.handshake_port {
                Some(port) => port,
                None => allocate_handshake_port(&args.handshake_host)?,
            };

            if args.data_parallel_size_local == Some(0) {
                if args.headless {
                    bail!("cannot combine `--headless` with `--data-parallel-size-local 0`");
                }

                let handshake_address = args.handshake_address(handshake_port);
                info!(
                    %handshake_address,
                    engine_count = args.data_parallel_size,
                    "running Rust frontend without a managed local Python engine"
                );
                let config = args.to_frontend_config(handshake_address);
                return vllm_server::serve(config, shutdown_signal()).await;
            }

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
                    _ = shutdown_signal() => ShutdownReason::Signal,

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
                vllm_server::serve(config, shutdown_signal_rx.map(|_| ()))
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
