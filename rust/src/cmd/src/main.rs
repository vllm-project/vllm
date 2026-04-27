mod cli;
mod logging;
mod managed_engine;

use std::env;
use std::process::ExitStatus;

use anyhow::{Context, Result, anyhow, bail};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::cli::{Cli, Command};
use crate::managed_engine::{ManagedEngineHandle, allocate_handshake_port};

const TOKIO_WORKER_THREADS_ENV: &str = "TOKIO_WORKER_THREADS";
const DEFAULT_MAX_TOKIO_WORKER_THREADS: usize = 32;

/// Cap the default number of Tokio worker threads if the user did not explicitly set
/// `TOKIO_WORKER_THREADS` to avoid spawning too many threads on machines with a large number of
/// CPUs, which may lead to excessive context switching and degraded performance.
fn tokio_worker_threads() -> Option<usize> {
    if env::var_os(TOKIO_WORKER_THREADS_ENV).is_some() {
        return None;
    }

    std::thread::available_parallelism()
        .map(|parallelism| {
            let available = parallelism.get();
            let worker_threads = available.min(DEFAULT_MAX_TOKIO_WORKER_THREADS);
            if worker_threads < available {
                info!(
                    available_parallelism = available,
                    capped_worker_threads = worker_threads,
                    "capping tokio worker threads, set {TOKIO_WORKER_THREADS_ENV} to override"
                );
            }
            worker_threads
        })
        .ok()
}

/// Reason that caused a managed `serve` session to stop.
#[derive(Debug)]
enum ShutdownReason {
    Signal,
    Server(anyhow::Error),
    EngineExited(ExitStatus),
}

/// Cancellation token tripped by Ctrl-C or SIGTERM.
fn shutdown_signal() -> CancellationToken {
    let token = CancellationToken::new();
    let shutdown = token.clone();

    tokio::spawn(async move {
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

        shutdown.cancel();
    });

    token
}

fn main() -> Result<()> {
    logging::init_tracing();
    let cli = Cli::parse();

    let mut runtime = tokio::runtime::Builder::new_multi_thread();
    runtime.enable_all();
    if let Some(worker_threads) = tokio_worker_threads() {
        runtime.worker_threads(worker_threads);
    }

    runtime
        .build()
        .context("failed to build Tokio runtime")?
        .block_on(async_main(cli))
}

async fn async_main(cli: Cli) -> Result<()> {
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

            let shutdown_timeout = args.runtime.shutdown_timeout();
            let engine_config = args.clone().into_managed_engine_config(handshake_port);
            let handshake_address = engine_config.handshake_address();

            let engine = ManagedEngineHandle::spawn(engine_config)
                .await
                .context("failed to start managed Python headless engine")?;

            let shutdown = shutdown_signal();

            let mut serve_task = if args.headless {
                info!("running managed Python headless engine without Rust frontend");
                let shutdown = shutdown.clone();
                tokio::spawn(async move {
                    shutdown.cancelled().await;
                    Ok(())
                })
            } else {
                let config = args.to_frontend_config(handshake_address);
                let shutdown = shutdown.clone();
                tokio::spawn(async move {
                    let result = vllm_server::serve(config, shutdown).await;
                    if result.is_ok() {
                        info!("OpenAI server shut down gracefully");
                    }
                    result
                })
            };

            let shutdown_reason = tokio::select! {
                biased;

                // Received shutdown signal via Ctrl-C or SIGTERM.
                _ = shutdown.cancelled() => ShutdownReason::Signal,

                // Engine process exited unexpectedly.
                status = engine.wait_for_exit() => {
                    warn!(%status, "managed Python headless engine exited, shutting down...");
                    ShutdownReason::EngineExited(status)
                }

                // Serve task exited unexpectedly.
                serve_result = &mut serve_task => {
                    let serve_result = serve_result.context("serve task join failed")?;
                    match serve_result {
                        Ok(()) => ShutdownReason::Server(anyhow!("OpenAI server shut down unexpectedly without error")),
                        Err(error) => ShutdownReason::Server(error),
                    }
                }
            };
            // Regardless of the shutdown reason, broadcast shutdown signal here to ensure that all
            // serving tasks are notified.
            shutdown.cancel();

            // Shutdown begins. Terminate the managed engine first.
            engine.shutdown(shutdown_timeout).await?;
            info!("managed engine shut down gracefully");
            // Wait for the API server to shut down gracefully by draining in-flight requests.
            if !matches!(shutdown_reason, ShutdownReason::Server(_)) {
                serve_task.await.context("serve task join failed")??;
            }

            match shutdown_reason {
                ShutdownReason::Signal => Ok(()),
                ShutdownReason::Server(error) => {
                    Err(error.context("OpenAI server shut down unexpectedly"))
                }
                ShutdownReason::EngineExited(status) => Err(anyhow!(
                    "managed Python headless engine exited unexpectedly with status {status}"
                )),
            }
        }
    }
}
