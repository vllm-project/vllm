// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod cli;

use std::env;
use std::ffi::OsStr;
use std::process::{ExitCode, ExitStatus};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use thiserror_ext::AsReport as _;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use vllm_managed_engine::ManagedEngineHandle;
use vllm_tracing::LogLevel;

use crate::cli::{BenchCommand, Cli, Command, ServeConfig};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const TOKIO_WORKER_THREADS_ENV: &str = "TOKIO_WORKER_THREADS";
const DEFAULT_MAX_TOKIO_WORKER_THREADS: usize = 32;

/// Cap the default number of Tokio worker threads if the user did not
/// explicitly set `TOKIO_WORKER_THREADS` to avoid spawning too many threads on
/// machines with a large number of CPUs, which may lead to excessive context
/// switching and degraded performance.
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
            tokio::signal::ctrl_c().await.expect("failed to install Ctrl-C signal handler");
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

/// Open-file soft limit the API server raises to, as Python's `set_ulimit`.
///
/// Not the hard limit: the managed engine and its workers inherit it, and
/// Python runs them at this value unless the inherited soft limit is higher.
const TARGET_NOFILE_SOFT_LIMIT: u64 = 65535;

/// Raise the open-file soft limit (`RLIMIT_NOFILE`) to
/// [`TARGET_NOFILE_SOFT_LIMIT`], within the hard limit, so high-concurrency
/// serving does not hit `EMFILE` under a 1024 default. A higher soft limit is
/// kept.
fn raise_nofile_limit() {
    let prev_soft = match rlimit::Resource::NOFILE.get() {
        Ok((soft, _hard)) => soft,
        Err(error) => {
            warn!(%error, "failed to read open-file limit");
            return;
        }
    };
    match rlimit::increase_nofile_limit(TARGET_NOFILE_SOFT_LIMIT) {
        Ok(new) if new > prev_soft => {
            info!(
                previous = prev_soft,
                soft_limit = new,
                "raised open-file soft limit"
            );
        }
        Ok(_) => {}
        Err(error) => {
            warn!(%error, soft_limit = prev_soft, "failed to raise open-file soft limit");
        }
    }
}

fn main() -> ExitCode {
    let process_label =
        match env::args_os().nth(1).as_deref().and_then(OsStr::to_str).unwrap_or_default() {
            "bench" => "Bench",
            "serve" | "frontend" => "RustFrontend",
            _ => "Rust",
        };
    // Parsing warns about no-op arguments before `--log-level` is known.
    let cli = {
        let _guard = vllm_tracing::set_default_tracing(process_label, LogLevel::from_env());
        Cli::parse()
    };
    vllm_tracing::init_tracing(process_label, cli.log_level(), cli.log_formatter());
    if matches!(
        cli.command,
        Command::Frontend(_) | Command::Serve(_) | Command::Render(_)
    ) {
        raise_nofile_limit();
    }

    let mut runtime = tokio::runtime::Builder::new_multi_thread();
    runtime.enable_all();
    if let Some(worker_threads) = tokio_worker_threads() {
        runtime.worker_threads(worker_threads);
    }

    let result = runtime
        .build()
        .context("failed to build Tokio runtime")
        .and_then(|runtime| runtime.block_on(async_main(cli)));

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            error!("process failed with error: {:#?}", error.as_report());
            ExitCode::FAILURE
        }
    }
}

async fn async_main(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Frontend(args) => args.into_config().into_server()?.serve(shutdown_signal()).await,
        Command::Bench(BenchCommand::Serve(bench_args)) => {
            vllm_bench::prepare_process();
            vllm_bench::run(bench_args).await
        }
        Command::Serve(args) => {
            let (engine_config, frontend, shutdown_timeout) = match args.resolve_config()? {
                ServeConfig::Frontend(config) => {
                    return config.into_server()?.serve(shutdown_signal()).await;
                }
                ServeConfig::ManagedEngine {
                    engine,
                    frontend,
                    shutdown_timeout,
                } => (engine, frontend, shutdown_timeout),
            };
            let engine = ManagedEngineHandle::spawn(engine_config)
                .await
                .context("failed to start managed Python headless engine")?;

            let shutdown = shutdown_signal();

            let mut serve_task = if let Some(config) = frontend {
                let shutdown = shutdown.clone();
                tokio::spawn(async move {
                    let result = config.into_server()?.serve(shutdown).await;
                    if result.is_ok() {
                        info!("OpenAI server shut down gracefully");
                    }
                    result
                })
            } else {
                info!("running managed Python headless engine without Rust frontend");
                let shutdown = shutdown.clone();
                tokio::spawn(async move {
                    shutdown.cancelled().await;
                    Ok(())
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
            // Regardless of the shutdown reason, broadcast shutdown signal here to ensure
            // that all serving tasks are notified.
            shutdown.cancel();
            let shutdown_deadline = Instant::now() + shutdown_timeout;

            // Wait for the API server to drain in-flight requests while the engine
            // keeps serving them, then stop the engine with the remaining budget, as
            // `vllm serve` does. Data parallel engines that drain on their own exit
            // one by one, and the first exit would fail the requests still running
            // on the others.
            let serve_result = if matches!(shutdown_reason, ShutdownReason::Server(_)) {
                Ok(())
            } else {
                serve_task.await.context("serve task join failed").and_then(|result| result)
            };
            let remaining = shutdown_deadline.saturating_duration_since(Instant::now());
            engine.shutdown(remaining).await?;
            info!("managed engine shut down gracefully");
            serve_result?;

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
        Command::Render(args) => {
            vllm_server::serve_render(args.into_config(), shutdown_signal()).await
        }
    }
}
