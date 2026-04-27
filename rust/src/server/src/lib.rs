#![feature(coroutines)]
#![feature(iterator_try_collect)]

//! Minimal OpenAI-compatible HTTP server above [`vllm_chat`].

mod config;
mod error;
mod grpc;
mod listener;
mod middleware;
mod routes;
mod state;
mod utils;

use std::sync::Arc;

use anyhow::{Context as _, Result};
use axum::serve::ListenerExt as _;
pub use config::{Config, CoordinatorMode, HttpListenerMode};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::either::Either;
use tokio_util::sync::CancellationToken;
use tonic::transport::Server as TonicServer;
use tracing::{info, trace};
use vllm_chat::{ChatLlm, LoadModelBackendsOptions, load_model_backends};
pub use vllm_chat::{ChatTemplateContentFormatOption, ParserSelection, RendererSelection};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::Llm;
use vllm_text::TextLlm;

use crate::listener::Listener;
use crate::routes::build_router;
use crate::state::AppState;

/// Build the shared application state for one configured model and one engine client.
async fn build_state(config: &Config) -> Result<Arc<AppState>> {
    // Load both backends from the same model metadata so they stay in sync.
    let loaded = load_model_backends(
        &config.model,
        LoadModelBackendsOptions {
            renderer: config.renderer,
            chat_template: config.chat_template.clone(),
            chat_template_content_format: config.chat_template_content_format,
            default_chat_template_kwargs: config
                .default_chat_template_kwargs
                .clone()
                .unwrap_or_default(),
        },
    )
    .await
    .context("failed to create chat/text backends")?;
    let text_backend = loaded.text_backend;
    let chat_backend = loaded.chat_backend;

    let coordinator_mode = config.effective_coordinator_mode(text_backend.is_moe());
    info!(
        engine_count = config.engine_count(),
        model_is_moe = text_backend.is_moe(),
        ?coordinator_mode,
        "resolved coordinator mode"
    );

    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        transport_mode: config.transport_mode.clone(),
        coordinator_mode,
        model_name: config.model.clone(),
        client_index: 0,
    })
    .await
    .context("failed to connect to engine core")?;

    let llm = Llm::new(client).with_log_stats(!config.disable_log_stats);
    let text = TextLlm::new(llm, text_backend);

    let chat = ChatLlm::new(text, chat_backend)
        .with_tool_call_parser(config.tool_call_parser.clone())
        .with_reasoning_parser(config.reasoning_parser.clone());

    Ok(Arc::new(
        AppState::new(config.model.clone(), chat).with_log_requests(config.enable_log_requests),
    ))
}

/// Run the OpenAI-compatible HTTP server until the supplied shutdown token is cancelled.
///
/// The server owns one `vllm-chat` facade, which in turn owns the lower `vllm-text` and
/// `vllm-llm` layers, and shuts them down before returning.
pub async fn serve(config: Config, shutdown: CancellationToken) -> Result<()> {
    config
        .validate()
        .context("invalid OpenAI frontend configuration")?;

    // Also check shutdown during the (potentially long) startup handshake.
    let state = tokio::select! {
        result = build_state(&config) => result?,
        _ = shutdown.cancelled() => return Ok(()),
    };
    let listener = Listener::bind(&config.listener_mode)
        .await
        .context("failed to bind listener for OpenAI server")?;
    let bind_address = listener.local_addr()?;
    let model = state.model_id.clone();
    let app = build_router(state.clone());

    // Optionally bind the gRPC Generate server on a separate port. Bind synchronously
    // here so bind errors (port in use, permission denied, ...) surface before we start
    // serving, rather than being deferred until shutdown. The gRPC listener follows the
    // same host as the HTTP listener so that enabling --grpc-port does not accidentally
    // expose the service on all interfaces when HTTP is intentionally local-only.
    let grpc_setup = if let Some(grpc_port) = config.grpc_port {
        let grpc_host = match &config.listener_mode {
            HttpListenerMode::BindTcp { host, .. } => host.as_str(),
            HttpListenerMode::BindUnix { .. } | HttpListenerMode::InheritedFd { .. } => "0.0.0.0",
        };
        let grpc_listener = TcpListener::bind((grpc_host, grpc_port))
            .await
            .with_context(|| format!("failed to bind gRPC listener on {grpc_host}:{grpc_port}"))?;
        let addr = grpc_listener.local_addr()?;
        let svc = grpc::GenerateServer::new(grpc::GenerateServiceImpl::new(state.clone()));
        info!(%addr, "starting gRPC server");
        Some((grpc_listener, svc))
    } else {
        None
    };

    info!(%bind_address, %model, "starting OpenAI server");

    // Set TCP_NODELAY on accepted connections to reduce latency.
    // By `tap_io` we will do this on every accepted connection.
    let listener = listener.tap_io(|io| {
        if let Either::Left(tcp_stream) = io
            && let Err(err) = tcp_stream.set_nodelay(true)
        {
            trace!(error = %err, "failed to enable TCP_NODELAY on accepted HTTP connection");
        }
    });

    // Run HTTP and gRPC concurrently under a child token of the caller's shutdown token.
    // Caller cancellation propagates into both protocols; if either protocol exits first,
    // we cancel this child token so its sibling also begins a graceful drain.
    let server_shutdown = shutdown.child_token();

    let http_fut = {
        let shutdown = server_shutdown.child_token();
        let server_shutdown = server_shutdown.clone();
        async move {
            let result = axum::serve(listener, app)
                .with_graceful_shutdown(shutdown.cancelled_owned())
                .await
                .context("HTTP server failed");
            server_shutdown.cancel();
            result
        }
    };

    let grpc_fut = {
        let shutdown = server_shutdown.child_token();
        let server_shutdown = server_shutdown.clone();
        async move {
            let Some((grpc_listener, svc)) = grpc_setup else {
                // No gRPC configured: just wait for shutdown so we do not race the
                // join! by resolving early and tripping the cancellation token.
                shutdown.cancelled_owned().await;
                return Ok(());
            };
            let result = TonicServer::builder()
                .add_service(svc)
                .serve_with_incoming_shutdown(
                    TcpListenerStream::new(grpc_listener),
                    shutdown.cancelled_owned(),
                )
                .await
                .context("gRPC server failed");
            server_shutdown.cancel();
            result
        }
    };

    let (http_res, grpc_res) = tokio::join!(http_fut, grpc_fut);
    http_res.and(grpc_res)?;

    state.shutdown().await
}
