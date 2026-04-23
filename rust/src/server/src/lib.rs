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

use std::future::Future;
use std::sync::Arc;

use anyhow::{Context as _, Result};
use axum::serve::ListenerExt as _;
pub use config::{Config, CoordinatorMode, HttpListenerMode};
use futures::FutureExt as _;
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

/// Run the OpenAI-compatible HTTP server until the supplied shutdown future resolves.
///
/// The server owns one `vllm-chat` facade, which in turn owns the lower `vllm-text` and
/// `vllm-llm` layers, and shuts them down before returning.
pub async fn serve<F>(config: Config, shutdown: F) -> Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    config
        .validate()
        .context("invalid OpenAI frontend configuration")?;

    let shutdown = Box::pin(shutdown).shared();

    // Also check shutdown during the (potentially long) startup handshake.
    let state = tokio::select! {
        result = build_state(&config) => result?,
        _ = shutdown.clone() => return Ok(()),
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

    // Run HTTP and gRPC concurrently under a shared cancellation token. If either
    // server exits — cleanly or with an error — we trip the token so the other
    // begins a graceful drain immediately. This avoids a partial-outage state where
    // one protocol keeps serving after the other has died.
    let internal_shutdown = CancellationToken::new();

    let http_fut = {
        let external = shutdown.clone();
        let internal = internal_shutdown.clone();
        async move {
            let signal = combined_shutdown(external, internal.clone());
            let result = axum::serve(listener, app)
                .with_graceful_shutdown(signal)
                .await
                .context("HTTP server failed");
            internal.cancel();
            result
        }
    };

    let grpc_fut = {
        let external = shutdown.clone();
        let internal = internal_shutdown.clone();
        async move {
            let Some((grpc_listener, svc)) = grpc_setup else {
                // No gRPC configured: just wait for shutdown so we do not race the
                // join! by resolving early and tripping the cancellation token.
                external.await;
                return Ok(());
            };
            let signal = combined_shutdown(external, internal.clone());
            let result = TonicServer::builder()
                .add_service(svc)
                .serve_with_incoming_shutdown(TcpListenerStream::new(grpc_listener), signal)
                .await
                .context("gRPC server failed");
            internal.cancel();
            result
        }
    };

    let (http_res, grpc_res) = tokio::join!(http_fut, grpc_fut);
    http_res.and(grpc_res)?;

    state.shutdown().await
}

/// Resolves when either the external shutdown future fires or the internal
/// cancellation token is tripped. Used to fan one shared shutdown signal out to
/// both server loops while also letting either loop pull the other down.
async fn combined_shutdown<F>(external: F, internal: CancellationToken)
where
    F: Future<Output = ()>,
{
    tokio::select! {
        _ = external => {}
        _ = internal.cancelled() => {}
    }
}
