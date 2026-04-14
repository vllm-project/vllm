#![feature(coroutines)]
#![feature(iterator_try_collect)]

//! Minimal OpenAI-compatible HTTP server above [`vllm_chat`].

mod config;
mod error;
mod middleware;
mod routes;
mod state;
mod utils;

use std::future::Future;
use std::net::TcpListener as StdTcpListener;
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;

use anyhow::{Context as _, Result};
use axum::serve::ListenerExt as _;
pub use config::{Config, CoordinatorMode, HttpListenerMode};
use socket2::Socket;
use tokio::net::TcpListener;
use tracing::{info, trace};
pub use vllm_chat::ParserSelection;
use vllm_chat::{ChatLlm, load_model_backends};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::Llm;
use vllm_text::TextLlm;

use crate::routes::build_router;
use crate::state::AppState;

/// Build the shared application state for one configured model and one engine client.
async fn build_state(config: &Config) -> Result<Arc<AppState>> {
    // Load both backends from the same model metadata so they stay in sync.
    let loaded = load_model_backends(&config.model)
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

    let mut shutdown = Box::pin(shutdown);

    // Also check shutdown during the (potentially long) startup handshake.
    let state = tokio::select! {
        result = build_state(&config) => result?,
        _ = &mut shutdown => return Ok(()),
    };
    let listener = bind_listener(&config.listener_mode)
        .await
        .context("failed to bind listener for OpenAI server")?;
    let bind_address = listener.local_addr()?;
    let model = state.model_id.clone();
    let app = build_router(state.clone());

    info!(%bind_address, model = %model, "starting OpenAI server");

    // Set TCP_NODELAY on accepted connections to reduce latency.
    let listener = listener.tap_io(|tcp_stream| {
        if let Err(err) = tcp_stream.set_nodelay(true) {
            trace!(error = %err, "failed to enable TCP_NODELAY on accepted HTTP connection");
        }
    });

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;

    state.shutdown().await
}

/// Construct the Tokio listener that matches the configured listener acquisition strategy.
async fn bind_listener(mode: &HttpListenerMode) -> Result<TcpListener> {
    match mode {
        HttpListenerMode::Bind { host, port } => {
            Ok(TcpListener::bind((host.as_str(), *port)).await?)
        }
        HttpListenerMode::InheritedFd { fd } => {
            // SAFETY: We trust the caller to only pass valid listener fds, and we only use this fd
            // once to create a single `TcpListener`.
            let owned_fd = unsafe { OwnedFd::from_raw_fd(*fd) };
            let socket = Socket::from(owned_fd);
            // The Python supervisor pre-binds the socket to reserve the port early, but Rust is
            // responsible for transitioning inherited TCP sockets into the listening state before
            // accepting connections.
            socket.listen(libc::SOMAXCONN)?;
            socket.set_nonblocking(true)?;
            let std_listener = StdTcpListener::from(socket);
            Ok(TcpListener::from_std(std_listener)?)
        }
    }
}
