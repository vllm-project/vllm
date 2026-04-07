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
pub use config::{Config, CoordinatorMode, HttpListenerMode};
use futures::FutureExt as _;
use socket2::Socket;
use tokio::net::TcpListener;
use tracing::info;
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

    let mut text = TextLlm::new(Llm::new(client), text_backend);
    if let Some(max_model_len) = config.max_model_len {
        text = text.with_max_model_len(max_model_len);
    }

    let mut chat = ChatLlm::new(text, chat_backend);
    if let Some(ref name) = config.tool_call_parser {
        chat = chat.with_tool_call_parser(name);
    }
    if let Some(ref name) = config.reasoning_parser {
        chat = chat.with_reasoning_parser(name);
    }

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
    // Wrap the shutdown signal in a `Shared` future so we can also check it
    // during the (potentially long) startup handshake, not only during HTTP
    // graceful shutdown.
    let shutdown = shutdown.shared();

    let state = tokio::select! {
        result = build_state(&config) => result?,
        _ = shutdown.clone() => return Ok(()),
    };
    let listener = bind_listener(&config.listener_mode).await?;
    let bind_address = listener.local_addr()?;
    let model = state.model_id.clone();
    let app = build_router(state.clone());

    info!(%bind_address, model = %model, "starting OpenAI server");

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
