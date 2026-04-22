#![feature(coroutines)]
#![feature(iterator_try_collect)]

//! Minimal OpenAI-compatible HTTP server above [`vllm_chat`].

mod config;
mod error;
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
use tokio_util::either::Either;
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

    let mut shutdown = Box::pin(shutdown);

    // Also check shutdown during the (potentially long) startup handshake.
    let state = tokio::select! {
        result = build_state(&config) => result?,
        _ = &mut shutdown => return Ok(()),
    };
    let listener = Listener::bind(&config.listener_mode)
        .await
        .context("failed to bind listener for OpenAI server")?;
    let bind_address = listener.local_addr()?;
    let model = state.model_id.clone();
    let app = build_router(state.clone());

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

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;

    state.shutdown().await
}
