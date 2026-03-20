#![feature(coroutines)]
#![feature(iterator_try_collect)]

//! Minimal OpenAI-compatible HTTP server above [`vllm_chat`].

mod config;
mod convert;
mod error;
mod routes;
mod state;

use std::sync::Arc;

use anyhow::Result;
pub use config::Config;
use tokio::net::TcpListener;
use tracing::info;
use vllm_chat::ChatLlm;
use vllm_chat::backends::hf::HfChatBackend;
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::Llm;
use vllm_text::TextLlm;
use vllm_text::backends::hf::HfTextBackend;

use crate::routes::build_router;
use crate::state::AppState;

/// Build the shared application state for one configured model and one engine client.
async fn build_state(config: &Config) -> Result<Arc<AppState>> {
    let text_backend = Arc::new(HfTextBackend::from_model(&config.model).await?);
    // Build chat on top of the already loaded text backend so tokenizer/model metadata stay
    // shared between raw completions and chat requests.
    let backend = Arc::new(HfChatBackend::from_text_backend((*text_backend).clone())?);
    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address: config.handshake_address.clone(),
        local_host: config.engine_local_host.clone(),
        ready_timeout: config.ready_timeout,
        client_index: 0,
    })
    .await?;

    let mut text = TextLlm::new(Llm::new(client), text_backend);
    if let Some(max_model_len) = config.max_model_len {
        text = text.with_max_model_len(max_model_len);
    }

    let mut chat = ChatLlm::new(text, backend);
    if let Some(ref name) = config.tool_call_parser {
        chat = chat.with_tool_call_parser(name);
    }
    if let Some(ref name) = config.reasoning_parser {
        chat = chat.with_reasoning_parser(name);
    }

    Ok(Arc::new(AppState::new(config.model.clone(), chat)))
}

/// Run the OpenAI-compatible HTTP server until the supplied shutdown future resolves.
///
/// The server owns one `vllm-chat` facade, which in turn owns the lower `vllm-text` and
/// `vllm-llm` layers, and shuts them down before returning.
pub async fn serve<F>(config: Config, shutdown: F) -> Result<()>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let state = build_state(&config).await?;
    let listener = TcpListener::bind(config.bind_address()).await?;
    let bind_address = listener.local_addr()?;
    let model = state.model_id.clone();
    let app = build_router(state.clone());

    info!(%bind_address, model = %model, "starting OpenAI server");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;

    state.shutdown().await
}
