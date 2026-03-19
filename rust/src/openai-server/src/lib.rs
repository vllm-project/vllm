#![feature(coroutines)]
#![feature(iterator_try_collect)]

//! Minimal OpenAI-compatible HTTP server above [`vllm_chat`].
//!
//! This crate keeps the northbound surface intentionally narrow:
//! one configured model, `GET /v1/models`, and streaming
//! `POST /v1/chat/completions`.

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

use crate::routes::build_router;
use crate::state::AppState;

/// Build the shared application state for one configured model and one engine client.
async fn build_state(config: &Config) -> Result<Arc<AppState>> {
    let backend = Arc::new(HfChatBackend::from_model(&config.model).await?);
    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address: config.handshake_address.clone(),
        local_host: config.engine_local_host.clone(),
        ready_timeout: config.ready_timeout,
        client_index: 0,
    })
    .await?;

    let mut chat = ChatLlm::new(Llm::new(client), backend);
    if let Some(ref name) = config.tool_call_parser {
        chat = chat.with_tool_call_parser(name);
    }
    if let Some(ref name) = config.reasoning_parser {
        chat = chat.with_reasoning_parser(name);
    }
    let chat = Arc::new(chat);
    Ok(Arc::new(AppState::new(config.model.clone(), chat)))
}

/// Run the OpenAI-compatible HTTP server until the supplied shutdown future resolves.
///
/// The server owns one `vllm-chat` stack and shuts it down before returning.
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
