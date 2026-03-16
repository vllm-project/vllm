mod config;
mod convert;
mod error;
mod routes;
mod state;

use std::sync::Arc;

use anyhow::{Result, anyhow};
pub use config::Config;
use tokio::net::TcpListener;
use tracing::info;
use vllm_chat::ChatLlm;
use vllm_chat::backends::hf::HfChatBackend;
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::Llm;

use crate::routes::build_router;
use crate::state::AppState;

async fn build_state(config: &Config) -> Result<Arc<AppState>> {
    let backend = Arc::new(HfChatBackend::from_model(&config.model).await?);
    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address: config.handshake_address.clone(),
        local_host: config.engine_local_host.clone(),
        ready_timeout: config.ready_timeout,
        client_index: 0,
    })
    .await?;

    let chat = Arc::new(ChatLlm::new(Llm::new(client), backend));
    Ok(Arc::new(AppState::new(config.model.clone(), chat)))
}

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

    let state = Arc::try_unwrap(state)
        .map_err(|_| anyhow!("openai server state still has outstanding references"))?;
    let chat = Arc::try_unwrap(state.chat)
        .map_err(|_| anyhow!("openai server chat still has outstanding references"))?;
    chat.shutdown().await?;

    Ok(())
}
