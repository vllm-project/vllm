mod config;
mod convert;
mod error;
mod routes;
mod state;

use std::sync::Arc;

use tokio::net::TcpListener;
use tracing::info;
use tracing_subscriber::EnvFilter;
use vllm_chat::ChatLlm;
use vllm_chat::backends::hf::HfChatBackend;
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::Llm;

use crate::config::Config;
use crate::routes::build_router;
use crate::state::AppState;

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let config = Config::parse();

    let backend = Arc::new(HfChatBackend::from_model(&config.model).await?);
    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address: config.handshake_address.clone(),
        local_host: config.engine_local_host.clone(),
        ready_timeout: config.ready_timeout,
        client_index: 0,
    })
    .await?;

    let chat = Arc::new(ChatLlm::new(Llm::new(client), backend));
    let state = Arc::new(AppState::new(config.model.clone(), chat));
    let app = build_router(state);

    let bind_address = config.bind_address();
    let listener = TcpListener::bind(&bind_address).await?;

    info!(%bind_address, model = %config.model, "starting OpenAI server");

    axum::serve(listener, app).await?;
    Ok(())
}
