use axum::{routing::get, Router};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokenizers::Tokenizer;
use tower_http::trace::TraceLayer;

use crate::grpc::VllmClient;
use crate::routes;

pub struct AppState {
    pub tokenizer: Tokenizer,
    pub grpc_client: Mutex<VllmClient>,
    pub model_name: String,
}

impl AppState {
    pub fn new(tokenizer: Tokenizer, grpc_client: VllmClient, model_name: String) -> Self {
        Self {
            tokenizer,
            grpc_client: Mutex::new(grpc_client),
            model_name,
        }
    }
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(routes::health_check))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn run_server(addr: &str, state: Arc<AppState>) -> Result<(), std::io::Error> {
    let app = create_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Server listening on {}", addr);
    axum::serve(listener, app).await
}
