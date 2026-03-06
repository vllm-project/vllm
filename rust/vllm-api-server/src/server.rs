use axum::{routing::{get, post}, Router};
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
    pub chat_template: Option<String>,
}

impl AppState {
    pub fn new(
        tokenizer: Tokenizer,
        grpc_client: VllmClient,
        model_name: String,
        chat_template: Option<String>,
    ) -> Self {
        Self {
            tokenizer,
            grpc_client: Mutex::new(grpc_client),
            model_name,
            chat_template,
        }
    }
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(routes::health_check))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn run_server(addr: &str, state: Arc<AppState>) -> Result<(), std::io::Error> {
    let app = create_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Server listening on {}", addr);
    axum::serve(listener, app).await
}
