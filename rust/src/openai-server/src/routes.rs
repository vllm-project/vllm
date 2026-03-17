mod chat_completions;
mod models;

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};
use tower_http::trace::TraceLayer;

use crate::state::AppState;

/// Build the minimal OpenAI-compatible router for one configured model.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/models", get(models::list_models))
        .route(
            "/v1/chat/completions",
            post(chat_completions::chat_completions),
        )
        .with_state(state)
        .layer(TraceLayer::new_for_http())
}

#[cfg(test)]
mod tests;
