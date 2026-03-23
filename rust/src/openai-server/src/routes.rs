mod chat_completions;
mod completions;
mod metrics;
mod models;
mod utils;

use std::sync::Arc;

use axum::Router;
use axum::middleware::from_fn;
use axum::routing::{get, post};
use tower_http::trace::TraceLayer;

use crate::state::AppState;

/// Build the minimal OpenAI-compatible router for one configured model.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/metrics", get(metrics::scrape))
        .route("/v1/models", get(models::list_models))
        .route("/v1/completions", post(completions::completions))
        .route(
            "/v1/chat/completions",
            post(chat_completions::chat_completions),
        )
        .with_state(state)
        .layer(from_fn(crate::metrics::track_http_metrics))
        .layer(TraceLayer::new_for_http())
}

#[cfg(test)]
mod tests;
