mod chat_completions;
mod completions;
mod is_sleeping;
mod load;
mod metrics;
mod models;
mod utils;

use std::sync::Arc;

use axum::Router;
use axum::middleware::{from_fn, from_fn_with_state};
use axum::routing::{get, post};
use tower_http::trace::TraceLayer;

use crate::middleware;
use crate::state::AppState;

/// Build the minimal OpenAI-compatible router for one configured model.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        // Health & monitoring
        .route("/metrics", get(metrics::scrape))
        .route("/load", get(load::load))
        // OpenAI-compatible endpoints
        .route("/v1/models", get(models::list_models))
        .route("/v1/completions", post(completions::completions))
        .route(
            "/v1/chat/completions",
            post(chat_completions::chat_completions),
        )
        // Dev-mode & admin
        // TODO: only expose this endpoint when in dev mode (`VLLM_SERVER_DEV_MODE=1`)
        .route("/is_sleeping", get(is_sleeping::is_sleeping))
        // State & middleware
        .with_state(state.clone())
        .layer(from_fn_with_state(state, middleware::track_server_load))
        .layer(from_fn(middleware::track_http_metrics))
        .layer(TraceLayer::new_for_http())
}

#[cfg(test)]
mod tests;
