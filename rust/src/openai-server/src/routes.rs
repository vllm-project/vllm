mod cache;
mod chat_completions;
mod completions;
mod health;
mod load;
mod metrics;
mod models;
mod sleep;
mod utils;

use std::sync::Arc;

use axum::Router;
use axum::middleware::{from_fn, from_fn_with_state};
use axum::routing::{get, post};
use tower_http::trace::TraceLayer;

use crate::middleware;
use crate::state::AppState;

fn server_dev_mode_enabled() -> bool {
    std::env::var("VLLM_SERVER_DEV_MODE")
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .is_some_and(|value| value != 0)
}

/// Build the minimal OpenAI-compatible router for one configured model.
pub fn build_router(state: Arc<AppState>) -> Router {
    build_router_with_dev_mode(state, server_dev_mode_enabled())
}

fn build_router_with_dev_mode(state: Arc<AppState>, dev_mode_enabled: bool) -> Router {
    let mut router = Router::new()
        // Health & monitoring
        .route("/health", get(health::health))
        .route("/metrics", get(metrics::scrape))
        .route("/load", get(load::load))
        // OpenAI-compatible endpoints
        .route("/v1/models", get(models::list_models))
        .route("/v1/completions", post(completions::completions))
        .route(
            "/v1/chat/completions",
            post(chat_completions::chat_completions),
        );

    if dev_mode_enabled {
        // Development-only
        router = router
            .route("/reset_prefix_cache", post(cache::reset_prefix_cache))
            .route("/reset_mm_cache", post(cache::reset_mm_cache))
            .route("/reset_encoder_cache", post(cache::reset_encoder_cache))
            .route("/sleep", post(sleep::sleep))
            .route("/wake_up", post(sleep::wake_up))
            .route("/is_sleeping", get(sleep::is_sleeping))
    }

    router
        .with_state(state.clone())
        .layer(from_fn_with_state(state, middleware::track_server_load))
        .layer(from_fn(middleware::track_http_metrics))
        .layer(TraceLayer::new_for_http())
}

#[cfg(test)]
mod tests;
