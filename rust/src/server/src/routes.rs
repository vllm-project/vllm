mod abort_requests;
mod cache;
mod collective_rpc;
mod health;
mod inference;
mod load;
mod lora;
mod metrics;
pub(crate) mod openai;
mod pause;
mod profile;
mod server_info;
mod sleep;
mod tokenize;
mod version;
mod world_size;

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware::{from_fn, from_fn_with_state};
use axum::routing::{get, post};
use tower_http::trace::TraceLayer;
use tracing::warn;

use crate::middleware;
use crate::state::AppState;

const DEFAULT_JSON_BODY_LIMIT_BYTES: usize = 32 * 1024 * 1024;

fn server_dev_mode_enabled() -> bool {
    std::env::var("VLLM_SERVER_DEV_MODE")
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .is_some_and(|value| value != 0)
}

fn runtime_lora_updating_enabled() -> bool {
    std::env::var("VLLM_ALLOW_RUNTIME_LORA_UPDATING")
        .ok()
        .is_some_and(|value| matches!(value.trim().to_lowercase().as_str(), "1" | "true"))
}

/// Build the minimal OpenAI-compatible router for one configured model.
pub fn build_router(state: Arc<AppState>) -> Router {
    let profiling_enabled = state.profiling_enabled;
    build_router_with_options(
        state,
        server_dev_mode_enabled(),
        runtime_lora_updating_enabled(),
        profiling_enabled,
    )
}

#[cfg(test)]
fn build_router_with_dev_mode(state: Arc<AppState>, dev_mode_enabled: bool) -> Router {
    build_router_with_dev_mode_and_lora(state, dev_mode_enabled, false)
}

#[cfg(test)]
fn build_router_with_dev_mode_and_lora(
    state: Arc<AppState>,
    dev_mode_enabled: bool,
    runtime_lora_updating_enabled: bool,
) -> Router {
    let profiling_enabled = state.profiling_enabled;
    build_router_with_options(
        state,
        dev_mode_enabled,
        runtime_lora_updating_enabled,
        profiling_enabled,
    )
}

#[cfg(test)]
pub(crate) fn build_router_with_profiling(state: Arc<AppState>) -> Router {
    build_router_with_options(state, false, false, true)
}

fn build_router_with_options(
    state: Arc<AppState>,
    dev_mode_enabled: bool,
    runtime_lora_updating_enabled: bool,
    profiling_enabled: bool,
) -> Router {
    let mut router = Router::new()
        // Health & monitoring
        .route("/health", get(health::health))
        .route("/metrics", get(metrics::scrape))
        .route("/load", get(load::load))
        .route("/version", get(version::version))
        // OpenAI-compatible endpoints
        .route("/v1/models", get(openai::list_models))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/chat/completions", post(openai::chat_completions))
        // vLLM specific endpoints
        .route("/tokenize", post(tokenize::tokenize))
        .route("/detokenize", post(tokenize::detokenize))
        .route("/inference/v1/generate", post(inference::generate));

    if runtime_lora_updating_enabled {
        router = router
            .route("/v1/load_lora_adapter", post(lora::load_lora_adapter))
            .route("/v1/unload_lora_adapter", post(lora::unload_lora_adapter));
    }

    if dev_mode_enabled {
        // Development-only
        router = router
            .route("/reset_prefix_cache", post(cache::reset_prefix_cache))
            .route("/reset_mm_cache", post(cache::reset_mm_cache))
            .route("/reset_encoder_cache", post(cache::reset_encoder_cache))
            .route("/collective_rpc", post(collective_rpc::collective_rpc))
            .route("/abort_requests", post(abort_requests::abort_requests))
            .route("/sleep", post(sleep::sleep))
            .route("/wake_up", post(sleep::wake_up))
            .route("/is_sleeping", get(sleep::is_sleeping))
            .route("/pause", post(pause::pause))
            .route("/resume", post(pause::resume))
            .route("/is_paused", get(pause::is_paused))
            .route("/server_info", get(server_info::server_info))
            .route("/get_world_size", get(world_size::get_world_size))
    }

    if profiling_enabled {
        warn!(
            "Profiler is enabled in the API server. \
             This should ONLY be used for local development!"
        );
        router = router
            .route("/start_profile", post(profile::start_profile))
            .route("/stop_profile", post(profile::stop_profile));
    }

    let enable_request_id_headers = state.api_server_options.enable_request_id_headers;
    let enable_api_key_auth = state.has_api_keys();
    let mut router = router
        .with_state(state.clone())
        .layer(DefaultBodyLimit::max(DEFAULT_JSON_BODY_LIMIT_BYTES))
        .layer(middleware::request_runtime_layer(state.clone()))
        .layer(from_fn_with_state(
            state.clone(),
            middleware::track_server_load,
        ))
        .layer(from_fn(middleware::track_http_metrics))
        .layer(middleware::cors_layer(&state.cors))
        .layer(from_fn(middleware::strip_cors_on_no_origin));

    if enable_api_key_auth {
        router = router.layer(from_fn_with_state(
            state.clone(),
            middleware::authenticate_api_key,
        ));
    }

    // Later layers wrap earlier ones. Keep tracing outside auth so rejected
    // requests are visible, while metrics/load only see authenticated traffic.
    router = router.layer(TraceLayer::new_for_http());

    if enable_request_id_headers {
        router = router.layer(from_fn(middleware::set_request_id_header));
    }

    router
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod http_client_tests;
