// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod abort_requests;
mod cache;
mod checkpoint;
mod collective_rpc;
mod health;
mod inference;
mod kv_event_sources;
mod load;
mod lora;
mod metrics;
pub(crate) mod openai;
mod output_log;
mod pause;
mod profile;
pub(super) mod render;
mod server_info;
mod sleep;
mod tokenize;
mod version;
mod weight_checker;
mod weight_transfer;
mod world_size;

use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::extract::MatchedPath;
use axum::http::{Method, StatusCode};
use axum::middleware::{from_fn, from_fn_with_state};
use axum::routing::{get, post};
use axum::{Json, Router};
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

use crate::middleware;
use crate::routes::openai::utils::types::{ErrorDetail, ErrorResponse};
use crate::state::AppState;

const SCALE_OUT_DISABLED_MESSAGE: &str =
    "Scale-out endpoints are disabled. Restart vllm serve with --enable-scale-out to enable them.";

/// Scale-out routes that Python registers together with
/// `/inference/v1/generate` but the Rust frontend does not implement yet.
const UNIMPLEMENTED_SCALE_OUT_ROUTES: &[&str] = &[
    "/inference/v1/abort_requests",
    "/v1/chat/completions/render",
    "/v1/completions/render",
    "/v1/messages/render",
    "/v1/responses/render",
    "/v1/chat/completions/derender",
    "/v1/completions/derender",
];

async fn disabled_scale_out() -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::FORBIDDEN,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: SCALE_OUT_DISABLED_MESSAGE.to_string(),
                error_type: "PermissionDeniedError".to_string(),
                param: None,
                code: StatusCode::FORBIDDEN.as_u16(),
            },
        }),
    )
}

/// Placeholder for a route the Python frontend registers but the Rust frontend
/// does not implement yet. Returning 501 instead of 404 tells clients that the
/// endpoint exists in vLLM and how to reach it.
async fn not_supported(method: Method, path: MatchedPath) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: format!(
                    "{method} {} is not supported by the Rust frontend yet. \
                     Set VLLM_USE_RUST_FRONTEND=0 to use the Python frontend.",
                    path.as_str()
                ),
                error_type: "NotImplementedError".to_string(),
                param: None,
                code: 501,
            },
        }),
    )
}

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
    let scale_out_endpoints_enabled = state.api_server_options.enable_scale_out;
    build_router_with_options(
        state,
        server_dev_mode_enabled(),
        runtime_lora_updating_enabled(),
        scale_out_endpoints_enabled,
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
    build_router_with_options(state, dev_mode_enabled, runtime_lora_updating_enabled, true)
}

#[cfg(test)]
fn build_router_with_scale_out_endpoints(
    state: Arc<AppState>,
    scale_out_endpoints_enabled: bool,
) -> Router {
    build_router_with_options(state, false, false, scale_out_endpoints_enabled)
}

fn build_router_with_options(
    state: Arc<AppState>,
    dev_mode_enabled: bool,
    runtime_lora_updating_enabled: bool,
    scale_out_endpoints_enabled: bool,
) -> Router {
    let mut router = Router::new()
        // Health & monitoring
        .route("/health", get(health::health))
        .route("/metrics", get(metrics::scrape))
        .route("/load", get(load::load))
        .route("/version", get(version::version))
        .route("/kv_event_sources", get(kv_event_sources::kv_event_sources))
        // OpenAI-compatible endpoints
        .route("/v1/models", get(openai::list_models))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/chat/completions", post(openai::chat_completions))
        // TODO: implement the Anthropic Messages API.
        .route("/v1/messages", post(not_supported))
        .route("/v1/messages/count_tokens", post(not_supported))
        // TODO: implement batched chat completions.
        .route("/v1/chat/completions/batch", post(not_supported))
        // TODO: implement the decisions APIs.
        .route("/v1/decisions", post(not_supported))
        .route("/v1/systemone", post(not_supported))
        // TODO: implement elastic EP scaling.
        .route("/scale_elastic_ep", post(not_supported))
        .route("/is_scaling_elastic_ep", post(not_supported))
        .route("/v1/responses", post(openai::create_responses))
        .route(
            "/v1/responses/{response_id}",
            get(openai::retrieve_response),
        )
        .route(
            "/v1/responses/{response_id}/cancel",
            post(openai::cancel_response),
        )
        // vLLM specific endpoints
        .route("/tokenize", post(tokenize::tokenize))
        .route("/detokenize", post(tokenize::detokenize));

    if scale_out_endpoints_enabled {
        router = router.route("/inference/v1/generate", post(inference::generate));
        // TODO: implement the remaining scale-out routes.
        for path in UNIMPLEMENTED_SCALE_OUT_ROUTES {
            router = router.route(path, post(not_supported));
        }
    } else {
        info!("{SCALE_OUT_DISABLED_MESSAGE}");
        router = router.route("/inference/v1/generate", post(disabled_scale_out));
        for path in UNIMPLEMENTED_SCALE_OUT_ROUTES {
            router = router.route(path, post(disabled_scale_out));
        }
    }

    if state.api_server_options.enable_tokenizer_info_endpoint {
        // TODO: implement `/tokenizer_info`.
        router = router.route("/tokenizer_info", get(not_supported));
    }

    // Process checkpoint lifecycle, which requires sleep mode.
    if state
        .engine_core_client()
        .ready_responses()
        .iter()
        .all(|ready| ready.enable_sleep_mode)
    {
        router = router
            .route("/checkpoint/prepare", post(checkpoint::prepare))
            .route("/checkpoint/restore", post(checkpoint::restore))
            .route("/checkpoint/status", get(checkpoint::status));
    }

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
            .route(
                "/init_weight_transfer_engine",
                post(weight_transfer::init_weight_transfer_engine),
            )
            .route(
                "/shutdown_weight_transfer_engine",
                post(weight_transfer::shutdown_weight_transfer_engine),
            )
            .route(
                "/start_weight_update",
                post(weight_transfer::start_weight_update),
            )
            .route(
                "/start_draft_weight_update",
                post(weight_transfer::start_draft_weight_update),
            )
            .route("/update_weights", post(weight_transfer::update_weights))
            .route(
                "/finish_weight_update",
                post(weight_transfer::finish_weight_update),
            )
            .route(
                "/update_weight_version",
                post(weight_transfer::update_weight_version),
            )
            .route("/weight_info", get(weight_transfer::weight_info))
            .route("/weight_checker", post(weight_checker::weight_checker))
            .route("/abort_requests", post(abort_requests::abort_requests))
            .route("/sleep", post(sleep::sleep))
            .route(
                "/release_kv_cache_memory",
                post(sleep::release_kv_cache_memory),
            )
            .route("/wake_up", post(sleep::wake_up))
            .route("/is_sleeping", get(sleep::is_sleeping))
            .route("/pause", post(pause::pause))
            .route("/resume", post(pause::resume))
            .route("/is_paused", get(pause::is_paused))
            .route("/server_info", get(server_info::server_info))
            .route("/get_world_size", get(world_size::get_world_size))
    }

    if let Some(profiler) = &state.profiler {
        warn!(
            mode = profiler,
            "profiler is enabled in the API server; \
             this should only be used for local development",
        );
        router = router
            .route("/start_profile", post(profile::start_profile))
            .route("/stop_profile", post(profile::stop_profile));
    }

    let enable_request_id_headers = state.api_server_options.enable_request_id_headers;
    let enable_api_key_auth = state.has_api_keys();
    let mut router = router
        .with_state(state.clone())
        .layer(match state.api_server_options.request_body_limit_bytes {
            Some(limit) => DefaultBodyLimit::max(limit),
            None => DefaultBodyLimit::disable(),
        })
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
