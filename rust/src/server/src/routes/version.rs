// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Serialize)]
pub(crate) struct VersionResponse {
    version: String,
    rust_frontend_version: &'static str,
}

/// Get engine and Rust frontend version metadata.
pub async fn version(
    State(state): State<Arc<AppState>>,
) -> Result<Json<VersionResponse>, ApiError> {
    let version = state
        .engine_client()
        .vllm_version()
        .map_err(|error| utility_call_error("vllm_version", error))?
        .to_string();

    Ok(Json(VersionResponse {
        version,
        rust_frontend_version: vllm_build_info::VERSION,
    }))
}
