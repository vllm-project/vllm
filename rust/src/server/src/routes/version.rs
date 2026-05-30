use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::state::AppState;

#[derive(Serialize)]
pub(crate) struct VersionResponse {
    version: String,
    rust_frontend_version: &'static str,
}

/// Get engine and Rust frontend version metadata.
pub async fn version(State(state): State<Arc<AppState>>) -> Json<VersionResponse> {
    let version = state.engine_core_client().vllm_version().to_string();

    Json(VersionResponse {
        version,
        rust_frontend_version: env!("CARGO_PKG_VERSION"),
    })
}
