use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::state::AppState;

#[derive(Serialize)]
pub(crate) struct VersionResponse {
    version: String,
}

pub async fn version(State(state): State<Arc<AppState>>) -> Json<VersionResponse> {
    let version = state.engine_core_client().vllm_version().to_string();

    Json(VersionResponse { version })
}
