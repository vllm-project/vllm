use std::sync::Arc;

use axum::Json;
use axum::extract::{Query, State};
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub(crate) struct WorldSizeParams {
    /// If true (default), returns the world size including data parallelism
    /// (TP * PP * DP). If false, returns the world size without data
    /// parallelism (TP * PP).
    #[serde(default = "default_true")]
    include_dp: bool,
}

const fn default_true() -> bool {
    true
}

#[derive(Serialize)]
pub(crate) struct WorldSizeResponse {
    world_size: u64,
}

/// Get the world size from the parallel config.
///
/// Currently reads static values captured during the engine startup handshake.
///
/// TODO: If the world size can change at runtime (e.g. elastic EP scaling,
/// DP rank recovery), this should be switched to either:
/// - A `call_utility("get_world_size", (include_dp,))` RPC to the Python
///   engine for live values (simple, adds one ZMQ round-trip per request), or
/// - A push-based approach where the engine sends config updates via the
///   output stream into shared state (zero per-request overhead, more complex).
pub async fn get_world_size(
    State(state): State<Arc<AppState>>,
    Query(params): Query<WorldSizeParams>,
) -> Result<Json<WorldSizeResponse>, ApiError> {
    let client = state.engine_core_client();

    let ws = client.world_size();

    let world_size = if params.include_dp {
        let dp = client.data_parallel_size();
        ws * dp
    } else {
        ws
    };

    Ok(Json(WorldSizeResponse { world_size }))
}
