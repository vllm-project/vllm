use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use rmpv::Value as MsgpackValue;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Debug, Deserialize)]
pub(crate) struct CollectiveRpcRequest {
    method: Option<String>,
    #[serde(default)]
    timeout: Option<f64>,
    #[serde(default)]
    args: Vec<JsonValue>,
    #[serde(default)]
    kwargs: BTreeMap<String, JsonValue>,
}

#[derive(Debug, Serialize)]
pub(crate) struct CollectiveRpcResponse {
    results: Vec<MsgpackValue>,
}

/// Execute a development-only collective RPC on the connected engine(s).
pub async fn collective_rpc(
    State(state): State<Arc<AppState>>,
    body: Result<Json<CollectiveRpcRequest>, JsonRejection>,
) -> Result<Json<CollectiveRpcResponse>, ApiError> {
    let Json(body) = body.map_err(|error| ApiError::json_parse_error(error.body_text()))?;
    let method = body.method.ok_or_else(|| {
        ApiError::invalid_request(
            "Missing 'method' in request body".to_string(),
            Some("method"),
        )
    })?;

    let results = state
        .engine_core_client()
        .collective_rpc(&method, body.timeout, body.args, body.kwargs)
        .await
        .map_err(|error| utility_call_error("collective_rpc", error))?;

    Ok(Json(CollectiveRpcResponse { results }))
}
