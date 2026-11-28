// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use serde::Serialize;
use serde_json::{Map, Value};
use vllm_engine_core_client::protocol::fault_tolerance::{
    FaultToleranceRequest, FaultToleranceStatusResponse, is_allowed_ft_instruction,
};

use crate::error::ApiError;
use crate::state::AppState;

#[derive(Debug, Serialize)]
pub(crate) struct ApplyResponse {
    message: &'static str,
    request_id: String,
}

fn is_python_falsy(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::Number(n) => n.as_f64() == Some(0.0),
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(m) => m.is_empty(),
    }
}

fn validate_payload(body: Value) -> Result<FaultToleranceRequest, ApiError> {
    let Value::Object(mut object) = body else {
        return Err(ApiError::invalid_request(
            "Request body must be a JSON object.".to_string(),
            None,
        ));
    };

    let instruction = match object.remove("instruction") {
        None => {
            return Err(ApiError::invalid_request(
                "'instruction' is required.".to_string(),
                None,
            ));
        }
        Some(value) if is_python_falsy(&value) => {
            return Err(ApiError::invalid_request(
                "'instruction' is required.".to_string(),
                None,
            ));
        }
        Some(Value::String(s)) if is_allowed_ft_instruction(&s) => s,
        Some(Value::String(s)) => {
            return Err(ApiError::invalid_request(
                format!("Invalid instruction: '{s}'."),
                None,
            ));
        }
        Some(other) => {
            return Err(ApiError::invalid_request(
                format!("Invalid instruction: '{other}'."),
                None,
            ));
        }
    };

    let params = match object.remove("params") {
        None => Map::new(),
        Some(Value::Object(params)) => params,
        Some(_) => {
            return Err(ApiError::invalid_request(
                "'params' must be an object.".to_string(),
                None,
            ));
        }
    };

    if let Some(abort) = params.get("abort")
        && !abort.is_boolean()
    {
        return Err(ApiError::invalid_request(
            "'abort' must be a boolean.".to_string(),
            None,
        ));
    }

    let request_id = match object.remove("request_id") {
        Some(Value::String(request_id)) if !request_id.is_empty() => request_id,
        _ => {
            return Err(ApiError::invalid_request(
                "'request_id' must be a non-empty string.".to_string(),
                None,
            ));
        }
    };

    Ok(FaultToleranceRequest {
        instruction,
        params,
        request_id,
    })
}

/// Dispatch a fault-tolerance recovery instruction in the background and
/// immediately return `202 Accepted`.
///
/// Original Python definition:
/// `vllm/entrypoints/serve/fault_tolerance/api_router.py`
/// (`process_fault_tolerance_instruction`).
pub async fn apply(
    State(state): State<Arc<AppState>>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<(StatusCode, Json<ApplyResponse>), ApiError> {
    let Json(body) = body?;
    let ft_request = validate_payload(body)?;
    let request_id = ft_request.request_id.clone();

    // Recovery runs cross-rank collective ops that only complete once every
    // rank has been dispatched. Run it in the background and return immediately
    // so the orchestrator can dispatch to all ranks without blocking;
    // completion is observed by polling GET /v1/fault_tolerance/status.
    state.engine_core_client().spawn_handle_fault(ft_request);

    Ok((
        StatusCode::ACCEPTED,
        Json(ApplyResponse {
            message: "Request accepted; poll /v1/fault_tolerance/status for updates.",
            request_id,
        }),
    ))
}

/// Return the locally-cached fault-tolerance status for the managed engine(s).
///
/// Original Python definition:
/// `vllm/entrypoints/serve/fault_tolerance/api_router.py` (`get_status`).
pub async fn status(State(state): State<Arc<AppState>>) -> Json<FaultToleranceStatusResponse> {
    Json(state.engine_core_client().fault_tolerance_status())
}
