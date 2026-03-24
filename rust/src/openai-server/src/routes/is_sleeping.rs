use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Serialize;
use thiserror_ext::AsReport as _;

use crate::error::ApiError;
use crate::state::AppState;

#[derive(Serialize)]
pub(crate) struct IsSleepingResponse {
    is_sleeping: bool,
}

pub async fn is_sleeping(
    State(state): State<Arc<AppState>>,
) -> Result<Json<IsSleepingResponse>, ApiError> {
    let is_sleeping = state
        .chat
        .engine_core_client()
        .is_sleeping()
        .await
        .map_err(|error| {
            ApiError::server_error(format!(
                "failed to query is_sleeping: {}",
                error.as_report()
            ))
        })?;

    Ok(Json(IsSleepingResponse { is_sleeping }))
}
