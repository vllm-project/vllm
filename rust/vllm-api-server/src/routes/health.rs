use axum::{extract::State, http::StatusCode, Json};
use serde::Serialize;
use std::sync::Arc;
use crate::server::AppState;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
}

pub async fn health_check(
    State(_state): State<Arc<AppState>>,
) -> (StatusCode, Json<HealthResponse>) {
    (
        StatusCode::OK,
        Json(HealthResponse {
            status: "healthy".to_string(),
        }),
    )
}
