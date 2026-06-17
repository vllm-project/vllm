use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;

use crate::state::AppState;

pub async fn health(State(state): State<Arc<AppState>>) -> StatusCode {
    if state.chat.engine_core_client().is_healthy() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}
