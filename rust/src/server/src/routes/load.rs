use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::state::AppState;

#[derive(Serialize)]
pub(crate) struct ServerLoadResponse {
    server_load: u64,
}

pub async fn load(State(state): State<Arc<AppState>>) -> Json<ServerLoadResponse> {
    Json(ServerLoadResponse {
        server_load: state.server_load(),
    })
}
