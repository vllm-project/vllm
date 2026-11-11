// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::state::AppState;

#[derive(Serialize)]
pub(crate) struct ServerLoadResponse {
    /// Frontend-local in-flight inference requests.
    server_load: u64,
    /// Remaining prefill work reported by the engine, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_tokens_remaining: Option<u64>,
    /// Monotonic counter for the sample above, so a reader can detect staleness.
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_seq: Option<u64>,
}

pub async fn load(State(state): State<Arc<AppState>>) -> Json<ServerLoadResponse> {
    let prefill = state.prefill_load();
    Json(ServerLoadResponse {
        server_load: state.server_load(),
        prefill_tokens_remaining: prefill.map(|load| load.remaining_tokens),
        sample_seq: prefill.map(|load| load.sample_seq),
    })
}
