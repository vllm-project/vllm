// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use vllm_engine_core_client::protocol::handshake::KvEventsConfig;

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

/// KV-event publisher config of each data-parallel rank, keyed by rank.
pub async fn kv_event_sources(
    State(state): State<Arc<AppState>>,
) -> Result<Json<BTreeMap<u32, KvEventsConfig>>, ApiError> {
    Ok(Json(
        state
            .engine_client()
            .ready_responses()
            .map_err(|error| utility_call_error("ready_responses", error))?
            .into_iter()
            .filter_map(|ready| Some((ready.data_parallel_rank, ready.kv_events_config.clone()?)))
            .collect(),
    ))
}
