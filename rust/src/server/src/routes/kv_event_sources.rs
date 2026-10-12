// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use vllm_engine_core_client::protocol::handshake::KvEventsConfig;

use crate::state::AppState;

/// KV-event publisher config of each data-parallel rank, keyed by rank.
pub async fn kv_event_sources(
    State(state): State<Arc<AppState>>,
) -> Json<BTreeMap<u32, KvEventsConfig>> {
    Json(
        state
            .engine_core_client()
            .ready_responses()
            .into_iter()
            .filter_map(|ready| Some((ready.data_parallel_rank, ready.kv_events_config.clone()?)))
            .collect(),
    )
}
