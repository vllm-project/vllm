// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::state::AppState;

/// One data-parallel rank's KV compatibility identity.
#[derive(Serialize)]
pub struct KvIdentity {
    kv_identity: String,
    components: Option<BTreeMap<String, String>>,
}

/// KV compatibility identity of each data-parallel rank, keyed by rank.
pub async fn kv_identity(State(state): State<Arc<AppState>>) -> Json<BTreeMap<u32, KvIdentity>> {
    Json(
        state
            .engine_core_client()
            .ready_responses()
            .into_iter()
            .filter_map(|ready| {
                let identity = KvIdentity {
                    kv_identity: ready.kv_identity.clone()?,
                    components: ready.kv_identity_components.clone(),
                };
                Some((ready.data_parallel_rank, identity))
            })
            .collect(),
    )
}
