// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

/// One data-parallel rank's KV compatibility identity.
#[derive(Serialize, Deserialize)]
pub struct KvIdentity {
    kv_identity: Option<String>,
    components: Option<BTreeMap<String, String>>,
}

/// Current KV compatibility identity of each data-parallel rank, keyed by
/// rank. Engines are asked on each call, as weight changes update it.
pub async fn kv_identity(
    State(state): State<Arc<AppState>>,
) -> Result<Json<BTreeMap<u32, KvIdentity>>, ApiError> {
    let identities: Vec<(u32, KvIdentity)> = state
        .engine_core_client()
        .call_utility("get_kv_identity", ())
        .await
        .map_err(|error| utility_call_error("get_kv_identity", error))?;
    Ok(Json(identities.into_iter().collect()))
}
