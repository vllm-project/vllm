// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Serialize;
use vllm_engine_core_client::protocol::handshake::{KvEventsConfig, PeerLookupInfo};

use crate::state::AppState;

/// One data-parallel rank: the KV-event publisher config, plus how peers reach
/// the rank for a KV prefix lookup when its connector serves them.
#[derive(Serialize)]
pub struct KvEventSource {
    #[serde(flatten)]
    kv_events: KvEventsConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    peer_lookup: Option<PeerLookupInfo>,
}

/// The Python `KVEventsConfig()` defaults: a disabled event source, reported
/// for a rank that serves peer lookups but publishes no KV events.
fn disabled_kv_events() -> KvEventsConfig {
    KvEventsConfig {
        enable_kv_cache_events: false,
        publisher: "null".to_string(),
        endpoint: "tcp://*:5557".to_string(),
        replay_endpoint: None,
        buffer_steps: 10_000,
        hwm: 100_000,
        max_queue_size: 100_000,
        topic: String::new(),
    }
}

/// KV-event source of each data-parallel rank, keyed by rank.
pub async fn kv_event_sources(
    State(state): State<Arc<AppState>>,
) -> Json<BTreeMap<u32, KvEventSource>> {
    Json(
        state
            .engine_core_client()
            .ready_responses()
            .into_iter()
            .filter_map(|ready| {
                let kv_events = match (&ready.kv_events_config, &ready.peer_lookup) {
                    (Some(config), _) => config.clone(),
                    (None, Some(_)) => disabled_kv_events(),
                    (None, None) => return None,
                };
                Some((
                    ready.data_parallel_rank,
                    KvEventSource {
                        kv_events,
                        peer_lookup: ready.peer_lookup.clone(),
                    },
                ))
            })
            .collect(),
    )
}
