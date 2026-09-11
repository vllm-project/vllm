// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use tokio::sync::OnceCell;
use tonic::{Request, Response, Status};
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_engine_core_client::protocol::kv_transfer::KvConnectorHandshakeEntry;

use crate::grpc::{KvTransferServer, pb, utility_status};
use crate::state::AppState;

pub(crate) type KvTransferGrpcService = KvTransferServer<KvTransferServiceImpl>;

/// gRPC KV transfer discovery service backed by the shared application state.
pub struct KvTransferServiceImpl {
    state: Arc<AppState>,
    /// Per-engine handshake entries, fetched from the engines on first use.
    /// The metadata is fixed once the KV caches are registered.
    kv_handshake_entries: OnceCell<Vec<Vec<KvConnectorHandshakeEntry>>>,
}

impl KvTransferServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self {
            state,
            kv_handshake_entries: OnceCell::new(),
        }
    }

    async fn kv_handshake_entries(&self) -> Result<&[Vec<KvConnectorHandshakeEntry>], Status> {
        self.kv_handshake_entries
            .get_or_try_init(|| async {
                self.client()
                    .get_kv_connector_handshake_entries()
                    .await
                    .map_err(|error| utility_status("get_kv_connector_handshake_entries", error))
            })
            .await
            .map(Vec::as_slice)
    }

    fn kv_engine_index(&self, engine_id: &str) -> Result<usize, Status> {
        self.client()
            .ready_responses()
            .iter()
            .position(|ready| {
                ready.kv_transfer_info.as_ref().is_some_and(|info| info.engine_id == engine_id)
            })
            .ok_or_else(|| {
                Status::not_found(format!("unknown KV transfer engine_id {engine_id:?}"))
            })
    }

    fn client(&self) -> &EngineCoreClient {
        self.state.engine_core_client()
    }
}

#[tonic::async_trait]
impl pb::kv_transfer_server::KvTransfer for KvTransferServiceImpl {
    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        let sources = self
            .client()
            .ready_responses()
            .into_iter()
            .filter_map(kv_event_source)
            .collect();
        Ok(Response::new(pb::GetKvEventSourcesResponse { sources }))
    }

    async fn get_kv_transfer_info(
        &self,
        _request: Request<pb::GetKvTransferInfoRequest>,
    ) -> Result<Response<pb::GetKvTransferInfoResponse>, Status> {
        let entries = self.kv_handshake_entries().await?;
        let engines = self
            .client()
            .ready_responses()
            .into_iter()
            .zip(entries)
            .filter_map(|(ready, entries)| kv_transfer_engine(ready, entries))
            .collect();
        Ok(Response::new(pb::GetKvTransferInfoResponse { engines }))
    }

    async fn get_kv_handshake_metadata(
        &self,
        request: Request<pb::GetKvHandshakeMetadataRequest>,
    ) -> Result<Response<pb::GetKvHandshakeMetadataResponse>, Status> {
        let engine_id = request.into_inner().engine_id;
        let index = self.kv_engine_index(&engine_id)?;
        let entries = self.kv_handshake_entries().await?;
        let ranks = entries
            .get(index)
            .map(|entries| entries.iter().map(kv_handshake_rank).collect())
            .unwrap_or_default();
        Ok(Response::new(pb::GetKvHandshakeMetadataResponse { ranks }))
    }
}

pub(crate) fn kv_event_source(response: &EngineCoreReadyResponse) -> Option<pb::KvEventSource> {
    let config = response.kv_events_config.as_ref()?;
    if !config.enable_kv_cache_events || config.publisher != "zmq" {
        return None;
    }

    Some(pb::KvEventSource {
        transport: "zmq".to_string(),
        endpoint: config.endpoint.clone(),
        topic: config.topic.clone(),
        replay_endpoint: config.replay_endpoint.clone().unwrap_or_default(),
        data_parallel_rank: Some(response.data_parallel_rank),
        encoding: "msgpack".to_string(),
        schema_version: 1,
        buffer_steps: config.buffer_steps,
        hwm: config.hwm,
        max_queue_size: config.max_queue_size,
    })
}

pub(super) fn kv_transfer_engine(
    ready: &EngineCoreReadyResponse,
    entries: &[KvConnectorHandshakeEntry],
) -> Option<pb::KvTransferEngine> {
    let info = ready.kv_transfer_info.as_ref()?;
    Some(pb::KvTransferEngine {
        engine_id: info.engine_id.clone(),
        connector: info.kv_connector.clone(),
        role: info.kv_role.clone(),
        data_parallel_rank: ready.data_parallel_rank,
        tensor_parallel_size: ready.tensor_parallel_size,
        pipeline_parallel_size: ready.pipeline_parallel_size,
        kv_block_size: ready.block_size.min(u64::from(u32::MAX)) as u32,
        compatibility_hash: entries
            .iter()
            .find_map(|entry| entry.compatibility_hash.clone())
            .unwrap_or_default(),
    })
}

pub(super) fn kv_handshake_rank(entry: &KvConnectorHandshakeEntry) -> pb::KvHandshakeRank {
    pb::KvHandshakeRank {
        pp_rank: entry.pp_rank,
        tp_rank: entry.tp_rank,
        compatibility_hash: entry.compatibility_hash.clone().unwrap_or_default(),
        encoding: "msgpack".to_string(),
        payload: entry.payload.to_vec(),
    }
}
