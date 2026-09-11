// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use tonic::{Request, Response, Status};
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;

use crate::grpc::{KvTransferServer, pb};
use crate::state::AppState;

pub(crate) type KvTransferGrpcService = KvTransferServer<KvTransferServiceImpl>;

/// gRPC KV transfer discovery service backed by the shared application state.
pub struct KvTransferServiceImpl {
    state: Arc<AppState>,
}

impl KvTransferServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl pb::kv_transfer_server::KvTransfer for KvTransferServiceImpl {
    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        let client = self.state.engine_core_client();
        let sources = client.ready_responses().into_iter().filter_map(kv_event_source).collect();
        Ok(Response::new(pb::GetKvEventSourcesResponse { sources }))
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
