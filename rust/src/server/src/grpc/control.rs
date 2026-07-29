// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use thiserror_ext::AsReport as _;
use tonic::{Request, Response, Status};
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;

use super::{ControlServer, pb};
use crate::state::AppState;

pub(crate) type ControlGrpcService = ControlServer<ControlServiceImpl>;

/// gRPC control service backed by the shared application state.
pub struct ControlServiceImpl {
    state: Arc<AppState>,
}

impl ControlServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }

    fn ready(&self) -> &EngineCoreReadyResponse {
        self.state.engine_core_client().ready_response()
    }

    fn parallelism_info(&self) -> pb::ParallelismInfo {
        let ready = self.ready();
        pb::ParallelismInfo {
            tensor_parallel_size: ready.tensor_parallel_size,
            pipeline_parallel_size: ready.pipeline_parallel_size,
            data_parallel_size: ready.data_parallel_size.min(u64::from(u32::MAX)) as u32,
            data_parallel_rank: ready.data_parallel_rank,
            decode_context_parallel_size: ready.decode_context_parallel_size,
        }
    }
}

const GRPC_API_VERSION: &str = "vllm";

#[tonic::async_trait]
impl pb::control_server::Control for ControlServiceImpl {
    async fn get_server_info(
        &self,
        _request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        let ready = self.ready();
        Ok(Response::new(pb::ServerInfo {
            engine_version: ready.vllm_version.clone(),
            api_version: GRPC_API_VERSION.to_string(),
            instance_id: ready.instance_id.clone(),
            parallelism: Some(self.parallelism_info()),
            max_model_len: self.state.engine_core_client().max_model_len(),
            kv_block_size: ready.block_size.min(u64::from(u32::MAX)) as u32,
            total_kv_blocks: self.state.engine_core_client().total_num_gpu_blocks(),
            max_running_requests: ready.max_num_seqs,
            max_batched_tokens: ready.max_num_batched_tokens,
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        let served = self.state.served_model_names();
        Ok(Response::new(pb::ModelInfo {
            model_id: self.state.chat.text().model_id().to_string(),
            served_model_name: self.state.primary_model_name().to_string(),
            served_model_aliases: served.iter().skip(1).cloned().collect(),
            // GenerateRequest accepts both prompt representations.
            supports_text_input: true,
            supports_token_ids_input: true,
            supports_multimodal: self.state.chat.supports_multimodal(),
            reasoning_parser: self
                .state
                .chat
                .reasoning_parser_name()
                .unwrap_or_default()
                .to_string(),
            tool_call_parser: self
                .state
                .chat
                .tool_call_parser_name()
                .unwrap_or_default()
                .to_string(),
        }))
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        let request_ids = request.into_inner().request_ids;
        if request_ids.is_empty() {
            return Ok(Response::new(pb::AbortResponse {}));
        }
        self.state
            .chat
            .abort(&request_ids)
            .await
            .map_err(|error| Status::internal(error.to_report_string()))?;
        Ok(Response::new(pb::AbortResponse {}))
    }
}
