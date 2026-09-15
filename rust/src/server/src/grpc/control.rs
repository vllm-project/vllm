// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use thiserror_ext::AsReport as _;
use tonic::{Request, Response, Status};
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;

use crate::grpc::pb::kv_transfer_server::KvTransfer as _;
use crate::grpc::pb::lora_server::Lora as _;
use crate::grpc::pb::rl_control_server::RlControl as _;
use crate::grpc::rl_control::rl_capabilities;
use crate::grpc::{
    ControlServer, KvTransferServiceImpl, LoraServiceImpl, RlControlServiceImpl, pb,
};
use crate::grpc_services::{GrpcServices, mounted_entries, service_entry};
use crate::state::AppState;

pub(crate) type ControlGrpcService = ControlServer<ControlServiceImpl>;

/// gRPC control service backed by the shared application state.
pub struct ControlServiceImpl {
    state: Arc<AppState>,
    kv_transfer: Arc<KvTransferServiceImpl>,
    rl_control: Arc<RlControlServiceImpl>,
    lora: Arc<LoraServiceImpl>,
}

impl ControlServiceImpl {
    pub fn new(
        state: Arc<AppState>,
        kv_transfer: Arc<KvTransferServiceImpl>,
        rl_control: Arc<RlControlServiceImpl>,
        lora: Arc<LoraServiceImpl>,
    ) -> Self {
        Self {
            state,
            kv_transfer,
            rl_control,
            lora,
        }
    }

    /// The `KvTransfer` implementation behind the deprecated `Control` aliases,
    /// or `Unimplemented` unless `--grpc-services all` serves them.
    fn kv_transfer(&self) -> Result<&KvTransferServiceImpl, Status> {
        self.aliased(&self.kv_transfer, GrpcServices::KV_TRANSFER)
    }

    /// The `RlControl` implementation behind the deprecated `Control` aliases,
    /// or `Unimplemented` unless `--grpc-services all` serves them.
    fn rl_control(&self) -> Result<&RlControlServiceImpl, Status> {
        self.aliased(&self.rl_control, GrpcServices::RL_CONTROL)
    }

    /// The `Lora` implementation behind the deprecated `Control.LoadLora` and
    /// `Control.UnloadLora` aliases, or `Unimplemented` unless
    /// `--grpc-services all` serves them.
    fn lora(&self) -> Result<&LoraServiceImpl, Status> {
        self.aliased(&self.lora, GrpcServices::LORA)
    }

    fn aliased<'a, T>(&self, target: &'a T, flag: GrpcServices) -> Result<&'a T, Status> {
        if self.state.grpc_control_aliases() {
            return Ok(target);
        }
        let entry = service_entry(flag)
            .ok_or_else(|| Status::unimplemented("the requested service is not mounted"))?;
        Err(Status::unimplemented(format!(
            "the deprecated vllm.Control aliases for {} are only served with `--grpc-services \
             all`; call {} directly",
            entry.service_name, entry.service_name,
        )))
    }

    fn ready(&self) -> &EngineCoreReadyResponse {
        self.state.engine_core_client().ready_response()
    }

    fn client(&self) -> &EngineCoreClient {
        self.state.engine_core_client()
    }

    fn parallelism_info(&self) -> pb::ParallelismInfo {
        let ready = self.ready();
        pb::ParallelismInfo {
            tensor_parallel_size: ready.tensor_parallel_size,
            pipeline_parallel_size: ready.pipeline_parallel_size,
            data_parallel_size: self.client().data_parallel_size().min(u32::MAX as usize) as u32,
            data_parallel_rank: ready.data_parallel_rank,
            decode_context_parallel_size: ready.decode_context_parallel_size,
            world_size: ready.world_size,
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
            max_loras: ready.max_loras,
            rl_capabilities: Some(rl_capabilities(&self.client().ready_responses())),
            services: mounted_entries(self.state.grpc_services())
                .map(|entry| entry.proto as i32)
                .collect(),
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
            supports_lora: self.ready().supports_lora,
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

    async fn list_loras(
        &self,
        request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        self.lora.list_loras(request).await
    }

    async fn load_lora(
        &self,
        request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        self.lora()?.load_lora(request).await
    }

    async fn unload_lora(
        &self,
        request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        self.lora()?.unload_lora(request).await
    }

    async fn get_kv_event_sources(
        &self,
        request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        self.kv_transfer()?.get_kv_event_sources(request).await
    }

    async fn pause_generation(
        &self,
        request: Request<pb::PauseGenerationRequest>,
    ) -> Result<Response<pb::PauseGenerationResponse>, Status> {
        self.rl_control()?.pause_generation(request).await
    }

    async fn resume_generation(
        &self,
        request: Request<pb::ResumeGenerationRequest>,
    ) -> Result<Response<pb::ResumeGenerationResponse>, Status> {
        self.rl_control()?.resume_generation(request).await
    }

    async fn is_paused(
        &self,
        request: Request<pb::IsPausedRequest>,
    ) -> Result<Response<pb::IsPausedResponse>, Status> {
        self.rl_control()?.is_paused(request).await
    }

    async fn sleep(
        &self,
        request: Request<pb::SleepRequest>,
    ) -> Result<Response<pb::SleepResponse>, Status> {
        self.rl_control()?.sleep(request).await
    }

    async fn wake_up(
        &self,
        request: Request<pb::WakeUpRequest>,
    ) -> Result<Response<pb::WakeUpResponse>, Status> {
        self.rl_control()?.wake_up(request).await
    }

    async fn is_sleeping(
        &self,
        request: Request<pb::IsSleepingRequest>,
    ) -> Result<Response<pb::IsSleepingResponse>, Status> {
        self.rl_control()?.is_sleeping(request).await
    }

    async fn init_weight_transfer_engine(
        &self,
        request: Request<pb::InitWeightTransferEngineRequest>,
    ) -> Result<Response<pb::InitWeightTransferEngineResponse>, Status> {
        self.rl_control()?.init_weight_transfer_engine(request).await
    }

    async fn start_weight_update(
        &self,
        request: Request<pb::StartWeightUpdateRequest>,
    ) -> Result<Response<pb::StartWeightUpdateResponse>, Status> {
        self.rl_control()?.start_weight_update(request).await
    }

    async fn start_draft_weight_update(
        &self,
        request: Request<pb::StartDraftWeightUpdateRequest>,
    ) -> Result<Response<pb::StartDraftWeightUpdateResponse>, Status> {
        self.rl_control()?.start_draft_weight_update(request).await
    }

    async fn update_weights(
        &self,
        request: Request<pb::UpdateWeightsRequest>,
    ) -> Result<Response<pb::UpdateWeightsResponse>, Status> {
        self.rl_control()?.update_weights(request).await
    }

    async fn finish_weight_update(
        &self,
        request: Request<pb::FinishWeightUpdateRequest>,
    ) -> Result<Response<pb::FinishWeightUpdateResponse>, Status> {
        self.rl_control()?.finish_weight_update(request).await
    }

    async fn update_weight_version(
        &self,
        request: Request<pb::UpdateWeightVersionRequest>,
    ) -> Result<Response<pb::UpdateWeightVersionResponse>, Status> {
        self.rl_control()?.update_weight_version(request).await
    }

    async fn get_weight_version(
        &self,
        request: Request<pb::GetWeightVersionRequest>,
    ) -> Result<Response<pb::GetWeightVersionResponse>, Status> {
        self.rl_control()?.get_weight_version(request).await
    }
}
