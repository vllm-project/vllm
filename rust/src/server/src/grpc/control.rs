// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use serde_json::Value as JsonValue;
use thiserror_ext::AsReport as _;
use tokio::sync::Mutex;
use tonic::{Request, Response, Status};
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_engine_core_client::protocol::utility::PauseMode as EnginePauseMode;

use super::{ControlServer, pb};
use crate::state::AppState;

pub(crate) type ControlGrpcService = ControlServer<ControlServiceImpl>;

/// gRPC control service backed by the shared application state.
pub struct ControlServiceImpl {
    state: Arc<AppState>,
    rl_lock: Mutex<()>,
}

impl ControlServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self {
            state,
            rl_lock: Mutex::new(()),
        }
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
            data_parallel_size: self.state.data_parallel_size().min(u32::MAX as usize) as u32,
            data_parallel_rank: ready.data_parallel_rank,
            decode_context_parallel_size: ready.decode_context_parallel_size,
        }
    }

    fn weight_transfer_backend(&self) -> Option<&str> {
        let responses = self.client().ready_responses();
        let backend = responses.first()?.weight_transfer_backend.as_deref()?;
        responses
            .iter()
            .all(|ready| ready.weight_transfer_backend.as_deref() == Some(backend))
            .then_some(backend)
    }

    fn sleep_mode_enabled(&self) -> bool {
        self.client().ready_responses().iter().all(|ready| ready.enable_sleep_mode)
    }

    fn draft_weight_updates_enabled(&self) -> bool {
        self.client()
            .ready_responses()
            .iter()
            .all(|ready| ready.supports_draft_weight_updates)
    }

    fn rl_capabilities(&self) -> pb::RlCapabilities {
        let backend = self.weight_transfer_backend();
        pb::RlCapabilities {
            weight_transfer_enabled: backend.is_some(),
            weight_transfer_backend: backend.unwrap_or_default().to_string(),
            sleep_mode_enabled: self.sleep_mode_enabled(),
            draft_weight_updates_enabled: self.draft_weight_updates_enabled(),
        }
    }

    fn require_weight_transfer(&self) -> Result<(), Status> {
        self.weight_transfer_backend().map(|_| ()).ok_or_else(|| {
            Status::failed_precondition(
                "weight transfer is not configured; start vLLM with --weight-transfer-config",
            )
        })
    }

    fn require_sleep_mode(&self) -> Result<(), Status> {
        self.sleep_mode_enabled().then_some(()).ok_or_else(|| {
            Status::failed_precondition(
                "sleep mode is not configured; start vLLM with --enable-sleep-mode",
            )
        })
    }

    async fn require_paused(&self) -> Result<(), Status> {
        let paused = self
            .client()
            .is_scheduler_paused()
            .await
            .map_err(|error| utility_status("is_scheduler_paused", error))?;
        paused
            .then_some(())
            .ok_or_else(|| Status::failed_precondition("pause generation before updating weights"))
    }
}

const GRPC_API_VERSION: &str = "vllm";

fn utility_status(method: &'static str, error: vllm_engine_core_client::Error) -> Status {
    Status::internal(format!("{method} failed: {}", error.to_report_string()))
}

fn pause_mode(mode: i32) -> Result<EnginePauseMode, Status> {
    match pb::PauseMode::try_from(mode) {
        Ok(pb::PauseMode::Unspecified | pb::PauseMode::Abort) => Ok(EnginePauseMode::Abort),
        Ok(pb::PauseMode::Wait) => Ok(EnginePauseMode::Wait),
        Ok(pb::PauseMode::Keep) => Ok(EnginePauseMode::Keep),
        Err(_) => Err(Status::invalid_argument("invalid pause mode")),
    }
}

fn json_object(bytes: &[u8], field: &'static str) -> Result<JsonValue, Status> {
    let value = serde_json::from_slice::<JsonValue>(bytes).map_err(|error| {
        Status::invalid_argument(format!(
            "{field} must contain valid JSON: {}",
            error.to_report_string()
        ))
    })?;
    if !value.is_object() {
        return Err(Status::invalid_argument(format!(
            "{field} must contain a JSON object"
        )));
    }
    Ok(value)
}

fn weight_version(value: String) -> Result<String, Status> {
    if value.trim().is_empty() {
        return Err(Status::invalid_argument("weight_version must not be empty"));
    }
    Ok(value)
}

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
            rl_capabilities: Some(self.rl_capabilities()),
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

    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        let client = self.state.engine_core_client();
        let sources = client.ready_responses().into_iter().filter_map(kv_event_source).collect();
        Ok(Response::new(pb::GetKvEventSourcesResponse { sources }))
    }

    async fn pause_generation(
        &self,
        request: Request<pb::PauseGenerationRequest>,
    ) -> Result<Response<pb::PauseGenerationResponse>, Status> {
        let request = request.into_inner();
        let mode = pause_mode(request.mode)?;
        let clear_cache = request.clear_cache.unwrap_or(true);
        let _guard = self.rl_lock.lock().await;
        self.client()
            .pause_scheduler(mode, clear_cache)
            .await
            .map_err(|error| utility_status("pause_generation", error))?;
        Ok(Response::new(pb::PauseGenerationResponse {}))
    }

    async fn resume_generation(
        &self,
        _request: Request<pb::ResumeGenerationRequest>,
    ) -> Result<Response<pb::ResumeGenerationResponse>, Status> {
        let _guard = self.rl_lock.lock().await;
        self.client()
            .resume_scheduler()
            .await
            .map_err(|error| utility_status("resume_generation", error))?;
        Ok(Response::new(pb::ResumeGenerationResponse {}))
    }

    async fn is_paused(
        &self,
        _request: Request<pb::IsPausedRequest>,
    ) -> Result<Response<pb::IsPausedResponse>, Status> {
        let paused = self
            .client()
            .is_scheduler_paused()
            .await
            .map_err(|error| utility_status("is_paused", error))?;
        Ok(Response::new(pb::IsPausedResponse { paused }))
    }

    async fn sleep(
        &self,
        request: Request<pb::SleepRequest>,
    ) -> Result<Response<pb::SleepResponse>, Status> {
        self.require_sleep_mode()?;
        let request = request.into_inner();
        let mode = pause_mode(request.mode)?;
        let level = request.level.unwrap_or(1);
        let _guard = self.rl_lock.lock().await;
        self.client()
            .sleep(level, mode)
            .await
            .map_err(|error| utility_status("sleep", error))?;
        Ok(Response::new(pb::SleepResponse {}))
    }

    async fn wake_up(
        &self,
        request: Request<pb::WakeUpRequest>,
    ) -> Result<Response<pb::WakeUpResponse>, Status> {
        self.require_sleep_mode()?;
        let tags = request.into_inner().tags;
        let tags = (!tags.is_empty()).then_some(tags);
        let _guard = self.rl_lock.lock().await;
        self.client()
            .wake_up(tags)
            .await
            .map_err(|error| utility_status("wake_up", error))?;
        Ok(Response::new(pb::WakeUpResponse {}))
    }

    async fn is_sleeping(
        &self,
        _request: Request<pb::IsSleepingRequest>,
    ) -> Result<Response<pb::IsSleepingResponse>, Status> {
        self.require_sleep_mode()?;
        let sleeping = self
            .client()
            .is_sleeping()
            .await
            .map_err(|error| utility_status("is_sleeping", error))?;
        Ok(Response::new(pb::IsSleepingResponse { sleeping }))
    }

    async fn init_weight_transfer_engine(
        &self,
        request: Request<pb::InitWeightTransferEngineRequest>,
    ) -> Result<Response<pb::InitWeightTransferEngineResponse>, Status> {
        self.require_weight_transfer()?;
        let init_info = json_object(&request.into_inner().init_info_json, "init_info_json")?;
        let _guard = self.rl_lock.lock().await;
        self.client()
            .init_weight_transfer_engine(init_info)
            .await
            .map_err(|error| utility_status("init_weight_transfer_engine", error))?;
        Ok(Response::new(pb::InitWeightTransferEngineResponse {}))
    }

    async fn start_weight_update(
        &self,
        _request: Request<pb::StartWeightUpdateRequest>,
    ) -> Result<Response<pb::StartWeightUpdateResponse>, Status> {
        self.require_weight_transfer()?;
        let _guard = self.rl_lock.lock().await;
        self.require_paused().await?;
        self.client()
            .start_weight_update()
            .await
            .map_err(|error| utility_status("start_weight_update", error))?;
        Ok(Response::new(pb::StartWeightUpdateResponse {}))
    }

    async fn start_draft_weight_update(
        &self,
        _request: Request<pb::StartDraftWeightUpdateRequest>,
    ) -> Result<Response<pb::StartDraftWeightUpdateResponse>, Status> {
        self.require_weight_transfer()?;
        if !self.draft_weight_updates_enabled() {
            return Err(Status::failed_precondition(
                "draft weight updates require a configured speculative draft model",
            ));
        }
        let _guard = self.rl_lock.lock().await;
        self.require_paused().await?;
        self.client()
            .start_draft_weight_update()
            .await
            .map_err(|error| utility_status("start_draft_weight_update", error))?;
        Ok(Response::new(pb::StartDraftWeightUpdateResponse {}))
    }

    async fn update_weights(
        &self,
        request: Request<pb::UpdateWeightsRequest>,
    ) -> Result<Response<pb::UpdateWeightsResponse>, Status> {
        self.require_weight_transfer()?;
        let update_info = json_object(&request.into_inner().update_info_json, "update_info_json")?;
        let _guard = self.rl_lock.lock().await;
        self.require_paused().await?;
        self.client()
            .update_weights(update_info)
            .await
            .map_err(|error| utility_status("update_weights", error))?;
        Ok(Response::new(pb::UpdateWeightsResponse {}))
    }

    async fn finish_weight_update(
        &self,
        request: Request<pb::FinishWeightUpdateRequest>,
    ) -> Result<Response<pb::FinishWeightUpdateResponse>, Status> {
        self.require_weight_transfer()?;
        let version = request.into_inner().weight_version.map(weight_version).transpose()?;
        let _guard = self.rl_lock.lock().await;
        self.require_paused().await?;
        self.client()
            .finish_weight_update()
            .await
            .map_err(|error| utility_status("finish_weight_update", error))?;
        if let Some(version) = version {
            self.client()
                .set_weight_version(&version)
                .await
                .map_err(|error| utility_status("update_weight_version", error))?;
        }
        Ok(Response::new(pb::FinishWeightUpdateResponse {}))
    }

    async fn update_weight_version(
        &self,
        request: Request<pb::UpdateWeightVersionRequest>,
    ) -> Result<Response<pb::UpdateWeightVersionResponse>, Status> {
        let version = weight_version(request.into_inner().weight_version)?;
        let _guard = self.rl_lock.lock().await;
        self.client()
            .set_weight_version(&version)
            .await
            .map_err(|error| utility_status("update_weight_version", error))?;
        Ok(Response::new(pb::UpdateWeightVersionResponse {}))
    }

    async fn get_weight_version(
        &self,
        _request: Request<pb::GetWeightVersionRequest>,
    ) -> Result<Response<pb::GetWeightVersionResponse>, Status> {
        let weight_version = self
            .client()
            .get_weight_version()
            .await
            .map_err(|error| utility_status("get_weight_version", error))?;
        Ok(Response::new(pb::GetWeightVersionResponse {
            weight_version,
        }))
    }
}

pub(super) fn kv_event_source(response: &EngineCoreReadyResponse) -> Option<pb::KvEventSource> {
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
