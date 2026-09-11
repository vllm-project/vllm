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

use crate::grpc::{RlControlServer, pb, utility_status};
use crate::grpc_services::{
    draft_weight_updates_enabled, sleep_mode_enabled, weight_transfer_backend,
};
use crate::state::AppState;

pub(crate) type RlControlGrpcService = RlControlServer<RlControlServiceImpl>;

/// gRPC reinforcement-learning control service backed by the shared application
/// state.
pub struct RlControlServiceImpl {
    state: Arc<AppState>,
    rl_lock: Mutex<()>,
}

impl RlControlServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self {
            state,
            rl_lock: Mutex::new(()),
        }
    }

    fn client(&self) -> &EngineCoreClient {
        self.state.engine_core_client()
    }

    fn require_weight_transfer(&self) -> Result<(), Status> {
        weight_transfer_backend(&self.client().ready_responses())
            .map(|_| ())
            .ok_or_else(|| {
                Status::failed_precondition(
                    "every engine needs the same weight transfer backend; start vLLM with \
                     --weight-transfer-config",
                )
            })
    }

    fn require_sleep_mode(&self) -> Result<(), Status> {
        sleep_mode_enabled(&self.client().ready_responses()).then_some(()).ok_or_else(|| {
            Status::failed_precondition(
                "sleep mode is not configured on every engine; start vLLM with --enable-sleep-mode",
            )
        })
    }
}

pub(crate) fn rl_capabilities(ready: &[&EngineCoreReadyResponse]) -> pb::RlCapabilities {
    let backend = weight_transfer_backend(ready);
    pb::RlCapabilities {
        weight_transfer_enabled: backend.is_some(),
        weight_transfer_backend: backend.unwrap_or_default().to_string(),
        sleep_mode_enabled: sleep_mode_enabled(ready),
        draft_weight_updates_enabled: draft_weight_updates_enabled(ready),
    }
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
impl pb::rl_control_server::RlControl for RlControlServiceImpl {
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
        if !draft_weight_updates_enabled(&self.client().ready_responses()) {
            return Err(Status::failed_precondition(
                "draft weight updates require a configured speculative draft model",
            ));
        }
        let _guard = self.rl_lock.lock().await;
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
