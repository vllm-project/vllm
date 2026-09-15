// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use thiserror_ext::AsReport as _;
use tonic::{Code, Request, Response, Status};
use vllm_engine_core_client::protocol::lora::LoraRequest;

use crate::config::LoraModulePath;
use crate::grpc::{LoraServer, pb};
use crate::lora::{LoadLoraError, LoraDisabledError, LoraPathAccessError, UnloadLoraError};
use crate::state::AppState;

pub(crate) type LoraGrpcService = LoraServer<LoraServiceImpl>;

/// gRPC LoRA lifecycle service backed by the shared application state.
pub struct LoraServiceImpl {
    state: Arc<AppState>,
}

impl LoraServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

fn lora_to_proto(adapter: &LoraRequest) -> pb::LoraAdapter {
    pb::LoraAdapter {
        lora_id: adapter.lora_int_id.min(i64::MAX as u64) as i64,
        lora_name: adapter.lora_name.clone(),
        source_path: adapter.lora_path.clone(),
    }
}

fn load_lora_status(error: LoadLoraError) -> Status {
    let code = match &error {
        LoadLoraError::Disabled(_) => Code::FailedPrecondition,
        LoadLoraError::InvalidAdapter { .. } => Code::InvalidArgument,
        LoadLoraError::PathAccess(LoraPathAccessError::InvalidPath { .. }) => Code::InvalidArgument,
        LoadLoraError::PathAccess(LoraPathAccessError::InvalidConfiguration { .. }) => {
            Code::Internal
        }
        LoadLoraError::AlreadyLoaded { .. } | LoadLoraError::BaseModelName { .. } => {
            Code::AlreadyExists
        }
        LoadLoraError::Engine { .. } | LoadLoraError::NotLoaded { .. } => Code::Internal,
    };
    Status::new(code, error.to_report_string())
}

fn unload_lora_status(error: UnloadLoraError) -> Status {
    let code = match &error {
        UnloadLoraError::Disabled(_) => Code::FailedPrecondition,
        UnloadLoraError::NotFound { .. } => Code::NotFound,
        UnloadLoraError::IntIdMismatch { .. }
        | UnloadLoraError::Engine { .. }
        | UnloadLoraError::NotRemoved { .. } => Code::Internal,
    };
    Status::new(code, error.to_report_string())
}

fn list_loras_status(error: LoraDisabledError) -> Status {
    Status::failed_precondition(error.to_report_string())
}

#[tonic::async_trait]
impl pb::lora_server::Lora for LoraServiceImpl {
    async fn load_lora(
        &self,
        request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        let request = request.into_inner();
        let module = LoraModulePath {
            name: request.lora_name,
            path: request.source_path,
            base_model_name: None,
            is_3d_lora_weight: false,
        };
        let adapter = self.state.load_lora(module, false).await.map_err(load_lora_status)?;
        Ok(Response::new(pb::LoadLoraResponse {
            adapter: Some(lora_to_proto(&adapter)),
        }))
    }

    async fn unload_lora(
        &self,
        request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        let name = request.into_inner().lora_name;
        if name.trim().is_empty() {
            return Err(Status::invalid_argument("lora_name is required"));
        }

        let adapter = self
            .state
            .list_loras()
            .await
            .map_err(list_loras_status)?
            .into_iter()
            .find(|adapter| adapter.lora_name == name)
            .ok_or_else(|| Status::not_found(format!("LoRA adapter `{name}` is not loaded")))?;
        let adapter = self
            .state
            .unload_lora(&name, Some(adapter.lora_int_id))
            .await
            .map_err(unload_lora_status)?;
        Ok(Response::new(pb::UnloadLoraResponse {
            adapter: Some(lora_to_proto(&adapter)),
        }))
    }

    async fn list_loras(
        &self,
        _request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        let mut adapters = self.state.list_loras().await.map_err(list_loras_status)?;
        adapters.sort_by(|left, right| left.lora_name.cmp(&right.lora_name));
        Ok(Response::new(pb::ListLorasResponse {
            adapters: adapters.iter().map(lora_to_proto).collect(),
        }))
    }
}
