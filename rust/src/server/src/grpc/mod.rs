// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! gRPC services backed by the shared application state.

mod control;
mod convert;
mod health;
mod inference;
mod kv_transfer;
mod rl_control;

use thiserror_ext::AsReport as _;
use tonic::Status;

/// Generated protobuf/gRPC types for the `vllm` package.
pub mod pb {
    tonic::include_proto!("vllm");
}

pub(crate) use control::ControlGrpcService;
pub use control::ControlServiceImpl;
pub(crate) use health::{mark_serving, monitor_health};
pub(crate) use inference::InferenceGrpcService;
pub use inference::InferenceServiceImpl;
pub use kv_transfer::KvTransferServiceImpl;
pub(crate) use kv_transfer::{KvTransferGrpcService, kv_event_source};
pub use pb::control_server::ControlServer;
pub use pb::inference_server::InferenceServer;
pub use pb::kv_transfer_server::KvTransferServer;
pub use pb::rl_control_server::RlControlServer;
pub(crate) use rl_control::RlControlGrpcService;
pub use rl_control::RlControlServiceImpl;

fn utility_status(method: &'static str, error: vllm_engine_core_client::Error) -> Status {
    Status::internal(format!("{method} failed: {}", error.to_report_string()))
}

#[cfg(test)]
mod tests;
