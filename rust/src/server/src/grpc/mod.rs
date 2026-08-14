// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! gRPC services backed by the shared application state.

mod control;
mod convert;
mod health;
mod inference;

/// Generated protobuf/gRPC types for the `vllm` package.
pub mod pb {
    tonic::include_proto!("vllm");
}

pub(crate) use control::ControlGrpcService;
pub use control::ControlServiceImpl;
pub(crate) use health::monitor_health;
pub(crate) use inference::InferenceGrpcService;
pub use inference::InferenceServiceImpl;
pub use pb::control_server::ControlServer;
pub use pb::inference_server::InferenceServer;

#[cfg(test)]
mod tests;
