// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! gRPC services backed by the shared application state.

mod convert;
mod health;
mod inference;
mod rl_control;

/// Generated protobuf/gRPC types for the `vllm` package.
pub mod pb {
    tonic::include_proto!("vllm");
}

pub(crate) use health::monitor_health;
pub(crate) use inference::InferenceGrpcService;
pub use inference::InferenceServiceImpl;
pub use pb::control_server::ControlServer;
pub use pb::inference_server::InferenceServer;
pub(crate) use rl_control::ControlGrpcService;
pub use rl_control::ControlServiceImpl;

#[cfg(test)]
mod tests;
