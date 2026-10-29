// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! gRPC services backed by the shared application state.

mod control;
mod convert;
mod health;
mod inference;
mod media;
mod nixl;

/// Generated protobuf/gRPC types for the `vllm` package.
pub use vllm_proto as pb;

pub(crate) use control::ControlGrpcService;
pub use control::ControlServiceImpl;
pub(crate) use health::monitor_health;
pub(crate) use inference::InferenceGrpcService;
pub use inference::InferenceServiceImpl;
#[cfg(feature = "nixl-media")]
pub(crate) use nixl::start as start_nixl_media;
pub use pb::control_server::ControlServer;
pub use pb::inference_server::InferenceServer;

#[cfg(test)]
mod tests;
