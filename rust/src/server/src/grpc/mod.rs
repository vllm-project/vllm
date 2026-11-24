// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! gRPC services backed by the shared application state.

mod control;
mod convert;
mod health;
mod inference;
mod kv_events;
mod media;
mod render;

/// Generated protobuf/gRPC types for the `vllm` package.
pub use vllm_proto as pb;

pub(crate) use control::ControlGrpcService;
pub use control::ControlServiceImpl;
pub(crate) use health::{EngineHealth, HealthMonitor};
pub(crate) use inference::InferenceGrpcService;
pub use inference::InferenceServiceImpl;
pub(crate) use kv_events::KvEventsGrpcService;
pub use kv_events::KvEventsServiceImpl;
pub use pb::control_server::ControlServer;
pub use pb::inference_server::InferenceServer;
pub use pb::kv_events_server::KvEventsServer;
pub(crate) use render::{RenderGrpcService, RenderServiceImpl};

#[cfg(test)]
mod tests;
