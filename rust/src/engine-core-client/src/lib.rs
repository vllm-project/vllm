// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod client;
mod coordinator;
mod error;
mod metrics;
pub mod mock_engine;
mod prompt_block_hasher;
pub mod runtime;
#[cfg(any(test, feature = "test-util"))]
pub mod test_utils;
mod transport;
mod utility;

pub use client::{
    AbortCause, CoordinatorMode, EngineCoreClient, EngineCoreClientConfig, EngineCoreOutputStream,
    EngineCoreStreamDelivery, EngineCoreStreamOutput, TransportMode,
};
pub use error::{Error, Result};
pub use transport::EngineId;
/// The wire types, from `vllm-engine-core-protocol`.
pub use vllm_engine_core_protocol as protocol;
pub use vllm_engine_core_protocol::ENGINE_CORE_DEAD_SENTINEL;

#[cfg(test)]
mod tests;
