// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Native ports of KV connectors (`vllm/distributed/kv_transfer/kv_connector/`)
//! for the Rust engine core: their scheduler halves, which implement the
//! scheduler's `KvConnector` trait on the Rust request, and the hot paths
//! that run per block.
//!
//! They are kept out of `vllm-engine-core`, which defines the trait, as
//! plugins are: the binding layer registers them.

pub mod mooncake_store;
pub mod mooncake_store_scheduler;
pub mod nixl;

#[cfg(test)]
mod tests;
