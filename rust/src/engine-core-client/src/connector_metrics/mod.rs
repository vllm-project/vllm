// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Descriptor-driven KV connector metrics for third-party and in-tree connectors.
//!
//! First-party Nixl / Mooncake keep typed DTOs and observe paths. Other
//! payloads (`Multi.other` / `Other`) are recorded here when a metrics descriptor
//! is available via the reserved stats key ``_metrics_descriptor`` (connectors
//! emit it once under ``VLLM_USE_RUST_FRONTEND``).

mod adapter;
pub(crate) mod descriptor;
mod dispatch;

pub(crate) use adapter::DescriptorDrivenAdapter;
pub(crate) use dispatch::observe_opaque_connector_stats;
