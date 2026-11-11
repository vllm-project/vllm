// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Replays of traces recorded from the Python scheduler with the Python
//! connectors, against the Rust scheduler with the native ones (see
//! `vllm-engine-core`'s `tests/README.md`).

mod mooncake_store_trace;
mod nixl_trace;
