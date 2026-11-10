// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Rust port of the vLLM v1 engine core: block hashing, KV cache management,
//! and scheduling. The Python implementation under `vllm/v1/core` is the
//! behavioral reference; every component here must produce bitwise identical
//! results.

pub mod connectors;
pub mod hash;
pub mod kv_cache;
pub mod sched;
pub mod structured_output;
#[cfg(test)]
mod trace_replay;
