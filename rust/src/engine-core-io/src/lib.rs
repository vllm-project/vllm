// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The engine core's side of the frontend transport: the ZMQ input and
//! output socket threads of `EngineCoreProc` (`process_input_sockets`,
//! `process_output_sockets`).
//!
//! The threads know nothing of the scheduler: they connect the sockets,
//! take part in the startup and DP coordinator protocol, and move frames.
//! Decoding requests is up to the caller; outputs come in the wire types of
//! `vllm-engine-core-client`, which the scheduler fills
//! (`vllm_engine_core::sched::wire`).
//!
//! TODO(rust-engine-core): the scheduler needs only the protocol types. Once
//! those are a crate of their own, the scheduler stops depending on the
//! client crate, and this crate is what keeps the transport (tokio, zeromq)
//! out of it.

pub mod input;
pub mod output;
mod stopped;
