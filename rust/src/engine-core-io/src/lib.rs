// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The engine core's side of the frontend transport: the ZMQ input and
//! output socket threads of `EngineCoreProc` (`process_input_sockets`,
//! `process_output_sockets`).
//!
//! The threads know nothing of the scheduler: they connect the sockets,
//! take part in the startup and DP coordinator protocol, and move frames.
//! Decoding requests is up to the caller; outputs come in the wire types of
//! `vllm-engine-core-protocol`, which the scheduler fills
//! (`vllm_engine_core::sched::wire`). This crate is what keeps the transport
//! (tokio, zeromq) out of the scheduler.

pub mod input;
pub mod output;
mod stopped;
