// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The Rust engine core's side of the frontend protocol: converting the
//! scheduler's outputs to the wire types of `vllm-engine-core-client`, and the
//! ZMQ socket threads of `EngineCoreProc`.
//!
//! The wire types are the Rust frontend's, so an engine core using this crate
//! serves the Rust frontend only.

pub mod convert;
pub mod output;
