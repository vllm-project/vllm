// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `vllm._rust_core`: Python bindings for `vllm-engine-core`.

use pyo3::prelude::*;

mod hash;
mod kv_cache;
mod mooncake;
mod nixl;
mod scheduler;

#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    hash::register(m)?;
    kv_cache::register(m)?;
    mooncake::register(m)?;
    nixl::register(m)?;
    scheduler::register(m)?;
    Ok(())
}
