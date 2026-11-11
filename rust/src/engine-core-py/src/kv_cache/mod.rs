// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Python views of the Rust scheduler's KV cache manager
//! (`vllm_engine_core::kv_cache`), for the Python code that reads it: KV
//! connectors, HiSparse, and callers inspecting the scheduler.

// The methods mirror the Python signatures.
#![allow(clippy::too_many_arguments)]

pub(crate) mod config;
pub(crate) mod events;
mod hisparse;
pub(crate) mod views;

use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<views::KVCacheBlocksView>()?;
    m.add_class::<views::Block>()?;
    m.add_class::<views::BlockList>()?;
    m.add_class::<views::BlockPoolView>()?;
    m.add_class::<views::CoordinatorView>()?;
    m.add_class::<views::SingleTypeManagerView>()?;
    m.add_class::<views::ReqToBlocks>()?;
    m.add_class::<views::FreeBlockQueue>()?;
    m.add_class::<hisparse::HiSparseCoordinatorView>()?;
    m.add_class::<hisparse::RequestStates>()?;
    Ok(())
}
