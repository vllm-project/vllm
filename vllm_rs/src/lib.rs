//! Rust-accelerated KV-cache block manager for vLLM v0.19.0+.
//!
//! Replaces the Python implementations of:
//!   - `vllm.v1.core.kv_cache_utils.KVCacheBlock`
//!   - `vllm.v1.core.kv_cache_utils.FreeKVCacheBlockQueue`
//!   - `vllm.v1.core.block_pool.BlockHashToBlockMap`
//!   - `vllm.v1.core.block_pool.BlockPool` (core hot methods only)
//!
//! Non-hot methods (cache_full_blocks, reset_prefix_cache, KV event queue,
//! metrics collector integration) stay on the Python side — they either call
//! into user code (Request.block_hashes, MetricsCollector) or emit complex
//! event dataclasses that are cheaper to build in Python.
use pyo3::prelude::*;

mod block;
mod hash_map;
mod pool;
mod scheduler_loop;

#[pymodule]
fn vllm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<block::KVCacheBlock>()?;
    m.add_class::<block::FreeKVCacheBlockQueue>()?;
    m.add_class::<hash_map::BlockHashToBlockMap>()?;
    m.add_class::<pool::BlockPool>()?;
    m.add_function(wrap_pyfunction!(scheduler_loop::scheduler_update_preamble, m)?)?;
    Ok(())
}
