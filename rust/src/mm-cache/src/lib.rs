// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Shared-memory multimodal processor cache (`--mm-processor-cache-type shm`).
//!
//! The frontend process (P0, single writer) owns a POSIX shm ring buffer; the
//! engine's worker processes (P1) are the readers. On a cache miss the
//! frontend stores the preprocessed [`MmKwargsItem`] in shm and sends the
//! engine a tiny "address item" instead; on a hit it skips re-sending the
//! data entirely. Oversize objects or a full buffer fall back to sending the
//! full item inline.
//!
//! Rust port of the Python originals:
//! - `SingleWriterShmRingBuffer` / `SingleWriterShmObjectStorage` /
//!   `MsgpackSerde` from
//!   `vllm/distributed/device_communicators/shm_object_storage.py`
//! - `ShmObjectStoreSenderCache` from `vllm/multimodal/cache/shm.py`
//!
//! [`MmKwargsItem`]: vllm_engine_core_client::protocol::multimodal::MmKwargsItem

mod error;
mod sender;
mod serde;
#[cfg(test)]
mod tests;

pub use error::{Error, Result};
pub use sender::{MmProcessorShmCache, MmShmCacheConfig};
