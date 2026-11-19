// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Single-writer, multi-reader object storage in POSIX shared memory.
//!
//! The writer process owns a POSIX shm ring buffer; reader processes attach
//! by name. Objects are serialized by the caller and stored as opaque bytes;
//! handles are signed with an HMAC over `(key, address, monotonic_id)` keyed
//! by a random secret at the start of the segment. Eviction is FIFO, driven
//! by a reader reference-count protocol: an object is reclaimable once its
//! reader count reaches `writer handouts * n_readers`.
//!
//! Rust port of `SingleWriterShmRingBuffer` / `SingleWriterShmObjectStorage`
//! from `vllm/distributed/device_communicators/shm_object_storage.py`.

mod error;
mod object_storage;
mod ring_buffer;

pub use error::{Error, Result};
pub use object_storage::{PutFailure, SingleWriterShmObjectStorage};
pub use ring_buffer::SingleWriterShmRingBuffer;
