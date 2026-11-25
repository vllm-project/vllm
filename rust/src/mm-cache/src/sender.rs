// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Frontend (P0) sender cache for `--mm-processor-cache-type shm`.
//!
//! Rust port of `ShmObjectStoreSenderCache` from
//! `vllm/multimodal/cache/shm.py`. The receiver side
//! (`ShmObjectStoreReceiverCache`) runs in the Python engine's worker
//! processes.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use parking_lot::Mutex;
use tracing::{debug, warn};

use super::serde::{serialize_mm_item_into, serialized_mm_item_len};
use crate::Result;
use vllm_engine_core_client::protocol::multimodal::{
    MmBatchedField, MmField, MmFieldElem, MmKwargValue, MmKwargsItem,
};
use vllm_shm_object_storage::{
    PutFailure, SingleWriterShmObjectStorage, SingleWriterShmRingBuffer,
};

/// Configuration for the writer-side multimodal processor shm cache.
#[derive(Debug)]
pub struct MmCacheConfig {
    /// POSIX shm object name (no leading slash), matching the
    /// `--mm-shm-cache-buffer-name` argument given to the Python engine.
    pub shm_name: String,
    /// Ring buffer size in bytes (`mm_processor_cache_gb * GiB`).
    pub data_buffer_size: usize,
    /// Per-object size cap in bytes (`mm_shm_cache_max_object_size_mb * MiB`).
    pub max_object_size: usize,
}

impl MmCacheConfig {
    /// Unique per-process name, e.g. "vllm-mm-shm-{uuid}".
    pub fn unique_shm_name() -> String {
        format!("vllm-mm-shm-{}", uuid::Uuid::new_v4())
    }
}

/// The cache which is used on P0 when SHM caching is enabled.
///
/// How to update each item:
///
/// - If the item is already in the cache, send a tiny "address item" in place
///   of the data to avoid unnecessary IPC.
///
/// - If the item is not in the cache, store the data in shared memory.
///
/// Send + Sync, internally locked.
pub struct MmProcessorShmCache<Metadata> {
    shm_name: String,
    inner: Mutex<SingleWriterShmObjectStorage<Metadata>>,
    /// Per-engine world size (TP*PP); 0 until `set_n_readers` is called.
    n_readers: AtomicU32,
    warned_oversize: AtomicBool,
}

impl<Metadata> MmProcessorShmCache<Metadata> {
    /// Create the shm segment (fails if the name already exists).
    pub fn create(config: MmCacheConfig) -> Result<Self> {
        let ring_buffer =
            SingleWriterShmRingBuffer::create(config.data_buffer_size, &config.shm_name)?;
        Ok(Self {
            shm_name: config.shm_name,
            inner: Mutex::new(SingleWriterShmObjectStorage::new(
                ring_buffer,
                config.max_object_size,
            )?),
            n_readers: AtomicU32::new(0),
            warned_oversize: AtomicBool::new(false),
        })
    }

    /// POSIX shm object name (no leading slash) backing this cache.
    pub fn shm_name(&self) -> &str {
        &self.shm_name
    }

    /// Per-engine world size; learned from the engine handshake.
    ///
    /// Until this is called, all operations are safe no-ops (`is_cached` is
    /// false, `put_or_inline` returns the inline item) because the free
    /// threshold cannot be computed.
    pub fn set_n_readers(&self, n_readers: u32) {
        self.n_readers.store(n_readers, Ordering::Release);
        self.inner.lock().set_n_readers(n_readers);
    }

    /// Remove the segment's name once every reader has opened it, so a crash
    /// cannot leak it; existing mappings stay valid.
    pub fn unlink_name(&self) {
        self.inner.lock().unlink_name();
    }

    /// Check if the item with the given hash is cached.
    pub fn is_cached(&self, hash: &str) -> bool {
        if self.n_readers() == 0 {
            return false;
        }
        self.inner.lock().is_cached(hash)
    }

    /// Bump the writer flag and return the signed address and its writer-local metadata.
    pub fn get_cached(&self, hash: &str) -> Option<(MmKwargsItem, Metadata)>
    where
        Metadata: Clone,
    {
        if self.n_readers() == 0 {
            return None;
        }
        let mut inner = self.inner.lock();
        let (address, monotonic_id, metadata) = inner.get_cached(hash)?;
        let signature = inner.make_signature(hash, address, monotonic_id);
        Some((address_as_item(address, monotonic_id, signature), metadata))
    }

    /// Miss path: put the item into shm and return the signed address item,
    /// or return the input item unchanged for inline sending on oversize /
    /// buffer-full / duplicate-key.
    ///
    /// Each of the `occurrences` issues one handle signed over `hash`.
    pub fn put_or_inline(
        &self,
        hash: &str,
        item: MmKwargsItem,
        metadata: Metadata,
        occurrences: NonZeroUsize,
    ) -> MmKwargsItem {
        if self.n_readers() == 0 {
            return item;
        }
        // Size the chunk with a counting write pass, then serialize directly
        // into the shm chunk: tensor bytes are copied exactly once.
        let payload_len = match serialized_mm_item_len(&item) {
            Ok(len) => len,
            Err(error) => {
                warn!(
                    hash,
                    error = %error,
                    "mm_input failed to serialize for shm cache; sending inline"
                );
                return item;
            }
        };

        let mut inner = self.inner.lock();
        match inner.put_with(hash, payload_len, metadata, occurrences, |buf| {
            serialize_mm_item_into(&item, buf)
        }) {
            Ok(Ok((address, monotonic_id))) => {
                let signature = inner.make_signature(hash, address, monotonic_id);
                address_as_item(address, monotonic_id, signature)
            }
            Err(error) => {
                warn!(
                    hash,
                    error = %error,
                    "mm_input failed to serialize for shm cache; sending inline"
                );
                item
            }
            // The duplicate-key case (concurrent insert) is benign, so we only
            // warn on the oversize case.
            Ok(Err(PutFailure::DuplicateKey)) => item,
            Ok(Err(PutFailure::Oversize)) => {
                if !self.warned_oversize.swap(true, Ordering::Relaxed) {
                    warn!(
                        hash,
                        "mm_input too large to cache; \
                         raise --mm-shm-cache-max-object-size-mb."
                    );
                }
                item
            }
            Ok(Err(PutFailure::BufferFull)) => {
                // Cache full and protected items prevent eviction.
                debug!(
                    hash,
                    "mm_input not cached; shm cache full, \
                     consider raising --mm-processor-cache-gb."
                );
                item
            }
        }
    }

    /// Clear the cache (used by /reset_mm_cache).
    pub fn clear(&self) {
        self.inner.lock().clear();
    }

    /// Number of cached multimodal items.
    pub fn cached_len(&self) -> usize {
        self.inner.lock().cached_len()
    }

    fn n_readers(&self) -> u32 {
        self.n_readers.load(Ordering::Acquire)
    }
}

/// Build the tiny "address item" sent to the engine in place of the real
/// data, matching `ShmObjectStoreSenderCache.address_as_item`.
///
/// The signature is the HMAC-SHA256 handle signature as a list of 32 ints
/// (0-255), which workers verify in get/touch before reading shared memory.
fn address_as_item(address: usize, monotonic_id: u32, signature: [u8; 32]) -> MmKwargsItem {
    fn elem(value: MmKwargValue) -> MmFieldElem {
        MmFieldElem {
            data: Some(value),
            field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
        }
    }

    BTreeMap::from([
        (
            "address".to_string(),
            elem(MmKwargValue::Int(address as i64)),
        ),
        (
            "monotonic_id".to_string(),
            elem(MmKwargValue::Int(monotonic_id as i64)),
        ),
        (
            "signature".to_string(),
            elem(MmKwargValue::List(
                signature.iter().map(|&byte| MmKwargValue::Int(i64::from(byte))).collect(),
            )),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use vllm_engine_core_client::protocol::tensor::WireTensor;

    fn create_cache<Metadata>(
        data_buffer_size: usize,
        max_object_size: usize,
    ) -> MmProcessorShmCache<Metadata> {
        MmProcessorShmCache::create(MmCacheConfig {
            shm_name: MmCacheConfig::unique_shm_name(),
            data_buffer_size,
            max_object_size,
        })
        .unwrap()
    }

    fn item_with_tensor(elems: usize) -> MmKwargsItem {
        BTreeMap::from([(
            "pixel_values".to_string(),
            MmFieldElem {
                data: Some(MmKwargValue::Tensor(
                    WireTensor::from_f32(vec![elems], vec![1.0; elems]).unwrap(),
                )),
                field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
            },
        )])
    }

    fn put(cache: &MmProcessorShmCache<()>, hash: &str, item: MmKwargsItem) -> MmKwargsItem {
        cache.put_or_inline(hash, item, (), NonZeroUsize::MIN)
    }

    /// The 32 signature bytes of an address item.
    fn signature_of(item: &MmKwargsItem) -> Vec<u8> {
        let Some(MmKwargValue::List(signature)) = item["signature"].data.as_ref() else {
            panic!("signature must be a list");
        };
        signature
            .iter()
            .map(|value| match value {
                MmKwargValue::Int(byte) => u8::try_from(*byte).expect("signature byte"),
                _ => panic!("signature bytes must be ints"),
            })
            .collect()
    }

    #[test]
    fn operations_are_noops_until_n_readers_is_set() {
        let cache = create_cache(1 << 20, 1 << 16);
        let item = item_with_tensor(8);

        assert!(!cache.is_cached("hash-a"));
        assert!(cache.get_cached("hash-a").is_none());

        // The item comes back unchanged for inline sending.
        let returned = put(&cache, "hash-a", item.clone());
        assert_eq!(returned, item);
        assert!(!cache.is_cached("hash-a"));
        assert_eq!(cache.cached_len(), 0);
    }

    #[test]
    fn put_returns_address_item_and_hit_returns_same_address() {
        let cache = create_cache(1 << 20, 1 << 16);
        cache.set_n_readers(2);

        let address_item =
            cache.put_or_inline("hash-a", item_with_tensor(8), "metadata", NonZeroUsize::MIN);
        assert_eq!(address_item.len(), 3);
        assert_eq!(address_item["address"].data, Some(MmKwargValue::Int(0)));
        assert_eq!(
            address_item["monotonic_id"].data,
            Some(MmKwargValue::Int(0))
        );
        assert!(
            address_item
                .values()
                .all(|elem| elem.field == MmField::Batched(MmBatchedField { keep_on_cpu: false }))
        );
        assert!(cache.is_cached("hash-a"));
        assert_eq!(cache.cached_len(), 1);

        // Hit path bumps the writer flag and re-uses the address and signature.
        let (hit_item, metadata) = cache.get_cached("hash-a").unwrap();
        assert_eq!(metadata, "metadata");
        assert_eq!(hit_item, address_item);
        assert!(cache.get_cached("hash-b").is_none());
    }

    #[test]
    fn address_items_are_signed_over_the_key() {
        let cache = create_cache(1 << 20, 1 << 16);
        cache.set_n_readers(1);

        let item = put(&cache, "hash-a", item_with_tensor(8));
        let signature = signature_of(&item);

        let inner = cache.inner.lock();
        assert_eq!(signature.as_slice(), inner.make_signature("hash-a", 0, 0));
        // Not over another key, and not over another handle.
        assert_ne!(signature.as_slice(), inner.make_signature("hash-b", 0, 0));
        assert_ne!(signature.as_slice(), inner.make_signature("hash-a", 0, 1));
    }

    #[test]
    fn put_with_duplicate_key_returns_inline_item() {
        let cache = create_cache(1 << 20, 1 << 16);
        cache.set_n_readers(1);

        let original = item_with_tensor(8);
        put(&cache, "hash-a", original.clone());
        let returned = put(&cache, "hash-a", original.clone());
        assert_eq!(returned, original);
    }

    #[test]
    fn oversize_item_falls_back_to_inline() {
        let cache = create_cache(1 << 20, 256);
        cache.set_n_readers(1);

        let original = item_with_tensor(1024); // 4 KiB tensor > 256 byte cap
        let returned = put(&cache, "hash-a", original.clone());
        assert_eq!(returned, original);
        assert!(!cache.is_cached("hash-a"));
    }

    #[test]
    fn full_buffer_with_unread_chunks_falls_back_to_inline() {
        // Only one small item fits in the buffer.
        let cache = create_cache(160, 1 << 16);
        cache.set_n_readers(1);

        let first = put(&cache, "hash-a", item_with_tensor(8));
        assert!(first.contains_key("address"));

        // Unread first chunk cannot be evicted, so the second item is inline.
        let original = item_with_tensor(8);
        let returned = put(&cache, "hash-b", original.clone());
        assert_eq!(returned, original);
        assert!(!cache.is_cached("hash-b"));
    }
}
