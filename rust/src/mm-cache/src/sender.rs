// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Frontend (P0) sender cache for `--mm-processor-cache-type shm`.
//!
//! Rust port of `ShmObjectStoreSenderCache` from
//! `vllm/multimodal/cache/shm.py`. The receiver side
//! (`ShmObjectStoreReceiverCache`) runs in the Python engine's worker
//! processes.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use parking_lot::Mutex;
use thiserror_ext::AsReport as _;
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
#[derive(Debug, Clone)]
pub struct MmShmCacheConfig {
    /// POSIX shm object name (no leading slash), matching the
    /// VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME env given to the Python engine.
    pub shm_name: String,
    /// Ring buffer size in bytes (`mm_processor_cache_gb * GiB`).
    pub data_buffer_size: usize,
    /// Per-object size cap in bytes (`mm_shm_cache_max_object_size_mb * MiB`).
    pub max_object_size: usize,
}

impl MmShmCacheConfig {
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
pub struct MmProcessorShmCache {
    shm_name: String,
    inner: Mutex<SingleWriterShmObjectStorage>,
    /// Per-engine world size (TP*PP); 0 until `set_n_readers` is called.
    n_readers: AtomicU32,
    warned_oversize: AtomicBool,
}

impl MmProcessorShmCache {
    /// Create the shm segment (fails if the name already exists).
    pub fn create(config: MmShmCacheConfig) -> Result<Self> {
        let ring_buffer =
            SingleWriterShmRingBuffer::create(config.data_buffer_size, &config.shm_name)?;
        Ok(Self {
            shm_name: config.shm_name,
            inner: Mutex::new(SingleWriterShmObjectStorage::new(
                ring_buffer,
                config.max_object_size,
            )),
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
    /// false, `touch` does nothing, `put_or_inline` returns the inline item)
    /// because the free threshold cannot be computed.
    pub fn set_n_readers(&self, n_readers: u32) {
        self.n_readers.store(n_readers, Ordering::Release);
        self.inner.lock().set_n_readers(n_readers);
    }

    /// Check if the item with the given hash is cached.
    pub fn is_cached(&self, hash: &str) -> bool {
        if self.n_readers() == 0 {
            return false;
        }
        self.inner.lock().is_cached(hash)
    }

    /// Pre-touch one hash of the current request (eviction protection if
    /// cached).
    ///
    /// Mirrors the `_merge_mm_kwargs` pre-touch: a probed-hit item is kept in
    /// the `_touched` set so it cannot be evicted before this request's
    /// `address_item_for_cached`/`put_or_inline` calls; the protection is
    /// released by `get_cached` or `release_touches`.
    pub fn touch(&self, hash: &str) {
        if self.n_readers() == 0 {
            return;
        }
        self.inner.lock().touch(hash);
    }

    /// Stop protecting touched items that were not looked up.
    ///
    /// Called after each request's merge completes (success or error),
    /// mirroring Python's `finally: release_sender_touches()`.
    pub fn release_touches(&self) {
        self.inner.lock().release_touches();
    }

    /// Number of items currently protected by `touch`, for tests.
    #[cfg(feature = "test-util")]
    pub fn touched_len(&self) -> usize {
        self.inner.lock().touched_len()
    }

    /// Hit path: bump writer_flag and build the signed address item; `None`
    /// if the hash is (no longer) cached.
    pub fn address_item_for_cached(&self, hash: &str) -> Option<MmKwargsItem> {
        if self.n_readers() == 0 {
            return None;
        }
        let mut inner = self.inner.lock();
        let (address, monotonic_id) = inner.get_cached(hash)?;
        let signature = inner.get_signature(hash)?;
        Some(address_as_item(address, monotonic_id, signature))
    }

    /// Miss path: put the item into shm and return the signed address item,
    /// or return the input item unchanged for inline sending on oversize /
    /// buffer-full / duplicate-key.
    ///
    /// `hash` is the cache key the issued handle is signed over.
    pub fn put_or_inline(&self, hash: &str, item: MmKwargsItem) -> MmKwargsItem {
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
                    error = %error.to_report_string(),
                    "mm_input failed to serialize for shm cache; sending inline"
                );
                return item;
            }
        };

        let mut inner = self.inner.lock();
        match inner.put_with(hash, payload_len, |buf| serialize_mm_item_into(&item, buf)) {
            Ok(Ok((address, monotonic_id))) => {
                let signature = inner.get_signature(hash).expect("put just inserted the key");
                address_as_item(address, monotonic_id, signature)
            }
            Err(error) => {
                warn!(
                    hash,
                    error = %error.to_report_string(),
                    "mm_input failed to serialize for shm cache; sending inline"
                );
                item
            }
            // The duplicate-key case (concurrent insert) is benign, so we only
            // warn on the oversize case. Subsequent UUID-only requests for an
            // oversize item will fail with a cache miss.
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

    /// Number of cached keys; the frontend shadow map uses this to mirror
    /// Python's `remove_dangling_items` pruning threshold.
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
    use std::sync::atomic::{AtomicU64, Ordering};

    use expect_test::expect;

    use super::*;
    use vllm_engine_core_client::protocol::tensor::WireTensor;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn create_cache(data_buffer_size: usize, max_object_size: usize) -> MmProcessorShmCache {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        MmProcessorShmCache::create(MmShmCacheConfig {
            shm_name: format!("vllm-mm-shm-test-sender-{}-{id}", std::process::id()),
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

    /// Assert the item's handle fields and snapshot its shape; the address,
    /// id, and (random, per-segment) signature are redacted to placeholders
    /// before snapshotting and checked separately.
    fn expect_address_item(item: &MmKwargsItem, address: i64, monotonic_id: i64) {
        let mut redacted = item.clone();
        for key in ["address", "monotonic_id"] {
            redacted.get_mut(key).expect("address item field").data = Some(MmKwargValue::Int(0));
        }
        redacted.get_mut("signature").expect("address item must carry a signature").data =
            Some(MmKwargValue::List(Vec::new()));
        expect![[r#"
            {
                "address": MmFieldElem {
                    data: Some(
                        Int(
                            0,
                        ),
                    ),
                    field: Batched(
                        MmBatchedField {
                            keep_on_cpu: false,
                        },
                    ),
                },
                "monotonic_id": MmFieldElem {
                    data: Some(
                        Int(
                            0,
                        ),
                    ),
                    field: Batched(
                        MmBatchedField {
                            keep_on_cpu: false,
                        },
                    ),
                },
                "signature": MmFieldElem {
                    data: Some(
                        List(
                            [],
                        ),
                    ),
                    field: Batched(
                        MmBatchedField {
                            keep_on_cpu: false,
                        },
                    ),
                },
            }"#]]
        .assert_eq(&format!("{redacted:#?}"));
        let Some(MmKwargValue::Int(actual_address)) = item["address"].data.as_ref() else {
            panic!("address must be an int");
        };
        let Some(MmKwargValue::Int(actual_id)) = item["monotonic_id"].data.as_ref() else {
            panic!("monotonic_id must be an int");
        };
        assert_eq!(*actual_address, address);
        assert_eq!(*actual_id, monotonic_id);

        let Some(MmKwargValue::List(signature)) = item["signature"].data.as_ref() else {
            panic!("signature must be a list");
        };
        assert_eq!(signature.len(), 32, "HMAC-SHA256 signature is 32 bytes");
        for value in signature {
            let MmKwargValue::Int(byte) = value else {
                panic!("signature bytes must be ints");
            };
            assert!((0..=255).contains(byte), "signature bytes are 0-255");
        }
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
        cache.touch("hash-a"); // no-op
        cache.release_touches(); // no-op
        assert_eq!(cache.address_item_for_cached("hash-a"), None);

        // The item comes back unchanged for inline sending.
        let returned = cache.put_or_inline("hash-a", item.clone());
        assert_eq!(returned, item);
        assert!(!cache.is_cached("hash-a"));
        assert_eq!(cache.cached_len(), 0);
    }

    #[test]
    fn put_returns_address_item_and_hit_returns_same_address() {
        let cache = create_cache(1 << 20, 1 << 16);
        cache.set_n_readers(2);

        let address_item = cache.put_or_inline("hash-a", item_with_tensor(8));
        expect_address_item(&address_item, 0, 0);
        assert!(cache.is_cached("hash-a"));
        assert_eq!(cache.cached_len(), 1);

        // Hit path bumps the writer flag and re-uses the address and signature.
        let hit_item = cache.address_item_for_cached("hash-a").unwrap();
        expect_address_item(&hit_item, 0, 0);
        assert_eq!(signature_of(&hit_item), signature_of(&address_item));
        assert_eq!(cache.address_item_for_cached("hash-b"), None);
    }

    #[test]
    fn address_items_are_signed_over_the_key() {
        let cache = create_cache(1 << 20, 1 << 16);
        cache.set_n_readers(1);

        let item = cache.put_or_inline("hash-a", item_with_tensor(8));
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
        cache.put_or_inline("hash-a", original.clone());
        let returned = cache.put_or_inline("hash-a", original.clone());
        assert_eq!(returned, original);
    }

    #[test]
    fn oversize_item_falls_back_to_inline() {
        let cache = create_cache(1 << 20, 256);
        cache.set_n_readers(1);

        let original = item_with_tensor(1024); // 4 KiB tensor > 256 byte cap
        let returned = cache.put_or_inline("hash-a", original.clone());
        assert_eq!(returned, original);
        assert!(!cache.is_cached("hash-a"));
    }

    #[test]
    fn full_buffer_with_unread_chunks_falls_back_to_inline() {
        // Only one small item fits in the buffer.
        let cache = create_cache(160, 1 << 16);
        cache.set_n_readers(1);

        let first = cache.put_or_inline("hash-a", item_with_tensor(8));
        assert!(first.contains_key("address"));

        // Unread first chunk cannot be evicted, so the second item is inline.
        let original = item_with_tensor(8);
        let returned = cache.put_or_inline("hash-b", original.clone());
        assert_eq!(returned, original);
        assert!(!cache.is_cached("hash-b"));
    }

    #[test]
    fn clear_allows_reinsertion() {
        let cache = create_cache(1 << 20, 1 << 16);
        cache.set_n_readers(1);

        cache.put_or_inline("hash-a", item_with_tensor(8));
        cache.clear();
        assert!(!cache.is_cached("hash-a"));
        assert_eq!(cache.cached_len(), 0);

        // After clearing, the next allocation restarts at address zero while
        // IDs keep increasing, so pre-clear handles fail the engine's ID check
        // once their chunk is reused.
        let address_item = cache.put_or_inline("hash-a", item_with_tensor(8));
        expect_address_item(&address_item, 0, 1);
    }
}
