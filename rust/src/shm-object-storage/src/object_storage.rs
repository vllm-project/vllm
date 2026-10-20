// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Writer-side key-value object storage on top of the shm ring buffer.
//!
//! Rust port of the writer half of `SingleWriterShmObjectStorage` from
//! `vllm/distributed/device_communicators/shm_object_storage.py`. The reader
//! half lives in the Python engine's worker processes.

use std::collections::{HashMap, HashSet};

use hmac::{Hmac, Mac as _};
use sha2::{Digest as _, Sha256};

use super::ring_buffer::SingleWriterShmRingBuffer;

/// HMAC-SHA256 handle signature, keyed by the shm segment's auth secret.
type HmacSha256 = Hmac<Sha256>;

/// Number of payload bytes reserved for the reader reference-count flag.
const FLAG_BYTES: usize = 4;

/// Why a `put` could not cache the item; the sender cache turns every variant
/// into an inline fallback, mirroring the Python sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PutFailure {
    /// The key already exists in the storage (concurrent duplicate insert).
    /// Benign: no warning is logged for this case.
    #[error("key already exists in the storage")]
    DuplicateKey,
    /// Serialized object size exceeds `max_object_size`.
    #[error("serialized object exceeds max object size")]
    Oversize,
    /// Not enough space in the buffer even after freeing unused chunks;
    /// protected items prevent eviction.
    #[error("shm cache full")]
    BufferFull,
}

/// One cached entry: the shm handle (address plus monotonic id).
#[derive(Debug, Clone)]
struct CachedEntry {
    address: usize,
    monotonic_id: u32,
}

/// A single-writer, multiple-reader object storage system built on top of a
/// shared memory ring buffer. Provides key-value storage with automatic memory
/// management and cross-process serialization support.
///
/// This storage system follows a FIFO (First-In-First-Out) eviction policy
/// where the oldest objects are automatically freed when memory runs low.
/// Memory is reclaimed based on reader reference counting - objects are only
/// freed when all readers have finished accessing them.
///
/// Handle Signatures:
/// - Handles are signed with an HMAC of (key, address, monotonic_id),
///   keyed by a random secret stored at the start of the shared memory
/// - The writer signs the handles it issues and rejects stale or forged
///   handles supplied by requests
/// - Readers verify the signature in get/touch before reading shared memory
///
/// Memory Layout per Object:
/// `[4-byte reference_count][0x01 format marker][msgpack body]`
pub struct SingleWriterShmObjectStorage {
    ring_buffer: SingleWriterShmRingBuffer,
    max_object_size: usize,
    n_readers: u32,
    /// Key-value mapping: key -> cached entry
    key_index: HashMap<String, CachedEntry>,
    /// Reverse mapping: monotonic_id -> key
    id_index: HashMap<u32, String>,
    /// Writer flag to track in-use status: monotonic_id -> count
    writer_flag: HashMap<u32, u64>,
    /// Items touch() keeps from being freed until they are looked up
    /// or release_touches() is called
    touched: HashSet<u32>,
}

impl SingleWriterShmObjectStorage {
    pub fn new(ring_buffer: SingleWriterShmRingBuffer, max_object_size: usize) -> Self {
        Self {
            ring_buffer,
            max_object_size,
            // Learned from the engine handshake; callers gate all operations
            // until `set_n_readers` is called because the free threshold
            // cannot be computed before that.
            n_readers: 0,
            key_index: HashMap::new(),
            id_index: HashMap::new(),
            writer_flag: HashMap::new(),
            touched: HashSet::new(),
        }
    }

    pub fn set_n_readers(&mut self, n_readers: u32) {
        self.n_readers = n_readers;
    }

    /// Clear the object storage.
    pub fn clear(&mut self) {
        self.ring_buffer.clear();
        self.key_index.clear();
        self.id_index.clear();
        self.writer_flag.clear();
        self.touched.clear();
        tracing::debug!("object storage cleared and reinitialized");
    }

    /// Check if the object with the given key is cached.
    pub fn is_cached(&self, key: &str) -> bool {
        self.key_index.contains_key(key)
    }

    /// Number of cached keys, for the sender cache's shadow-map pruning.
    pub fn cached_len(&self) -> usize {
        self.key_index.len()
    }

    /// Get the cached object by key if it exists, bumping its writer flag and
    /// releasing its touch protection.
    pub fn get_cached(&mut self, key: &str) -> Option<(usize, u32)> {
        let entry = self.key_index.get(key)?;
        let (address, monotonic_id) = (entry.address, entry.monotonic_id);
        self.increment_writer_flag(monotonic_id);
        self.touched.remove(&monotonic_id);
        Some((address, monotonic_id))
    }

    /// Stop protecting touched items that were not looked up.
    pub fn release_touches(&mut self) {
        self.touched.clear();
    }

    /// Number of items currently protected by `touch`, for tests.
    #[cfg(feature = "test-util")]
    pub fn touched_len(&self) -> usize {
        self.touched.len()
    }

    /// Sign the handle of a cached object so readers can verify it.
    pub fn get_signature(&self, key: &str) -> Option<[u8; 32]> {
        let entry = self.key_index.get(key)?;
        Some(self.make_signature(key, entry.address, entry.monotonic_id))
    }

    /// HMAC-SHA256 over `sha256(key) + address_le8 + monotonic_id_le4`,
    /// keyed by the secret stored at the start of the shared memory.
    pub fn make_signature(&self, key: &str, address: usize, monotonic_id: u32) -> [u8; 32] {
        let mut payload = Sha256::digest(key.as_bytes()).to_vec();
        payload.extend_from_slice(&(address as u64).to_le_bytes());
        payload.extend_from_slice(&monotonic_id.to_le_bytes());
        let mut mac = HmacSha256::new_from_slice(&self.ring_buffer.auth_secret())
            .expect("HMAC accepts keys of any size");
        mac.update(&payload);
        mac.finalize().into_bytes().into()
    }

    /// Touch an existing cached item.
    ///
    /// For writers (`ShmObjectStoreSenderCache`): protect the item from
    /// eviction until get_cached or release_touches.
    pub fn touch(&mut self, key: &str) {
        let Some(entry) = self.key_index.get(key) else {
            return;
        };
        self.touched.insert(entry.monotonic_id);
    }

    /// Store a key-value pair in the object storage.
    ///
    /// Attempts to free `2 * max_object_size` bytes using FIFO order when the
    /// ring buffer runs out of space during a put() operation.
    ///
    /// `body` is the serialized object payload as produced by
    /// `super::serde::serialize_mm_item`: the 0x01 format marker followed by
    /// the msgpack body.
    pub fn put(&mut self, key: &str, body: &[u8]) -> Result<(usize, u32), PutFailure> {
        match self.put_with::<std::convert::Infallible>(key, body.len(), |buf| {
            buf.copy_from_slice(body);
            Ok(())
        }) {
            Ok(result) => result,
        }
    }

    /// Store a key-value pair, serializing the payload into the chunk in
    /// place through `write` instead of copying a pre-built buffer.
    ///
    /// On a `write` error the chunk is left unrecorded — it has no
    /// writer-flag entry, so the next eviction pass reclaims it immediately.
    pub fn put_with<E>(
        &mut self,
        key: &str,
        payload_len: usize,
        write: impl FnOnce(&mut [u8]) -> std::result::Result<(), E>,
    ) -> std::result::Result<Result<(usize, u32), PutFailure>, E> {
        if self.key_index.contains_key(key) {
            return Ok(Err(PutFailure::DuplicateKey));
        }

        let buffer_size = FLAG_BYTES + payload_len;
        // Sanity checks
        if buffer_size > self.max_object_size {
            return Ok(Err(PutFailure::Oversize));
        }

        // Allocate new buffer, freeing unused chunks on first failure.
        let (address, monotonic_id) = match self.ring_buffer.allocate_buf(buffer_size) {
            Ok(allocated) => allocated,
            Err(_) => {
                self.free_unused();
                // try again after freeing up space
                match self.ring_buffer.allocate_buf(buffer_size) {
                    Ok(allocated) => allocated,
                    Err(_) => return Ok(Err(PutFailure::BufferFull)),
                }
            }
        };

        // Write data to buffer: `[reader_count = 0][payload]`.
        self.ring_buffer.write_payload(address, 0, &0_i32.to_le_bytes());
        self.ring_buffer.write_payload_with(address, FLAG_BYTES, payload_len, write)?;
        self.increment_writer_flag(monotonic_id);

        // Update key index
        self.key_index.insert(
            key.to_string(),
            CachedEntry {
                address,
                monotonic_id,
            },
        );
        self.id_index.insert(monotonic_id, key.to_string());
        Ok(Ok((address, monotonic_id)))
    }

    /// Set the in-use flag for the writer.
    fn increment_writer_flag(&mut self, id: u32) {
        *self.writer_flag.entry(id).or_insert(0) += 1;
    }

    /// Free unused buffers in the ring buffer.
    fn free_unused(&mut self) {
        // try to free up 2*max_object_size bytes of space in the ring buffer,
        // since the buffer might be fragmented
        let budget = 2 * self.max_object_size;
        let Self {
            ring_buffer,
            writer_flag,
            n_readers,
            touched,
            ..
        } = self;
        let n_readers = u64::from(*n_readers);
        let freed_ids = ring_buffer.free_buf(
            |id, reader_count| {
                // Items touch() keeps from being freed until they are looked
                // up or release_touches() is called.
                if touched.contains(&id) {
                    return false;
                }
                // An item is freeable iff
                // `reader_count >= writer_flag[id] * n_readers`.
                let writer_count = *writer_flag.get(&id).unwrap_or(&0);
                u64::try_from(reader_count).unwrap_or(0) >= writer_count * n_readers
            },
            Some(budget),
        );
        // update the metadata after freeing up space
        for freed_id in freed_ids {
            if let Some(key) = self.id_index.remove(&freed_id) {
                self.key_index.remove(&key);
            }
            self.writer_flag.remove(&freed_id);
        }
    }

    /// Set the reader count of the chunk at `address`, simulating reader
    /// reads from the Python engine workers, for tests.
    #[cfg(test)]
    fn set_reader_count(&mut self, address: usize, count: i32) {
        self.ring_buffer.write_payload(address, 0, &count.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::ring_buffer::MD_SIZE;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn create_storage(
        data_buffer_size: usize,
        max_object_size: usize,
    ) -> SingleWriterShmObjectStorage {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = format!("vllm-mm-shm-test-storage-{}-{id}", std::process::id());
        let ring_buffer = SingleWriterShmRingBuffer::create(data_buffer_size, &name).unwrap();
        let mut storage = SingleWriterShmObjectStorage::new(ring_buffer, max_object_size);
        storage.set_n_readers(1);
        storage
    }

    fn body(len: usize) -> Vec<u8> {
        vec![0xAB; len]
    }

    #[test]
    fn put_get_cached_and_touch_track_protection() {
        let mut storage = create_storage(4096, 1024);

        let (address, id) = storage.put("hash-a", &body(100)).unwrap();
        assert!(storage.is_cached("hash-a"));
        assert!(!storage.is_cached("hash-b"));
        assert_eq!(storage.writer_flag[&id], 1);

        // Hit path bumps the writer flag and returns the same address.
        assert_eq!(storage.get_cached("hash-a"), Some((address, id)));
        assert_eq!(storage.writer_flag[&id], 2);
        assert_eq!(storage.get_cached("hash-b"), None);

        // Touch protects from eviction without bumping the writer flag;
        // get_cached releases the protection again.
        storage.touch("hash-a");
        storage.touch("hash-b"); // no-op for unknown keys
        assert_eq!(storage.writer_flag[&id], 2);
        assert!(storage.touched.contains(&id));
        storage.get_cached("hash-a");
        assert_eq!(storage.writer_flag[&id], 3);
        assert!(!storage.touched.contains(&id));

        // The payload starts with a zeroed reader-count flag.
        assert_eq!(storage.ring_buffer.reader_count(address), 0);
        let (_, size) = storage.ring_buffer.chunk_header(address);
        assert_eq!(size, MD_SIZE + FLAG_BYTES + 100);
    }

    #[test]
    fn release_touches_drops_protection() {
        let mut storage = create_storage(4096, 1024);
        let (_, id) = storage.put("hash-a", &body(10)).unwrap();

        storage.touch("hash-a");
        assert!(storage.touched.contains(&id));
        storage.release_touches();
        assert!(storage.touched.is_empty());
    }

    #[test]
    fn get_signature_signs_over_the_key() {
        let mut storage = create_storage(4096, 1024);
        storage.put("hash-a", &body(10)).unwrap();

        let signature = storage.get_signature("hash-a").unwrap();
        let expected = storage.make_signature("hash-a", 0, 0);
        assert_eq!(signature, expected);
        // Not over another key, and not over another handle.
        assert_ne!(signature, storage.make_signature("hash-b", 0, 0));
        assert_ne!(signature, storage.make_signature("hash-a", 0, 1));
        assert_eq!(storage.get_signature("hash-b"), None);

        // Another segment has a different secret, hence a different signature.
        let mut other = create_storage(4096, 1024);
        other.put("hash-a", &body(10)).unwrap();
        assert_ne!(other.get_signature("hash-a").unwrap(), signature);
    }

    #[test]
    fn put_rejects_duplicate_key_without_touching_flags() {
        let mut storage = create_storage(4096, 1024);
        let (_, id) = storage.put("hash-a", &body(10)).unwrap();
        let flags_before = storage.writer_flag[&id];

        assert_eq!(
            storage.put("hash-a", &body(10)),
            Err(PutFailure::DuplicateKey)
        );
        assert_eq!(storage.writer_flag[&id], flags_before);
    }

    #[test]
    fn put_rejects_oversize_object() {
        let mut storage = create_storage(4096, 64);
        assert_eq!(storage.put("hash-a", &body(100)), Err(PutFailure::Oversize));
        assert!(!storage.is_cached("hash-a"));
    }

    #[test]
    fn put_evicts_fully_read_chunks_when_full() {
        // Two 108-byte bodies fit; the third needs eviction.
        let mut storage = create_storage(320, 256);
        let (address_a, id_a) = storage.put("hash-a", &body(108)).unwrap();
        storage.put("hash-b", &body(108)).unwrap();

        // Both chunks are unread: eviction is blocked and the put fails.
        assert_eq!(
            storage.put("hash-c", &body(108)),
            Err(PutFailure::BufferFull)
        );

        // hash-a is fully read (reader_count = writer_flag * n_readers = 1).
        storage.set_reader_count(address_a, 1);
        storage.put("hash-c", &body(108)).unwrap();
        assert!(!storage.is_cached("hash-a"));
        assert!(storage.is_cached("hash-b"));
        assert!(storage.is_cached("hash-c"));
        assert!(!storage.writer_flag.contains_key(&id_a));
        assert!(!storage.id_index.contains_key(&id_a));
    }

    #[test]
    fn touch_blocks_eviction_until_get_cached_or_release() {
        let mut storage = create_storage(320, 256);
        let (address_a, _) = storage.put("hash-a", &body(108)).unwrap();
        let (address_b, _) = storage.put("hash-b", &body(108)).unwrap();

        // Fully read, but touched: eviction is still blocked.
        storage.set_reader_count(address_a, 1);
        storage.touch("hash-a");
        assert_eq!(
            storage.put("hash-c", &body(108)),
            Err(PutFailure::BufferFull)
        );

        // Looking the item up releases the touch; with the writer-flag bump
        // matched by a second read, eviction proceeds.
        storage.get_cached("hash-a");
        storage.set_reader_count(address_a, 2);
        storage.put("hash-c", &body(108)).unwrap();
        assert!(!storage.is_cached("hash-a"));

        // Same for release_touches.
        storage.set_reader_count(address_b, 1);
        storage.touch("hash-b");
        assert_eq!(
            storage.put("hash-d", &body(108)),
            Err(PutFailure::BufferFull)
        );
        storage.release_touches();
        storage.put("hash-d", &body(108)).unwrap();
        assert!(!storage.is_cached("hash-b"));
    }

    #[test]
    fn eviction_respects_writer_flag_threshold() {
        let mut storage = create_storage(320, 256);
        let (address_a, _) = storage.put("hash-a", &body(108)).unwrap();
        storage.put("hash-b", &body(108)).unwrap();

        // One read is not enough: writer_flag is 2 after the hit.
        storage.get_cached("hash-a");
        storage.set_reader_count(address_a, 1);
        assert_eq!(
            storage.put("hash-c", &body(108)),
            Err(PutFailure::BufferFull)
        );

        // A second read releases the chunk.
        storage.set_reader_count(address_a, 2);
        storage.put("hash-c", &body(108)).unwrap();
        assert!(!storage.is_cached("hash-a"));
    }

    #[test]
    fn clear_empties_all_indices() {
        let mut storage = create_storage(4096, 1024);
        let (_, id) = storage.put("hash-a", &body(10)).unwrap();
        storage.touch("hash-a");
        storage.clear();

        assert!(!storage.is_cached("hash-a"));
        assert_eq!(storage.cached_len(), 0);
        assert!(storage.writer_flag.is_empty());
        assert!(storage.id_index.is_empty());
        assert!(!storage.touched.contains(&id));
    }
}
