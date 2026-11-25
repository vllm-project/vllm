// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Writer-side key-value object storage on top of the shm ring buffer.
//!
//! Rust port of the writer half of `SingleWriterShmObjectStorage` from
//! `vllm/distributed/device_communicators/shm_object_storage.py`. The reader
//! half lives in the Python engine's worker processes.

use std::collections::HashMap;
use std::num::NonZeroUsize;

use hmac::{Hmac, Mac as _};
use sha2::{Digest as _, Sha256};

use super::ring_buffer::{MD_SIZE, SingleWriterShmRingBuffer};
use crate::error::{Error, Result};

/// HMAC-SHA256 handle signature, keyed by the shm segment's auth secret.
type HmacSha256 = Hmac<Sha256>;

/// Number of payload bytes reserved for the reader reference-count flag.
const FLAG_BYTES: usize = 4;

/// Why a `put` could not store the object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PutFailure {
    /// The key already exists in the storage.
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

/// A shm handle and its writer-local metadata, evicted together.
struct CachedEntry<Metadata> {
    address: usize,
    monotonic_id: u32,
    metadata: Metadata,
}

/// Writer-side key-value store of serialized objects in the shm ring buffer.
///
/// Each object is one chunk whose payload is `[4-byte reader_count][bytes]`.
/// Readers increment `reader_count` after each read. The writer counts the
/// handles it hands out per chunk in `writer_flag`, and a chunk is reclaimable
/// once `reader_count >= writer_flag * n_readers`, i.e. every reader has read
/// every handout. Reclamation is FIFO and only happens when a `put` runs out
/// of space, so one unread chunk at the head blocks all later ones.
///
/// Handles are `(address, monotonic_id)` signed with an HMAC over
/// `(key, address, monotonic_id)`, keyed by the secret at the start of the
/// segment; readers verify the signature before reading.
pub struct SingleWriterShmObjectStorage<Metadata> {
    ring_buffer: SingleWriterShmRingBuffer,
    max_object_size: usize,
    n_readers: u32,
    /// Key-value mapping: key -> cached entry
    key_index: HashMap<String, CachedEntry<Metadata>>,
    /// Reverse mapping: monotonic_id -> key
    id_index: HashMap<u32, String>,
    /// Writer flag to track in-use status: monotonic_id -> count
    writer_flag: HashMap<u32, u64>,
    /// Outstanding `touch` count per monotonic id; a touched item is not
    /// freed until every touch is released with `untouch`.
    touched: HashMap<u32, usize>,
}

impl<Metadata> SingleWriterShmObjectStorage<Metadata> {
    pub fn new(ring_buffer: SingleWriterShmRingBuffer, max_object_size: usize) -> Result<Self> {
        // Readers decode the chunk size header as a signed 32-bit integer.
        let limit = i32::MAX as usize - MD_SIZE;
        if max_object_size > limit {
            return Err(Error::MaxObjectSizeTooLarge {
                max_object_size,
                limit,
            });
        }
        Ok(Self {
            ring_buffer,
            max_object_size,
            // Learned from the engine handshake; callers gate all operations
            // until `set_n_readers` is called because the free threshold
            // cannot be computed before that.
            n_readers: 0,
            key_index: HashMap::new(),
            id_index: HashMap::new(),
            writer_flag: HashMap::new(),
            touched: HashMap::new(),
        })
    }

    /// See [`SingleWriterShmRingBuffer::unlink_name`].
    pub fn unlink_name(&mut self) {
        self.ring_buffer.unlink_name();
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

    /// Number of cached keys.
    pub fn cached_len(&self) -> usize {
        self.key_index.len()
    }

    /// Get the cached handle and metadata together, bumping its writer flag.
    pub fn get_cached(&mut self, key: &str) -> Option<(usize, u32, Metadata)>
    where
        Metadata: Clone,
    {
        let entry = self.key_index.get(key)?;
        let hit = (entry.address, entry.monotonic_id, entry.metadata.clone());
        self.increment_writer_flag(entry.monotonic_id);
        Some(hit)
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

    /// Protect a cached item from eviction until a matching `untouch`.
    ///
    /// Touches nest, so concurrent requests can each protect the same item.
    /// Returns the item's monotonic id to pass to `untouch`, or `None` if
    /// `key` is not cached.
    pub fn touch(&mut self, key: &str) -> Option<u32> {
        let id = self.key_index.get(key)?.monotonic_id;
        *self.touched.entry(id).or_insert(0) += 1;
        Some(id)
    }

    /// Release one `touch` of the item with monotonic id `id`.
    ///
    /// A no-op if the item is not touched, e.g. after `clear`.
    pub fn untouch(&mut self, id: u32) {
        if let Some(count) = self.touched.get_mut(&id) {
            *count -= 1;
            if *count == 0 {
                self.touched.remove(&id);
            }
        }
    }

    /// Store a key-value pair in the object storage.
    ///
    /// Attempts to free `2 * max_object_size` bytes using FIFO order when the
    /// ring buffer runs out of space during a put() operation.
    pub fn put(
        &mut self,
        key: &str,
        body: &[u8],
        metadata: Metadata,
    ) -> std::result::Result<(usize, u32), PutFailure> {
        match self.put_with::<std::convert::Infallible>(
            key,
            body.len(),
            metadata,
            NonZeroUsize::MIN,
            |buf| {
                buf.copy_from_slice(body);
                Ok(())
            },
        ) {
            Ok(result) => result,
        }
    }

    /// Store a key-value pair, serializing the payload into the chunk in
    /// place through `write` instead of copying a pre-built buffer. Each
    /// handout accounts for one handle issued for the resulting chunk.
    ///
    /// On a `write` error the chunk is left unrecorded — it has no
    /// writer-flag entry, so the next eviction pass reclaims it immediately.
    pub fn put_with<E>(
        &mut self,
        key: &str,
        payload_len: usize,
        metadata: Metadata,
        handouts: NonZeroUsize,
        write: impl FnOnce(&mut [u8]) -> std::result::Result<(), E>,
    ) -> std::result::Result<std::result::Result<(usize, u32), PutFailure>, E> {
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
        self.writer_flag.insert(monotonic_id, handouts.get() as u64);

        // Update key index
        self.key_index.insert(
            key.to_string(),
            CachedEntry {
                address,
                monotonic_id,
                metadata,
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
                // Touched items stay until every touch is released.
                if touched.contains_key(&id) {
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

    fn create_storage<Metadata>(
        data_buffer_size: usize,
        max_object_size: usize,
    ) -> SingleWriterShmObjectStorage<Metadata> {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = format!("vllm-mm-shm-test-storage-{}-{id}", std::process::id());
        let ring_buffer = SingleWriterShmRingBuffer::create(data_buffer_size, &name).unwrap();
        let mut storage = SingleWriterShmObjectStorage::new(ring_buffer, max_object_size).unwrap();
        storage.set_n_readers(1);
        storage
    }

    fn body(len: usize) -> Vec<u8> {
        vec![0xAB; len]
    }

    #[test]
    fn put_get_cached_and_touch_track_protection() {
        let mut storage = create_storage(4096, 1024);

        let (address, id) = storage.put("hash-a", &body(100), ()).unwrap();
        assert!(storage.is_cached("hash-a"));
        assert!(!storage.is_cached("hash-b"));
        assert_eq!(storage.writer_flag[&id], 1);

        // Hit path bumps the writer flag and returns the same address.
        assert_eq!(storage.get_cached("hash-a"), Some((address, id, ())));
        assert_eq!(storage.writer_flag[&id], 2);
        assert_eq!(storage.get_cached("hash-b"), None);

        // Touch does not bump the writer flag, and ignores unknown keys.
        assert_eq!(storage.touch("hash-a"), Some(id));
        assert_eq!(storage.touch("hash-b"), None);
        assert_eq!(storage.writer_flag[&id], 2);

        // The payload starts with a zeroed reader-count flag.
        assert_eq!(storage.ring_buffer.reader_count(address), 0);
        let (_, size) = storage.ring_buffer.chunk_header(address);
        assert_eq!(size, MD_SIZE + FLAG_BYTES + 100);
    }

    #[test]
    fn get_signature_signs_over_the_key() {
        let mut storage = create_storage(4096, 1024);
        storage.put("hash-a", &body(10), ()).unwrap();

        let signature = storage.get_signature("hash-a").unwrap();
        let expected = storage.make_signature("hash-a", 0, 0);
        assert_eq!(signature, expected);
        // Not over another key, and not over another handle.
        assert_ne!(signature, storage.make_signature("hash-b", 0, 0));
        assert_ne!(signature, storage.make_signature("hash-a", 0, 1));
        assert_eq!(storage.get_signature("hash-b"), None);

        // Another segment has a different secret, hence a different signature.
        let mut other = create_storage(4096, 1024);
        other.put("hash-a", &body(10), ()).unwrap();
        assert_ne!(other.get_signature("hash-a").unwrap(), signature);
    }

    #[test]
    fn put_rejects_duplicate_key_without_touching_flags() {
        let mut storage = create_storage(4096, 1024);
        let (_, id) = storage.put("hash-a", &body(10), ()).unwrap();
        let flags_before = storage.writer_flag[&id];

        assert_eq!(
            storage.put("hash-a", &body(10), ()),
            Err(PutFailure::DuplicateKey)
        );
        assert_eq!(storage.writer_flag[&id], flags_before);
    }

    #[test]
    fn new_rejects_max_object_size_beyond_the_chunk_size_header() {
        let name = format!("vllm-mm-shm-test-storage-max-{}", std::process::id());
        let ring_buffer = SingleWriterShmRingBuffer::create(4096, &name).unwrap();
        let result = SingleWriterShmObjectStorage::<()>::new(ring_buffer, i32::MAX as usize);
        assert!(matches!(result, Err(Error::MaxObjectSizeTooLarge { .. })));
    }

    #[test]
    fn put_rejects_oversize_object() {
        let mut storage = create_storage(4096, 64);
        assert_eq!(
            storage.put("hash-a", &body(100), ()),
            Err(PutFailure::Oversize)
        );
        assert!(!storage.is_cached("hash-a"));
    }

    #[test]
    fn put_evicts_fully_read_chunks_when_full() {
        use std::sync::Arc;

        // Two 108-byte bodies fit; the third needs eviction.
        let mut storage = create_storage(320, 256);
        let metadata = Arc::new(());
        let cached_metadata = Arc::downgrade(&metadata);
        let (address_a, id_a) = storage.put("hash-a", &body(108), metadata).unwrap();
        storage.put("hash-b", &body(108), Arc::new(())).unwrap();

        // Both chunks are unread: eviction is blocked and the put fails.
        assert_eq!(
            storage.put("hash-c", &body(108), Arc::new(())),
            Err(PutFailure::BufferFull)
        );

        // hash-a is fully read (reader_count = writer_flag * n_readers = 1).
        storage.set_reader_count(address_a, 1);
        storage.put("hash-c", &body(108), Arc::new(())).unwrap();
        assert!(cached_metadata.upgrade().is_none());
        assert!(!storage.is_cached("hash-a"));
        assert!(storage.is_cached("hash-b"));
        assert!(storage.is_cached("hash-c"));
        assert!(!storage.writer_flag.contains_key(&id_a));
        assert!(!storage.id_index.contains_key(&id_a));
    }

    #[test]
    fn touch_blocks_eviction_until_every_touch_is_released() {
        let mut storage = create_storage(320, 256);
        let (address_a, id_a) = storage.put("hash-a", &body(108), ()).unwrap();
        storage.put("hash-b", &body(108), ()).unwrap();

        // Two concurrent requests touch hash-a, and one looks it up. The
        // lookup is read as well, so only the touches block eviction.
        storage.touch("hash-a");
        storage.touch("hash-a");
        storage.get_cached("hash-a");
        storage.set_reader_count(address_a, 2);
        assert_eq!(
            storage.put("hash-c", &body(108), ()),
            Err(PutFailure::BufferFull)
        );

        // One request finishing keeps the other's protection.
        storage.untouch(id_a);
        assert_eq!(
            storage.put("hash-c", &body(108), ()),
            Err(PutFailure::BufferFull)
        );

        storage.untouch(id_a);
        storage.put("hash-c", &body(108), ()).unwrap();
        assert!(!storage.is_cached("hash-a"));
    }

    #[test]
    fn eviction_respects_writer_flag_threshold() {
        let mut storage = create_storage(320, 256);
        storage.set_n_readers(2);
        let (address_a, _) = storage
            .put_with("hash-a", 108, (), NonZeroUsize::new(3).unwrap(), |buf| {
                buf.fill(0xAB);
                Ok::<_, std::convert::Infallible>(())
            })
            .unwrap()
            .unwrap();
        storage.put("hash-b", &body(108), ()).unwrap();

        // Three initial handouts plus one later hit require eight worker reads.
        storage.get_cached("hash-a");
        storage.set_reader_count(address_a, 7);
        assert_eq!(
            storage.put("hash-c", &body(108), ()),
            Err(PutFailure::BufferFull)
        );

        storage.set_reader_count(address_a, 8);
        storage.put("hash-c", &body(108), ()).unwrap();
        assert!(!storage.is_cached("hash-a"));
    }

    #[test]
    fn clear_empties_all_indices() {
        let mut storage = create_storage(4096, 1024);
        let (_, id) = storage.put("hash-a", &body(10), ()).unwrap();
        storage.touch("hash-a");
        storage.clear();

        assert!(!storage.is_cached("hash-a"));
        assert_eq!(storage.cached_len(), 0);
        assert!(storage.writer_flag.is_empty());
        assert!(storage.id_index.is_empty());
        assert!(storage.touched.is_empty());
        // Releasing a touch taken before the clear is a no-op.
        storage.untouch(id);
    }
}
