// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Writer side of the POSIX shared-memory ring buffer.
//!
//! Rust port of `SingleWriterShmRingBuffer` (writer-only) from
//! `vllm/distributed/device_communicators/shm_object_storage.py`.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::os::unix::fs::OpenOptionsExt;

use memmap2::MmapMut;

use crate::error::{Error, Result, bail_shm_storage_create};

const ID_NBYTES: usize = 4;
const SIZE_NBYTES: usize = 4;
/// 4 bytes for id, 4 bytes for buffer size.
pub const MD_SIZE: usize = ID_NBYTES + SIZE_NBYTES;
/// Size of the handle-signing secret stored at the start of the segment.
pub const AUTH_SECRET_BYTES: usize = 32;
/// Physical offset of the ring buffer chunks inside the segment; logical
/// addresses are unchanged, so every chunk lives at `DATA_OFFSET + logical`.
pub const DATA_OFFSET: usize = AUTH_SECRET_BYTES;
/// Exclusive monotonic id bound, so `2**31 - 1` is the max value.
const ID_MAX: u32 = 1 << 31;

/// Marker for `MemoryError: "Not enough space in the data buffer"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("not enough space in the data buffer, try freeing up space first")]
pub struct NotEnoughSpace;

/// A single-writer, multiple-reader ring buffer implementation using shared
/// memory. This struct provides a thread-safe ring buffer where one process
/// can write data while multiple processes/threads can read from it.
///
/// Architecture:
/// - Uses shared memory for cross-process communication
/// - Maintains metadata for each allocated buffer chunk in the writer process
/// - Supports custom "is_free_fn" functions to determine when buffers can be
///   reused
/// - Each buffer chunk contains: `[4-byte id][4-byte size][actual_data]`
///
/// Key Concepts:
/// - monotonic_id_start/end: Track the range of active buffer IDs
/// - data_buffer_start/end: Track the physical memory range in use
/// - Automatic wraparound when reaching buffer end
/// - Lazy garbage collection based on is_free_fn checks
///
/// Thread Safety:
/// - Single writer: Only one process/thread should write (allocate_buf)
/// - Multiple readers: Multiple processes/threads can read (access_buf)
/// - Reader synchronization handled by is_free_fn callback
/// - Writer handles garbage collection (free_buf) based on reader feedback
///
/// Shared Memory Layout:
/// `[32-byte handle-signing secret][ring buffer chunks...]`
///
/// Memory Layout per Buffer Chunk:
/// `[4-byte monotonic_id][4-byte chunk_size][actual_data...]`
/// ^metadata_start                         ^data_start
///
/// The monotonic_id ensures data integrity - readers can verify they're
/// accessing the correct data even after buffer wraparound or reuse.
pub struct SingleWriterShmRingBuffer {
    /// POSIX shm object name without the leading slash, as reported to the
    /// Python engine via `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME`.
    name: String,
    data_buffer_size: usize,
    map: MmapMut,
    /// monotonic_id -> start address.
    metadata: HashMap<u32, usize>,
    monotonic_id_end: u32,
    monotonic_id_start: u32,
    data_buffer_start: usize,
    data_buffer_end: usize,
}

impl SingleWriterShmRingBuffer {
    /// Create a new shared memory buffer, failing if `name` already exists.
    ///
    /// Python's `shared_memory.SharedMemory(name=X, create=True)` maps to the
    /// file `/dev/shm/X` (tmpfs), so a plain exclusive-create file plus
    /// `memmap2` is layout-compatible with it.
    pub fn create(data_buffer_size: usize, name: &str) -> Result<Self> {
        if data_buffer_size == 0 {
            bail_shm_storage_create!(name = name.to_string(), "buffer size must be positive");
        }
        let path = format!("/dev/shm/{name}");

        let create_result = (|| {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)?;
            file.set_len((data_buffer_size + DATA_OFFSET) as u64)?;
            // SAFETY: `file` is a fresh shm file of at least the mapped size.
            unsafe { MmapMut::map_mut(&file) }
        })();
        let map = match create_result {
            Ok(map) => map,
            Err(error) => {
                // On failure after creation, unlink the segment we created.
                let _ = std::fs::remove_file(&path);
                bail_shm_storage_create!(name = name.to_string(), "{error}");
            }
        };

        let mut buffer = Self {
            name: name.to_string(),
            data_buffer_size,
            map,
            metadata: HashMap::new(),
            monotonic_id_end: 0,
            monotonic_id_start: 0,
            data_buffer_start: 0,
            data_buffer_end: 0,
        };
        buffer.init_auth_secret();
        tracing::debug!(name, data_buffer_size, "created new shared memory buffer");
        Ok(buffer)
    }

    /// Write the random handle-signing secret at the start of the segment.
    fn init_auth_secret(&mut self) {
        let secret: [u8; AUTH_SECRET_BYTES] = rand::random();
        self.map[..AUTH_SECRET_BYTES].copy_from_slice(&secret);
    }

    /// Return the current handle-signing secret.
    pub fn auth_secret(&self) -> [u8; AUTH_SECRET_BYTES] {
        self.map[..AUTH_SECRET_BYTES].try_into().expect("secret region is 32 bytes")
    }

    /// Clear the ring buffer.
    pub fn clear(&mut self) {
        self.metadata.clear();
        // Keep IDs increasing so handles issued before the clear (e.g. for
        // requests still draining) fail the ID check once their chunk is
        // reused, instead of resolving to a different object.
        self.monotonic_id_start = self.monotonic_id_end;
        self.data_buffer_start = 0;
        self.data_buffer_end = 0;
    }

    /// Allocate a buffer of `MD_SIZE` + `size` bytes in the shared memory.
    /// Memory layout:
    /// `[4-byte monotonic_id][4-byte size][buffer data...]`
    ///
    /// Returns the logical start address and the assigned monotonic id.
    pub fn allocate_buf(
        &mut self,
        size: usize,
    ) -> std::result::Result<(usize, u32), NotEnoughSpace> {
        assert!(size > 0, "size must be greater than 0");
        let size = size + MD_SIZE; // add metadata size to the buffer size
        // reset to beginning if the buffer does not have enough contiguous space
        let buffer_end_reset = self.data_buffer_end % self.data_buffer_size;
        let buffer_end_reset = if buffer_end_reset + size > self.data_buffer_size {
            (self.data_buffer_end / self.data_buffer_size + 1) * self.data_buffer_size
        } else {
            // no reset needed
            self.data_buffer_end
        };

        // check if we have enough space in the data buffer
        // i.e. if the new end (self.data_buffer_end + size)
        // exceeds the start of the data buffer
        let occupied_size_new = buffer_end_reset + size - self.data_buffer_start;
        if occupied_size_new > self.data_buffer_size {
            return Err(NotEnoughSpace);
        }
        self.data_buffer_end = buffer_end_reset;

        // first 4 bytes as the monotonic id, next 4 bytes as the size
        let offset = self.chunk_offset(self.data_buffer_end);
        self.write_bytes(offset, &self.monotonic_id_end.to_le_bytes());
        self.write_bytes(offset + ID_NBYTES, &(size as u32).to_le_bytes());

        // record metadata
        self.metadata.insert(self.monotonic_id_end, self.data_buffer_end);
        // update buffer and monotonic id indices
        let current_buffer_end = self.data_buffer_end;
        let current_id_end = self.monotonic_id_end;
        self.data_buffer_end += size;
        self.monotonic_id_end = (self.monotonic_id_end + 1) % ID_MAX;
        Ok((current_buffer_end, current_id_end))
    }

    /// Read the `[4-byte monotonic_id][4-byte size]` chunk header at the
    /// logical `address`.
    pub fn chunk_header(&self, address: usize) -> (u32, usize) {
        let offset = self.chunk_offset(address);
        let id = self.read_i32(offset) as u32;
        let size = self.read_i32(offset + ID_NBYTES) as usize;
        (id, size)
    }

    /// Read the reader reference count stored in the first 4 bytes of the
    /// chunk payload at the logical `address`.
    ///
    /// Readers in other processes update this field concurrently, so the read
    /// is done byte-wise with volatile semantics.
    pub fn reader_count(&self, address: usize) -> i32 {
        let offset = self.chunk_offset(address) + MD_SIZE;
        let bytes = &self.map[offset..offset + 4];
        let mut out = [0u8; 4];
        for (i, byte) in out.iter_mut().enumerate() {
            // SAFETY: `offset + i` stays within the mapped segment; volatile
            // because readers in other processes mutate these bytes.
            *byte = unsafe { std::ptr::read_volatile(bytes.as_ptr().add(i)) };
        }
        i32::from_le_bytes(out)
    }

    /// Write `data` at `payload_offset` bytes into the payload of the chunk at
    /// the logical `address`.
    pub fn write_payload(&mut self, address: usize, payload_offset: usize, data: &[u8]) {
        let result = self.write_payload_with(address, payload_offset, data.len(), |buf| {
            buf.copy_from_slice(data);
            Ok::<(), std::convert::Infallible>(())
        });
        let Ok(()) = result;
    }

    /// Write `len` payload bytes at `payload_offset` through `write`, which
    /// receives the mutable chunk slice, instead of copying a pre-built
    /// buffer.
    pub fn write_payload_with<E>(
        &mut self,
        address: usize,
        payload_offset: usize,
        len: usize,
        write: impl FnOnce(&mut [u8]) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        let offset = self.chunk_offset(address) + MD_SIZE + payload_offset;
        assert!(
            offset + len <= self.segment_size(),
            "payload write out of bounds"
        );
        write(&mut self.map[offset..offset + len])
    }

    /// Free buffers in FIFO order while `is_free_fn` accepts them, reclaiming
    /// at most `nbytes` worth of chunks.
    ///
    /// This is a no-op in shared memory, but we need to keep track of the
    /// metadata.
    ///
    /// If freed memory spreads across the end and start of the ring buffer,
    /// the actual freed memory will be in two segments. In this case there
    /// still might not be a contiguous space of `nbytes` available.
    ///
    /// Unlike the Python original, which hands the whole payload view to
    /// `is_free_fn`, the callback receives the monotonic id and the chunk's
    /// reader reference count, which is all the writer's free check needs.
    ///
    /// Returns the monotonic ids of the freed chunks.
    pub fn free_buf(
        &mut self,
        is_free_fn: impl Fn(u32, i32) -> bool,
        nbytes: Option<usize>,
    ) -> Vec<u32> {
        // if nbytes is None, free up the maximum size of the ring buffer
        let nbytes = nbytes.unwrap_or(self.data_buffer_size);
        let mut freed_bytes = 0;
        let mut freed_ids = Vec::new();
        while let Some(&address) = self.metadata.get(&self.monotonic_id_start) {
            if freed_bytes >= nbytes {
                break;
            }
            let (_, size) = self.chunk_header(address);
            if !is_free_fn(self.monotonic_id_start, self.reader_count(address)) {
                // there are still readers, we cannot free the buffer
                break;
            }
            // check passed, we can free the buffer
            self.metadata.remove(&self.monotonic_id_start);
            self.monotonic_id_start = (self.monotonic_id_start + 1) % ID_MAX;
            if let Some(&next_address) = self.metadata.get(&self.monotonic_id_start) {
                // pointing to the start addr of next allocation
                self.data_buffer_start += (next_address + self.data_buffer_size
                    - self.data_buffer_start)
                    % self.data_buffer_size;
            } else {
                // no remaining allocation, reset to zero
                self.data_buffer_start = 0;
                self.data_buffer_end = 0;
            }
            freed_bytes += size;
            freed_ids.push(self.monotonic_id_start.wrapping_sub(1) % ID_MAX);
        }

        tracing::debug!(
            freed_bytes,
            monotonic_id_start = self.monotonic_id_start,
            monotonic_id_end = self.monotonic_id_end,
            "freed space in the ring buffer",
        );

        // buffer wrap around
        if self.data_buffer_start >= self.data_buffer_size {
            self.data_buffer_start -= self.data_buffer_size;
            self.data_buffer_end -= self.data_buffer_size;
        }

        freed_ids
    }

    fn chunk_offset(&self, address: usize) -> usize {
        DATA_OFFSET + address % self.data_buffer_size
    }

    /// Total mapped size: the 32-byte secret region plus the data buffer.
    fn segment_size(&self) -> usize {
        DATA_OFFSET + self.data_buffer_size
    }

    fn write_bytes(&mut self, offset: usize, data: &[u8]) {
        debug_assert!(offset + data.len() <= self.segment_size());
        self.map[offset..offset + data.len()].copy_from_slice(data);
    }

    fn read_i32(&self, offset: usize) -> i32 {
        i32::from_le_bytes(self.map[offset..offset + 4].try_into().expect("4-byte read"))
    }
}

impl Drop for SingleWriterShmRingBuffer {
    fn drop(&mut self) {
        // `MmapMut` unmaps on drop; the writer also unlinks the segment.
        let _ = std::fs::remove_file(format!("/dev/shm/{}", self.name));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_name() -> String {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("vllm-mm-shm-test-ring-{}-{id}", std::process::id())
    }

    fn create_buffer(size: usize) -> SingleWriterShmRingBuffer {
        SingleWriterShmRingBuffer::create(size, &unique_name()).expect("create ring buffer")
    }

    #[test]
    fn create_fails_when_name_exists() {
        let name = unique_name();
        let _buffer = SingleWriterShmRingBuffer::create(1024, &name).unwrap();
        let result = SingleWriterShmRingBuffer::create(1024, &name);
        let error = match result {
            Ok(_) => panic!("creating an existing segment must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("File exists"), "{error}");
    }

    #[test]
    fn linear_allocation_records_metadata_and_header() {
        // Scenario 1: Simple Linear Allocation
        let mut buffer = create_buffer(100);

        let (address, id) = buffer.allocate_buf(20).unwrap();
        assert_eq!((address, id), (0, 0));
        assert_eq!(buffer.chunk_header(address), (0, 28));

        let (address, id) = buffer.allocate_buf(30).unwrap();
        assert_eq!((address, id), (28, 1));
        assert_eq!(buffer.chunk_header(address), (1, 38));
    }

    #[test]
    fn free_buf_advances_start_and_resets_when_empty() {
        // Scenario 2: Memory Reclamation
        let mut buffer = create_buffer(100);
        buffer.allocate_buf(20).unwrap();
        buffer.allocate_buf(30).unwrap();

        // Free only the first chunk; freeing stops at the first chunk that
        // still has readers.
        let freed = buffer.free_buf(|id, _| id == 0, None);
        assert_eq!(freed, vec![0]);
        assert_eq!(buffer.monotonic_id_start, 1);
        assert_eq!(buffer.data_buffer_start, 28);

        let freed = buffer.free_buf(|_, _| true, None);
        assert_eq!(freed, vec![1]);
        assert_eq!(buffer.data_buffer_start, 0);
        assert_eq!(buffer.data_buffer_end, 0);
    }

    #[test]
    fn allocation_wraps_around_after_reclamation() {
        // Scenario 3: Wraparound Allocation
        let mut buffer = create_buffer(100);
        buffer.allocate_buf(20).unwrap();
        buffer.allocate_buf(30).unwrap();
        buffer.free_buf(|_, _| true, None);

        // 48 bytes do not fit after the freed tail, so allocation wraps.
        let (address, id) = buffer.allocate_buf(40).unwrap();
        assert_eq!((address, id), (0, 2));
        assert_eq!(buffer.chunk_header(address), (2, 48));
    }

    #[test]
    fn allocation_fails_when_buffer_is_full() {
        // Scenario 4: Error Handling - Out of Space
        let mut buffer = create_buffer(100);
        buffer.allocate_buf(20).unwrap();
        buffer.allocate_buf(30).unwrap();
        buffer.free_buf(|id, _| id == 0, None);

        // occupied_size_new = end + size - start = 66 + 28 - 28 = 66 fits.
        let (address, _) = buffer.allocate_buf(20).unwrap();
        assert_eq!(address, 66);
        // Next chunk wraps to offset 100 but 100 + 28 - 28 = 100 fits exactly.
        let (address, _) = buffer.allocate_buf(20).unwrap();
        assert_eq!(address, 100);
        // No space left at all.
        assert_eq!(buffer.allocate_buf(1), Err(NotEnoughSpace));
    }

    #[test]
    fn free_buf_respects_nbytes_budget() {
        let mut buffer = create_buffer(256);
        buffer.allocate_buf(20).unwrap(); // chunk size 28
        buffer.allocate_buf(20).unwrap();
        buffer.allocate_buf(20).unwrap();

        // Budget below one chunk still frees the first chunk (Python frees
        // while freed_bytes < nbytes, checking before freeing).
        let freed = buffer.free_buf(|_, _| true, Some(10));
        assert_eq!(freed, vec![0]);
    }

    #[test]
    fn free_buf_skips_live_chunks_in_fifo_order() {
        let mut buffer = create_buffer(256);
        buffer.allocate_buf(20).unwrap();
        buffer.allocate_buf(20).unwrap();
        buffer.allocate_buf(20).unwrap();

        // Chunk 0 is still referenced: nothing is freed even though chunk 1
        // would be free.
        let freed = buffer.free_buf(|id, _| id == 1, None);
        assert!(freed.is_empty());
        assert_eq!(buffer.monotonic_id_start, 0);
    }

    #[test]
    fn reader_count_is_read_from_payload_start() {
        let mut buffer = create_buffer(256);
        let (address, _) = buffer.allocate_buf(20).unwrap();
        buffer.write_payload(address, 0, &7_i32.to_le_bytes());
        assert_eq!(buffer.reader_count(address), 7);
    }

    #[test]
    fn clear_keeps_ids_increasing() {
        let mut buffer = create_buffer(100);
        buffer.allocate_buf(20).unwrap();
        buffer.clear();

        // Addresses restart at zero, but IDs keep increasing so handles
        // issued before the clear fail the ID check once their chunk is
        // reused, instead of resolving to a different object.
        let (address, id) = buffer.allocate_buf(20).unwrap();
        assert_eq!((address, id), (0, 1));
    }

    #[test]
    fn segment_starts_with_a_random_auth_secret() {
        let buffer = create_buffer(100);
        let secret = buffer.auth_secret();
        assert_ne!(secret, [0u8; AUTH_SECRET_BYTES]);
        assert_ne!(secret, create_buffer(100).auth_secret());
    }

    #[test]
    fn chunk_writes_never_touch_the_secret_region() {
        let mut buffer = create_buffer(100);
        let secret = buffer.auth_secret();

        let (address, _) = buffer.allocate_buf(20).unwrap();
        buffer.write_payload(address, 0, &[0xFF; 20]);
        assert_eq!(buffer.auth_secret(), secret);
    }
}
