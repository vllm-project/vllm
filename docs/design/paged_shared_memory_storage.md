# Paged Shared Memory Storage

## Overview

Shared memory allows multiple processes to access the same physical memory region, enabling zero-copy data exchange 
without serialization. In vLLM V1, the multi‑process architecture (API Server, Engine Core, GPU Workers) benefits 
greatly from this mechanism for passing large multimodal tensors.

**Why paged shared memory?**
- Multimodal data (images, video frames) vary widely in size; per‑request allocation/deallocation causes fragmentation 
- and overhead.
- Using system‑level `shm_open`/`mmap` and `pin_memory`/`unpin_memory` repeatedly adds latency.
- **Paged SHM** pre‑allocates a large, fixed pool divided into pages (default: 1MB, chosen to saturate H2D bandwidth) 
- and manages blocks via a server, avoiding fragmentation and reducing per‑operation overhead.

## Architecture

### Processes and Data Paths

- **API Server**: Handles HTTP requests, tokenization, multimodal input loading. It writes multimodal tensors into SHM 
- and sends lightweight metadata (UUID, block list, shape, dtype) to the Engine Core over ZMQ.
- **Engine Core**: Schedules requests and dispatches work to GPU Workers.
- **GPU Workers**: Each GPU process executes model forward passes. It reads tensor data from SHM (to GPU or CPU) and 
- releases the SHM blocks when done.

Data flows:

```
ZMQ Path (small metadata only):
[API Server] --ZMQ IPC--> [GPU Worker]  (metadata, no tensor data)

Shared Memory Path (tensor data):
[API Server] --SHM write--> [PagedShmServer] --SHM read--> [GPU Worker] --H2D--> [GPU]
```

The actual tensor bytes are **offloaded** from the ZMQ hot path, eliminating serialization and large‑message overhead.

### PagedShmServer & Client

- **PagedShmServer**: A standalone process that owns the SHM pool. It handles block allocation, reference counting, LRU 
- eviction, and client requests via ZMQ RPCs.
- **PagedShmClient**: A lightweight client that connects to the server. It provides `open_write`/`close_write` 
- (write lock) and `open_read`/`close_read` (read lock) primitives, plus high‑level helpers for bytes/NumPy/PyTorch.

**Typical Write Flow (API Server):**
1. Call `open_write(items)` to atomically allocate blocks for a batch of tensors.
2. Copy tensor data into the allocated SHM blocks (e.g., via `storage.write()`).
3. Call `close_write(uuid)` to finalize the write, making the item readable. If `generate_read_token=True`, a read token 
4. is created and a read reference is automatically held.

**Typical Read Flow (GPU Worker):**
1. Call `open_read(uuid_or_token)` to acquire a read lock and obtain block information. If the item is still being written, the call may wait (with timeout).
2. Read the data from SHM into CPU or GPU memory.
3. Call `close_read(uuid_or_token)` to release the lock and, if a token is used, destroy it. The server will then cache the item (if cacheable) or free the blocks.

## Integration with Multimodal Tensor IPC

The `PagedShmTensorIPC` class wraps the client to automatically handle large tensors in `mm_inputs` during request processing.

- **`write()`**: Scans `mm_inputs` for tensors larger than `block_size`, batches `open_write` requests, submits asynchronous copies to SHM, and replaces the original tensor with a `PagedShmTensor` metadata object (containing the read token, shape, dtype, and block list). The metadata is small enough to be sent via ZMQ.
- **`read()`**: On the worker side, extracts `PagedShmTensor` from the received metadata, waits for the write to complete (with timeout), reads the tensor data from SHM to the target device, and reconstructs the full tensor. By default, it also destroys the read token (`auto_release=True`). For TP, set `auto_release=False` and call `release_token()` once after all ranks have finished reading.

This design ensures that only the metadata traverses the ZMQ path, while the heavy tensor data stays in SHM, drastically reducing latency and CPU overhead under load.

## Thread Safety & Performance

- The server uses a ZMQ ROUTER socket and handles concurrent requests; clients use a pool of REQ sockets for thread‑safe access.
- Asynchronous writes overlap SHM copy with ZMQ round‑trip, further hiding latency.
- LRU eviction on the server keeps memory usage bounded; frequently accessed items remain cached.

## Configuration

Enabled via `multimodal_config.enable_paged_shm = True`. Key parameters:
- `paged_shm_size`: Total size of the SHM pool (in bytes).
- `paged_shm_block_size`: Page size (default 1MB). Must be large enough to amortize H2D transfer overhead.

The server is started automatically when the `ModelConfig` indicates SHM is enabled; clients connect to the published IPC address.
