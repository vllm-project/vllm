# Paged Shared Memory Storage

## Overview

Shared memory allows multiple processes to access the same physical memory region, enabling zero-copy data exchange without serialization. In vLLM V1, the multi‑process architecture (API Server, Engine Core, GPU Workers) benefits greatly from this mechanism for passing large multimodal tensors.

**Why paged shared memory?**

- Multimodal data (images, video frames) vary widely in size; per‑request allocation/deallocation causes fragmentation and overhead.
- Using system‑level `shm_open`/`mmap` and `pin_memory`/`unpin_memory` repeatedly adds latency.
- **Paged SHM** pre‑allocates a large, fixed pool divided into pages (default: 1MB, chosen to saturate H2D bandwidth) and manages blocks via a server, avoiding fragmentation and reducing per‑operation overhead.

## Architecture

### Processes and Data Paths

- **API Server**: Handles HTTP requests, tokenization, multimodal input loading. It writes multimodal tensors into SHM and sends lightweight metadata (UUID, block list, shape, dtype) to the Engine Core over ZMQ.
- **Engine Core**: Schedules requests and dispatches work to GPU Workers.
- **GPU Workers**: Each GPU process executes model forward passes. It reads tensor data from SHM (to GPU or CPU) and releases the SHM blocks when done.

Data flows:

```text
ZMQ Path (small metadata only):
[API Server] --ZMQ IPC--> [GPU Worker]  (metadata, no tensor data)

Shared Memory Path (tensor data):
[API Server] --SHM write--> [PagedShmServer] --SHM read--> [GPU Worker] --H2D--> [GPU]
```

The actual tensor bytes are **offloaded** from the ZMQ hot path, eliminating serialization and large‑message overhead.

### PagedShmServer & Client

- **PagedShmServer**: A standalone process that owns the SHM pool. It handles block allocation, reference counting, LRU eviction, and client requests via ZMQ RPCs.
- **PagedShmClient**: A lightweight client that connects to the server. It provides `open_write`/`close_write` (write lock) and `open_read`/`close_read` (read lock) primitives, plus high‑level helpers for bytes/NumPy/PyTorch.

**Typical Write Flow (API Server):**

1. Call `open_write(items)` to atomically allocate blocks for a batch of tensors.
2. Copy tensor data into the allocated SHM blocks (e.g., via `storage.write()`).
3. Call `close_write(uuid)` to finalize the write, making the item readable. If `generate_read_token=True`, a read token is created and a read reference is automatically held.

**Typical Read Flow (GPU Worker):**

1. Call `open_read(uuid_or_token)` to acquire a read lock and obtain block information. If the item is still being written, the call may wait (with timeout).
2. Read the data from SHM into CPU or GPU memory.
3. **Note**: The read lock is **not** released immediately after reading. Instead, the release of the read token (via `close_read`) is deferred to the `PagedShmTensorTracker`, which is invoked when the request is freed (e.g., after generation completes). This design avoids synchronizing the GPU stream, since the H2D transfer may still be in flight when `read()` returns. The tracker ensures the token is destroyed only after all workers have finished using the data.
