# Paged Shared Memory Storage

## Overview

Shared memory enables multiple processes to access the same physical memory region, allowing zero‑copy data exchange without serialization overhead. In vLLM V1, the multi‑process architecture—comprising the API Server, Engine Core, and GPU Workers—benefits significantly from this mechanism, especially for passing large multi‑modal tensors (images, video frames, audio).

**Why paged shared memory?**

- Multi‑modal inputs vary widely in size; per‑request allocation and deallocation cause fragmentation and high latency.
- Using system‑level `shm_open`/`mmap` and `pin_memory`/`unpin_memory` on each request adds unnecessary overhead.
- **Paged SHM** pre‑allocates a large, fixed pool divided into pages (default: 1MB, chosen to saturate H2D bandwidth) and manages blocks via a central server. This eliminates fragmentation and reduces per‑operation cost, while also enabling efficient caching and reference counting.

## Architecture

### Processes and Data Paths

- **API Server**  
  Handles HTTP requests, tokenization, and multi‑modal pre‑processing (aka the `Renderer`). After the MultiModal Processor obtains the multi‑modal item, the API Server serializes it into **lightweight metadata** (i.e., prompt updates, tensor dtype and shape) and **raw tensor data**, then writes the raw tensors to SHM via the sender cache, returning a UUID. The UUID, along with other minimal request metadata, is sent to Engine Core over ZMQ.

- **Engine Core**  
  Schedules requests and dispatches work to GPU Workers. It forwards the UUID (and other metadata) to the workers without touching the tensor bytes.

- **GPU Workers**  
  Each GPU process executes model forward passes. It reads the raw tensors from SHM and deserializes the metadata via the receiver cache to reconstruct the full `MultiModalProcessorCacheOutItem` with negligible serialization/deserialization overhead. The read lock is released immediately after the data is copied.

Data flows:

```text
ZMQ IPC Path (without SHM):
[Multi-modal Tensor] ─ZMQ IPC→ [CPU Buffer] ─(pin & H2D)→ [GPU]
    (API Server)               (GPU Worker)

Shared Memory Path (with SHM):
[Multi-modal Tensor] ─Shared Memory→ [read via receiver cache] ─→ [GPU]
    (API Server)                           (GPU Worker)         (H2D)
```

The tensor bytes are **offloaded** from the ZMQ hot path, eliminating serialization and large‑message overhead. Only the UUID and minimal request metadata travel over ZMQ.

### PagedShmServer & Client

- **PagedShmServer**  
  A standalone process that owns the SHM pool. It handles block allocation, reference counting, LRU eviction, and client requests via ZMQ RPCs. The server is the single source of truth for block ownership.

- **PagedShmClient**  
  A lightweight client that connects to the server. It provides:
    - `open_write` / `close_write` (write lock)
    - `open_read` / `close_read` (read lock)
    - High‑level helpers for writing/reading `bytes`, `numpy.ndarray`, and `torch.Tensor`.

**Typical Write Flow (API Server):**

1. Call `open_write(items)` to atomically allocate blocks for a batch of tensors.
2. Copy tensor data into the allocated SHM blocks (e.g., via `storage.write()`).
3. Call `close_write(uuid)` to finalize the write, making the item readable. If `generate_read_token=True`, a read token is created and a read reference is automatically held (useful for immediate sharing).

**Typical Read Flow (GPU Worker):**

1. Call `open_read(uuid_or_token)` to acquire a read lock and obtain the block list. If the item is still being written, the call may wait (with a configurable timeout).
2. Read the data from SHM into CPU or GPU memory (e.g., using `storage.read_to_tensor()`).
3. After reading, the read lock is **released immediately** by calling `close_read(token)`. This is safe because the data has been copied to the target device; any asynchronous H2D transfers should be synchronized upfront if needed. The immediate release simplifies the lifecycle and avoids extra reference counting.

### Serialization Format for Multi‑Modal Items

`PagedShmCache` uses a dedicated serialization format to pack arbitrary multi‑modal items into block‑aligned chunks.

**Format layout** (as implemented in `serial_utils.py`):

```text
Chunks:   [ Meta ]   [   Data 0   ]        [ Data 1  ...]
Blocks:   +----------+----------+----------+----------+ ...
          Block 0    Block 1    Block 2    Block 3
```

- Each chunk starts at a new block boundary (block‑aligned).
- A chunk may occupy multiple consecutive blocks.
- Padding may exist at the end of the last block of each chunk.

**Metadata Chunk** consists of:

1. A **10‑byte header**:
   - 2 bytes: magic number `M0` (identifies vLLM paged shared memory format version 0).
   - 4 bytes: total size of the entire metadata chunk (including the header).
   - 4 bytes: total number of chunks (metadata + all data chunks).
2. For each data chunk, a **metadata entry**:
   - 4 bytes: original length of the data chunk (unsigned int, little‑endian).
   - 1 byte : type flag (`0` → bytes‑like data, `1` → `torch.Tensor`).

When only a single data chunk exists (e.g., a small tensor), the encoding function returns `None` to signal that shared memory transfer is unnecessary; the item is sent directly via ZMQ.

## Higher‑Level Processor Cache: `PagedShmCache`

### Usage

Add the following arguments when starting vLLM:

```bash
vllm serve Qwen/Qwen2-VL-2B-Instruct \
    --mm-processor-cache-type paged_shm \
    --mm-processor-cache-gb 4
```

| Argument                              | Description                                                 |
|---------------------------------------|-------------------------------------------------------------|
| `--mm-processor-cache-type paged_shm` | Enables Paged Shared Memory caching for multi‑modal inputs. |
| `--mm-processor-cache-gb 4`           | Allocates a 4GB SHM pool (adjust as needed).                |

### Overview

`PagedShmCache` is a high‑level abstraction built on `PagedShmClient` that integrates with the multi‑modal processing pipeline (`BaseMultiModalProcessorCache`). It provides:

- **Serialization** – Encodes arbitrary multi‑modal items (tensors + prompt updates) into block‑aligned chunks using MessagePack, with the metadata format described above.
- **Asynchronous writes** – Submits write tasks to a thread pool, allowing the API Server to return immediately without waiting for SHM allocation to complete.
- **Efficient reads** – Supports CUDA streams and pinned memory to overlap data transfer with computation. Reads are synchronous but stream‑synchronized.

The cache automatically manages block allocation, reference counting, and cleanup through the underlying server.

### Two Cache Roles

- **Sender (P0) cache** (`PagedShmSenderCache`)  
  Used by the API Server (Renderer) to store items. When a new item arrives, it encodes and writes it asynchronously, returning `(None, updates)` to indicate that the item is now in SHM and the caller can proceed without the raw data.

- **Receiver (P1) cache** (`PagedShmReceiverCache`)  
  Used by GPU Workers to retrieve items. It waits for the writer to complete (if necessary), reads the data, and returns the deserialized item. After the forward pass, the worker releases the read lock immediately.

## Performance Considerations

- **Block size**: 1MB is chosen to maximize H2D bandwidth efficiency (matches typical CUDA transfer granularity).
- **Zero‑copy reads**: When reading to GPU, the data can be directly copied from SHM to GPU memory without intermediate CPU buffers.
- **Pinned memory**: The receiver cache automatically uses pinned memory to further improve transfer speed.
- **Reference counting**: The server keeps track of active readers and writers to prevent premature eviction.
- **LRU eviction**: When the pool is full, the server evicts the least recently used items, freeing blocks for new writes.
