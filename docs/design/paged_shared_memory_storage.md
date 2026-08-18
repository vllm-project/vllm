# Paged Shared Memory Storage

## Shared Memory

Shared memory enables multiple processes to access the same memory region, allowing efficient data sharing without 
serialization overhead.

vLLM V1 adopts a multi-process architecture to separate concerns and maximize throughput:

- **API Server Process**: Handles HTTP requests (e.g., the OpenAI-compatible API), performs input processing 
- (tokenization, multi-modal data loading), and streams results back to clients. It communicates with the engine core 
- process(es) via ZMQ sockets.

- **Engine Core Process**: Runs the scheduler, manages KV cache, and coordinates model execution across GPU workers. It 
- maintains a busy loop that continuously schedules requests and dispatches work to the GPU workers.

- **GPU Worker Processes**: Each GPU is managed by a dedicated worker process. The worker loads model weights, executes 
- forward passes, and manages GPU memory. Workers communicate with the engine core process that owns them.

Additionally, shared memory supports `pin_memory`, which enables faster CPU–GPU data transfers.

The data transfer paths are as follows:

```
ZMQ IPC Path:
[Multi-modal Tensor] ─ZMQ IPC→ [CPU Buffer] ─Pin Memory ─→ [GPU]
    (API Server)               (GPU Worker)            (H2D)

Shared Memory Path:
[Multi-modal Tensor] ─Shared Memory→ [swap_blocks_batch] ─→ [GPU]
    (API Server)                           (GPU Worker) (H2D)
```

## Paged Shared Memory

Multimodal data sizes are not uniform, which can lead to memory fragmentation. Using the system's allocation and 
deallocation functions, along with pin_memory and unpin_memory, also incurs some overhead. Therefore, we pre-allocate a 
large shared memory pool and divide it into pages. The default page size is 1 MB, as it saturates the H2D transfer 
bandwidth.

We use a centralized PagedShmServer to manage the paged shared memory. It allocates the shared memory segment at startup 
and releases it during shutdown. Other processes read from and write to the shared memory but do not own it, which 
eliminates the risk of shared memory leaks.


To avoid race conditions caused by multiple processes simultaneously reading from and writing to paged shared memory, 
the client must first request permission from the Server before performing any read or write operations. 

A typical workflow is as follows:

1. The API Server uses `open_write` to allocate blocks for a batch of items to be written.
2. The API Server writes the multimodal data into the paged shared memory.
3. The API Server uses `close_write` to indicate that the write operation is complete, allowing other clients to read the data.
4. The GPU Worker uses `open_read` to wait for the write operation to complete.
5. The GPU Worker reads the data from the shared memory.
6. The GPU Worker uses `close_read` to release the shared memory.