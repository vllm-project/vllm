Architecture Overview
=====================

High-Level System Architecture
------------------------------


LMCache extends an LLM inference engine (e.g., vLLM) with a multi-tier KV cache storage system spanning GPU memory, CPU memory, and disk/remote backends. The diagram below illustrates how KV cache blocks move across these layers.

**Multi-Tier Storage Architecture**

LMCache implements a hierarchical storage system with three distinct tiers:

* **GPU Memory**: Holds the active working set of KV caches that are currently being used by the model
* **CPU DRAM**: Acts as a "hot cache" for recently used KV chunks, using pinned memory for efficient GPU-CPU transfers
* **Local storage (e.g., local disk, NVMe GDS)**: Provides a large tier for local KV caching (e.g. for long documents)
* **Remote storage (e.g., Redis, Mooncake, InfiniStore)**: Persistent storage for KV caches. Reliable but not as performant as previous tiers.

**Data Flow and Operations**

When the model generates new key-value (KV) cache chunks on the GPU, LMCache can:

1. **Offload overflow KV caches** from GPU to CPU DRAM, freeing precious GPU memory
2. **Asynchronously write** KV caches from CPU to disk or remote storage using LRU eviction policies
3. **Prefetch hot KV caches** from disk/remote storage back to CPU when needed
4. **On-demand reuse** of cached segments by moving them from CPU back to GPU

This architecture enables LMCache to significantly reduce prefill delays and GPU memory pressure while maintaining high performance through intelligent cache management.

.. mermaid::
   :align: center

   flowchart TB
       subgraph "LLM Engine (with LMCache Integration)"
         direction TB
         GPU["GPU Memory"]
         CPU["CPU DRAM"]
         GPU -- "Offload overflow KV" --> CPU
         CPU -- "On-demand reuse" --> GPU
       end
       Disk[(Disk Storage Backend)]
       Remote[(Remote Storage Backend)]
       CPU -- "Async write (LRU evict)" --> Disk
       CPU -- "Async upload" --> Remote
       Disk -- "Prefetch hot KV" --> CPU
       Remote -- "Fetch on reuse" --> CPU

Two modes
---------

**Storage Mode (KV cache offloading)**
   LMCache acts as a persistent KV store, optimizing for high reuse across queries or sessions. It offloads infrequently used KV blocks from GPU memory and persists popular caches across sessions, boosting cache hit rates for "hot" content. KV caches survive beyond single inference calls and even process restarts when backed by disk or external storage.


.. mermaid::

    sequenceDiagram
    participant Main as LLM Inference Thread
    participant DiskTask as Disk Offload Task
    participant RemoteTask as Remote Offload Task
    Main->>Main: New KV chunk created (GPU memory)
    Main->>Main: Copy KV chunk to CPU buffer
    par Disk backend offload
        Main--)DiskTask: Spawn async disk write task
        DiskTask-->>DiskTask: Compress & save chunk to disk
    and Remote backend offload
        Main--)RemoteTask: Spawn async remote upload task
        RemoteTask-->>RemoteTask: Send chunk to remote store
    end
    Main-->>Main: Continue with next inference (no blocking)


**Transport Mode (Prefill-decode disaggregation)**
   Focuses on accelerating distributed inference by routing KV cache data between nodes in real-time. Enables prefill-decode disaggregation where one server computes KV for prompts and delivers them to another server for generation without recomputation. Uses peer-to-peer channels with communication libraries like NIXL for low-latency, high-bandwidth transfers.


Core Components
---------------

**LLM Inference Engine Integration Module (Connector)**
   Integrated into the LLM engine (vLLM), the Connector taps into the paged KV memory manager. During prompt processing, it checks if token sequences were seen before:

   * **Cache hit**: Fetches precomputed KV cache chunks from LMCache, bypassing computation
   * **Cache miss**: Model computes KV as usual, then Connector hands newly-generated KV data to LMCache for storage

**Cache Index (Token Database)**
   Maintains an internal index mapping token sequences to cached KV entries and their locations. Enables cross-request and cross-instance cache lookups with configurable chunking strategy (default 256 tokens) and hashing scheme.

**Memory Object & Allocator**
   Manages KV cache entries as MemoryObj instances using a custom memory allocator within LocalCPUBackend. Ensures pinned memory for fast GPU↔CPU transfers, NUMA-aware allocation, and interfaces with eviction policies (LRU by default).

**Asynchronous Offloading**
   Offloading / loading the KV cache chunks in an asynchronous manner to avoid blocking inference threads and GPU cycles.

**Remote Connectors**
   Plugin-based system for remote backends (Redis, Mooncake, NiXL). Uses generic RemoteBackend wrapper that delegates operations to connector implementations, supporting dynamic loading of custom backends.

LMCache Controller
------------------

The Controller provides a management API for runtime cache operations:

* **Lookup**: Query cache entries for given token sequences and their locations
* **Clear**: Purge KV cache entirely or for specific entries
* **Compress/Decompress**: On-demand compression using CacheGen or decompression to full precision
* **Move**: Migrate caches to specified locations for cache warming or optimization
* **Pin/Unpin**: Mark cache entries as persistent to prevent eviction
* **Health & Finish Checks**: Report worker health and confirm completion of async operations

The Controller coordinates with all LMCache workers in the system, providing centralized management for both single-instance and distributed deployments.


