Async Loading
=============

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

This document explains the principle, benefits, differences from vLLM PR `19330 <https://github.com/vllm-project/vllm/pull/19330>`_, and limitations of the LMCache ``async_loading`` feature.
It focuses on LMCache v1 integration with vLLM and the internal storage pipeline.

Key change of components in this feature include:

- LMCache async lookup client/server (ZMQ-based)
- Storage manager orchestrating backends and concurrency
- Cache engine async API entrypoints
- vLLM adapter integration points

Principle and Theory
--------------------

At a high level, ``async_loading`` decouples scheduler-side lookup from worker-side prefetch/retrieval, allowing overlap between I/O and computation while preserving prefix-based correctness.

- The scheduler sends lookup requests with token chunk hashes and offsets.
- Worker-side servers perform tiered ``batched_async_contains`` over available backends and eagerly launch non-blocking batched get operations for hit prefixes.
- Completion is tracked via an ``EventManager`` to safely deliver loaded memory objects back to the requesting path.
- A weighted semaphore with an ``AsyncSerializer`` prevents allocator deadlocks by shaping concurrency according to chunk budget.

The following Mermaid sequence diagram illustrates the end-to-end flow:

.. mermaid::

   sequenceDiagram
     autonumber
     participant S as Scheduler (vLLM)
     participant LC as LMCacheAsyncLookupClient
     participant WS as LMCacheAsyncLookupServer (Worker)
     participant SM as StorageManager
     participant BE as Backends (LocalCPU/LocalDisk/FSConnector)
     participant EM as EventManager

     S->>LC: lookup(token_ids, lookup_id, request_configs)
     note right of LC: Hashes + offsets via TokenDatabase
     LC->>WS: ZMQ PUSH multipart [lookup_id, hashes, offsets, configs]
     WS->>SM: async_lookup_and_prefetch(lookup_id, keys, cum_chunk_lengths)
     SM->>BE: batched_async_contains(lookup_id, keys, pin=True)
     alt prefix hit across tiers
       BE-->>SM: num_hit_chunks (per tier)
       SM->>BE: batched_get_non_blocking(lookup_id, hit_prefix)
       BE-->>SM: Future[List[MemoryObj]]
       SM->>EM: add_event(EventType.LOADING, lookup_id, gather_all)
       SM-->>WS: send_response_to_scheduler(lookup_id, retrieved_length)
       WS-->>LC: ZMQ PUSH [lookup_id, num_hit_tokens]
     else cache miss
       SM-->>WS: send_response_to_scheduler(lookup_id, 0)
       WS-->>LC: ZMQ PUSH [lookup_id, 0]
     end


Architecture (Worker Side)
--------------------------

.. mermaid::
   :align: center

   flowchart LR
       subgraph Worker
         direction TB
         A["LMCacheAsyncLookupServer<br/>ZMQ PULL/PUSH"]
         B["StorageManager<br/>Async loop (thread)"]
         C["AsyncSerializer<br/>WeightedSemaphore"]
         D["EventManager<br/>EventType.LOADING"]
       end

       subgraph Backends
         E["LocalCPUBackend<br/>contains/get"]
         F["LocalDiskBackend<br/>async contains/get"]
         G["FSConnector<br/>remote FS"]
       end

       A --> B
       B --> C
       B --> D
       B -.contains/get.-> E
       B -.contains/get.-> F
       B -.contains/get.-> G

       style E fill:#dff,stroke:#333,stroke-width:1px
       style F fill:#ffd,stroke:#333,stroke-width:1px
       style G fill:#dfd,stroke:#333,stroke-width:1px


Benefits
--------

- Performance overlap
    - **I/O–Compute Overlap**: Decoupling lookup/prefetch from loading enables fetching KV chunks while vLLM continues scheduling/computation.
- Robustness and error handling
    - **Event-driven Synchronization**: ``EventManager`` ensures safe hand-off of futures and avoids race conditions between threads and the async loop.
    - **Backpressure & Deadlock Avoidance**: ``AsyncSerializer`` with a weighted semaphore caps concurrent chunk retrievals based on allocator budget, preventing starvation or allocator lockups.
    - **Graceful Miss Path**: Immediate response with ``None`` hit tokens when nothing is retrievable; worker returns quickly without stalling the scheduler.

Comparison with vLLM Load Failure Recovery feature
---------------------------------------------------

The `VLLM_PR_19330 <https://github.com/vllm-project/vllm/pull/19330>`_ introduces a fault recovery mechanism for vLLM's KV connector infrastructure that enables graceful handling of KV cache load failures by automatically detecting failed block loads and rescheduling only affected requests for recomputation from a valid prefix.
By contrast, LMCache’s ``async_loading`` is an externalized caching layer with its own client/server, storage backends, and concurrency control.

Limitations
-----------

- Only works with vllm merged `VLLM_PR_23620 <https://github.com/vllm-project/vllm/pull/23620>`_
- Backend support constraint: This feature currently requires backends that implement ``batched_async_contains``; limited to a few backends, e.g.:
    - ``LocalCpuBackend``
    - ``LocalDiskBackend``
    - ``S3Connector``
    - ``FSConnector``
    - ``RedisConnector/RedisClusterConnector``

Future Work
-----------

- Introduce a default ``batched_async_contains`` implementation, so all backends can support ``async_loading``.
- Add metrics and observability to track the number of asynchronous lookup requests and the number of occupied ``MemoryObj`` instances.
- Improve the lookup framework by passing vLLM prefix cache hit tokens so that async lookup can skip loading parts already hit in vLLM.
