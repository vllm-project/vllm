Layerwise KV Transfer
=====================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


The storage and loading of KV Cache on a layer granularity is a key optimization that allows for forward pass to "stagger" through its computation as each layer's KV Cache is received instead of only waiting to begin after the entire loading

CacheBlend is implemented on top of the layerwise codepath in order to pipeline recompute and loading to mask the latency of loading KV Cache.

.. image:: /_static/basic_codepath.svg
   :alt: Basic Codepath
   :class: scalable

.. raw:: html

   <div style="text-align:center; margin:1em 0;">
     <a href="/_static/full_layerwise_diagram.svg" target="_blank">
       <img src="/_static/full_layerwise_diagram.svg"
            style="display:block; margin:auto; max-width:100%; height:auto;"/>
     </a>
     <div style="font-size:0.9em; color:#555; margin-top:0.5em;">
       Click to open full-size
     </div>
   </div>

Architecture Overview
---------------------

**CacheEngine**
  The main orchestrator containing two primary generators:
  
  * **Retrieval Generator** (N + 2 yields): Handles layer-by-layer KV cache loading with on-demand memory allocation
  * **Storage Generator** (N + 1 yields): Manages layer-by-layer KV cache saving with upfront CPU memory allocation

**LayerwiseGPUConnector** 
  Manages GPU-CPU memory transfers with dedicated CUDA streams:
  
  * **Load GPU Buffer**: Temporary GPU memory for CPU→GPU transfers (``use_gpu: true``)
  * **Store GPU Buffer**: Temporary GPU memory for GPU→CPU transfers (``use_gpu: true``)
  * **Nested Generators**: ``batched_to_gpu()`` and ``batched_from_gpu()`` handle actual memory operations

**StorageManager**
  Handles persistent storage operations:
  
  * ``layerwise_batched_get()``: Asynchronous retrieval with ``.result()`` for request-level concurrency
  * ``batched_put()``: Stores memory objects to persistent backends

Execution Flow
~~~~~~~~~~~~~~

The layerwise pipeline follows a numbered execution sequence:

**1. start_load_kv()**
   * Initializes Retrieval Generator via ``lmcache_engine.retrieve_layer()``
   * Performs setup (1st ``next()``) and loads layer 0 (2nd ``next()``)
   * Creates ``layerwise_retrievers`` list for ongoing layer processing

**2. wait_for_layer_load()** (repeated for each layer)
   * Advances Retrieval Generator via ``next()`` to process layer i
   * Triggers ``StorageManager.layerwise_batched_get()`` for async cache retrieval
   * Calls GPU Load Generator's ``batched_to_gpu()`` to transfer memory objects to GPU
   * **Last request in batch**: Synchronizes ``current_stream.wait_stream(load_stream)``

**3. save_kv_layer()** (repeated for each layer)
   * **First call only**: Creates Storage Generator with upfront CPU memory allocation
   * Advances Storage Generator via ``next()`` to process layer i
   * Calls GPU Store Generator's ``batched_from_gpu()`` to transfer GPU data to CPU
   * **First request in batch**: Synchronizes ``store_stream.wait_stream(current_stream)``

**4. wait_for_save()**
   * Finalizes Storage Generator with last ``next()`` call
   * Completes all ``StorageManager.batched_put()`` operations
   * Performs GPU Store Generator cleanup

Key Optimizations
~~~~~~~~~~~~~~~~~

**Pipelined Memory Operations**
  The system overlaps layer N+1 computation with layer N storage.

**Stream Synchronization**
  Three CUDA streams coordinate operations:
  
  * ``current_stream``: vLLM's forward pass computation
  * ``load_stream``: KV cache loading operations  
  * ``store_stream``: KV cache storing operations

**Batch-Level Coordination**
  Multiple requests are processed together with specialized synchronization:
  
  * **First request**: Provides store stream synchronization to prevent GPU buffer corruption
  * **Last request**: Provides load stream synchronization to ensure KV cache availability

**Memory Allocation Strategies**
  * **Retrieval**: Layer-by-layer allocation
  * **Storage**: Upfront allocation for all layers

**Cache Key Management**
  Multi-layer cache engine keys use ``split_layers(N)`` to create per-layer kubernetes_deployment

Configuration
~~~~~~~~~~~~~

Enable layerwise caching by setting:

.. code-block:: yaml

   use_layerwise: true

The system automatically selects appropriate layerwise GPU connectors based on configuration:

* ``VLLMPagedMemLayerwiseGPUConnector``: For standard layerwise operations  
* ``VLLMBufferLayerwiseGPUConnector``: When blending is enabled
