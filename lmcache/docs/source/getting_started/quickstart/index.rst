.. _quickstart_examples:

More Examples
=============

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


This section provides quick examples to help you get started with LMCache's key features.

KV Cache Offloading
-------------------

KV cache offloading allows you to move KV caches from GPU memory to CPU memory or other storage devices. This feature is particularly useful when:

- There are requests shares the same prefix (e.g., long system prompt, reusing chat history in chat applications, or caching offline-processed data).
- The GPU memory is limited to save all the KV caches.

By offloading KV caches, LMCache can reduce both time-to-first-token (TTFT) and GPU cycles.

See :ref:`offload_kv_cache` for more details.

KV Cache Sharing
----------------

KV cache sharing enables sharing the KV cache across different LLM instances. This feature is beneficial when:

- There are multiple LLM instances running in the same system.
- The requests that share the same prefix may go to different LLM instances.

Sharing KV caches also reduces TTFT and GPU computation by eliminating redundant calculations across different LLM instances.

See :ref:`share_kv_cache` for more details.

Disaggregated Prefill
---------------------

Disaggregated prefill separates the prefill and decode phases across different compute resources. This approach:

- Allows specialized hardware allocation for each phase of inference
- Enables more efficient resource utilization in distributed settings
- Improves overall system throughput by optimizing for the different computational patterns of prefill vs. decode

This architecture is particularly valuable in large-scale deployment scenarios where maximizing resource efficiency and keeping a stable generation speed are both important.

See :ref:`disaggregated_prefill` for more details.

Standalone Starter
------------------

The LMCache Standalone Starter allows you to run LMCacheEngine as a standalone service without vLLM or GPU dependencies. This is particularly useful for:

- Testing and development environments
- CPU-only deployments
- Distributed cache scenarios
- Integration with custom applications

See :ref:`standalone_starter` for more details.

Detailed Examples
-----------------

.. toctree::
   :maxdepth: 1

   offload_kv_cache
   share_kv_cache
   disaggregated_prefill 
   multimodality
   standalone_starter