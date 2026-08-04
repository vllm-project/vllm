.. _legacy:

Legacy (In-Process Mode)
========================

.. warning::

   These pages document LMCache's original **in-process mode**, where LMCache
   ran *inside* the inference engine process (e.g. via ``LMCacheConnectorV1`` on
   vLLM). In-process mode is **deprecated**; new deployments should use
   :doc:`Multiprocess (MP) mode <../mp/index>`.

Background
----------

LMCache began as an in-process library embedded directly in the serving engine.
The **multiprocess refactor** moved LMCache into a standalone ``lmcache server``
and made an **asynchronous prefetching architecture** the default -- a
``LOOKUP`` followed by background L2→L1 loads -- alongside process isolation,
shared caching across engine instances, and multi-tier (L1/L2) storage.

MP is now the recommended mode and is on track to support essentially
everything in-process mode did. The pages below are kept for users still on
in-process mode and as a historical reference while MP closes any remaining
gaps; where a feature already has an MP equivalent, prefer the MP docs.

.. toctree::
   :hidden:
   :maxdepth: 1

   /getting_started/quickstart/index
   /kv_cache/storage_backends/index
   /kv_cache/async_loading
   /kv_cache/caching_policies
   /kv_cache/p2p_sharing
   /non_kv_cache/encoder_cache
   /disaggregated_prefill/nixl/index
   /disaggregated_prefill/shared_storage
   /kv_cache_optimizations/compression/index
   /kv_cache_optimizations/layerwise
   /kv_cache_management/index
   /kv_cache_optimizations/blending
   /api_reference/multimodality
   /api_reference/storage_backends
   /api_reference/dynamic_connector
   /api_reference/configurations
   /internal_api_server/internal_api_server
   /controller/index
   /production/observability/index
   /production/docker_deployment
   /production/performance_tuning
   /production/kv_cache_events
   /developer_guide/architecture
   /developer_guide/integration
   /developer_guide/usage/index
   /developer_guide/usage/basic_check
   /developer_guide/extending_lmcache/storage_plugins
   /developer_guide/extending_lmcache/remote_storage_plugins
