# SPDX-License-Identifier: Apache-2.0
"""LMCache integration for NVIDIA TensorRT-LLM via the KV Cache Connector API.

Two modes are supported:

**In-process mode** (``lmcache``)
    The LMCache engine runs inside the TRT-LLM process. Simple to set
    up; no extra services required.

    .. code-block:: yaml

       kv_connector_config:
         connector: lmcache

**Multi-process mode** (``lmcache-mp``)
    The LMCache engine runs as a standalone ZMQ server, providing
    process isolation and shared KV caching across multiple TRT-LLM
    instances on the same node.

    1. Start the LMCache server::

        python -m lmcache.v1.multiprocess.server \\
            --host 0.0.0.0 --port 5555 \\
            --l1-size-gb 10 --eviction-policy LRU --max-workers 4

    2. Use the ``lmcache-mp`` preset with ``server_url``::

        kv_connector_config:
          connector: lmcache-mp
          server_url: tcp://localhost:5555

Both modes can also be configured explicitly instead of via presets::

    # In-process
    kv_connector_config:
      connector_module: lmcache.integration.tensorrt_llm.tensorrt_adapter
      connector_scheduler_class: LMCacheKvConnectorScheduler
      connector_worker_class: LMCacheKvConnectorWorker

    # Multi-process
    kv_connector_config:
      connector_module: lmcache.integration.tensorrt_llm.tensorrt_mp_adapter
      connector_scheduler_class: LMCacheMPKvConnectorScheduler
      connector_worker_class: LMCacheMPKvConnectorWorker
"""

# tensorrt_llm is an optional dependency. Guard the import so importing
# ``lmcache.integration.tensorrt_llm`` does not crash when the package
# is absent (e.g. core LMCache unit tests, doc builds).
try:
    # First Party
    from lmcache.integration.tensorrt_llm.tensorrt_adapter import (
        LMCacheKvConnectorScheduler,
        LMCacheKvConnectorWorker,
        destroy_engine,
    )

    __all__ = [
        "LMCacheKvConnectorScheduler",
        "LMCacheKvConnectorWorker",
        "destroy_engine",
    ]
except ImportError:
    __all__ = []

try:
    # First Party
    from lmcache.integration.tensorrt_llm.tensorrt_mp_adapter import (
        LMCacheMPKvConnectorScheduler,
        LMCacheMPKvConnectorWorker,
    )

    __all__ += [
        "LMCacheMPKvConnectorScheduler",
        "LMCacheMPKvConnectorWorker",
    ]
except ImportError:
    pass
