vLLM Dynamic Connector
======================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


Upstream Integration:
~~~~~~~~~~~~~~~~~~~~~

LMCache integration with official upstream vLLM was introduced in `early February 2025 <https://github.com/vllm-project/vllm/pull/12953>`_.

vLLM imports the connector from the lmcache package and wraps it in `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py <https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py>`_:

.. code-block:: python

    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

This means that any updates to LMCache connector need to be synced/updated in the upstream vLLM. 

Example usage of vLLM upstream connector: 

**Pythonic Transfer Config:** 

.. code-block:: python

    from vllm.config import KVTransferConfig
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

**Command Line Transfer Configs:** 

.. code-block:: bash

    vllm serve "YOUR_MODEL" \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Dynamic Connector:
~~~~~~~~~~~~~~~~~~

`In June 2025 <https://github.com/vllm-project/vllm/pull/18142>`_, vLLM supports dynamic loading of KV connector implementations so we can directly reference connectors from the LMCache package without having to update vLLM. 

Example usage of dynamic connector from LMCache: 

**Pythonic Transfer Config:** 

.. code-block:: python

    from vllm.config import KVTransferConfig
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1Dynamic",
        kv_role="kv_both",
        kv_connector_module_path="lmcache.integration.vllm.lmcache_connector_v1",
    )

**Command Line Transfer Config:** 

.. code-block:: bash

    vllm serve "YOUR_MODEL" \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1Dynamic","kv_role":"kv_both","kv_connector_module_path":"lmcache.integration.vllm.lmcache_connector_v1"}'

This allows LMCache to modify/develop connectors and quickly plug-and-play. 

Any custom adapters will be documented here in the future as well as possible deprecations to the upstream connector. 


