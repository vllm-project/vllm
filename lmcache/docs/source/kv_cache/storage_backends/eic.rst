EIC
===

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/index`.


EIC(Elastic Instant Cache) is a distributed database designed for LLM KV Cache. It supports RDMA, GDR and has the capabilities of distributed disaster tolerance and expansion.
You can understand the principles and architecture of EIC through these articles: 

* https://mp.weixin.qq.com/s/tasDqXf0Gxr3o_WCJ2IJUQ
* https://mp.weixin.qq.com/s/b_4YhTa96Zeklh23lv8qBw

Deploy EIC
----------

You can visit the official link https://console.volcengine.com/eic and deploy EIC KVCache on your compute cluster with web UI. In addition, we provide particular image in volcano engine, which integrates various optimizations based on the official image.
You may use tests/v1/storage_backend/test_eic.py to detect the connectivity of EIC.

Deploy Model With EIC
---------------------

You can enable EIC KVCache offload with the official interface, such as

.. code-block:: bash

   export LMCACHE_CONFIG_FILE=/workspace/config/remote-eic.yaml
   export VLLM_USE_V1=1

   python3 -m vllm.entrypoints.openai.api_server \
     ... \
     --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Example ``config.yaml``:

.. code-block:: yaml

    chunk_size: 256
    remote_url: "eic://your-eic-endpoint"
    eic_instance_id: "your-eic-instance-id"
    eic_flag_file: "your-eic-config-path"


For more details, you can see https://www.volcengine.com/docs/85848/1749188.