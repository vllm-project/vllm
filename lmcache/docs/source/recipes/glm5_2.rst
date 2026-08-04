.. _recipe_glm5_2:

GLM 5.1/5.2
===========

A large Mixture-of-Experts model using **Dynamic Sparse Attention (DSA)**, shared
by the **GLM-5.1 / GLM-5.2** series. Like DeepSeek-V4-Flash, the sparse-attention path
splits the model's layers into more than one KV cache group; the
``LMCacheMPConnector`` stores and retrieves each group in its own block size, so
KV reuse works without extra flags.

Validated models
----------------

- `zai-org/GLM-5.2-FP8 <https://huggingface.co/zai-org/GLM-5.2-FP8>`_ (8 GPUs)

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `GLM-5.2 in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture ``GlmMoeDsaForCausalLM``). See also the
      `vLLM GLM-5.2 recipe <https://recipes.vllm.ai/zai-org/GLM-5.2>`_.

      **Status:** Validated with LMCache (vLLM 0.23.0 + LMCache 0.4.7).

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server \
             --port 6555 \
             --max-workers 8 \
             --l1-size-gb 100 \
             --eviction-policy LRU \
             --chunk-size 1024

      |

      Start vLLM with the LMCache MP connector (8 GPUs):

      .. code-block:: bash

         vllm serve zai-org/GLM-5.2-FP8 \
             --tensor-parallel-size 8 \
             --tool-call-parser glm47 \
             --enable-auto-tool-choice \
             --reasoning-parser glm45 \
             --no-enable-prefix-caching \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":6555}}'

      |

      ``--tool-call-parser glm47``, ``--enable-auto-tool-choice``, and
      ``--reasoning-parser glm45`` are GLM-5.2's serving requirements (see the
      vLLM recipe). ``--no-enable-prefix-caching`` routes all KV reuse through
      LMCache rather than vLLM's in-engine prefix cache. The server's
      ``--port 6555`` must match ``lmcache.mp.port`` in the connector config;
      ``--max-workers`` is set to the tensor-parallel size. Adjust
      ``--tensor-parallel-size`` to match your hardware. For the generic
      LMCache + vLLM wiring (ports, remote hosts), see
      :doc:`../getting_started/quickstart`.

      If there are any issues with vLLM setup, please refer to the
      `vLLM Recipes <https://docs.vllm.ai/projects/recipes/en/latest/index.html>`_
      for more details.

   .. tab-item:: SGLang

      **Status:** Not validated with LMCache.

   .. tab-item:: TRT-LLM

      **Status:** Supported. See :doc:`../getting_started/quickstart` for TRT-LLM + LMCache setup.

CacheBlend support
------------------

Not validated.

Compression support
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Method
     - Status
     - Notes
   * - :doc:`CacheGen <../kv_cache_optimizations/compression/cachegen>`
     - Not validated
     -

Caveats
-------

- **Dynamic Sparse Attention KV groups.** GLM-5.2's DSA path splits the model's
  layers into more than one KV cache group with different block geometries.
  LMCache stores and retrieves each group in its own block size; no extra flags
  are required beyond the launch commands above.
