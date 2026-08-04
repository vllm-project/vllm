.. _recipe_mixtral:

Mixtral
=======

Validated models
----------------

- `mistralai/Mixtral-8x7B-v0.1 <https://huggingface.co/mistralai/Mixtral-8x7B-v0.1>`_
- `mistralai/Mixtral-8x7B-Instruct-v0.1 <https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1>`_

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `MixtralForCausalLM in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture `MixtralForCausalLM`).

      **Status:** Validated with LMCache.

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server --l1-size-gb 100 --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector:

      **Mixtral-8x7B-v0.1** (4 GPUs):

      .. code-block:: bash

         vllm serve mistralai/Mixtral-8x7B-v0.1 \
             --tensor-parallel-size 4 \
             --trust-remote-code \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |
   
      **Mixtral-8x7B-Instruct-v0.1** (4 GPUs):

      .. code-block:: bash

         vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
             --tensor-parallel-size 4 \
             --trust-remote-code \
             --enable-auto-tool-choice \
             --tool-call-parser mistral \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'
      
      |

      Adjust ``--tensor-parallel-size`` to match your hardware. For the
      generic LMCache + vLLM wiring (ports, remote hosts),
      see :doc:`../getting_started/quickstart`.

   .. tab-item:: SGLang

      **Status:** Not validated with LMCache.

   .. tab-item:: TRT-LLM

      **Status:** Supported. See :doc:`../getting_started/quickstart` for TRT-LLM + LMCache setup.

CacheBlend support
------------------

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

None known.