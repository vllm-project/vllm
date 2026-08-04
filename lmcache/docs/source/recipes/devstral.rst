.. _recipe_devstral:

Mistral / Devstral
==================

Validated models
----------------

- `mistralai/Devstral-2-123B-Instruct-2512 <https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512>`_

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `Mistral / Devstral in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture ``MistralForCausalLM``).

      **Status:** Validated with LMCache.

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server --l1-size-gb 100 --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector:

      .. code-block:: bash

         vllm serve mistralai/Devstral-2-123B-Instruct-2512 \
             --tensor-parallel-size 2 \
             --enable-auto-tool-choice \
             --tool-call-parser mistral \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      Adjust ``--tensor-parallel-size`` to match your hardware. For the
      generic LMCache + vLLM wiring (ports, remote hosts),
      see :doc:`../getting_started/quickstart`.

      If there are any issues with vLLM setup, please refer to the
      `vLLM Recipes <https://docs.vllm.ai/projects/recipes/en/latest/index.html>`_
      for more details.

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
