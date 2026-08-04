.. _recipe_phi3:

Phi-3 / Phi-4
=============

Validated models
----------------

- `microsoft/Phi-4-mini-instruct <https://huggingface.co/microsoft/Phi-4-mini-instruct>`_
- `microsoft/Phi-3-medium-128k-instruct <https://huggingface.co/microsoft/Phi-3-medium-128k-instruct>`_

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `Phi3ForCausalLM in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture `Phi3ForCausalLM`).

      **Status:** Validated with LMCache.

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server --l1-size-gb 100 --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector:

      **Phi-4-mini-instruct** (1 GPU):

      .. code-block:: bash

         vllm serve microsoft/Phi-4-mini-instruct \
             --trust-remote-code \
             --enable-auto-tool-choice \
             --tool-call-parser phi4_mini_json \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      **Phi-3-medium-128k-instruct** (1 GPU):

      .. code-block:: bash

         vllm serve microsoft/Phi-3-medium-128k-instruct \
             --trust-remote-code \
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