.. _recipe_gpt_oss:

gpt-oss
=======

Validated models
----------------

- `openai/gpt-oss-120b <https://huggingface.co/openai/gpt-oss-120b>`_
- `openai/gpt-oss-20b <https://huggingface.co/openai/gpt-oss-20b>`_

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `GPT-OSS in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture ``GptOssForCausalLM``).

      **Status:** Validated with LMCache.

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server --l1-size-gb 100 --eviction-policy LRU

      |

      **gpt-oss-120b** (2 GPUs):

      .. code-block:: bash

         vllm serve openai/gpt-oss-120b \
             --tensor-parallel-size 2 \
             --enable-auto-tool-choice \
             --tool-call-parser openai \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      **gpt-oss-20b** (1 GPU):

      .. code-block:: bash

         vllm serve openai/gpt-oss-20b \
             --enable-auto-tool-choice \
             --tool-call-parser openai \
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
