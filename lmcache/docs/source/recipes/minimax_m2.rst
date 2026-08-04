.. _recipe_minimax_m2:

MiniMax M2 series
=================

Validated models
----------------

- `MiniMaxAI/MiniMax-M2 <https://huggingface.co/MiniMaxAI/MiniMax-M2>`_
- `MiniMaxAI/MiniMax-M2.5 <https://huggingface.co/MiniMaxAI/MiniMax-M2.5>`_
- `MiniMaxAI/MiniMax-M2.7 <https://huggingface.co/MiniMaxAI/MiniMax-M2.7>`_

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `MiniMax-M2 in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture ``MiniMaxM2ForCausalLM``).

      **Status:** Validated with LMCache.

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server --l1-size-gb 100 --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector:

      **MiniMax-M2** (8 GPUs):

      .. code-block:: bash

         vllm serve MiniMaxAI/MiniMax-M2 \
             --tensor-parallel-size 8 \
             --trust-remote-code \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      **MiniMax-M2.5** (4 GPUs):

      .. code-block:: bash

         vllm serve MiniMaxAI/MiniMax-M2.5 \
             --tensor-parallel-size 4 \
             --trust-remote-code \
             --enable-auto-tool-choice \
             --tool-call-parser minimax_m2 \
             --reasoning-parser minimax_m2 \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      **MiniMax-M2.7** (4 GPUs):

      .. code-block:: bash

         vllm serve MiniMaxAI/MiniMax-M2.7 \
             --tensor-parallel-size 4 \
             --trust-remote-code \
             --enable-auto-tool-choice \
             --tool-call-parser minimax_m2 \
             --reasoning-parser minimax_m2 \
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

      **Engine documentation:**
      `MiniMax-M2 SGLang cookbook
      <https://docs.sglang.io/cookbook/autoregressive/MiniMax/MiniMax-M2>`_,
      `MiniMax M2.5/M2.1/M2 usage guide
      <https://docs.sglang.io/docs/basic_usage/minimax_m2>`_.

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
