.. _recipe_qwen3:

Qwen3 MoE
=========

Validated models
----------------

- `Qwen/Qwen3-235B-A22B <https://huggingface.co/Qwen/Qwen3-235B-A22B>`_
- `Qwen/Qwen3-30B-A3B <https://huggingface.co/Qwen/Qwen3-30B-A3B>`_
- `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 <https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8>`_
- `Qwen/Qwen3-Coder-30B-A3B-Instruct <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct>`_

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `Qwen3 MoE in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture ``Qwen3MoeForCausalLM``).

      **Status:** Validated with LMCache.

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server --l1-size-gb 100 --eviction-policy LRU

      |

      **Qwen3-235B-A22B** (4 GPUs, expert parallel):

      .. code-block:: bash

         vllm serve Qwen/Qwen3-235B-A22B \
             --tensor-parallel-size 4 \
             --enable-expert-parallel \
             --enable-auto-tool-choice \
             --tool-call-parser hermes \
             --reasoning-parser qwen3 \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      **Qwen3-30B-A3B** (1 GPU):

      .. code-block:: bash

         vllm serve Qwen/Qwen3-30B-A3B \
             --enable-auto-tool-choice \
             --tool-call-parser hermes \
             --reasoning-parser qwen3 \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      **Qwen3-Coder-480B-A35B-Instruct-FP8** (8 GPUs, expert parallel):

      .. code-block:: bash

         vllm serve Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
             --tensor-parallel-size 8 \
             --enable-expert-parallel \
             --enable-auto-tool-choice \
             --tool-call-parser qwen3_coder \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      **Qwen3-Coder-30B-A3B-Instruct** (1 GPU):

      .. code-block:: bash

         vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
             --enable-auto-tool-choice \
             --tool-call-parser qwen3_coder \
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
