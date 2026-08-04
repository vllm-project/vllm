.. _recipe_gemma4:

Gemma 4
=======

Validated models
----------------

- `google/gemma-4-31B-it <https://huggingface.co/google/gemma-4-31B-it>`_
- `google/gemma-4-12B-it <https://huggingface.co/google/gemma-4-12B-it>`_
- `google/gemma-4-E4B-it <https://huggingface.co/google/gemma-4-E4B-it>`_

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `Gemma 4 in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#multimodal-language-models>`_
      (architectures ``Gemma4ForConditionalGeneration`` for 31B/E4B and
      ``Gemma4UnifiedForConditionalGeneration`` for 12B).

      **Status:** Validated with LMCache.

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server --l1-size-gb 100 --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector:

      .. code-block:: bash

         vllm serve google/gemma-4-31B-it \
             --tensor-parallel-size 2 \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      The smaller ``google/gemma-4-12B-it`` and ``google/gemma-4-E4B-it`` run on
      a single GPU:

      .. code-block:: bash

         vllm serve google/gemma-4-12B-it \
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

- **Hybrid KV cache with heterogeneous block sizes.** Gemma 4 interleaves
  sliding-window and full-attention layers whose head dimensions differ
  (sliding 256, full 512), so vLLM unifies the physical page size by giving the
  two attention types different ``block_size``\ s (e.g. ``google/gemma-4-E4B-it``:
  sliding 32, full 16). LMCache stores and retrieves each KV cache group in its
  own block size; no extra flags are required.
- **Cross-layer KV sharing.** ``google/gemma-4-E4B-it`` reuses some layers' KV
  caches across layers. LMCache stores the cache-owning layers only; the sharing
  layers' KV lives in the same blocks and is restored automatically.
