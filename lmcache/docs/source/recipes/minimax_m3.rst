.. _recipe_minimax_m3:

MiniMax M3
==========

Validated models
----------------

- `MiniMaxAI/MiniMax-M3 <https://huggingface.co/MiniMaxAI/MiniMax-M3>`_

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `MiniMax-M3 in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture ``MiniMaxM3SparseForConditionalGeneration``).

      **Status:** Validated with LMCache.

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server --l1-size-gb 100 --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector (8 GPUs):

      .. code-block:: bash

         vllm serve MiniMaxAI/MiniMax-M3 \
             --tensor-parallel-size 8 \
             --trust-remote-code \
             --block-size 128 \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      ``--block-size 128`` is **required** for this model (see Caveats); the
      smaller defaults fail vLLM's KV-cache init. ``--trust-remote-code`` loads
      M3's custom architecture. Adjust ``--tensor-parallel-size`` to your
      hardware — M3's weights need eight 140 GB-class GPUs. For the generic
      LMCache + vLLM wiring (ports, remote hosts), see
      :doc:`../getting_started/quickstart`.

      If there are any issues with vLLM setup, please refer to the
      `vLLM Recipes <https://docs.vllm.ai/projects/recipes/en/latest/index.html>`_
      for more details.

   .. tab-item:: SGLang

      **Status:** Not validated with LMCache.

   .. tab-item:: TRT-LLM

      **Status:** Not validated with LMCache.

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

- **Sparse attention with a lightning indexer.** M3 runs grouped-query full
  attention plus a DeepSeek-style sparse-attention indexer. Each sparse layer
  owns two paged caches — the main K/V (rank-5) and a key-only indexer cache
  (rank-3) — which vLLM places in a single ``UniformTypeKVCacheSpecs`` engine
  group. LMCache detects both layouts and stores/retrieves each as its own
  group; the indexer keys travel with the K/V because they cannot be recomputed
  from the cached K/V on a hit.
- **``--block-size 128`` is required.** M3's indexer uses ``sparse_block_size =
  128``; vLLM cannot reconcile the default block size (16) or 64 across the
  full-attention and sparse kernels and aborts KV-cache init with
  ``No common block size``. Use 128.
- **LMCache chunk size must be a multiple of the block size.** The default
  chunk size (256) already satisfies 128, so no extra flag is needed; if you
  pass ``--chunk-size`` to the server, keep it a multiple of 128.
