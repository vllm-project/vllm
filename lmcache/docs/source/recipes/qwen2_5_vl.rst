.. _recipe_qwen2_5_vl:

Qwen2.5-VL (multimodal)
=======================

Validated models
----------------

- `Qwen/Qwen2.5-VL-3B-Instruct <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>`_

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `Qwen2.5-VL in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#multimodal-language-models>`_
      (architecture ``Qwen2_5_VLForConditionalGeneration``).

      **Status:** Validated with LMCache (image inputs, MP mode).

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server --l1-size-gb 20 --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector:

      .. code-block:: bash

         vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
             --tensor-parallel-size 1 \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      No multimodal-specific LMCache configuration is needed. Image identity
      is folded into the cache keys automatically: vLLM emits identical
      placeholder token ids for every image, so LMCache overwrites each
      placeholder span with a value derived from the image's content hash
      (``mm_hash``) before hashing. Two requests with the same text but
      different images therefore get distinct cache entries, while repeating
      the same image and prompt hits the cache.

      If there are any issues with vLLM setup, please refer to the
      `vLLM Recipes <https://docs.vllm.ai/projects/recipes/en/latest/index.html>`_
      for more details.

   .. tab-item:: SGLang

      **Status:** Not validated with LMCache.

   .. tab-item:: TRT-LLM

      **Status:** Not validated with LMCache.

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

- **KV cache only.** LMCache caches the decoder KV for multimodal requests.
  Vision-encoder outputs are a separate cache: see
  :doc:`../non_kv_cache/encoder_cache` (in-process mode only; not yet
  available in MP mode).
- **Cross-user sharing.** Cache entries are content-addressed: two users
  sending the byte-identical image and prompt share an entry. For hard
  per-user or per-tenant isolation, set ``cache_salt`` on the request --
  salted requests never share entries, even for identical content.
- **Cache hits require identical images.** The key derives from the image
  content hash, so a re-encoded or resized variant of the same picture is a
  different image and misses the cache.
