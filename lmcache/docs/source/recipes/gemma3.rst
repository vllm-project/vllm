.. _recipe_gemma3:

Gemma 3
=======

Validated models
----------------

- `google/gemma-3-4b-it <https://huggingface.co/google/gemma-3-4b-it>`_

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `Gemma 3 in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#multimodal-language-models>`_
      (architecture ``Gemma3ForConditionalGeneration``).

      **Status:** Validated with LMCache.

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server --l1-size-gb 100 --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector:

      .. code-block:: bash

         vllm serve google/gemma-3-4b-it \
             --tensor-parallel-size 1 \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      Gemma 3 interleaves local (sliding-window) and global (full) attention
      layers, so vLLM keeps its **hybrid KV cache manager** on and exposes
      multiple KV cache groups. LMCache stores and retrieves all of them through
      its hybrid memory allocator support -- ``LMCacheMPConnector`` advertises
      ``SupportsHMA``, so vLLM does not auto-disable the hybrid manager and no
      extra configuration is required.

      ``google/gemma-3-4b-it`` is a gated model; authenticate with the Hugging
      Face Hub (e.g. set ``HF_TOKEN``) before serving. Adjust
      ``--tensor-parallel-size`` to match your hardware. For the generic LMCache
      + vLLM wiring (ports, remote hosts), see
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

- **Gated model.** ``google/gemma-3-4b-it`` requires accepting the license on
  Hugging Face and authenticating (e.g. ``HF_TOKEN``) before it can be served.
- **Hybrid attention.** Gemma 3 is a hybrid (sliding-window + full-attention)
  model. LMCache transfers every KV cache group via its hybrid memory allocator
  support, so caching works transparently. This applies to the standard paged
  attention used by Gemma 3; Mamba / linear-attention hybrids (whose recurrent
  state caches LMCache cannot yet transfer) are not supported.
