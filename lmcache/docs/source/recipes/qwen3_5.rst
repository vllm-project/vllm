.. _recipe_qwen3_5:

Qwen3.5 / Qwen3.6 series
========================

A hybrid architecture interleaving Mamba / Gated-DeltaNet (GDN) linear-attention
layers with full-attention layers, shared by the **Qwen3.5 and Qwen3.6**
series. LMCache reinterprets the recurrent state caches as opaque pages at
registration time; see :doc:`../mp/hybrid_models` for the general handling of
Mamba / linear-attention models.

Validated models
----------------

- `Qwen/Qwen3.6-27B <https://huggingface.co/Qwen/Qwen3.6-27B>`_ (1 GPU)
- `Qwen/Qwen3.5-0.8B <https://huggingface.co/Qwen/Qwen3.5-0.8B>`_ (1 GPU)

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `Qwen3.5 in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture ``Qwen3_5ForConditionalGeneration``).

      **Status:** Validated with LMCache.

      Every model in this family needs the same three settings: the ``align``
      Mamba cache mode, prefix caching, and a chunk size matched to vLLM's
      *unified block size*. That block size is model-specific — vLLM logs
      ``Setting attention block size to N tokens`` at startup:

      .. list-table::
         :header-rows: 1
         :widths: 50 25 25

         * - Model
           - Unified block size ``N``
           - GPUs
         * - ``Qwen/Qwen3.6-27B``
           - 784
           - 1
         * - ``Qwen/Qwen3.5-0.8B``
           - 544
           - 1

      Set the LMCache server's ``--chunk-size`` to that ``N`` (or a multiple of
      it), and vLLM's ``--max-num-batched-tokens`` to ``2N-1`` (the largest value
      below ``2N``). ``N`` is also valid but serializes prefill under load — see
      the note below.

      **Qwen3.6-27B** (1 GPU, ``N = 784`` → ``2N-1 = 1567``):

      .. code-block:: bash

         lmcache server --chunk-size 784 --l1-size-gb 100 --eviction-policy LRU

      .. code-block:: bash

         vllm serve Qwen/Qwen3.6-27B \
             --enable-prefix-caching \
             --mamba-cache-mode align \
             --max-num-batched-tokens 1567 \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      **Qwen3.5-0.8B** (1 GPU, ``N = 544`` → ``2N-1 = 1087``): identical to the
      above, with ``--chunk-size 544`` and ``--max-num-batched-tokens 1087``.

      ``--mamba-cache-mode align`` is required (GDN does not support the
      ``all`` mode). ``--max-num-batched-tokens`` must be in ``[N, 2N)`` (at
      least the unified block size and below twice it) — LMCache raises at
      engine startup otherwise. ``align`` snapshots the Mamba state at
      scheduler-step ends on a block boundary, and the scheduler splits prefills
      into whole ``N``-token blocks. **Prefer the maximum, ``2N-1``:** a single
      request still advances exactly one block per step (``2N-1 < 2N``), so the
      per-block snapshot LMCache stores is preserved, *and* the spare ``N-1``
      budget lets decodes co-schedule with a prefill block. Setting it to exactly
      ``N`` makes the per-step budget equal to one block, so once any request is
      decoding (consuming ≥1 token of the budget) no new request can start
      prefill — execution serializes to one request at a time. (Benchmarked on
      Qwen3.6-27B: at ``N`` a cold / low-hit run ran ~7× slower with GPU batch
      stuck at 1; ``2N-1`` restored full batching. With a warm LMCache cache
      (~97 % hit) the gap is small since little prefill remains, but ``2N-1`` is
      the safe default.) If vLLM reports *"max_num_seqs exceeds available Mamba
      cache blocks"* at ``2N-1``, lower ``--max-num-seqs`` to ≤ that count (each
      decode sequence needs one Mamba block) or raise ``--gpu-memory-utilization``.

      For the generic LMCache + vLLM wiring (ports, remote hosts), see
      :doc:`../getting_started/quickstart`.

   .. tab-item:: SGLang

      **Status:** Not validated with LMCache.

   .. tab-item:: TRT-LLM

      **Status:** Supported. See :doc:`../getting_started/quickstart` for TRT-LLM + LMCache setup.

CacheBlend support
------------------

Not supported: the hybrid groups' cached pages are byte-opaque (see Caveats).

Compression support
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Method
     - Status
     - Notes
   * - :doc:`CacheGen <../kv_cache_optimizations/compression/cachegen>`
     - Not supported
     - Hybrid groups' cached pages are byte-opaque.

Caveats
-------

- Generation is **not bit-exact** between a cached and a fresh run: GDN
  backends do not support vLLM's batch-invariant mode. Expect score-level
  equivalence, not token-level (the CI gate is the ``hma_lm_eval_qwen3_5``
  gsm8k store-vs-retrieve comparison).
- Cached pages for the Mamba and full-attention groups are byte-opaque views,
  so content-aware processing does not apply, and cache entries must not be
  shared across engines with different attention backends or kernel block
  sizes.
- vLLM's Mamba prefix caching in ``align`` mode is experimental.
- ``Qwen/Qwen3.6-27B`` is a vision-language model (it loads a vision tower);
  the LMCache validation covers **text** generation (the ``hma_lm_eval_qwen3_5``
  gsm8k store-vs-retrieve gate). Caching of image/video KV is not validated.
