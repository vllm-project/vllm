.. _recipe_kimi_linear:

Kimi-Linear
===========

A hybrid architecture from Moonshot AI that interleaves **Kimi Delta Attention
(KDA)** linear-attention layers with **Multi-head Latent Attention (MLA)**
full-attention layers. Like other Mamba / linear-attention hybrids, the KDA
layers keep a recurrent **state cache** (a convolution + delta-net state)
instead of a paged key/value cache; LMCache reinterprets that state as an opaque
page at registration time, so prefix caching and KV reuse work end to end. See
:doc:`../mp/hybrid_models` for the general handling of Mamba / linear-attention
models.

Validated models
----------------

- `moonshotai/Kimi-Linear-48B-A3B-Instruct
  <https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct>`_
  (48B-parameter MoE, 3B active; 2 GPUs)

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `Kimi-Linear in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture ``KimiLinearForCausalLM``).

      **Status:** Validated with LMCache.

      As a Mamba / linear-attention hybrid, Kimi-Linear needs the same three
      settings as the other GDN hybrids: the ``align`` Mamba cache mode, prefix
      caching, and a chunk size matched to vLLM's *unified block size* ``N``.
      That block size is model- and parallelism-specific — vLLM logs
      ``Setting attention block size to N tokens`` at startup. Because the KDA
      state is sharded across tensor-parallel ranks (while the MLA cache is
      not), ``N`` depends on ``--tensor-parallel-size``; read it from the log
      for your own configuration:

      .. list-table::
         :header-rows: 1
         :widths: 50 25 25

         * - Model
           - Unified block size ``N``
           - GPUs (TP)
         * - ``moonshotai/Kimi-Linear-48B-A3B-Instruct``
           - 944
           - 2

      Set the LMCache server's ``--chunk-size`` to that ``N`` (or a multiple of
      it), and vLLM's ``--max-num-batched-tokens`` to ``2N-1`` (the largest
      value below ``2N``).

      .. note::

         ``N`` **scales inversely with the tensor-parallel size** for this
         model, so changing ``--tensor-parallel-size`` changes both derived
         flags. The MLA cache is *replicated* across TP ranks (its per-rank
         size is fixed), while the KDA state is *sharded* (its per-rank size is
         divided by the TP degree). vLLM picks ``N`` so an attention page is at
         least as large as a KDA state page, so a smaller per-rank KDA state
         yields a smaller ``N`` — and vice versa. Concretely, going from
         **TP=2 → TP=1** doubles the per-rank KDA state, so ``N`` doubles from
         ``944`` to ``1888``; ``--chunk-size`` (``= N``) and
         ``--max-num-batched-tokens`` (``= 2N-1``, i.e. ``3775``) both double to
         match. Always re-read ``N`` from the startup log after changing the TP
         degree — do not scale it by hand.

      Start the LMCache MP server (``N = 944``):

      .. code-block:: bash

         lmcache server --chunk-size 944 --l1-size-gb 100 --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector
      (2 GPUs, ``N = 944`` → ``2N-1 = 1887``):

      .. code-block:: bash

         vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
             --tensor-parallel-size 2 \
             --trust-remote-code \
             --enable-prefix-caching \
             --mamba-cache-mode align \
             --max-num-batched-tokens 1887 \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      **Why these settings:**

      - ``--mamba-cache-mode align`` and ``--enable-prefix-caching`` are
        **required**. ``align`` is the only Mamba cache mode the KDA backend
        supports, and prefix caching must be on for LMCache to store and reuse
        the recurrent state.
      - ``--chunk-size`` (server) must be a multiple of the unified block size
        ``N`` — ``--chunk-size N`` is the simplest choice. LMCache raises at
        engine startup if it is not.
      - ``--max-num-batched-tokens`` must be in ``[N, 2N)``. ``align`` snapshots
        the KDA state only at the *end* of each scheduler step, on a block
        boundary, and the scheduler splits prefills into whole ``N``-token
        blocks — so the per-step token budget must cover at least one block but
        stay below two (a larger budget could advance past a boundary and leave
        no snapshot for LMCache to store). Prefer the maximum, ``2N-1``: a
        single request still advances exactly one block per step, and the spare
        ``N-1`` budget lets decodes co-schedule with a prefill block instead of
        serializing behind it. Setting it to exactly ``N`` makes the per-step
        budget one block, so once any request is decoding no new request can
        start prefill.
      - ``--trust-remote-code`` loads Kimi-Linear's custom modeling code.
      - ``--tensor-parallel-size 2`` shards the weights across two GPUs. Adjust
        it to your hardware — but note it changes ``N`` and the two derived
        flags (see the note above).

      No attention-backend or ``--no-disable-hybrid-kv-cache-manager`` flag is
      needed; ``LMCacheMPConnector`` advertises hybrid support and vLLM
      auto-selects the KDA and MLA backends. For the generic LMCache + vLLM
      wiring (ports, remote hosts), see :doc:`../getting_started/quickstart`.

   .. tab-item:: SGLang

      **Status:** Not validated with LMCache.

   .. tab-item:: TRT-LLM

      **Status:** Not validated with LMCache.

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

- Generation is **not guaranteed bit-exact** between a cached and a fresh run
  under concurrent load: KDA / GDN linear-attention backends do not support
  vLLM's batch-invariant mode, so kernel results can vary with batch
  composition. Validate with a score-level comparison, not a token-level diff.
- Cached pages for the KDA and MLA groups are byte-opaque views, so
  content-aware processing (CacheGen, CacheBlend) does not apply, and cache
  entries must not be shared across engines with different attention backends or
  kernel block sizes.
- vLLM's Mamba prefix caching in ``align`` mode is experimental upstream.
