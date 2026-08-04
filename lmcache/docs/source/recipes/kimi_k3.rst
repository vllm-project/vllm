.. _recipe_kimi_k3:

Kimi K3
=======

Moonshot AI's flagship hybrid Mixture-of-Experts model. Like
:doc:`Kimi-Linear <kimi_linear>`, it interleaves **Kimi Delta Attention (KDA)**
linear-attention layers with **Multi-head Latent Attention (MLA)**
full-attention layers. The KDA layers keep a recurrent **state cache** (a
convolution + delta-net state) instead of a paged key/value cache; LMCache
reinterprets that state as an opaque page at registration time, so prefix
caching and KV reuse work end to end. See :doc:`../mp/hybrid_models` for the
general handling of Mamba / linear-attention models.

Validated models
----------------

- `moonshotai/Kimi-K3 <https://huggingface.co/moonshotai/Kimi-K3>`_
  (8× NVIDIA B300)

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `Kimi K3 in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_.

      **Status:** Validated with LMCache. Verified on NVIDIA B300 (8 GPUs).

      .. note::

         **Version requirements.** Kimi K3 support has not landed in a stable
         vLLM release yet — use the upstream pre-release Docker image for now
         (a stable vLLM release with K3 support is expected to follow, likely
         0.26.1 or 0.27):

         .. code-block:: bash

            docker pull vllm/vllm-openai:kimi-k3

         On the LMCache side, use a **nightly build from 2026-07-27 or newer**
         (see :doc:`../getting_started/installation`); stable LMCache releases
         include K3 support starting from **0.5.3**.

      As a Mamba / linear-attention hybrid, Kimi K3 needs the same three
      settings as the other KDA / GDN hybrids: the ``align`` Mamba cache mode,
      prefix caching, and a chunk size matched to vLLM's *unified block size*
      ``N`` (see :ref:`mamba-block-size` for how ``N`` is determined). For
      Kimi K3 at ``--tensor-parallel-size 8`` the unified block size is
      ``N = 768``:

      .. list-table::
         :header-rows: 1
         :widths: 50 25 25

         * - Model
           - Unified block size ``N``
           - GPUs (TP)
         * - ``moonshotai/Kimi-K3``
           - 768
           - 8

      ``N`` depends on the tensor-parallel size (the KDA state is sharded
      across TP ranks while the MLA cache is not), so after changing
      ``--tensor-parallel-size`` always re-read ``N`` from vLLM's
      ``Setting attention block size to N tokens`` startup log line and
      re-derive ``--chunk-size`` and ``--max-num-batched-tokens`` from it.

      Start the LMCache MP server (``--chunk-size`` = ``N`` = 768):

      .. code-block:: bash

         lmcache server \
             --port 6555 \
             --chunk-size 768 \
             --max-workers 4 \
             --l1-size-gb 100 \
             --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector (8 GPUs):

      .. code-block:: bash

         export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1

         vllm serve moonshotai/Kimi-K3 \
             --trust-remote-code \
             --load-format dummy \
             --moe-backend auto \
             --gpu-memory-utilization 0.95 \
             --tensor-parallel-size 8 \
             --no-enable-flashinfer-autotune \
             --enable-auto-tool-choice \
             --tool-call-parser kimi_k3 \
             --reasoning-parser kimi_k3 \
             --enable-prefix-caching \
             --mamba-cache-mode align \
             --max-num-batched-tokens 1500 \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":6555}}'

      |

      **Why these settings** — LMCache-side (required):

      - ``--mamba-cache-mode align`` and ``--enable-prefix-caching`` are
        **required**. ``align`` is the only Mamba cache mode the KDA backend
        supports, and prefix caching must be on for LMCache to store and reuse
        the recurrent state.
      - ``--chunk-size`` (server) must be a multiple of the unified block size
        ``N = 768`` — ``--chunk-size N`` is the simplest choice. LMCache
        raises at engine startup if it is not.
      - ``--max-num-batched-tokens`` must be in ``[N, 2N)`` — here
        ``[768, 1536)``; the validated value is ``1500``. ``align`` snapshots
        the KDA state only on a block boundary at the end of a scheduler step,
        so the per-step token budget must cover at least one block but stay
        below two. See :doc:`../mp/hybrid_models` for the full rationale.
      - The server's ``--port 6555`` must match ``lmcache.mp.port`` in the
        connector config.

      Model-side (Kimi K3 serving requirements):

      - ``VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1`` enables the fused
        latent-MoE tail path for Kimi K3.
      - ``--tool-call-parser kimi_k3``, ``--reasoning-parser kimi_k3``, and
        ``--enable-auto-tool-choice`` wire up K3's tool-calling and reasoning
        output formats.
      - ``--trust-remote-code`` loads Kimi K3's custom modeling code;
        ``--moe-backend auto`` lets vLLM pick the MoE kernel backend;
        ``--no-enable-flashinfer-autotune`` skips FlashInfer autotuning at
        startup.
      - ``--load-format dummy`` initializes random weights so the serving
        stack can be validated without downloading the full checkpoint —
        **drop it to serve the real weights**.
      - ``--tensor-parallel-size 8`` shards the weights across eight GPUs.
        Adjust it to your hardware — but note it changes ``N`` and the two
        derived flags (see above).

      No attention-backend or ``--no-disable-hybrid-kv-cache-manager`` flag is
      needed; ``LMCacheMPConnector`` advertises hybrid support and vLLM
      auto-selects the KDA and MLA backends. For the generic LMCache + vLLM
      wiring (ports, remote hosts), see :doc:`../getting_started/quickstart`.

      If there are any issues with vLLM setup, please refer to the
      `vLLM Recipes <https://docs.vllm.ai/projects/recipes/en/latest/index.html>`_
      for more details.

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

- **Pre-release engine support.** Until vLLM ships K3 in a stable release, pin
  both sides: the ``vllm/vllm-openai:kimi-k3`` Docker image on the vLLM side
  and an LMCache nightly from 2026-07-27 or newer (stable from 0.5.3).
- Generation is **not guaranteed bit-exact** between a cached and a fresh run
  under concurrent load: KDA / GDN linear-attention backends do not support
  vLLM's batch-invariant mode, so kernel results can vary with batch
  composition. Validate with a score-level comparison, not a token-level diff.
- Cached pages for the KDA and MLA groups are byte-opaque views, so
  content-aware processing (CacheGen, CacheBlend) does not apply, and cache
  entries must not be shared across engines with different attention backends
  or kernel block sizes.
- vLLM's Mamba prefix caching in ``align`` mode is experimental upstream.
