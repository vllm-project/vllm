Hybrid Attention Models
=======================

Some models interleave more than one attention type across their layers — most
commonly **sliding-window attention** on most layers and **full attention** on a
few. vLLM serves these with its *hybrid KV cache manager*, which splits the
model's layers into multiple **KV cache groups** (one per attention behavior).

The LMCache multiprocess connector (``LMCacheMPConnector``) supports these
hybrid models: it stores and retrieves the KV cache for every group, so prefix
caching and KV reuse work the same way they do for plain models.

.. contents::
   :local:
   :depth: 2

Validated hybrid models
-----------------------

Recipe pages for the validated hybrid-attention architectures:

.. list-table::
   :header-rows: 1
   :widths: 34 34 32

   * - Model
     - Attention layout
     - Recipe
   * - Gemma 3
     - Sliding-window + full
     - :doc:`/recipes/gemma3`
   * - Gemma 4
     - Sliding-window + full
     - :doc:`/recipes/gemma4`
   * - gpt-oss
     - Sliding-window + full
     - :doc:`/recipes/gpt_oss`
   * - Qwen3.5 / Qwen3.6 series
     - Mamba / GDN + full
     - :doc:`/recipes/qwen3_5`
   * - Kimi-Linear
     - KDA linear-attention + MLA full
     - :doc:`/recipes/kimi_linear`
   * - Kimi K3
     - KDA linear-attention + MLA full
     - :doc:`/recipes/kimi_k3`
   * - DeepSeek-V4-Flash
     - Sparse-MLA (multiple KV groups)
     - :doc:`/recipes/deepseek_v4_flash`
   * - GLM 5.1/5.2
     - Dynamic Sparse Attention (multiple KV groups)
     - :doc:`/recipes/glm5_2`
   * - MiniMax-M3
     - Sparse attention + lightning indexer (mixed KV formats in one group)
     - :doc:`/recipes/minimax_m3`

.. toctree::
   :hidden:
   :maxdepth: 1

   /recipes/gemma3
   /recipes/gemma4
   /recipes/gpt_oss
   /recipes/qwen3_5
   /recipes/kimi_linear
   /recipes/kimi_k3
   /recipes/deepseek_v4_flash
   /recipes/glm5_2
   /recipes/minimax_m3

What Works
----------

Models whose layers all use **standard paged attention** — including hybrids
that mix sliding-window and full attention — are supported with no special
configuration. Examples:

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Model family
     - Attention layout
     - Status
   * - Gemma 2 / Gemma 3
     - Interleaved sliding-window + full
     - Supported
   * - gpt-oss
     - Interleaved sliding-window + full
     - Supported
   * - Qwen3.5 (and other Gated-DeltaNet hybrids)
     - Interleaved Mamba/GDN + full
     - Supported (see below)
   * - Llama, Qwen2/Qwen3 (dense), Mistral, …
     - Single attention type
     - Supported

Just point vLLM at the LMCache server as usual (see :doc:`/getting_started/quickstart`); LMCache
detects the model's KV cache groups automatically at registration time.

.. note::

   Because ``LMCacheMPConnector`` advertises hybrid support to vLLM, vLLM keeps
   its hybrid KV cache manager **enabled** for these models (it does not fall
   back to a single unified group). You do not need
   ``--no-disable-hybrid-kv-cache-manager`` or any related flag.

Object-group separation
-----------------------

At KV-cache registration LMCache buckets a hybrid model's layers into **object
groups** — the unit it stores and retrieves as one object. By default
(``--separate-object-groups``, on) each distinct cross-chunk attention window
becomes its own object group: full-attention layers form one group, and each
sliding-window size (mamba / GDN included) forms another. Pass
``--no-separate-object-groups`` to keep every layer in a single full-attention
object group instead (the previous behavior).

.. code-block:: bash

   # default: one object group per attention window
   lmcache server --chunk-size 256 --l1-size-gb 100

   # opt out: a single full-attention object group for all layers
   lmcache server --chunk-size 256 --l1-size-gb 100 --no-separate-object-groups

The flag is transparent to correctness — prefix caching and KV reuse behave the
same either way, and a non-hybrid model (a single attention behavior) always
resolves to one object group regardless of the setting. Separation organizes
storage by attention window so that each group's cross-chunk window is tracked
independently.

Mamba / Linear-Attention Hybrids
--------------------------------

Models that interleave **Mamba / Gated-DeltaNet (GDN) linear-attention layers**
with full attention — the Qwen3.5 and Qwen3.6 series (``Qwen/Qwen3.5-0.8B``,
``Qwen/Qwen3.6-27B``, …), Qwen3-Next, Kimi-Linear
(``moonshotai/Kimi-Linear-48B-A3B-Instruct``), Kimi K3
(``moonshotai/Kimi-K3``), and other GDN hybrids — are supported.
Unlike a paged key/value cache, their linear-attention layers keep a recurrent
**state cache** (a convolution + SSM state). LMCache reinterprets that state as
an opaque page at registration time, so prefix caching and KV reuse work end to
end without any model-specific transfer code.

This section is the **general procedure for any such model**. The only
per-model variable is the *unified block size* ``N`` (step 1); everything else
is identical across models.

.. _mamba-block-size:

Step 1 — find the model's unified block size ``N``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``N`` is the **single number** that drives every other setting: the LMCache
server's ``--chunk-size`` and vLLM's ``--max-num-batched-tokens`` are both
derived from it (step 2). Get it wrong and LMCache raises at engine startup.

For a Mamba / GDN hybrid, vLLM forces **one** block size across all KV cache
groups, chosen large enough that an attention page is at least as big as a
Mamba state page. It depends on the model's head dimensions and GDN state size,
so it is **model-specific — never assume a value, read it from the model**.
vLLM prints it once at startup::

    INFO ... interface.py:670] Setting attention block size to 784 tokens to
    ensure that attention page size is >= mamba page size.

You do not need LMCache, a full serving run, or the weights to be quantized to
read it — just launch vLLM until the line appears, then stop. The snippet below
does exactly that and prints ``N``:

.. code-block:: bash

   MODEL=Qwen/Qwen3.6-27B
   LOG=$(mktemp)

   # Launch vLLM just far enough to size the KV cache; cheap settings only.
   vllm serve "$MODEL" \
       --mamba-cache-mode align --enable-prefix-caching \
       --max-model-len 8192 --gpu-memory-utilization 0.5 \
       --port 8011 > "$LOG" 2>&1 &
   VLLM_PID=$!

   # Wait for the block-size line (or a fatal error), then stop vLLM.
   until grep -qiE "Setting attention block size|Error|Traceback" "$LOG"; do
       sleep 3
   done
   grep -i "Setting attention block size" "$LOG"
   kill "$VLLM_PID"

The number in ``to N tokens`` is your ``N``. Values grow with model size; for
example:

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
   * - ``moonshotai/Kimi-Linear-48B-A3B-Instruct``
     - 944
     - 2
   * - ``moonshotai/Kimi-K3``
     - 768
     - 8

Step 2 — derive the three required flags from ``N``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. **LMCache server** ``--chunk-size`` **= N** (or any multiple of ``N``). This
   is the rule the connector enforces: LMCache's chunk size must be a multiple
   of vLLM's unified block size, or registration fails::

       lmcache server --chunk-size 784 --l1-size-gb 100 --eviction-policy LRU

#. **vLLM** ``--max-num-batched-tokens`` **in [N, 2·N)** — setting it equal to
   ``N`` is the simple, always-valid choice. Outside this range LMCache raises
   at engine startup. ``align`` mode snapshots the Mamba state only at the
   *end* of each scheduler step, so each prefill step must advance exactly one
   block; a larger budget would let a step skip block boundaries, leaving no
   snapshot for LMCache to store at those prefixes.

#. **vLLM** ``--mamba-cache-mode align --enable-prefix-caching`` — ``align`` is
   mandatory (GDN backends do not support the ``all`` mode)::

       vllm serve <model> \
           --enable-prefix-caching --mamba-cache-mode align \
           --max-num-batched-tokens 784 \
           --kv-transfer-config \
           '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

So for a freshly-probed model the whole derivation is just: read ``N`` (step 1),
then pass ``--chunk-size N`` to the server and ``--max-num-batched-tokens N`` to
vLLM.

No ``--no-disable-hybrid-kv-cache-manager`` or attention-backend flag is needed;
``LMCacheMPConnector`` advertises hybrid support and vLLM auto-selects the GDN
backend.

Caveats
^^^^^^^

- Generation is **not bit-exact** between a cached and a fresh run: GDN
  backends do not support vLLM's batch-invariant mode. Validate with a
  **score-level** comparison (see `Verifying Correctness`_), not a token-level
  diff.
- The cached pages are **byte-opaque**, so content-aware features (CacheGen
  compression, CacheBlend) do not apply, and cache entries must not be shared
  across engines with different attention backends or kernel block sizes.
- Several of these models are **vision-language** (they load a vision tower).
  The validated, supported path is **text** KV caching; image/video KV caching
  is not validated.
- vLLM's Mamba prefix caching in ``align`` mode is marked experimental upstream.

See the :doc:`Qwen3.5 / Qwen3.6 recipe <../recipes/qwen3_5>` for the validated
end-to-end commands and the per-model block sizes.

Verifying Correctness
---------------------

To convince yourself that a hybrid model's KV is being cached and reused
correctly, you can compare a cold run against a run served from LMCache:

#. Run an evaluation (e.g. ``lm_eval`` on ``gsm8k``) against vLLM + LMCache.
   This computes the KV cache and **stores** it in LMCache.
#. Reset *only* vLLM's local prefix cache, leaving the LMCache-managed cache
   intact (requires launching vLLM with ``VLLM_SERVER_DEV_MODE=1``)::

       curl -X POST http://localhost:8000/reset_prefix_cache

   Omit the ``reset_external=true`` query parameter so the LMCache cache is
   preserved.
#. Re-run the same evaluation. vLLM now misses in its local cache, so the prefix
   KV is **retrieved** from LMCache. The score should match the first run.

The project ships this as the ``hma_lm_eval`` continuous-integration test (see
``.buildkite/k3_tests/multiprocess``).

See Also
--------

- :doc:`/getting_started/quickstart` — launching the LMCache server and a vLLM client.
- Design notes on how groups are detected and addressed:
  ``docs/design/integration/vllm/hybrid-kv-cache-groups.md`` in the source tree.
