Hidden states
=============

The **HiddenStateStore** caches per-token *hidden-state* tensors
(intermediate activations from an inference forward pass) alongside
the KV cache, keyed by the same chunk keys. It exists to support any
system where a downstream stage needs an upstream stage's
intermediate activations for the cached prefix and cannot
reconstruct them from KV alone — for example a vLLM-Omni
``thinker -> talker`` pipeline, where the talker consumes the
thinker's hidden states for tokens whose KV the engine restored
without re-running prefill.

Without this store, restoring KV alone causes downstream stages to
see truncated or wrong activations for the cached prefix. With it,
the talker (or any equivalent consumer) gets a contiguous
``[num_cached_prefix_tokens, hidden_dim]`` tensor matching the KV
prefix LMCache restored.

Enabling it
-----------

Set three keys in the LMCache YAML (``LMCACHE_CONFIG_FILE``):

.. code-block:: yaml

   chunk_size: 256

   enable_hidden_state_cache: true
   max_hidden_state_cpu_size: 4          # GiB, pinned CPU pool size
   # hidden_state_layers: [0, 1]         # optional allowlist (see below)

- ``enable_hidden_state_cache`` — master switch. When ``false`` (the
  default), ``engine.hidden_state_store`` is ``None`` and integrators
  must skip all HS APIs on that worker.
- ``max_hidden_state_cpu_size`` — pinned-CPU pool size in GiB,
  dedicated to hidden states. Must be ``> 0`` when the store is
  enabled. The pool is **independent** of the KV pool; HS allocator
  pressure never evicts KV.
- ``hidden_state_layers`` — optional allowlist of ``layer_idx``
  values accepted on ``store_hidden_states``. Leave unset to accept
  all layer indices. Use this only when you know exactly which
  hook indices the worker writes (e.g. matching a fork's hybrid
  multimodal hook list).

Using it from a worker
----------------------

The store is exposed on the engine (not as engine-level methods):

.. code-block:: python

   hs = engine.hidden_state_store
   if hs is None:
       return  # HS caching disabled; nothing to do

   # Store: hidden_states corresponds to token_ids[token_offset:]
   hs.store_hidden_states(
       token_ids,                # full prefix (same as used for KV)
       hidden_states,            # [len(token_ids) - token_offset, hidden_dim]
       layer_idx=0,
       token_offset=num_computed,  # 0 for non-incremental callers
   )

   # Retrieve: contiguous prefix tensor or None on full miss
   restored = hs.retrieve_hidden_states(token_ids, layer_idx=0)

Notes:

- ``token_ids`` is always the **full** prefix so chunk keys align
  with KV exactly. ``token_offset`` lets incremental callers (such
  as vLLM-Omni) pass only the newly computed rows in
  ``hidden_states`` without zero-padding the already-cached prefix.
- ``layer_idx`` is the *storage* layer index. Callers that need
  several intermediate tensors per request (e.g. multimodal hook
  outputs *plus* the main text hidden state) call
  ``store_hidden_states`` once per distinct ``layer_idx`` and
  retrieve per layer on restore.
- ``retrieve_hidden_states`` is **prefix-strict**: it returns the
  longest contiguous CPU ``float32`` prefix where every chunk has
  both KV and HS for the requested ``layer_idx``, and stops at the
  first chunk where either is missing.

Eviction model
--------------

The store implements a **coupled-but-asymmetric** eviction rule:

- **KV evicted ⇒ HS evicted.** On every ``retrieve_hidden_states``
  the store asks the KV ``StorageManager`` whether KV is still
  present for each chunk key. If KV is gone, the orphan HS entry
  is dropped and the returned prefix ends there.
- **HS evicted ⇏ KV evicted.** Pressure on the HS pinned pool only
  evicts HS entries (its own LRU), never KV.
- **Restore stops at first missing HS or KV chunk** for the
  requested layer.

Design notes (user-visible)
---------------------------

- **Separate pinned pool from KV.** KV tensors and hidden-state
  tensors have heterogeneous shapes and dtypes, so they get
  independent allocators. ``max_hidden_state_cpu_size`` sizes the
  HS pool only.
- **Same chunk keys as KV.** The store reuses the engine's
  ``TokenDatabase`` so HS chunks share the exact ``CacheEngineKey``
  as the matching KV chunks. This is what enables the lazy coupled
  eviction check.
- **Lazy coupled-eviction check.** No callbacks or shared indices
  are required between KV and HS — the store asks
  ``storage_manager.contains(key)`` per chunk on retrieve. Works
  uniformly for any backend that implements ``contains()``.

The full internal design (class layering, code paths, follow-up
work) lives at :file:`docs/design/v1/hidden_state_store.md` in the
source tree.
