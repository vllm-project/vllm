# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 38 — TurboQuant `_continuation_prefill` memory fix.

Problem
-------
On vLLM v0.19.2rc1.dev134+ the `TurboQuantAttentionImpl._continuation_prefill`
method allocates ~500 MiB of TRANSIENT FP16 tensors for a single deep-prefix
chunk (prod config: Qwen3.6-35B-A3B-FP8, 2×A5000, max-model-len 262144,
`kv-cache-dtype=turboquant_k8v4`):

  * `k_cached_trim = ... .contiguous()`  — ~128 MiB FP16 copy of dequant K
  * `v_cached_trim = ... .contiguous()`  — ~128 MiB FP16 copy of dequant V
  * `k_full = torch.cat([k_cached_trim, key_chunk])` — ~128 MiB new alloc
  * `v_full = torch.cat([v_cached_trim, val_chunk])` — ~128 MiB new alloc

At prefix ≥ 99k on a 22.86 GiB saturated GPU these transients collide with
fragmentation overhead and fail — reproducible on two integration runs as
`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 52.00 MiB.
GPU 0 has ... 27.94 MiB is free` at `turboquant_attn.py:776 v_full =
torch.cat(...)`.

P22 (legacy) attempted to pre-allocate dequant buffers with a 3-D shape
`(Hk, D, max_alloc_len)`, but the dev134 engine uses 4-D slicing
`k_buf[:, :, :alloc_len, :]` which raises IndexError on a 3-D tensor —
effectively our P22 prealloc was being ignored and the engine was
falling back to its own lazy `torch.empty((1, Hk, alloc_len, D), ...)`.

Fix (this module)
-----------------
1. Correctly-shaped 4-D K/V dequant pre-allocs (attached by
   `vllm._genesis.kernels.dequant_buffer.ensure_turboquant_buffers` via
   `TurboQuantBufferManager.get_or_create_p38_dequant_4d`). Shape
   `(1, Hk, max_alloc_len, D)` — matches dev134's internal expectation
   so the `k_buf.shape[2] < alloc_len` fallback check resolves False
   and our prealloc is the one that gets used.

2. Persistent shared K_full / V_full workspace of shape
   `(max_model_len + max_num_batched_tokens, Hk, D)` FP16, attached as
   `layer._tq_k_full_buf` / `layer._tq_v_full_buf`. Shared across all
   TQ layers (sequential forward) → single pair per model.

3. Re-implementation of `_continuation_prefill` (this module's
   `_genesis_continuation_prefill`) that:
     - uses the prealloced 4-D K/V dequant buffers (step 1) instead of
       per-call `torch.empty`;
     - skips BOTH `.contiguous()` copies by writing the dequant prefix
       directly into K_full / V_full via `copy_()` (PyTorch's copy_
       accepts non-contiguous source → contiguous destination);
     - skips BOTH `torch.cat` calls by appending the per-chunk K/V
       directly into the shared workspace's `[cached_len:seq_len]`
       slice;
     - reuses P32's `_tq_cu_q`/`_tq_cu_k` scratch when present to avoid
       the per-call `torch.tensor([0, q_len])` host→device transfer
       (prod already runs P32).

Net budget on prod config (Qwen3.6-35B-A3B-FP8, TP=2, max_model_len
262144, max_num_batched_tokens 4096):

  P38 persistent memory per rank:
    K_dequant 4-D:   128 MiB  (1, 2, 262144, 128) fp16
    V_dequant 4-D:   128 MiB
    K_full:          130 MiB  (266240, 2, 128) fp16 (max_model_len + max_bt + align)
    V_full:          130 MiB
    ------------
    Total:           516 MiB persistent per rank

  Previous transient PEAK per rank (on deep prefix continuation):
    k_cached_trim:   128 MiB
    v_cached_trim:   128 MiB
    k_full:          128 MiB
    v_full:          128 MiB
    ------------
    Peak:            512 MiB transient (plus allocator fragmentation)

  Peak GPU usage: same bytes but PERSISTENT and PROFILER-VISIBLE. Because
  vLLM's memory profiler measures `max_memory_allocated()` after warmup,
  moving a transient peak into a persistent allocation shrinks KV cache
  capacity by the same amount → predictable KV sizing, no fragmentation
  collision. Concretely this lets us raise `--gpu-memory-utilization`
  from 0.80 back toward 0.92-0.94 at `--max-num-batched-tokens 4096`
  while still fitting 262k max-model-len single-request stably.

Compatibility
-------------
- NVIDIA CUDA SM 8.0+: applied.
- AMD / CPU / pre-Ampere: skipped (TurboQuant itself is CUDA-only).
- Non-flash-attn fallback (SDPA): replacement still writes through the
  persistent buffers — just uses `F.scaled_dot_product_attention` on
  the same views instead of `flash_attn_varlen`.
- `not tq_config.key_fp8` (MSE-quant key) branch: uses in-place copy
  from `k_cached[0, :, :cached_len, :].transpose(0, 1)` into
  `k_full[:cached_len]` via a single `copy_()` that PyTorch handles as a
  non-contiguous → contiguous transfer without extra alloc. The FP32
  rotation itself (`@ Pi`) still allocates — that's out of scope here
  (turboquant_k8v4 path has `key_fp8=True` and skips the rotation).

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.2 implementation
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
import torch.nn.functional as F

from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least

log = logging.getLogger("genesis.wiring.p38_tq_continuation_memory")

_GENESIS_P38_MARKER_ATTR = "_genesis_p38_wrapped"

# v7.48 (2026-04-27): one-time mode-decision log gate.
_P38_MODE_LOGGED = False

# Class-name candidate list — mirrors P22. If upstream renames, add here.
_CANDIDATE_TQ_IMPL_NAMES = (
    "TurboQuantAttentionImpl",
)


def should_apply() -> bool:
    """Match TurboQuantBufferManager gates."""
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return True


def _import_tq_impl() -> Any | None:
    """Resolve the TurboQuant attention impl class or return None."""
    try:
        import importlib
        mod = importlib.import_module(
            "vllm.v1.attention.backends.turboquant_attn"
        )
    except ImportError as e:
        log.info("[Genesis P38] turboquant_attn module not importable: %s", e)
        return None
    except Exception as e:
        log.warning("[Genesis P38] unexpected error importing TQ module: %s", e)
        return None
    for name in _CANDIDATE_TQ_IMPL_NAMES:
        impl = getattr(mod, name, None)
        if impl is not None:
            return impl
    log.info(
        "[Genesis P38] none of %s found in turboquant_attn",
        list(_CANDIDATE_TQ_IMPL_NAMES),
    )
    return None


def _resolve_flash_attn_available() -> bool:
    """Probe the same `_HAS_FLASH_ATTN` used by upstream to stay in sync."""
    try:
        import importlib
        mod = importlib.import_module(
            "vllm.v1.attention.backends.turboquant_attn"
        )
        return bool(getattr(mod, "_HAS_FLASH_ATTN", False))
    except Exception:
        return False


def _resolve_triton_kernel():
    """Return the `_tq_full_dequant_kv` Triton kernel or None."""
    try:
        import importlib
        mod = importlib.import_module(
            "vllm.v1.attention.backends.turboquant_attn"
        )
        return getattr(mod, "_tq_full_dequant_kv", None)
    except Exception:
        return None


def _resolve_fp8_e4b15():
    """Return the `_use_fp8_e4b15` helper or None."""
    try:
        import importlib
        mod = importlib.import_module(
            "vllm.v1.attention.backends.turboquant_attn"
        )
        return getattr(mod, "_use_fp8_e4b15", None)
    except Exception:
        return None


def _genesis_continuation_prefill(
    self,
    layer: Any,
    query: torch.Tensor,   # (q_len, Hq, D)
    key_chunk: torch.Tensor,  # (q_len, Hk, D)
    val_chunk: torch.Tensor,  # (q_len, Hk, D)
    kv_cache: torch.Tensor,   # (num_blocks, block_size, Hk, slot_size)
    block_table: torch.Tensor,  # (1, max_num_blocks)
    cached_len: int,
    seq_len: int,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Genesis P38 replacement for `TurboQuantAttentionImpl._continuation_prefill`.

    Behaviour-identical to upstream dev134 except:
      * K/V dequant buffers are 4-D persistent pools (prealloc'd by P22/P38,
        not `torch.empty` per call);
      * K_full/V_full are persistent shared pools filled in-place — no
        `.contiguous()` copies and no `torch.cat` peaks.

    Raises nothing that upstream wouldn't; on any unexpected state (e.g.
    pool capacity exceeded — shouldn't happen since sizing is
    max_model_len + max_num_batched_tokens), falls back to the upstream
    allocation pattern so correctness is preserved.
    """
    import triton

    q_len, Hq, D = query.shape
    Hk = key_chunk.shape[1]
    device = query.device
    block_size = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)

    mse_bytes = self._mse_bytes
    val_data_bytes = self._val_data_bytes

    # ── 1. Dequant buffers (prealloc 4-D, matching engine shape exactly) ──
    alloc_len = math.ceil(cached_len / block_size) * block_size
    buf_shape = (1, Hk, alloc_len, D)

    # v7.48 (2026-04-27): SHARED vs PER-LAYER toggle via GENESIS_BUFFER_MODE.
    # Default "shared" — single buffer pair across all 36 attention layers
    # (sequential exec ⇒ safe). Saves ~13-25 GB on long-context. Set
    # GENESIS_BUFFER_MODE=per_layer (or _MODE_P38=per_layer) for legacy
    # per-layer attached buffers if the shared path ever regresses.
    from vllm._genesis.buffer_mode import buffer_mode_for, log_mode_decision
    mode = buffer_mode_for("P38")
    global _P38_MODE_LOGGED
    if not _P38_MODE_LOGGED:
        log_mode_decision("P38",  mode,
                          "K/V dequant + K_full/V_full workspace pool")
        _P38_MODE_LOGGED = True

    if mode == "shared":
        # v7.48 fix: ONE namespace per (Hk, D, dtype, device) — max-size
        # alloc at first use, then slice. Earlier rounding strategy plowed
        # multiple registry entries (one per rounded_len) which never GC'd
        # — defeated the purpose of singleton pool. Now we allocate ONCE
        # under TQ_MAX_MODEL_LEN and reuse forever via slicing.
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
        # Maximum size we'll ever need (env-controllable; defaults to engine
        # max_model_len). Rounded to block_size so dequant kernel grid math
        # stays clean.
        _max_env = os.environ.get("GENESIS_TQ_MAX_MODEL_LEN", "262144")
        try:
            max_tq_len = int(_max_env)
        except ValueError:
            max_tq_len = 262144
        max_tq_len = math.ceil(max_tq_len / block_size) * block_size
        # Single namespace per layer-shape signature — same across all 36
        # attention layers because Hk/D/dtype/device identical for them.
        ns_k = f"p38_k_dequant|Hk={Hk}|D={D}|fp16"
        ns_v = f"p38_v_dequant|Hk={Hk}|D={D}|fp16"
        max_shape = (1, Hk, max_tq_len, D)
        k_buf = GPB.get_or_create(ns_k, max_shape, torch.float16, device,
                                  zero_init=False)
        v_buf = GPB.get_or_create(ns_v, max_shape, torch.float16, device,
                                  zero_init=False)
    else:
        # Legacy per-layer path — retained for rollback safety.
        k_buf = getattr(layer, "_tq_k_dequant_buf", None)
        v_buf = getattr(layer, "_tq_v_dequant_buf", None)
        # P49 runtime shape-compat check (v7.8): if buffers exist but have
        # unexpected Hk / D / ndim, fall back to fresh alloc. Covers the
        # "same TurboQuantAttentionImpl class name, different shape
        # convention" drift scenario (different model, upstream refactor).
        need_fresh = (
            k_buf is None or v_buf is None
            or k_buf.dim() != 4 or v_buf.dim() != 4
            or k_buf.shape[1] != Hk        # v7.8: Hk mismatch
            or k_buf.shape[3] != D         # v7.8: D mismatch
            or k_buf.shape[2] < alloc_len
        )
        if need_fresh:
            # Fresh 4-D alloc — mirrors upstream exactly.
            k_buf = torch.empty(buf_shape, dtype=torch.float16, device=device)
            v_buf = torch.empty(buf_shape, dtype=torch.float16, device=device)
            layer._tq_k_dequant_buf = k_buf
            layer._tq_v_dequant_buf = v_buf

    k_cached = k_buf[:, :, :alloc_len, :].zero_()
    v_cached = v_buf[:, :, :alloc_len, :].zero_()

    # ── 2. Triton dequant kernel — unchanged from upstream ──
    grid = (alloc_len, 1 * Hk)
    tq_kernel = _resolve_triton_kernel()
    fp8_e4b15_fn = _resolve_fp8_e4b15()
    if tq_kernel is None or fp8_e4b15_fn is None:
        # Can't proceed — fall back to original method. This hits only
        # if upstream moved the kernel symbol. Import the original via
        # the saved reference attached to our wrapper.
        # Use `getattr(cls, name, None)` on the class (walks MRO) rather
        # than `self.__class__.__dict__.get(name)` which sees only the
        # direct class and returns None on subclasses — which would then
        # AttributeError on the `_genesis_p38_original` lookup.
        impl_method = getattr(
            self.__class__, "_continuation_prefill", None,
        )
        original = getattr(impl_method, "_genesis_p38_original", None)
        if original is None:
            raise RuntimeError(
                "[Genesis P38] Triton kernel / fp8 helper not resolvable AND "
                "no saved original method — engine is in an unexpected "
                "state."
            )
        return original(
            self, layer, query, key_chunk, val_chunk, kv_cache,
            block_table, cached_len, seq_len, Pi, centroids,
        )

    tq_kernel[grid](
        kv_cache,
        block_table,
        centroids,
        k_cached,
        v_cached,
        k_cached.stride(0),
        k_cached.stride(1),
        k_cached.stride(2),
        v_cached.stride(0),
        v_cached.stride(1),
        v_cached.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_HEADS=Hk,
        MSE_BYTES=mse_bytes,
        KPS=self.tq_config.key_packed_size,
        VQB=self.tq_config.effective_value_quant_bits,
        VAL_DATA_BYTES=val_data_bytes,
        MSE_BITS=self.tq_config.key_mse_bits,
        KEY_FP8=1 if self.tq_config.key_fp8 else 0,
        BLOCK_D=BLOCK_D,
        NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
        FP8_E4B15=fp8_e4b15_fn(device.index or 0),
        num_warps=4,
    )

    # ── 3. Persistent K_full / V_full workspace ──
    qdtype = query.dtype

    # v7.48 (2026-04-27): same SHARED vs PER-LAYER toggle for the K_full /
    # V_full workspace pair (shape `(seq_len, Hk, D)`). On Qwen3.6-MoE this
    # is the larger of the two prealloc footprints (full sequence × heads
    # × dim, fp16). Shared singleton via GenesisPreallocBuffer is the
    # default; per-layer attached attribute path retained for rollback.
    if mode == "shared":
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
        # ONE namespace per shape signature — max-size alloc once, slice forever.
        # Use the same TQ_MAX env as dequant buffers above for consistency.
        ns_k_full = f"p38_k_full|Hk={Hk}|D={D}|{qdtype}"
        ns_v_full = f"p38_v_full|Hk={Hk}|D={D}|{qdtype}"
        full_max_shape = (max_tq_len, Hk, D)
        k_full_buf = GPB.get_or_create(ns_k_full, full_max_shape, qdtype, device,
                                       zero_init=False)
        v_full_buf = GPB.get_or_create(ns_v_full, full_max_shape, qdtype, device,
                                       zero_init=False)
        use_persistent = True
    else:
        k_full_buf = getattr(layer, "_tq_k_full_buf", None)
        v_full_buf = getattr(layer, "_tq_v_full_buf", None)
        use_persistent = (
            k_full_buf is not None and v_full_buf is not None
            and k_full_buf.shape[0] >= seq_len
            and k_full_buf.shape[1] == Hk and k_full_buf.shape[2] == D
            and k_full_buf.dtype == qdtype
            and k_full_buf.device == device
        )

    if use_persistent:
        # Take views; no new allocation.
        k_full = k_full_buf[:seq_len]
        v_full = v_full_buf[:seq_len]

        # Prefix: dequant'd K/V — handle the MSE-rotation branch too.
        if not self.tq_config.key_fp8:
            # FP32 rotation still allocates a temporary (cached_len × Hk, D)
            # FP32 tensor — out of scope for P38 (k8v4 hot path skips this).
            # We route through the same `copy_` path as the fp8 branch.
            k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D).float()
            k_flat = k_flat @ Pi
            k_rotated = (
                k_flat.to(torch.float16).reshape(Hk, cached_len, D).transpose(0, 1)
            )
            # k_rotated is already (cached_len, Hk, D) FP16 — copy into workspace.
            k_full[:cached_len].copy_(k_rotated)
            # Free FP32 temps eagerly — not strictly necessary (refcount=0
            # after exit of local scope), but CPython's GC is best-effort.
            del k_flat, k_rotated
        else:
            # (1, Hk, cached_len, D) → transpose(0, 1) → (Hk, 1, cached_len, D)
            # Actually upstream does k_cached[0, :, :cached_len, :] → (Hk,
            # cached_len, D); then .transpose(0, 1) → (cached_len, Hk, D).
            # copy_ accepts non-contiguous source → contiguous destination.
            src = k_cached[0, :, :cached_len, :].transpose(0, 1)
            k_full[:cached_len].copy_(src)

        src_v = v_cached[0, :, :cached_len, :].transpose(0, 1)
        v_full[:cached_len].copy_(src_v)

        # Chunk: new K/V — copy into workspace[cached_len:seq_len].
        k_full[cached_len:seq_len].copy_(key_chunk)
        v_full[cached_len:seq_len].copy_(val_chunk)
    else:
        # Fallback: pool not wired (e.g. AMD/CPU tests) — used to do
        # `.contiguous()` + torch.cat. v7.51.2: replaced with the same
        # pre-allocate-then-slice pattern from upstream PR vllm#40941
        # (and from our main `use_persistent` branch above) to avoid the
        # double-alloc spike of torch.cat. Behaviour-equivalent; just
        # eliminates one allocation peak per call.
        if not self.tq_config.key_fp8:
            k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D).float()
            k_flat = k_flat @ Pi
            k_cached_trim = (
                k_flat.to(torch.float16).reshape(Hk, cached_len, D).transpose(0, 1)
            )
        else:
            k_cached_trim = (
                k_cached[0, :, :cached_len, :].transpose(0, 1).contiguous()
            )
        v_cached_trim = (
            v_cached[0, :, :cached_len, :].transpose(0, 1).contiguous()
        )
        # Pre-allocate full-length workspace once; copy cached prefix and
        # new chunk into the right slices. No torch.cat allocation peak.
        k_full = torch.empty((seq_len, Hk, D), dtype=qdtype, device=device)
        v_full = torch.empty((seq_len, Hk, D), dtype=qdtype, device=device)
        k_full[:cached_len].copy_(k_cached_trim.to(qdtype))
        k_full[cached_len:seq_len].copy_(key_chunk)
        v_full[:cached_len].copy_(v_cached_trim.to(qdtype))
        v_full[cached_len:seq_len].copy_(val_chunk)

    # ── 4. Attention — flash-attn fast path or SDPA fallback ──
    if _resolve_flash_attn_available():
        # Reuse P32 cu_seqlens scratch if attached; else fresh tensors
        # (small, 2 × int32 = 8 bytes — negligible churn).
        cu_q = getattr(layer, "_tq_cu_q", None)
        cu_k = getattr(layer, "_tq_cu_k", None)
        if cu_q is not None and cu_k is not None:
            # In-place write — pointer stable, CUDA-graph safe.
            cu_q[0] = 0
            cu_q[1] = q_len
            cu_k[0] = 0
            cu_k[1] = seq_len
            cu_seqlens_q = cu_q
            cu_seqlens_k = cu_k
        else:
            cu_seqlens_q = torch.tensor(
                [0, q_len], device=device, dtype=torch.int32,
            )
            cu_seqlens_k = torch.tensor(
                [0, seq_len], device=device, dtype=torch.int32,
            )
        return self._flash_attn_varlen(
            q=query,
            k=k_full,
            v=v_full,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=q_len,
            max_seqlen_k=seq_len,
        )

    # SDPA fallback — build transposes + causal mask.
    q_t = query.transpose(0, 1).unsqueeze(0)
    k_t = k_full.transpose(0, 1).unsqueeze(0)
    v_t = v_full.transpose(0, 1).unsqueeze(0)
    q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len
    k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = k_pos <= q_pos
    out = F.scaled_dot_product_attention(
        q_t, k_t, v_t,
        attn_mask=mask, scale=self.scale,
        enable_gqa=(Hk < Hq),
    )
    return out[0].transpose(0, 1)


def apply() -> tuple[str, str]:
    """Rebind `TurboQuantAttentionImpl._continuation_prefill`.

    Never raises. Returns (status, reason).
    """
    if not should_apply():
        return "skipped", "platform: NVIDIA SM 8.0+ required for TurboQuant"

    impl_cls = _import_tq_impl()
    if impl_cls is None:
        return "skipped", "TurboQuant backend not available"

    if not hasattr(impl_cls, "_continuation_prefill"):
        return "skipped", (
            "TurboQuantAttentionImpl._continuation_prefill not present "
            "(upstream may have refactored)"
        )

    # P49 interface contract validation (v7.8): verify that the impl
    # class has the EXACT attrs + methods our replacement body reads.
    # If upstream refactored attribute names — or if this is a different
    # model class with the same name — we skip rather than rebinding
    # into a class that would crash on first forward.
    try:
        from vllm._genesis.interface_guard import (
            validate_impl,
        )
        validate_impl(
            impl_cls,
            role="TurboQuantAttentionImpl (P38 _continuation_prefill)",
            required_attrs={
                # Class-level attributes (non-instance-only are fine at
                # class lookup since they are defined in __init__ but
                # documented on the class).
                # No class-level attrs required — check instance-level
                # via required_methods + method signature.
            },
            required_methods=[
                "_continuation_prefill",
                "_flash_attn_varlen",
            ],
        )
    except Exception as e:
        # ImportError — guard module missing, treat as safe (skip).
        # GenesisInterfaceMismatch — signature drift, skip gracefully.
        if "GenesisInterfaceMismatch" in type(e).__name__:
            return "skipped", f"P49 interface drift: {e}"
        # Any other exception is a bug in the guard itself — log + continue.
        log.warning(
            "[Genesis P38] interface guard check errored (%s); "
            "proceeding without pre-flight validation", e,
        )

    original = impl_cls._continuation_prefill

    if getattr(original, _GENESIS_P38_MARKER_ATTR, False):
        # Already our wrapper. Ensure the original preserved attr is
        # intact (would only be missing if a test mutated the module-
        # level fn). Don't OVERWRITE with our wrapper-as-"original".
        return "applied", "already wrapped (idempotent)"

    # Stamp marker + save original on the new function. Guard against
    # double-stamping (would happen only if apply() was called from
    # two racing import paths in the same process — still benign but
    # preserves the FIRST captured original so revert() restores the
    # real upstream fn, never our own wrapper).
    if not getattr(_genesis_continuation_prefill, "_genesis_p38_original", None):
        setattr(_genesis_continuation_prefill, "_genesis_p38_original", original)
    setattr(_genesis_continuation_prefill, _GENESIS_P38_MARKER_ATTR, True)

    impl_cls._continuation_prefill = _genesis_continuation_prefill

    log.info(
        "[Genesis P38] rebound TurboQuantAttentionImpl._continuation_prefill "
        "(persistent K_full/V_full buffers replace torch.cat peak)"
    )
    return "applied", (
        "class method replaced (persistent K_full/V_full workspace, "
        "no .contiguous()/torch.cat transient peaks)"
    )


def is_applied() -> bool:
    """Post-apply assertion helper. True if our replacement is bound."""
    impl_cls = _import_tq_impl()
    if impl_cls is None:
        return False
    method = getattr(impl_cls, "_continuation_prefill", None)
    if method is None:
        return False
    return getattr(method, _GENESIS_P38_MARKER_ATTR, False)


def revert() -> bool:
    """Restore the original method. For tests only."""
    impl_cls = _import_tq_impl()
    if impl_cls is None:
        return False
    method = getattr(impl_cls, "_continuation_prefill", None)
    if method is None:
        return False
    if not getattr(method, _GENESIS_P38_MARKER_ATTR, False):
        return False
    original = getattr(method, "_genesis_p38_original", None)
    if original is None:
        return False
    impl_cls._continuation_prefill = original
    return True
