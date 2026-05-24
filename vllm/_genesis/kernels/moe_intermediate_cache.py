# SPDX-License-Identifier: Apache-2.0
"""Genesis MoE intermediate-cache shared pool (Patch 37).

Problem
-------
`fused_marlin_moe._fused_marlin_moe` in upstream vLLM allocates two
intermediate tensors **lazily, per-call**:

  - `intermediate_cache13`: shape
    `(M * num_topk * max(w13_num_shards * N, K),)` in `hidden_states.dtype`.
  - `intermediate_cache2`:  shape `(M * num_topk, N)` in `hidden_states.dtype`.

For Qwen3.6-35B-A3B at chunked-prefill M=4096, num_topk=8, N=2816, K=2048,
w13_num_shards=2, fp16:

  - cache13: 4096 × 8 × max(5632, 2048) = 184M elements × 2 B  ≈ **369 MiB**
  - cache2:  4096 × 8 × 2816 = 92M elements × 2 B              ≈ **184 MiB**

Per MoE-layer call peak: ~553 MiB. With 30 MoE layers executing
sequentially per step, the allocator churns 553 MiB alloc/free 30 times
per step. Even with `expandable_segments=True` this creates:
  (a) profiler-invisible pressure (KV cache may be over-sized, #40420
      class of OOMs when activations hit the ceiling).
  (b) allocator slab fragmentation when interleaved with Marlin kernel
      workspace allocations.

Fix
---
Shared pool: ONE `cache13` tensor and ONE `cache2` tensor per
`(max_M_times_topk, N, K, num_shards, dtype, device)` configuration. All
MoE layers reuse the same pool because they execute sequentially — no
race condition. The pool is sized for `max_num_batched_tokens` which is
the upper bound imposed by the chunked-prefill scheduler.

On overflow (should not happen in steady state — scheduler caps M) we
fall back to a per-call `torch.empty` to preserve correctness.

dynamo-safety
-------------
This is called from within `_fused_marlin_moe` which is traced by
`aot_compile_fullgraph`. We therefore:

  1. Keep key construction to PRIMITIVE int-tuples only
     (`device.index`, `dtype.itemsize` — both int). No `str(device)`,
     no `str(dtype)`, no `os.environ`, no `log.*`.
  2. Use `@torch._dynamo.allow_in_graph` on the entry functions so
     dynamo treats the allocation as an opaque graph node with a
     concrete tensor output (shape derived from primitive args).
  3. Gracefully fall back to `torch.empty(...)` on any failure path
     (should-apply false, overflow, exception) — caller's behaviour
     is identical to upstream.

Retirement
----------
Upstream intends to resolve this at the FusedMoE layer level rather than
per-call (see PR #40655 pattern for the TQ decode case). When that lands
for MoE intermediates, the `upstream_drift_markers` in
`vllm/_genesis/wiring/patch_37_moe_intermediate_cache.py` will catch the
signature and skip.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

log = logging.getLogger("genesis.moe_intermediate_cache")


# ─── Env pinning (module-import; NO env reads in hot path) ──────────────
_ENV_ENABLE = "GENESIS_ENABLE_P37"
_ENV_MAX_BT_OVERRIDE = "GENESIS_MOE_MAX_BATCHED_TOKENS"


def _read_env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _read_env_max_bt_override() -> Optional[int]:
    env = os.environ.get(_ENV_MAX_BT_OVERRIDE, "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    return None


# Resolved at module import — these are const across process lifetime.
_ENABLED_AT_IMPORT: bool = _read_env_enabled()
_MAX_BT_OVERRIDE: Optional[int] = _read_env_max_bt_override()


# ─── Pool registries (module-level, not class-level — dynamo-friendlier) ─
# Keys use only int primitives for dynamo safety.
_CACHE13_POOLS: dict[tuple, torch.Tensor] = {}
_CACHE2_POOLS: dict[tuple, torch.Tensor] = {}

# Should-apply cache (set by warm_up() at apply_all register time).
_SHOULD_APPLY_CACHED: Optional[bool] = None


def warm_up() -> bool:
    """Called OUTSIDE compile region (at apply_all register time). Probes
    platform capability and caches the result so forward-path functions
    don't have to do device-property queries (which dynamo rejects).

    Returns the cached boolean.
    """
    global _SHOULD_APPLY_CACHED
    try:
        from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least
        _SHOULD_APPLY_CACHED = bool(
            is_nvidia_cuda() and is_sm_at_least(8, 0)
        )
    except Exception as e:
        log.info("[Genesis P37] warm_up probe failed: %s", e)
        _SHOULD_APPLY_CACHED = False
    return _SHOULD_APPLY_CACHED


def should_apply() -> bool:
    """Fast path readable from inside traced regions.

    Consults `_SHOULD_APPLY_CACHED` (set by warm_up at register time)
    AND the env-gate resolved at module import. Both must be True for
    the pool to engage.
    """
    if _SHOULD_APPLY_CACHED is None:
        # warm_up was never called (e.g. helper imported in isolation).
        # Resolve now. This path shouldn't fire in production because
        # apply_all calls warm_up() before any forward.
        warm_up()
    return bool(_SHOULD_APPLY_CACHED) and _ENABLED_AT_IMPORT


def clear_for_tests() -> None:
    """TESTS ONLY. Clear pools + reset should_apply cache + re-read env.

    Re-reads `_ENABLED_AT_IMPORT` from the env so tests that monkeypatch
    `GENESIS_ENABLE_P37` see the new value. Without this, the import-
    time freeze made test fixtures that set the env AFTER import
    silently ineffective on the very path they meant to cover.
    """
    global _SHOULD_APPLY_CACHED, _ENABLED_AT_IMPORT, _MAX_BT_OVERRIDE
    _CACHE13_POOLS.clear()
    _CACHE2_POOLS.clear()
    _SHOULD_APPLY_CACHED = None
    # Re-read the env so tests can toggle on/off without full reimport.
    _ENABLED_AT_IMPORT = _read_env_enabled()
    _MAX_BT_OVERRIDE = _read_env_max_bt_override()


def _resolve_max_batched_tokens(m_hint: Optional[int] = None) -> int:
    """Resolve max_num_batched_tokens for pool sizing.

    [Genesis P73 fix v7.42] Now consults central prealloc_budget resolver
    which probes vllm scheduler_config. Back-compat: GENESIS_MOE_MAX_BATCHED_TOKENS
    still wins as domain-specific override.

    Priority:
      1. `GENESIS_MOE_MAX_BATCHED_TOKENS` env (pinned at module import).
      2. `m_hint` arg rounded up to power-of-2 (stabilise pool key).
      3. Central P73 resolver (vllm scheduler_config / global env override).
      4. Conservative default 4096 (final fallback).
    """
    if _MAX_BT_OVERRIDE is not None:
        return _MAX_BT_OVERRIDE
    if m_hint is not None and m_hint > 0:
        # Round up to next power of 2 to stabilise the key across calls
        # of slightly different M values.
        p = 1
        while p < m_hint:
            p <<= 1
        # Floor by central resolver (prevents chunk-overflow regression)
        try:
            from vllm._genesis.prealloc_budget import resolve_token_budget
            return max(p, resolve_token_budget(domain_env=_ENV_MAX_BT_OVERRIDE))
        except Exception:
            return max(p, 4096)
    try:
        from vllm._genesis.prealloc_budget import resolve_token_budget
        return resolve_token_budget(domain_env=_ENV_MAX_BT_OVERRIDE)
    except Exception:
        return 4096


# ─── Core allocator — DYNAMO-SAFE int-tuple keys ────────────────────────
# `@torch._dynamo.allow_in_graph` makes dynamo treat this function as
# an opaque node: it is invoked in the graph, inputs/outputs are tracked
# as tensors/constants, but the body (dict ops + torch.empty branch) is
# NOT traced. Perfect fit for «cache-hit returns same tensor, cache-miss
# allocates once».

try:
    from torch._dynamo import allow_in_graph as _allow_in_graph
except Exception:
    def _allow_in_graph(fn):
        return fn


def _pool_key_cache13(
    max_m_times_topk_times_dimmax: int,
    device_index: int,
    dtype_itemsize: int,
) -> tuple[int, int, int]:
    return (max_m_times_topk_times_dimmax, device_index, dtype_itemsize)


def _pool_key_cache2(
    max_m_times_topk: int,
    n: int,
    device_index: int,
    dtype_itemsize: int,
) -> tuple[int, int, int, int]:
    return (max_m_times_topk, n, device_index, dtype_itemsize)


@_allow_in_graph
def acquire_cache13(
    M: int,
    num_topk: int,
    w13_num_shards: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    device: torch.device,
    max_batched_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Return the `intermediate_cache13` 1D tensor for this call.

    On pool-hit: returns the shared pool tensor (same `data_ptr` as all
    previous calls with same config → allocator-fragmentation-free,
    CUDA-graph-safe reuse). The caller's `_resize_cache` will slice to
    the actually-needed `M * num_topk * max(w13_num_shards * N, K)`.

    On pool-miss: allocates the pool for `(max_batched_tokens, ...)`
    then returns it.

    On overflow (actual needed > pool size): falls back to a fresh
    `torch.empty` to preserve correctness. This shouldn't fire in
    steady state but is a safety net.

    On platform incompatibility / disabled env / any error: fresh
    `torch.empty` identical to upstream behaviour.
    """
    needed = M * num_topk * max(w13_num_shards * N, K)
    if not should_apply():
        return torch.empty((needed,), dtype=dtype, device=device)

    max_m = _resolve_max_batched_tokens(M)
    pool_elems = max_m * num_topk * max(w13_num_shards * N, K)

    if needed > pool_elems:
        # Overflow — fresh alloc for this call. Doesn't poison the pool.
        return torch.empty((needed,), dtype=dtype, device=device)

    dev_idx = device.index if device.index is not None else 0
    key = _pool_key_cache13(pool_elems, dev_idx, dtype.itemsize)
    pool = _CACHE13_POOLS.get(key)
    if pool is None:
        pool = torch.empty((pool_elems,), dtype=dtype, device=device)
        _CACHE13_POOLS[key] = pool
    return pool


@_allow_in_graph
def acquire_cache2(
    M: int,
    num_topk: int,
    N: int,
    dtype: torch.dtype,
    device: torch.device,
    max_batched_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Return the `intermediate_cache2` 2D tensor for this call.

    Same semantics as `acquire_cache13` but 2D-shape
    `(max_batched_tokens * num_topk, N)`. Caller's `_resize_cache`
    sub-shapes it to `(M * num_topk, N)`.
    """
    needed_m = M * num_topk
    if not should_apply():
        return torch.empty((needed_m, N), dtype=dtype, device=device)

    max_m = _resolve_max_batched_tokens(M)
    pool_m = max_m * num_topk

    if needed_m > pool_m:
        return torch.empty((needed_m, N), dtype=dtype, device=device)

    dev_idx = device.index if device.index is not None else 0
    key = _pool_key_cache2(pool_m, N, dev_idx, dtype.itemsize)
    pool = _CACHE2_POOLS.get(key)
    if pool is None:
        pool = torch.empty((pool_m, N), dtype=dtype, device=device)
        _CACHE2_POOLS[key] = pool
    return pool


# ─── Diagnostics (called from OUTSIDE compile region — log-safe) ────────
def get_registry_info() -> dict:
    out: dict = {
        "cache13_pools": [],
        "cache2_pools": [],
        "total_bytes": 0,
    }
    for key, t in _CACHE13_POOLS.items():
        b = t.element_size() * t.numel()
        out["total_bytes"] += b
        out["cache13_pools"].append({
            "key": key, "shape": list(t.shape), "bytes": b,
        })
    for key, t in _CACHE2_POOLS.items():
        b = t.element_size() * t.numel()
        out["total_bytes"] += b
        out["cache2_pools"].append({
            "key": key, "shape": list(t.shape), "bytes": b,
        })
    return out


# ─── Back-compat class facade for tests that prefer class methods ──────
class GenesisMoEIntermediateCacheManager:
    """Thin facade over module-level functions + registries for tests
    and ergonomic callers."""

    @classmethod
    def warm_up(cls) -> bool:
        return warm_up()

    @classmethod
    def should_apply(cls) -> bool:
        return should_apply()

    @classmethod
    def acquire_cache13(cls, *args, **kwargs) -> torch.Tensor:
        return acquire_cache13(*args, **kwargs)

    @classmethod
    def acquire_cache2(cls, *args, **kwargs) -> torch.Tensor:
        return acquire_cache2(*args, **kwargs)

    @classmethod
    def clear_for_tests(cls) -> None:
        clear_for_tests()

    @classmethod
    def get_registry_info(cls) -> dict:
        return get_registry_info()
