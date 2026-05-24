# SPDX-License-Identifier: Apache-2.0
"""GatedDeltaNet core_attn_out pre-allocation — Patch 28 correct redo.

Architecture (resolves CRIT-HW-1 per master plan Part 5)
---------------------------------------------------------
**Pre-allocation happens at `GatedDeltaNet.__init__`**, NEVER lazy in
forward. `forward_cuda` does a pure tensor slice + in-place zero, which
is fully `torch.dynamo`-traceable.

Why:
  - `GatedDeltaNet.forward_cuda` is NOT isolated by `splitting_ops` — the
    inner custom op `vllm::gdn_attention_core` IS in `splitting_ops`, but
    the surrounding Python code in `forward_cuda` is compiled by
    `torch.dynamo` / AOT-compile.
  - ANY Python helper call from inside `forward_cuda` must be
    dynamo-traceable. Dict lookups, `str(device)`, `os.environ`,
    `logging.Logger.info`, device-capability probes → all Unsupported.
  - Lesson from P19 revert: lazy alloc in forward → −30% throughput,
    188× stdev, CUDA graph recaptures.

Solution
--------
  1. Monkey-patch `GatedDeltaNet.__init__` to CALL `attach_buffer(self,
     hint)` at the end. The init path runs eagerly (before any compile
     trace), so we can do whatever we want there: device probes,
     env reads, dict lookups, logging.
  2. `attach_buffer` sets `self._genesis_gdn_core_attn_buf` to EITHER a
     pre-allocated tensor of shape `(max_num_tokens, Hv, Hv_dim)` OR
     `None` on incompatible platforms.
  3. The `forward_cuda` text-patch (see wiring/patch_28_gdn_core_attn.py)
     reads the attribute and slices:

         core_attn_out = (
             self._genesis_gdn_core_attn_buf[:num_tokens].zero_()
             if self._genesis_gdn_core_attn_buf is not None
             else torch.zeros(
                 (num_tokens, self.num_v_heads // self.tp_size,
                  self.head_v_dim),
                 dtype=hidden_states.dtype,
                 device=hidden_states.device,
             )
         )

     Both branches are pure tensor operations — fully dynamo-compatible.
     The `is not None` check resolves at trace time against a constant
     attribute value: dynamo sees a concrete tensor OR None and picks
     one branch to compile (no ambiguity).

Platform compatibility
----------------------
  - NVIDIA CUDA SM ≥ 8.0: pre-alloc engaged.
  - AMD / XPU / CPU: `attach_buffer` sets attribute to None; forward
    falls through to upstream-equivalent `torch.zeros` path.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

log = logging.getLogger("genesis.gdn_core_attn_manager")


# ─── Env budget (resolved ONCE at import, dynamo-safe) ──────────────────────
_ENV_MAX_BT = "GENESIS_GDN_MAX_BATCHED_TOKENS"


def _read_env_budget() -> Optional[int]:
    env = os.environ.get(_ENV_MAX_BT, "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    return None


_ENV_BUDGET: Optional[int] = _read_env_budget()
_DEFAULT_MAX_BT: int = 4096


def resolve_max_batched_tokens(hint: Optional[int] = None) -> int:
    """Choose max_num_tokens for prealloc. Called at __init__ time only.

    [Genesis P73 fix v7.42] Now delegates to central `prealloc_budget.resolve_token_budget`
    which consults vllm scheduler_config IF env not set. Back-compat: still
    honors GENESIS_GDN_MAX_BATCHED_TOKENS as a domain-specific override.
    Resolves the chunk-overflow root cause (5664-token chunk vs hardcoded 4096).
    """
    if hint is not None and hint > 0:
        return int(hint)
    # Back-compat: domain env still wins if explicitly set
    if _ENV_BUDGET is not None:
        return _ENV_BUDGET
    # Try central resolver (consults vllm scheduler_config)
    try:
        from vllm._genesis.prealloc_budget import resolve_token_budget
        return resolve_token_budget(domain_env=_ENV_MAX_BT)
    except Exception as e:
        log.warning(
            "[Genesis P28] prealloc_budget resolver failed (%s); "
            "falling back to legacy default %d", e, _DEFAULT_MAX_BT,
        )
        return _DEFAULT_MAX_BT


# ─── Should-apply cache ─────────────────────────────────────────────────────
_SHOULD_APPLY_CACHED: Optional[bool] = None


def should_apply() -> bool:
    """True on NVIDIA CUDA SM ≥ 8.0 — cached at first call (init-time)."""
    global _SHOULD_APPLY_CACHED
    if _SHOULD_APPLY_CACHED is None:
        try:
            from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least
            _SHOULD_APPLY_CACHED = bool(
                is_nvidia_cuda() and is_sm_at_least(8, 0)
            )
        except Exception as e:
            log.info("[Genesis P28] should_apply probe failed: %s", e)
            _SHOULD_APPLY_CACHED = False
    return _SHOULD_APPLY_CACHED


def _reset_pin_for_tests() -> None:
    """TESTS ONLY."""
    global _SHOULD_APPLY_CACHED
    _SHOULD_APPLY_CACHED = None


# ─── Shared buffer registry ─────────────────────────────────────────────────
# Keyed by (max_num_tokens, num_v_heads, head_v_dim, device_type, device_index,
# dtype_str). Shared across ALL GDN layers so N_layers × MiB bloat becomes
# 1 × MiB only (layers execute sequentially per step).


class _BufferRegistry:
    _cache: dict[tuple, torch.Tensor] = {}

    @classmethod
    def get_or_create(
        cls,
        max_num_tokens: int,
        num_v_heads: int,
        head_v_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # All key components are primitive (int/str) — dict lookup is cheap
        # and purely Python-level. This method is NOT called from forward.
        dev_type = device.type
        dev_index = device.index if device.index is not None else -1
        key = (max_num_tokens, num_v_heads, head_v_dim, dev_type, dev_index, str(dtype))
        buf = cls._cache.get(key)
        if buf is None:
            buf = torch.zeros(
                (max_num_tokens, num_v_heads, head_v_dim),
                dtype=dtype, device=device,
            )
            cls._cache[key] = buf
            log.info(
                "[Genesis P28] GDN core_attn_out prealloc: "
                "max_num_tokens=%d Hv=%d Hv_dim=%d device=%s dtype=%s "
                "bytes=%d",
                max_num_tokens, num_v_heads, head_v_dim, device, dtype,
                buf.element_size() * buf.numel(),
            )
        return buf

    @classmethod
    def clear_for_tests(cls) -> None:
        cls._cache.clear()

    @classmethod
    def get_registry_info(cls) -> dict[str, Any]:
        out = []
        total_bytes = 0
        for key, t in cls._cache.items():
            b = t.element_size() * t.numel()
            total_bytes += b
            out.append({
                "max_num_tokens": key[0], "num_v_heads": key[1],
                "head_v_dim": key[2],
                "device_type": key[3], "device_index": key[4],
                "dtype": key[5],
                "bytes": b,
            })
        return {"entries": out, "total_bytes": total_bytes}


# ─── Init-time attacher ─────────────────────────────────────────────────────

_ATTR_NAME = "_genesis_gdn_core_attn_buf"


def attach_buffer(module: Any, hint: Optional[int] = None) -> None:
    """Attach `_genesis_gdn_core_attn_buf` to a GatedDeltaNet instance.

    Called ONCE per module from a monkey-patched `__init__`. All work
    here is eager — no compile trace, so device probes + logging + dict
    ops are all safe. The resulting attribute is EITHER a pre-allocated
    tensor OR None (platform incompatible).

    The forward path then reads the attribute directly (pure tensor op).
    """
    # Platform guard
    if not should_apply():
        setattr(module, _ATTR_NAME, None)
        return

    # Resolve shape from module attributes. These are set in
    # GatedDeltaNet.__init__ before we attach, so they're available.
    num_v_heads_total = getattr(module, "num_v_heads", None)
    tp_size = getattr(module, "tp_size", 1)
    head_v_dim = getattr(module, "head_v_dim", None)
    if num_v_heads_total is None or head_v_dim is None:
        # Module not fully initialized (edge case): skip, forward will fallback.
        setattr(module, _ATTR_NAME, None)
        log.warning(
            "[Genesis P28] attach_buffer: module missing num_v_heads/head_v_dim; "
            "forward will fall back to eager allocation"
        )
        return

    num_v_heads_per_rank = num_v_heads_total // tp_size
    max_num_tokens = resolve_max_batched_tokens(hint)

    # Device + dtype resolution. At __init__ time the module may not yet
    # have been moved to device; we defer the first allocation until
    # `attach_buffer` is re-called post-move, OR we probe from a param.
    device = _guess_module_device(module)
    dtype = _guess_module_dtype(module)

    if device is None or dtype is None:
        # Still CPU — save the hint for later but don't allocate on wrong device.
        setattr(module, _ATTR_NAME, None)
        return

    try:
        buf = _BufferRegistry.get_or_create(
            max_num_tokens=max_num_tokens,
            num_v_heads=num_v_heads_per_rank,
            head_v_dim=head_v_dim,
            device=device,
            dtype=dtype,
        )
    except Exception as e:
        log.warning("[Genesis P28] attach_buffer alloc failed: %s — fallback", e)
        setattr(module, _ATTR_NAME, None)
        return

    setattr(module, _ATTR_NAME, buf)


def _guess_module_device(module: Any) -> Optional[torch.device]:
    """Pick a device from the module's parameters / submodules."""
    for p in module.parameters(recurse=True):
        if p.device.type != "cpu" or p.device.type == "cuda":
            return p.device
    return None


def _guess_module_dtype(module: Any) -> Optional[torch.dtype]:
    """Pick a dtype from the module's parameters."""
    for p in module.parameters(recurse=True):
        return p.dtype
    return None


def warm_up(hint: Optional[int] = None) -> int:
    """Call from `apply_all` register time to force device-probe + env
    resolution outside any traced region.
    """
    should_apply()  # populates cache
    return resolve_max_batched_tokens(hint)


# ─── Back-compat class facade (tests still use GdnCoreAttnManager) ─────────
class GdnCoreAttnManager:
    """Back-compat facade over the module-level functions + registry.

    Tests import `GdnCoreAttnManager` for classmethods like `get_or_create`,
    `acquire_slice`, `clear_for_tests`, `get_registry_info`, `should_apply`.
    The actual work is delegated to the module-level `_BufferRegistry` and
    helpers.
    """

    @classmethod
    def should_apply(cls) -> bool:
        return should_apply()

    @classmethod
    def get_or_create(
        cls,
        num_tokens_max: int,
        num_v_heads: int,
        head_v_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not should_apply():
            return None
        if isinstance(device, str):
            device = torch.device(device)
        return _BufferRegistry.get_or_create(
            max_num_tokens=num_tokens_max,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def acquire_slice(
        cls,
        num_tokens: int,
        num_v_heads: int,
        head_v_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
        num_tokens_max: Optional[int] = None,
    ) -> torch.Tensor:
        """EAGER-only API for tests / fallback code. NOT called from
        forward path anymore (forward uses `self._genesis_gdn_core_attn_buf`
        directly).
        """
        max_n = resolve_max_batched_tokens(num_tokens_max)
        if num_tokens > max_n or not should_apply():
            return torch.zeros(
                (num_tokens, num_v_heads, head_v_dim),
                device=device, dtype=dtype,
            )
        if isinstance(device, str):
            device = torch.device(device)
        buf = _BufferRegistry.get_or_create(
            max_num_tokens=max_n,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            device=device,
            dtype=dtype,
        )
        slice_ = buf[:num_tokens]
        slice_.zero_()
        return slice_

    @classmethod
    def clear_for_tests(cls) -> None:
        _BufferRegistry.clear_for_tests()
        _reset_pin_for_tests()

    @classmethod
    def get_registry_info(cls) -> dict[str, Any]:
        return _BufferRegistry.get_registry_info()
