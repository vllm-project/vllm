# SPDX-License-Identifier: Apache-2.0
"""Genesis P73 — single source of truth for prealloc M-dim budget.

All Genesis prealloc patches (P28 GDN core_attn_out, P26 TQ prefill output,
P37 MoE intermediate cache, P44 TQ mixed-batch attn_out, P39a FLA KKT, etc.)
that size on `M = tokens-per-step` MUST resolve their budget through
`resolve_token_budget()`. This guarantees that when operators raise
`--max-num-batched-tokens` (e.g. via P72 unblock), every pool grows to
match — no overflow on chunked-prefill chunks larger than legacy 4096
default.

================================================================
WHY THIS EXISTS
================================================================

Pre-v7.42, P28 hardcoded `_DEFAULT_MAX_BT = 4096` and never read the live
scheduler config. With P72 raising the scheduler's batched cap to 8192,
the chunked-prefill scheduler dispatched a 5664-token chunk that overran
the 4096-slot buffer at `gdn_linear_attn.py forward_cuda` →
`setStorage … out of bounds` crash on long-context (180K) prefill.

This module fixes the bug class by funneling ALL prealloc budget queries
through a single resolver that consults (in priority order):

  1. Explicit `hint` arg (caller knows best — typically scheduler_config).
  2. `GENESIS_PREALLOC_TOKEN_BUDGET` env (operator override).
  3. Domain-specific env (back-compat: GENESIS_GDN_MAX_BATCHED_TOKENS,
     GENESIS_TQ_MAX_BATCHED_TOKENS, GENESIS_MOE_MAX_BATCHED_TOKENS).
  4. `vllm.config.get_current_vllm_config().scheduler_config
        .max_num_batched_tokens` (live config — set during engine init).
  5. Conservative default 4096 (preserves prior behavior; bump
     globally via env if operator wants 8192).

================================================================
DYNAMO SAFETY
================================================================

Resolved ONCE per process at first call and CACHED. NO env reads /
config probes inside traced regions. The cache invalidates only on
`reset_for_tests()` (test-only).

================================================================
RUNTIME GUARD — assert_fits()
================================================================

To prevent silent buffer overflows like the v7.42 P28 incident, callers
SHOULD invoke `assert_fits(num_tokens, where=...)` at allocation slice
time. This raises a clear actionable error if a chunk exceeds the
resolved budget — much better than a deep `setStorage out of bounds`
trace from torch internals.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("genesis.prealloc_budget")


_CACHED: Optional[int] = None
_DEFAULT_FALLBACK = 4096  # only used if NO config / env signal at all


def _probe_vllm_config() -> Optional[int]:
    """Try to read scheduler_config.max_num_batched_tokens from live vLLM.

    Returns None if vLLM not initialized yet (e.g. called at import time).
    """
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        if cfg is None:
            return None
        sched = getattr(cfg, "scheduler_config", None)
        if sched is None:
            return None
        v = getattr(sched, "max_num_batched_tokens", None)
        return int(v) if v else None
    except Exception:
        return None


def resolve_token_budget(
    hint: Optional[int] = None,
    domain_env: Optional[str] = None,
) -> int:
    """Single source of truth for per-step token budget.

    Args:
        hint: caller-supplied value (highest priority). Typically passed
              from the scheduler config at allocation site.
        domain_env: domain-specific env var name (back-compat with existing
                    GENESIS_GDN_MAX_BATCHED_TOKENS, etc.). Consulted
                    AFTER GENESIS_PREALLOC_TOKEN_BUDGET to allow per-pool
                    overrides if needed.

    Returns:
        Positive int representing the maximum tokens any prealloc may
        encounter in a single step.
    """
    global _CACHED

    # Hint always wins (no caching — caller may have step-specific signal)
    if hint is not None and hint > 0:
        return int(hint)

    # Cached after first non-hint resolution
    if _CACHED is not None:
        return _CACHED

    # Priority 1: global override
    env = os.environ.get("GENESIS_PREALLOC_TOKEN_BUDGET", "")
    if env.isdigit() and int(env) > 0:
        _CACHED = int(env)
        log.info(
            "[Genesis P73] token budget resolved → %d "
            "(via GENESIS_PREALLOC_TOKEN_BUDGET)", _CACHED,
        )
        return _CACHED

    # Priority 2: domain-specific env (back-compat with existing P28/P37
    # _ENV_*_MAX_BT vars)
    if domain_env:
        de = os.environ.get(domain_env, "")
        if de.isdigit() and int(de) > 0:
            _CACHED = int(de)
            log.info(
                "[Genesis P73] token budget resolved → %d (via %s)",
                _CACHED, domain_env,
            )
            return _CACHED

    # Priority 3: live vLLM config
    cfg_v = _probe_vllm_config()
    if cfg_v is not None and cfg_v > 0:
        _CACHED = cfg_v
        log.info(
            "[Genesis P73] token budget resolved → %d "
            "(via vllm scheduler_config.max_num_batched_tokens)", _CACHED,
        )
        return _CACHED

    # Priority 4: conservative default
    _CACHED = _DEFAULT_FALLBACK
    log.info(
        "[Genesis P73] token budget resolved → %d (default fallback). "
        "Set GENESIS_PREALLOC_TOKEN_BUDGET to override.", _CACHED,
    )
    return _CACHED


def assert_fits(num_tokens: int, where: str) -> None:
    """Runtime guard — raises clearly if a chunk would overflow a pool.

    Replaces silent torch `setStorage out of bounds` deep-stack with a
    Genesis-attributed error pointing to the actionable env var.

    Args:
        num_tokens: actual chunk size encountered at runtime
        where: human-readable callsite (e.g. "P28 GDN core_attn_out forward_cuda")
    """
    budget = resolve_token_budget()
    if num_tokens > budget:
        raise RuntimeError(
            f"[Genesis P73 prealloc overflow] {where}: chunk={num_tokens} > "
            f"budget={budget}. Either:\n"
            f"  (1) set GENESIS_PREALLOC_TOKEN_BUDGET={num_tokens} (or higher) "
            f"and restart, OR\n"
            f"  (2) lower --max-num-batched-tokens, OR\n"
            f"  (3) enable P74 chunk-clamp (caps prefill chunks to budget)."
        )


def reset_for_tests() -> None:
    """TESTS ONLY — clear cache so subsequent resolve_token_budget()
    re-reads env / config."""
    global _CACHED
    _CACHED = None


def get_cached() -> Optional[int]:
    """Diagnostic: read current cached budget without triggering resolution."""
    return _CACHED
