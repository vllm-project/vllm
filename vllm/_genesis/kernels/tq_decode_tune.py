# SPDX-License-Identifier: Apache-2.0
"""TurboQuant decode stage1 env-driven tunables (Patch 18b).

Context
-------
Upstream launcher in `triton_turboquant_decode.py` (~line 554) hardcodes:

    BLOCK_KV    = 4
    num_warps   = 1
    num_stages  = 1

These are H100-tuned. On A5000 (SM 8.6) at long context (e.g. 160k tokens
× NUM_KV_SPLITS=32 → ~5k tokens per split), BLOCK_KV=4 means ~1250 tile
iters per program with 1 warp and no pipelining — underutilizing Ampere
SMs that can schedule more warps per tile.

Design
------
Default behaviour = upstream (env-unset → same 4/1/1 literals). Opt-in A/B
tuning via:

    VLLM_TQ_DECODE_BLOCK_KV     (int, 1..64 — kernel block size along KV dim)
    VLLM_TQ_DECODE_NUM_WARPS    (int, 1/2/4/8)
    VLLM_TQ_DECODE_NUM_STAGES   (int, 1..8)

Invalid values fall back to the upstream default (NEVER raise — Genesis
guards).

Platform compatibility
----------------------
  NVIDIA CUDA SM 8.0+  → primary target (A5000 especially)
  AMD ROCm             → TurboQuant not ported to ROCm (skip)
  Intel XPU            → TurboQuant not ported (skip)
  CPU                  → no Triton path (skip)

Integration
-----------
apply_all orchestrator inspects env → either leaves upstream alone (no
override) or emits a monkey-patch that overrides the three literals in
the kernel launch.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("genesis.tq_decode_tune")


# Upstream (H100-tuned) defaults — preserve exactly.
UPSTREAM_BLOCK_KV: int = 4
UPSTREAM_NUM_WARPS: int = 1
UPSTREAM_NUM_STAGES: int = 1

_VALID_BLOCK_KV = {1, 2, 4, 8, 16, 32, 64}
_VALID_NUM_WARPS = {1, 2, 4, 8}


def get_block_kv_override() -> Optional[int]:
    """Parse VLLM_TQ_DECODE_BLOCK_KV. Returns None if unset/invalid.

    Whitelisted: {1, 2, 4, 8, 16, 32, 64}. Other values ignored silently.
    """
    env = os.environ.get("VLLM_TQ_DECODE_BLOCK_KV", "").strip()
    if not env:
        return None
    if not env.isdigit():
        return None
    v = int(env)
    if v not in _VALID_BLOCK_KV:
        return None
    return v


def get_num_warps_override() -> Optional[int]:
    """Parse VLLM_TQ_DECODE_NUM_WARPS. Returns None if unset/invalid.

    Whitelisted: {1, 2, 4, 8}.
    """
    env = os.environ.get("VLLM_TQ_DECODE_NUM_WARPS", "").strip()
    if not env:
        return None
    if not env.isdigit():
        return None
    v = int(env)
    if v not in _VALID_NUM_WARPS:
        return None
    return v


def get_num_stages_override() -> Optional[int]:
    """Parse VLLM_TQ_DECODE_NUM_STAGES. Returns None if unset/invalid.

    Valid range 1..8.
    """
    env = os.environ.get("VLLM_TQ_DECODE_NUM_STAGES", "").strip()
    if not env:
        return None
    if not env.isdigit():
        return None
    v = int(env)
    if 1 <= v <= 8:
        return v
    return None


def resolve_decode_tune() -> tuple[int, int, int]:
    """Resolve the final (block_kv, num_warps, num_stages) tuple.

    Environment overrides take precedence; otherwise upstream defaults
    are returned unchanged.

    Returns:
        (block_kv, num_warps, num_stages) — always a fully-populated tuple
        of valid ints.
    """
    block_kv = get_block_kv_override() or UPSTREAM_BLOCK_KV
    num_warps = get_num_warps_override() or UPSTREAM_NUM_WARPS
    num_stages = get_num_stages_override() or UPSTREAM_NUM_STAGES
    return block_kv, num_warps, num_stages


def has_any_override() -> bool:
    """True iff at least one of the three tunables has a valid env override."""
    return (
        get_block_kv_override() is not None
        or get_num_warps_override() is not None
        or get_num_stages_override() is not None
    )


def should_apply() -> bool:
    """Platform guard — TurboQuant is NVIDIA CUDA + SM 8.0+ only.

    Matches the guard in dequant_buffer.TurboQuantBufferManager.should_apply.
    """
    from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return True


def log_selected_tune() -> None:
    """Log the resolved tuning at engine start."""
    if not should_apply():
        return
    block_kv, num_warps, num_stages = resolve_decode_tune()
    if has_any_override():
        log.info(
            "[Genesis P18b] TQ decode stage1 TUNE OVERRIDE: "
            "BLOCK_KV=%d num_warps=%d num_stages=%d "
            "(upstream default %d/%d/%d)",
            block_kv, num_warps, num_stages,
            UPSTREAM_BLOCK_KV, UPSTREAM_NUM_WARPS, UPSTREAM_NUM_STAGES,
        )
    else:
        log.info(
            "[Genesis P18b] TQ decode stage1 using upstream defaults "
            "(BLOCK_KV=%d num_warps=%d num_stages=%d)",
            block_kv, num_warps, num_stages,
        )
