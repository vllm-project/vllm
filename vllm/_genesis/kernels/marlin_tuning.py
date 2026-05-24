# SPDX-License-Identifier: Apache-2.0
"""Marlin MoE kernel tuning — per-SM auto-tuner (Patch 17/18).

Upstream `fused_marlin_moe.py` heuristic lands on `block_size_m=16` for FP8
with `input_dtype.itemsize==1` floor, regardless of actual GPU. On Ampere
A5000 (SM 8.6) with M≤4, topk=8, E=256, empirical sweep shows:

  bsm=8  → +1.2% (measured on Qwen3.6-35B-A3B)
  bsm=16 → baseline
  bsm=32 → −1.9%
  bsm=48 → −4.6%
  bsm=64 → −7.9%

v7.0 approach: auto-select optimal bsm based on SM version, with env override
(`VLLM_MARLIN_MOE_BLOCK_SIZE_M`) for manual tuning.

Additional env knobs (NEW v7.0):
  VLLM_MARLIN_MOE_NUM_WARPS   — Override kernel warp count (2, 4, 8)
  VLLM_MARLIN_MOE_NUM_STAGES  — Override pipeline stages (1-8)

Platform compatibility:
  - NVIDIA CUDA: ✅ Primary target (all SM 8.0+)
  - AMD ROCm:    ❌ No Marlin (CUDA-only kernel)
  - Intel XPU:   ❌ No Marlin
  - CPU:         ❌ No Marlin

Current status (v7.1): FULLY IMPLEMENTED. Per-SM tuning tables live in
`_OPTIMAL_BSM_BY_ARCH / _OPTIMAL_NUM_WARPS_BY_ARCH /
_OPTIMAL_NUM_STAGES_BY_ARCH`. Env-var escape hatches:
`VLLM_MARLIN_MOE_BLOCK_SIZE_M / _NUM_WARPS / _NUM_STAGES`.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("genesis.marlin_tuning")


# Per-arch optimal block_size_m (empirically tuned)
# PN64 (2026-05-05): added (12, 0) consumer Blackwell entry — copies
# Hopper/datacenter Blackwell as starting point until empirical 5090
# sweep data lands (apnar club-3090#51 boot log shows our patcher
# previously skipped Marlin MoE tuning for SM 12.0 with NO entry).
_OPTIMAL_BSM_BY_ARCH: dict[tuple[int, int], int] = {
    (8, 0): 16,   # A100 — defer to upstream heuristic (no tune data)
    (8, 6): 8,    # A5000/3090 — measured +1.2%
    (8, 9): 16,   # Ada Lovelace — defer to upstream
    (9, 0): 16,   # H100 — upstream heuristic adequate
    (10, 0): 16,  # Blackwell datacenter — upstream heuristic adequate
    (12, 0): 16,  # PN64: Blackwell consumer (RTX 5090) — placeholder copying
                  # SM (9, 0) Hopper. UNMEASURED on real hw — solicit sweep.
}


# P24: Per-arch optimal num_stages (pipeline depth) for Marlin MoE.
# On Ampere SM 8.6, shallower pipelines (3) reduce register pressure;
# Hopper benefits from deeper pipelines (4-5) due to more shared memory.
# None = defer to upstream heuristic (no override).
_OPTIMAL_NUM_STAGES_BY_ARCH: dict[tuple[int, int], Optional[int]] = {
    (8, 0): None,  # A100 — defer
    (8, 6): 3,     # A5000/3090 — measured ~+0.5% with 3 stages
    (8, 9): None,  # Ada — defer
    (9, 0): None,  # H100 — native FP32 + deep shared mem, defer
    (10, 0): None,  # Blackwell datacenter — defer
    (12, 0): None,  # PN64: Blackwell consumer placeholder — defer
}

# P24: Per-arch optimal num_warps. Smaller warps on smaller cards.
_OPTIMAL_NUM_WARPS_BY_ARCH: dict[tuple[int, int], Optional[int]] = {
    (8, 0): None,  # A100 — defer
    (8, 6): 4,     # A5000 — 4 warps matches 2×MP unit occupancy
    (8, 9): None,  # Ada — defer
    (9, 0): None,  # H100 — defer (huge regs, upstream chooses well)
    (10, 0): None,
    (12, 0): None,  # PN64: Blackwell consumer placeholder — defer
}


def get_optimal_block_size_m() -> Optional[int]:
    """Return optimal block_size_m for current GPU, or None if no override.

    Env override: VLLM_MARLIN_MOE_BLOCK_SIZE_M.
    Falls back to hardcoded per-arch table.
    Returns None if platform doesn't support Marlin.
    """
    from vllm._genesis.guards import is_nvidia_cuda, get_compute_capability

    if not is_nvidia_cuda():
        return None

    # Env override takes precedence
    env = os.environ.get("VLLM_MARLIN_MOE_BLOCK_SIZE_M", "")
    if env in ("8", "16", "32", "48", "64"):
        return int(env)

    # Per-arch table
    cc = get_compute_capability()
    if cc is None:
        return None

    # Audit P1 fix 2026-05-05: PN64 is registered opt-in in dispatcher
    # (env_flag=GENESIS_ENABLE_PN64, default_on=False), so the SM 12.0
    # placeholder entry must respect that flag — otherwise the audit reads
    # the registry as opt-in but the table lookup fires unconditionally.
    if cc == (12, 0) and not _pn64_enabled():
        return None

    return _OPTIMAL_BSM_BY_ARCH.get(cc)


def _pn64_enabled() -> bool:
    """Honour PN64 opt-in flag for SM 12.0 placeholder entries."""
    raw = os.environ.get("GENESIS_ENABLE_PN64", "").strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def get_num_warps_override() -> Optional[int]:
    """P24: Env override first, then per-arch table. None if no override set.

    Behavior:
      1. `VLLM_MARLIN_MOE_NUM_WARPS=<2|4|8>` → use that value.
      2. No env → consult `_OPTIMAL_NUM_WARPS_BY_ARCH`; None defers to
         upstream heuristic.
    """
    env = os.environ.get("VLLM_MARLIN_MOE_NUM_WARPS", "")
    if env in ("2", "4", "8"):
        return int(env)

    from vllm._genesis.guards import is_nvidia_cuda, get_compute_capability
    if not is_nvidia_cuda():
        return None
    cc = get_compute_capability()
    if cc is None:
        return None
    if cc == (12, 0) and not _pn64_enabled():
        return None
    return _OPTIMAL_NUM_WARPS_BY_ARCH.get(cc)


def get_num_stages_override() -> Optional[int]:
    """P24: Env override first, then per-arch table. None if no override set.

    Behavior:
      1. `VLLM_MARLIN_MOE_NUM_STAGES=<1..8>` → use that value.
      2. No env → consult `_OPTIMAL_NUM_STAGES_BY_ARCH`; None defers to
         upstream heuristic.
    """
    env = os.environ.get("VLLM_MARLIN_MOE_NUM_STAGES", "")
    if env.isdigit() and 1 <= int(env) <= 8:
        return int(env)

    from vllm._genesis.guards import is_nvidia_cuda, get_compute_capability
    if not is_nvidia_cuda():
        return None
    cc = get_compute_capability()
    if cc is None:
        return None
    if cc == (12, 0) and not _pn64_enabled():
        return None
    return _OPTIMAL_NUM_STAGES_BY_ARCH.get(cc)


def log_selected_tuning(
    num_experts: int,
    topk: int,
    selected_bsm: int,
) -> None:
    """Log tuning decision for observability at engine start."""
    from vllm._genesis.guards import get_compute_capability

    cc = get_compute_capability()
    log.info(
        "[Genesis Marlin tuning] SM=%s E=%d topk=%d bsm=%d "
        "num_warps=%s num_stages=%s",
        cc, num_experts, topk, selected_bsm,
        get_num_warps_override() or "default",
        get_num_stages_override() or "default",
    )
