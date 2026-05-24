# SPDX-License-Identifier: Apache-2.0
"""FP8 kernel dispatcher — Ampere/Ada/Hopper path selector (Patch 1/2).

Upstream `TritonBlockFP8ScaledMMKernel` assumes SM ≥ 89 (Ada+) for native
FP8 support. On Ampere (SM 8.6), silently produces wrong results. Upstream
provides Marlin as fallback but doesn't auto-route to it.

This module provides the smart dispatcher that:
  1. On SM ≥ 8.9 (Ada/Hopper/Blackwell): use Triton block FP8 (native)
  2. On SM < 8.9 (Ampere consumer/datacenter): force Marlin fallback
  3. On non-NVIDIA: no-op (other backends handle)

Platform compatibility:
  - NVIDIA CUDA SM 8.0 / 8.6  → Force Marlin (Patch 1/2 territory)
  - NVIDIA CUDA SM 8.9+       → Native Triton FP8 (no override needed)
  - AMD ROCm                  → Skip (different FP8 path)
  - Intel XPU                 → Skip (different FP8 path)
  - CPU                       → Skip (no FP8)

Current status (v7.1): FULLY IMPLEMENTED, applied. Helpers consulted
from `apply_all.py::apply_patch_1_2_fp8_dispatcher` at register time —
operator sees `Using MARLIN Fp8 MoE backend` in logs for SM < 8.9.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging

log = logging.getLogger("genesis.fp8_dispatcher")


def requires_marlin_fp8_fallback() -> bool:
    """True if current platform needs Marlin fallback for FP8 MoE.

    Returns:
        True on NVIDIA SM < 8.9 (Ampere) — Triton FP8 unreliable, use Marlin.
        False on NVIDIA SM ≥ 8.9 (Ada/Hopper/Blackwell) — native FP8 works.
        False on non-NVIDIA — different backend, doesn't apply.
    """
    from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least

    if not is_nvidia_cuda():
        return False

    # Ampere (SM 8.0/8.6/8.7) needs fallback
    # Ada (8.9), Hopper (9.0), Blackwell (10.0) have native FP8
    return not is_sm_at_least(8, 9)


def fp8_triton_kernel_supported() -> bool:
    """True if current GPU supports Triton block FP8 natively.

    Mirror of upstream `TritonBlockFP8ScaledMMKernel.is_supported()` logic.
    """
    from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least

    if not is_nvidia_cuda():
        return False
    return is_sm_at_least(8, 9)


def should_skip_triton_fp8(compute_capability: tuple[int, int] | None = None) -> bool:
    """True if Triton FP8 kernel should be bypassed on this GPU.

    Args:
        compute_capability: Optional explicit (major, minor). If None,
            queries current platform.

    Returns:
        True if SM < 8.9 (should fall back to Marlin).
    """
    if compute_capability is None:
        from vllm._genesis.guards import get_compute_capability
        compute_capability = get_compute_capability()

    if compute_capability is None:
        return False  # Non-NVIDIA — different path

    return compute_capability < (8, 9)


def log_dispatcher_decision() -> None:
    """Log the FP8 kernel routing decision at engine start."""
    from vllm._genesis.guards import get_compute_capability

    cc = get_compute_capability()
    if cc is None:
        log.info("[Genesis FP8 dispatcher] Not NVIDIA — skipping")
        return

    if requires_marlin_fp8_fallback():
        log.info(
            "[Genesis FP8 dispatcher] SM %s → Marlin fallback required "
            "(Triton block FP8 unsupported on Ampere)", cc
        )
    else:
        log.info(
            "[Genesis FP8 dispatcher] SM %s → Native Triton FP8 (no override)", cc
        )
