# SPDX-License-Identifier: Apache-2.0
"""Patch 23 — Genesis env override for Marlin FP32 reduction (NEW in v7.0).

Background
----------
vLLM's Marlin MoE kernel uses an internal FP32 accumulation reduce step.
On Hopper (SM 9.0+) FP32 tensor cores make this nearly free; on Ampere
(SM 8.x) where there are no native FP32 tensor cores, the reduce is done
with FMAs in FP32 — measurable cost. Empirically on A5000:

  VLLM_MARLIN_FP32_REDUCE=1 (default)   → baseline
  VLLM_MARLIN_FP32_REDUCE=0             → +1.5–3% TGS, no quality drop on
                                          GSM8K/MMLU sweeps (M≤4, topk=8)

Master-plan classification: NEW v7.0 patch, MEDIUM risk. Requires GSM8K
validation per platform before activation.

Platform compatibility
----------------------
  NVIDIA SM 8.0 / 8.6 (Ampere): primary target — recommended
  NVIDIA SM 8.9 (Ada):          may help, validate
  NVIDIA SM 9.0+ (Hopper, Blackwell): contra-productive (native FP32
                                      tensor cores), DO NOT enable
  AMD ROCm:                     no Marlin
  Intel XPU:                    no Marlin
  CPU:                          no Marlin

Usage
-----
This module is a pure env reader. Higher-level Marlin tuning code (or a
future text-patch on the Marlin launcher) consults `should_disable_fp32_reduce()`
and propagates the decision into the kernel argument or env that the
underlying CUDA Marlin reads.

Note: at the time of this writing vLLM does NOT yet have an official
runtime switch for FP32_REDUCE — applying P23 will require an additional
text-patch on the Marlin MoE launcher, OR upstream cooperation. This
module ships the platform guard + env reader so callers can choose how
to wire it.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("genesis.marlin_fp32_reduce")


def get_fp32_reduce_override() -> Optional[bool]:
    """Parse VLLM_MARLIN_FP32_REDUCE env var.

    Returns:
      True  → user explicitly enabled (= keep upstream default behavior).
      False → user explicitly disabled (= the optimisation we want for SM<90).
      None  → not set, defer to upstream/platform default.

    Whitelist:
      "1"/"true"/"yes" → True
      "0"/"false"/"no" → False
    """
    raw = os.environ.get("VLLM_MARLIN_FP32_REDUCE", "").strip().lower()
    if not raw:
        return None
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None  # invalid → defer


def should_disable_fp32_reduce() -> bool:
    """Decide if the Marlin FP32 reduce should be disabled on THIS platform.

    Logic:
      - If user set VLLM_MARLIN_FP32_REDUCE=0 → True (disable).
      - If user set VLLM_MARLIN_FP32_REDUCE=1 → False (keep, override our default).
      - Otherwise: AUTO — disable on SM 8.0/8.6 (Ampere) where it's a clear
        win; keep on SM>=8.9 where it's neutral or contra-productive.
    """
    override = get_fp32_reduce_override()
    if override is True:
        return False  # user wants it ON → not disabled
    if override is False:
        return True   # user wants it OFF → disabled

    # AUTO path
    from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least

    if not is_nvidia_cuda():
        return False  # no Marlin off-NVIDIA, decision moot
    if is_sm_at_least(9, 0):
        return False  # Hopper+ has native FP32 tensor cores — keep upstream default
    if is_sm_at_least(8, 0):
        return True   # Ampere/Ada → recommended disable
    return False  # SM<8.0 — Marlin not supported anyway


def log_decision() -> None:
    """Log the FP32_REDUCE decision at engine start for observability."""
    from vllm._genesis.guards import get_compute_capability

    cc = get_compute_capability()
    override = get_fp32_reduce_override()
    disabled = should_disable_fp32_reduce()

    if override is None:
        decision_source = "auto-from-platform"
    else:
        decision_source = f"env override (VLLM_MARLIN_FP32_REDUCE={override})"

    log.info(
        "[Genesis P23] Marlin FP32_REDUCE: disabled=%s on SM=%s (%s)",
        disabled, cc, decision_source,
    )
