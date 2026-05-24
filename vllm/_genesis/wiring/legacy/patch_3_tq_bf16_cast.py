# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 3 — TurboQuant BF16→FP8 cast fix for Ampere.

Problem
-------
Triton's `convert_custom_float8_sm80` (the kernel that backs
`.to(tl.float8e4b15)`) only accepts FP16 / FP32 input. When TurboQuant
runs on Ampere (SM 8.x) and the model uses BF16 weights, the kernel at
`v1/attention/ops/triton_turboquant_store.py:191` does
`k_vals.to(tl.float8e4b15)` directly on a BF16 tensor and crashes inside
the Triton compilation pipeline.

Reference: vLLM PR [#39908](https://github.com/vllm-project/vllm/pull/39908).

Fix
---
Insert an explicit BF16 → FP16 cast before the FP8 cast on the e4b15 path:

    if FP8_E4B15:
        k_fp8 = k_vals.to(tl.float16).to(tl.float8e4b15)
    else:
        k_fp8 = k_vals.to(tl.float8e4nv)

The `.to(tl.float8e4nv)` (Hopper/Ada path) already accepts BF16 directly.

Platform compatibility
----------------------
  NVIDIA SM 8.0 / 8.6 (Ampere): primary target — fix is mandatory
  NVIDIA SM 8.9 / 9.0 / 10.0:   FP8_E4B15 false, no-op
  AMD ROCm:                     no Triton FP8 path
  Intel XPU / CPU:              no Triton FP8 path

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import (
    is_nvidia_cuda,
    is_sm_at_least,
    resolve_vllm_file,
    vllm_install_root,
)
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p3_tq_bf16_cast")

GENESIS_P3_MARKER = "Genesis P3 TQ BF16->FP8 Ampere fix v7.0"

UPSTREAM_DRIFT_MARKERS = [
    # If an upstream fix merges, the file will contain a staircase cast
    # to a wider float type before the FP8 conversion. Genesis P3 uses
    # the FP16 staircase (gated on FP8_E4B15); upstream PR #39988 (active
    # as of 2026-04-29) goes the FP32 route unconditionally. Watch for
    # BOTH so the drift detector catches whichever variant lands.
    "tl.float16).to(tl.float8e4b15)",  # P3 / PR #39908 form
    "tl.float32).to(tl.float8e4b15)",  # PR #39988 form
    "tl.float32).to(tl.float8e5m2)",   # PR #39988 sibling cast site
    "PR #39908",
    "PR #39988",
]


_OLD = (
    "    k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0)\n"
    "    k_fp8 = k_vals.to(tl.float8e4b15) if FP8_E4B15 else k_vals.to(tl.float8e4nv)"
)

_NEW = (
    "    k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0)\n"
    "    # [Genesis P3] BF16->FP8 cast crashes on SM<89 (convert_custom_float8_sm80\n"
    "    # only accepts fp16/fp32). Cast to fp16 first on FP8_E4B15 path. PR #39908.\n"
    "    if FP8_E4B15:\n"
    "        k_fp8 = k_vals.to(tl.float16).to(tl.float8e4b15)\n"
    "    else:\n"
    "        k_fp8 = k_vals.to(tl.float8e4nv)"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/ops/triton_turboquant_store.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P3 TurboQuant BF16->FP8 cast (Ampere fix)",
        target_file=target,
        marker=GENESIS_P3_MARKER,
        sub_patches=[
            TextPatch(
                name="p3_bf16_fp8_cast",
                anchor=_OLD,
                replacement=_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def apply() -> tuple[str, str]:
    """Apply P3 wiring. Never raises."""
    if not is_nvidia_cuda():
        return "skipped", "non-NVIDIA — Triton FP8 not in path"
    # The bug only fires on SM<89 in practice. SM>=89 has FP8_E4B15=False
    # so the patch is a no-op there. Apply unconditionally for SM>=80
    # to also cover any future re-routing.
    if not is_sm_at_least(8, 0):
        return "skipped", "SM<8.0 — TurboQuant requires Ampere+"

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "triton_turboquant_store.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "BF16->FP8 cast guard inserted"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"
