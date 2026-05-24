# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 24 — fused_moe `num_warps` / `num_stages` overlay.

Problem
-------
`vllm/model_executor/layers/fused_moe/fused_moe.py::get_default_config`
hard-codes two tile-selection numbers for the Triton fused_moe kernel:

  - Line 1255-1256 (fp8_w8a8 block-quant path): `num_warps=4, num_stages=3
    if not ROCm else 2`.
  - Line 1301-1308 (bf16/fp16/fp8-per-tensor path): `num_warps=4 if M<=128
    else 8`, `num_stages=4 if M<=32 else 3` (CUDA).

On Ampere SM 8.6 (A5000) the M-based branch is workable, but we want to
give operators a uniform env-knob for re-tuning when upstream's heuristic
turns out wrong on a specific GPU (Ampere vs Hopper vs Blackwell profiles
differ substantially).

Fix
---
Both fields are set from a small scope in get_default_config; we text-
patch in a call to our Genesis helpers AFTER upstream chooses its
default — the overrides win when `_OPTIMAL_NUM_WARPS_BY_ARCH` /
`_OPTIMAL_NUM_STAGES_BY_ARCH` has an entry for this GPU (or when the env
var is set). Otherwise our helpers return None and upstream's pick stays.

Platform compatibility:
  - NVIDIA CUDA + SM 8.6 (A5000/3090): auto-select applies.
  - Other NVIDIA arches: helpers return None → upstream behavior.
  - AMD ROCm / Intel XPU / CPU: helpers short-circuit on non-NVIDIA
    `is_nvidia_cuda()` guard.
  - Marlin path (`moe_wna16_marlin_gemm`): not affected — Marlin is a
    CUDA op that doesn't accept Triton autotune parameters. This patch
    is relevant when the engine falls back to Triton fused_moe (e.g.
    on smaller-moe batches or when Marlin-incompatible quant types).

Upstream drift detection: if `num_warps = _genesis_override` or
`_genesis_num_warps_override` appears, skip.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging

# Audit A-19 (2026-05-05): tightly coupled subpatches — both apply
# or both stay un-applied. Shared marker is acceptable here because the
# subpatches together form one logical fix; partial application is not
# desired anyway. _AUDIT_A19_EXEMPT documents this intentional design.
_AUDIT_A19_EXEMPT = True  # tightly coupled subpatches

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p24_moe_tune")

GENESIS_P24_MARKER = "Genesis P24 fused_moe num_warps/num_stages v7.0"

UPSTREAM_DRIFT_MARKERS = [
    "_genesis_num_warps_override",
    "_genesis_num_stages_override",
    "genesis_override_moe_tune",
    # PR #41184 (bnellnm, OPEN 2026-04-29) — massive MoE Refactor:
    # FusedMoE/MoERunner inversion + class rename FusedMoE →
    # RoutedExperts + ~80 files touched. If this lands our anchor in
    # `fused_moe.py` may not exist anymore. Watch for the canonical
    # rename signatures so P24 self-retires (anchor mismatch already
    # makes it skip cleanly, but explicit drift detection gives better
    # operator log messaging):
    "class RoutedExperts",
    "RoutedExperts(",
    "from vllm.model_executor.layers.fused_moe.routed_experts",
]


# Anchor 1: fp8_w8a8 block-quant config dict (line 1255-1256 in baseline).
_OLD_FP8_CFG = (
    "        config = {\n"
    "            \"BLOCK_SIZE_M\": 16 if M <= 64 else 64,\n"
    "            \"BLOCK_SIZE_N\": block_shape[0],\n"
    "            \"BLOCK_SIZE_K\": block_shape[1],\n"
    "            \"GROUP_SIZE_M\": 1 if M <= 16 else 32,\n"
    "            \"SPLIT_K\": 1,\n"
    "            \"num_warps\": 4,\n"
    "            \"num_stages\": 3 if not current_platform.is_rocm() else num_stages_rocm,\n"
    "        }"
)

_NEW_FP8_CFG = (
    "        config = {\n"
    "            \"BLOCK_SIZE_M\": 16 if M <= 64 else 64,\n"
    "            \"BLOCK_SIZE_N\": block_shape[0],\n"
    "            \"BLOCK_SIZE_K\": block_shape[1],\n"
    "            \"GROUP_SIZE_M\": 1 if M <= 16 else 32,\n"
    "            \"SPLIT_K\": 1,\n"
    "            \"num_warps\": 4,\n"
    "            \"num_stages\": 3 if not current_platform.is_rocm() else num_stages_rocm,\n"
    "        }\n"
    "        # [Genesis P24] per-SM / env override for num_warps + num_stages\n"
    "        try:\n"
    "            from vllm._genesis.kernels.marlin_tuning import (\n"
    "                get_num_warps_override as _genesis_num_warps_override,\n"
    "                get_num_stages_override as _genesis_num_stages_override,\n"
    "            )\n"
    "            _nw = _genesis_num_warps_override()\n"
    "            _ns = _genesis_num_stages_override()\n"
    "            if _nw is not None:\n"
    "                config[\"num_warps\"] = _nw\n"
    "            if _ns is not None:\n"
    "                config[\"num_stages\"] = _ns\n"
    "        except Exception:\n"
    "            pass"
)


# Anchor 2: general-default config block (lines 1310-1318 in baseline).
_OLD_GEN_CFG = (
    "        config = {\n"
    "            \"BLOCK_SIZE_M\": block_m,\n"
    "            \"BLOCK_SIZE_N\": block_n,\n"
    "            \"BLOCK_SIZE_K\": block_k,\n"
    "            \"GROUP_SIZE_M\": group_m,\n"
    "            \"SPLIT_K\": 1,\n"
    "            \"num_warps\": num_warps,\n"
    "            \"num_stages\": num_stages,\n"
    "        }\n"
    "    return config"
)

_NEW_GEN_CFG = (
    "        config = {\n"
    "            \"BLOCK_SIZE_M\": block_m,\n"
    "            \"BLOCK_SIZE_N\": block_n,\n"
    "            \"BLOCK_SIZE_K\": block_k,\n"
    "            \"GROUP_SIZE_M\": group_m,\n"
    "            \"SPLIT_K\": 1,\n"
    "            \"num_warps\": num_warps,\n"
    "            \"num_stages\": num_stages,\n"
    "        }\n"
    "        # [Genesis P24] per-SM / env override for num_warps + num_stages\n"
    "        try:\n"
    "            from vllm._genesis.kernels.marlin_tuning import (\n"
    "                get_num_warps_override as _genesis_num_warps_override,\n"
    "                get_num_stages_override as _genesis_num_stages_override,\n"
    "            )\n"
    "            _nw = _genesis_num_warps_override()\n"
    "            _ns = _genesis_num_stages_override()\n"
    "            if _nw is not None:\n"
    "                config[\"num_warps\"] = _nw\n"
    "            if _ns is not None:\n"
    "                config[\"num_stages\"] = _ns\n"
    "        except Exception:\n"
    "            pass\n"
    "    return config"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/fused_moe/fused_moe.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P24 fused_moe num_warps/num_stages overlay",
        target_file=target,
        marker=GENESIS_P24_MARKER,
        sub_patches=[
            TextPatch(
                name="p24_fp8_cfg_overlay",
                anchor=_OLD_FP8_CFG,
                replacement=_NEW_FP8_CFG,
                required=False,  # soft: only one of the two configs may
                                 # match if upstream refactors one branch.
            ),
            TextPatch(
                name="p24_general_cfg_overlay",
                anchor=_OLD_GEN_CFG,
                replacement=_NEW_GEN_CFG,
                required=False,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def apply() -> tuple[str, str]:
    """Apply P24 wiring. Never raises."""
    # P52 (v7.9): MoE-active dispatch gate. The num_warps overlay only affects
    # Triton fused_moe — dense models never hit this code path, so the
    # text-patch is dead weight there.
    try:
        from vllm._genesis.model_detect import is_moe_model, log_skip
        if not is_moe_model():
            log_skip(
                "P24 MoE num_warps/num_stages overlay",
                "dense model (no fused_moe dispatch)",
            )
            return "skipped", "P52 dispatch: model has no MoE layers"
    except Exception as e:
        log.debug("[Genesis P24] model_detect probe failed (proceeding): %s", e)

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "fused_moe.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", (
            "num_warps / num_stages overlay wired into get_default_config "
            "(active only on Triton fused_moe path; Marlin unaffected)"
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"
