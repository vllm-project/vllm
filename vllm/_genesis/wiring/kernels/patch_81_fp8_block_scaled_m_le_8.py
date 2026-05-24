# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 81 — fp8 block-scaled MM low-M decode tuning.

Backport of upstream PR vllm-project/vllm#40925 (tonyliu312, OPEN as of
2026-04-26). Specializes the default `w8a8_triton_block_scaled_mm` config
for M ≤ 8 (single-request decode + short MTP-style draft batches) where
the prior `BLOCK_SIZE_M=64` default wasted 98% of the M-dimension.

================================================================
WHAT THIS FIXES
================================================================

`vllm/model_executor/layers/quantization/utils/fp8_utils.py`'s
`w8a8_triton_block_scaled_mm` falls back to a hardcoded default config
when no pre-tuned `configs/N=*,K=*,device_name=*.json` matches the GPU.
The default uses BLOCK_SIZE_M=64. For single-request decode (M=1) or
MTP K=3 verify (M=4) this wastes 98%-94% of M-dim work — paying for
unused tile rows.

PR #40925 specializes ONLY the M ≤ 8 branch:
  - `BLOCK_SIZE_M = 16` (4× less wasted M-dim)
  - `num_stages = 3` (deeper pipeline, on non-ROCm)
Larger M is unchanged to keep blast radius small.

Empirical (per upstream PR test on GB10 sm_121):
  - Before: 5.45 t/s median decode
  - After: 6.73 t/s median decode = **+23%**

Genesis applicability: HIGH — we're on Qwen3.6-A3B FP8 + max_num_seqs=2
(M=1 typical) + MTP K=3 (M=4 verify), and we have NO pre-tuned JSON
for our (N, K, RTX A5000) tuple in `configs/`. So we hit the default
branch on every fp8 block-scaled MM call.

================================================================
GENESIS APPROACH
================================================================

Mirror upstream exactly via single TextPatch on the default-config dict.
Anchor on the unique `# Default config` comment + the BLOCK_SIZE_M=64
literal that follows it (verified single occurrence in our pin).

================================================================
ENV
================================================================

GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8=1

================================================================
RISK
================================================================

LOW — branches on M, leaves M > 8 path EXACTLY as before. Pre-tuned
JSON configs for (N, K, device) short-circuit before this code path,
so any host with a tuned JSON is unchanged.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Backport of: vllm#40925 (tonyliu312, OPEN at backport time).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.p81_fp8_block_scaled_m_le_8")

GENESIS_P81_MARKER = "Genesis P81 fp8 block-scaled M<=8 backport vllm#40925 v7.47"


# Anchor on the exact 9-line config dict block. Single occurrence in fp8_utils.py.

P81_OLD = (
    "        # Default config\n"
    "        # Block-wise quant: BLOCK_SIZE_N must be divisible by block_size[0]\n"
    "        # BLOCK_SIZE_K must be divisible by block_size[1]\n"
    "        config = {\n"
    "            \"BLOCK_SIZE_M\": 64,\n"
    "            \"BLOCK_SIZE_N\": block_size[0],\n"
    "            \"BLOCK_SIZE_K\": block_size[1],\n"
    "            \"GROUP_SIZE_M\": 32,\n"
    "            \"num_warps\": 4,\n"
    "            \"num_stages\": 2,\n"
    "        }\n"
)

P81_NEW = (
    "        # Default config\n"
    "        # Block-wise quant: BLOCK_SIZE_N must be divisible by block_size[0]\n"
    "        # BLOCK_SIZE_K must be divisible by block_size[1]\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis P81 backport vllm#40925] M-aware tuning for low-M decode.\n"
    "        # Default BLOCK_SIZE_M=64 wastes 98% of M-dim for single-request\n"
    "        # decode (M=1) and short MTP-style draft batches (M=4). Specialize\n"
    "        # only M<=8 case to keep blast radius small; larger M unchanged.\n"
    "        # num_stages=3 gated to non-ROCm (MI300 LDS pressure at [128,128]).\n"
    "        # CREDIT: tonyliu312 vllm#40925 (OPEN at backport time)\n"
    "        # Empirical: +23% median decode TPS on GB10 sm_121 (DeepSeek-V4)\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        if M <= 8:\n"
    "            _genesis_p81_block_m = 16\n"
    "            _genesis_p81_num_stages = 2 if current_platform.is_rocm() else 3\n"
    "        else:\n"
    "            _genesis_p81_block_m, _genesis_p81_num_stages = 64, 2\n"
    "        config = {\n"
    "            \"BLOCK_SIZE_M\": _genesis_p81_block_m,\n"
    "            \"BLOCK_SIZE_N\": block_size[0],\n"
    "            \"BLOCK_SIZE_K\": block_size[1],\n"
    "            \"GROUP_SIZE_M\": 32,\n"
    "            \"num_warps\": 4,\n"
    "            \"num_stages\": _genesis_p81_num_stages,\n"
    "        }\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/quantization/utils/fp8_utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P81 fp8_utils.py — fp8 block-scaled M<=8 tuning backport",
        target_file=str(target),
        marker=GENESIS_P81_MARKER,
        sub_patches=[
            TextPatch(
                name="p81_fp8_block_scaled_m_le_8",
                anchor=P81_OLD,
                replacement=P81_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P81",
            # Detect upstream merge: the new config dict will reference
            # `block_m` or `_genesis_p81_block_m`. We probe for the
            # `if M <= 8:` literal which only appears post-merge.
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P81 — fp8 block-scaled M<=8 tuning."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P81")
    log_decision("P81", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/.../quantization/utils/fp8_utils.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P81] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"

    # Drift detector: if upstream merged, the `if M <= 8:` literal appears
    # in fp8_utils.py without our Genesis marker.
    if "if M <= 8:" in content and "[Genesis P81" not in content:
        return "skipped", (
            "`if M <= 8:` literal present without Genesis marker — "
            "upstream PR #40925 may have merged equivalent fix"
        )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: surface SKIPPED as skipped (was masked as applied)
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: {failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )
    return "applied", (
        "P81 applied: w8a8_triton_block_scaled_mm default config now uses "
        "BLOCK_SIZE_M=16 + num_stages=3 for M<=8 (single-request decode / "
        "short MTP draft batches). Backport of vllm#40925 (tonyliu312, OPEN). "
        "Expected: +23% median decode TPS on hosts without pre-tuned JSON."
    )
