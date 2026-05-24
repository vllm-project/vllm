# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 98 — revert WorkspaceManager indirection in TQ decode.

Diagnosis (2026-04-28): NEW vllm nightly (sha256:7923b48047be) introduced
WorkspaceManager indirection in turboquant_attn.py `_decode_attention`
hot path (PR #40941 chain merged). On 2× A5000 + Qwen3.6-35B-A3B-FP8
PROD this caused ~17% TPS regression: 200+ TPS → 167 TPS sustained.

Root cause: every decode call goes through:
  current_workspace_manager().get_simultaneous(
      ((B, Hq, S, D + 1), torch.float32),
      ((B, Hq, D), query.dtype),
      ((B, Hq), torch.float32),
  )
which requires Python attribute lookup + shape tuple hashing + 3 tensor
returns × N layers × per-step. On 64-layer model with MTP K=3 spec-decode
that's ~256 Python lookups per token. The "saves 60× memory at long
context" benefit of WorkspaceManager is real but the Python indirection
cost outweighs it on Ampere small-batch single-stream workload.

This patch reverts to the OLD per-layer cached buffer pattern:
  mid_o_buf = getattr(layer, "_tq_mid_o_buf", None)
  output_buf = getattr(layer, "_tq_output_buf", None)
  lse_buf = getattr(layer, "_tq_lse_buf", None)

The kernel itself (triton_turboquant_decode_attention) lazily allocates
buffers on first call when None passed, then reuses via `buf_holder`
parameter. So `None, None, None` is fully safe — kernel handles it.

Memory cost: O(num_layers) extra dequant buffers. For Qwen3.6-A3B with
64 layers and 4K decode batch: ~16 MB × 64 layers = ~1 GB. Acceptable
on 2× A5000 24GB at our PROD config (320K context with TQ k8v4 leaves
~3GB headroom).

Status: opt-in via `GENESIS_ENABLE_P98=1`. Default OFF — only enable on
hardware where Python indirection cost > memory savings (Ampere
small-batch single-stream). DO NOT enable on H100/H200 high-concurrency
workloads where WorkspaceManager amortizes better.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original ref: vllm#40941 (we revert).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p98_tq_workspace_revert")


GENESIS_P98_MARKER = (
    "Genesis P98 TQ WorkspaceManager revert (vllm#40941 perf hotfix) v7.62.14"
)


# ─── Sub-patch 1: revert _decode_attention WorkspaceManager block ───────

P98_DECODE_OLD = (
    "        # Acquire shared decode scratch buffers from WorkspaceManager.\n"
    "        # Layers execute sequentially so one set of buffers is sufficient.\n"
    "        # Falls back to kernel-internal allocation if workspace unavailable.\n"
    "        B = query.shape[0]\n"
    "        D = self.head_size\n"
    "        S = self.max_num_kv_splits\n"
    "        Hq = self.num_heads\n"
    "        mid_o_buf = output_buf = lse_buf = None\n"
    "        if is_workspace_manager_initialized():\n"
    "            # output_buf in query dtype — matches the in-kernel fp16 cast in stage2.\n"
    "            mid_o_buf, output_buf, lse_buf = (\n"
    "                current_workspace_manager().get_simultaneous(\n"
    "                    ((B, Hq, S, D + 1), torch.float32),\n"
    "                    ((B, Hq, D), query.dtype),\n"
    "                    ((B, Hq), torch.float32),\n"
    "                )\n"
    "            )\n"
)

P98_DECODE_NEW = (
    "        # [Genesis P98 vllm#40941 perf revert] Skip WorkspaceManager.\n"
    "        # NEW upstream pattern adds Python lookup overhead per decode call\n"
    "        # × N layers × per-step. On Ampere small-batch single-stream\n"
    "        # workload the indirection cost outweighs memory savings.\n"
    "        # Restore per-layer cached buffer (OLD pattern) — kernel lazily\n"
    "        # allocates and stores on `buf_holder=layer` on first call.\n"
    "        mid_o_buf = output_buf = lse_buf = None\n"
    "        if layer is not None:\n"
    "            mid_o_buf = getattr(layer, \"_tq_mid_o_buf\", None)\n"
    "            output_buf = getattr(layer, \"_tq_output_buf\", None)\n"
    "            lse_buf = getattr(layer, \"_tq_lse_buf\", None)\n"
)


# ─── Sub-patch 2: revert continuation prefill dequant WorkspaceManager ───

P98_DEQUANT_OLD = (
    "        # Use WorkspaceManager for dequant buffers.\n"
    "        # Shared across all layers — saves 60× memory at long context.\n"
    "        # Required for CUDA Graph capture (per-layer growth incompatible with CG).\n"
    "        k_buf, v_buf = current_workspace_manager().get_simultaneous(\n"
    "            (buf_shape, torch.float16),\n"
    "            (buf_shape, torch.float16),\n"
    "        )\n"
    "        # Skip .zero_() — kernel writes all positions up to cached_len,\n"
    "        # and we only read [:cached_len] afterwards.\n"
    "        k_cached = k_buf[:, :, :alloc_len, :]\n"
    "        v_cached = v_buf[:, :, :alloc_len, :]\n"
)

P98_DEQUANT_NEW = (
    "        # [Genesis P98 vllm#40941 perf revert] Skip WorkspaceManager.\n"
    "        # Per-layer cached buffer (OLD pattern) avoids Python indirection.\n"
    "        import torch as _genesis_p98_torch\n"
    "        if not hasattr(layer, \"_tq_k_dequant_buf\") or layer._tq_k_dequant_buf.shape != buf_shape:\n"
    "            k_buf = _genesis_p98_torch.empty(buf_shape, dtype=_genesis_p98_torch.float16, device=device)\n"
    "            v_buf = _genesis_p98_torch.empty(buf_shape, dtype=_genesis_p98_torch.float16, device=device)\n"
    "            layer._tq_k_dequant_buf = k_buf\n"
    "            layer._tq_v_dequant_buf = v_buf\n"
    "        else:\n"
    "            k_buf = layer._tq_k_dequant_buf\n"
    "            v_buf = layer._tq_v_dequant_buf\n"
    "        # Skip .zero_() — kernel writes all positions up to cached_len,\n"
    "        # and we only read [:cached_len] afterwards.\n"
    "        k_cached = k_buf[:, :, :alloc_len, :]\n"
    "        v_cached = v_buf[:, :, :alloc_len, :]\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P98 turboquant_attn.py — revert WorkspaceManager (perf hotfix)"
        ),
        target_file=str(target),
        marker=GENESIS_P98_MARKER,
        sub_patches=[
            TextPatch(
                name="p98_decode_workspace_revert",
                anchor=P98_DECODE_OLD,
                replacement=P98_DECODE_NEW,
                required=True,
            ),
            TextPatch(
                name="p98_dequant_workspace_revert",
                anchor=P98_DEQUANT_OLD,
                replacement=P98_DEQUANT_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P98",
            # Upstream-side markers if vllm reverts WorkspaceManager itself:
            "perf-hotfix workspace revert",
            # PR #41123 (cderinbogaz, OPEN 2026-04-29) — narrower
            # competitor to JartX #39931. If #41123 lands first it
            # rewrites the same _decode_attention region we revert.
            # Anchor-shape signatures of #41123:
            "_align_hybrid_block_size",
            "TQ packed-slot size",
            "UNIFORM_SINGLE_TOKEN_DECODE",
            # PR #41212 (KungYork, OPEN) — sharing buffers via
            # `tensor.set_()` (DSv4 specific) introduces a different
            # contract for cross-model buffer ownership. If upstream
            # generalizes this to TQ workspace, our revert will need
            # to mirror the new ownership semantics:
            "topk_indices_buffer.set_(",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P98 — TurboQuant WorkspaceManager revert."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P98")
    log_decision("P98", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "turboquant_attn.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P98] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"

    # Drift-detection: if no `current_workspace_manager` calls in source,
    # then the upstream WorkspaceManager is already gone — patch unnecessary.
    if "current_workspace_manager" not in content:
        return "skipped", (
            "current_workspace_manager not in source — upstream may have "
            "already reverted WorkspaceManager indirection. P98 unnecessary."
        )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: surface SKIPPED as skipped (was masked as applied)
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )

    return (
        "applied",
        "P98 v7.62.14 applied: turboquant_attn.py _decode_attention + "
        "continuation prefill dequant now use per-layer cached buffers "
        "(OLD pattern, pre-vllm#40941). Removes Python indirection from "
        "decode hot path. Expected: +15-25% TPS recovery on Ampere "
        "small-batch single-stream workload."
    )


def is_applied() -> bool:
    if vllm_install_root() is None:
        return False
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file) as f:
            return patcher.marker in f.read()
    except Exception:
        return False
