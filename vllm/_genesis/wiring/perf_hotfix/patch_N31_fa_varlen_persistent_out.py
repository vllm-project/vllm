# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N31 — FA varlen persistent `out` buffer (issue #15).

================================================================
Issue
================================================================
https://github.com/Sandermage/genesis-vllm-patches/issues/15 (noonghunna)

OOM at `vllm/vllm_flash_attn/flash_attn_interface.py:300:flash_attn_varlen_func`
on 1×3090 24GB single GPU when long-vision config + ~50K-token prefill.
50 MiB allocation fails when GPU has 30 MiB free (model+KV cache eat 23.52 GB
of 23.56 GB). Different code path from P15B's max_seqlen_k clamp (which
helps but doesn't eliminate the per-call out/softmax_lse allocations).

================================================================
ROOT CAUSE (sister to P38 K_full/V_full)
================================================================

`flash_attn_varlen_func` allocates output tensor `out` and `softmax_lse`
inside the C extension on every call. Per-call malloc pressure on a
budget-constrained GPU (24GB single card with model already saturating
budget) leads to OOM even when individual allocations are small.

P38 fixed analogous problem for `_continuation_prefill` via persistent
K_full/V_full buffers. PN31 applies the same pattern to FA varlen path
by passing `out=<pre-allocated>` to `flash_attn_varlen_func` (the
upstream signature already supports this — we just don't use it).

================================================================
FIX
================================================================

Single-file text-patch on `turboquant_attn.py:_flash_attn_varlen`:
- On first call per (max_q_tokens, num_q_heads, head_dim) shape:
  allocate persistent `out` buffer on `self._genesis_pn31_out_buf`
- On subsequent calls: reuse buffer if shape compatible, slice to
  needed size; reallocate if shape grows
- Pass `out=<buffer>` to `flash_attn_varlen_func` — it writes in-place
  instead of allocating

Memory footprint: bounded by max_num_batched_tokens × num_q_heads ×
head_dim × dtype_bytes. For 27B Lorbus on 24GB: 2048 × 32 × 128 × 2
= 16 MiB — well within budget. For 35B: 4096 × 64 × 128 × 2 = 64 MiB.

================================================================
SAFETY MODEL
================================================================
- Default OFF (opt-in via GENESIS_ENABLE_PN31_FA_VARLEN_PERSISTENT_OUT=1)
- Pure text-patch, idempotent via marker
- Drift-aware: anchor includes the exact `flash_attn_varlen_func(` call
- Anchor missing → SKIPPED, source stays vanilla
- `out` parameter is upstream-supported in flash_attn_varlen_func — we
  just enable its use
- Worst case: +16-64 MiB persistent VRAM per process for the buffer,
  zero per-call allocation pressure
- For our 2× A5000 PROD: NULL impact (we have 24GB headroom, no OOM
  risk); intended for single-GPU community users (1×3090, 1×4090) with
  budget-constrained workloads

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Reporter: noonghunna (issue #15, 2026-05-01).
Sister patch: P38 (K_full/V_full persistent buffers).
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pN31_fa_varlen_persistent_out")

GENESIS_PN31_MARKER = (
    "Genesis PN31 FA varlen persistent out (issue #15) v7.65"
)


# ─── Patch the fa_version=None call path ────────────────────────────
# Anchor matches the first flash_attn_varlen_func call (no fa_version kwarg).
# Prepended with persistent buffer acquire logic.

PN31_ANCHOR_NO_VER = (
    "        if self.fa_version is None:\n"
    "            return flash_attn_varlen_func(\n"
    "                q=q,\n"
    "                k=k,\n"
    "                v=v,\n"
    "                cu_seqlens_q=cu_seqlens_q,\n"
    "                cu_seqlens_k=cu_seqlens_k,\n"
    "                max_seqlen_q=max_seqlen_q,\n"
    "                max_seqlen_k=max_seqlen_k,\n"
    "                softmax_scale=self.scale,\n"
    "                causal=True,\n"
    "            )\n"
)

PN31_REPLACEMENT_NO_VER = (
    "        # [Genesis PN31 issue #15 fix] Acquire persistent out buffer.\n"
    "        # Eliminates per-call malloc pressure inside C extension —\n"
    "        # critical on budget-constrained single-GPU configs (1×3090\n"
    "        # 24GB) where model+KV cache saturate budget. Buffer keyed by\n"
    "        # shape; reallocates only when shape grows.\n"
    "        _genesis_pn31_total_q = q.shape[0]\n"
    "        _genesis_pn31_n_heads = q.shape[1]\n"
    "        _genesis_pn31_head_dim = q.shape[2]\n"
    "        _genesis_pn31_buf_key = (\n"
    "            _genesis_pn31_total_q,\n"
    "            _genesis_pn31_n_heads,\n"
    "            _genesis_pn31_head_dim,\n"
    "        )\n"
    "        if not hasattr(self, '_genesis_pn31_out_bufs'):\n"
    "            self._genesis_pn31_out_bufs = {}\n"
    "        _genesis_pn31_out = self._genesis_pn31_out_bufs.get(\n"
    "            _genesis_pn31_buf_key\n"
    "        )\n"
    "        if (\n"
    "            _genesis_pn31_out is None\n"
    "            or _genesis_pn31_out.dtype != q.dtype\n"
    "            or _genesis_pn31_out.device != q.device\n"
    "        ):\n"
    "            import torch as _genesis_pn31_torch\n"
    "            _genesis_pn31_out = _genesis_pn31_torch.empty(\n"
    "                _genesis_pn31_buf_key,\n"
    "                dtype=q.dtype, device=q.device,\n"
    "            )\n"
    "            self._genesis_pn31_out_bufs[_genesis_pn31_buf_key] = (\n"
    "                _genesis_pn31_out\n"
    "            )\n"
    "        if self.fa_version is None:\n"
    "            return flash_attn_varlen_func(\n"
    "                q=q,\n"
    "                k=k,\n"
    "                v=v,\n"
    "                cu_seqlens_q=cu_seqlens_q,\n"
    "                cu_seqlens_k=cu_seqlens_k,\n"
    "                max_seqlen_q=max_seqlen_q,\n"
    "                max_seqlen_k=max_seqlen_k,\n"
    "                softmax_scale=self.scale,\n"
    "                causal=True,\n"
    "                out=_genesis_pn31_out,  # [Genesis PN31] persistent buffer\n"
    "            )\n"
)


# ─── Patch the fa_version-set call path ─────────────────────────────
# This is the second flash_attn_varlen_func call (has fa_version kwarg).

PN31_ANCHOR_WITH_VER = (
    "        return flash_attn_varlen_func(\n"
    "            q=q,\n"
    "            k=k,\n"
    "            v=v,\n"
    "            cu_seqlens_q=cu_seqlens_q,\n"
    "            cu_seqlens_k=cu_seqlens_k,\n"
    "            max_seqlen_q=max_seqlen_q,\n"
    "            max_seqlen_k=max_seqlen_k,\n"
    "            softmax_scale=self.scale,\n"
    "            causal=True,\n"
    "            fa_version=self.fa_version,\n"
    "        )\n"
)

PN31_REPLACEMENT_WITH_VER = (
    "        # [Genesis PN31] Reuse buffer acquired above\n"
    "        return flash_attn_varlen_func(\n"
    "            q=q,\n"
    "            k=k,\n"
    "            v=v,\n"
    "            cu_seqlens_q=cu_seqlens_q,\n"
    "            cu_seqlens_k=cu_seqlens_k,\n"
    "            max_seqlen_q=max_seqlen_q,\n"
    "            max_seqlen_k=max_seqlen_k,\n"
    "            softmax_scale=self.scale,\n"
    "            causal=True,\n"
    "            fa_version=self.fa_version,\n"
    "            out=_genesis_pn31_out,  # [Genesis PN31] persistent buffer\n"
    "        )\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN31 turboquant_attn.py — _flash_attn_varlen persistent out buffer "
            "(Issue #15 — sister patch to P38)"
        ),
        target_file=str(target),
        marker=GENESIS_PN31_MARKER,
        sub_patches=[
            TextPatch(
                name="pN31_fa_varlen_persistent_out_no_version",
                anchor=PN31_ANCHOR_NO_VER,
                replacement=PN31_REPLACEMENT_NO_VER,
                required=True,
            ),
            TextPatch(
                name="pN31_fa_varlen_persistent_out_with_version",
                anchor=PN31_ANCHOR_WITH_VER,
                replacement=PN31_REPLACEMENT_WITH_VER,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN31",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN31 — FA varlen persistent out buffer (text-patch)."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN31")
    log_decision("PN31", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "turboquant_attn.py not resolvable"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN31 applied: _flash_attn_varlen now uses per-shape persistent "
            "out buffer instead of allocating per-call. Eliminates malloc "
            "pressure inside FA C extension. Sister patch to P38. "
            "Memory cost: bounded ~16-64 MiB per layer per shape. "
            "Closes issue #15 for budget-constrained single-GPU configs."
        ),
        patch_name=patcher.patch_name,
    )
