# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N29 — GDN chunk_o scale-fold (vllm#41446 pattern (c)).

================================================================
Source PR
================================================================
https://github.com/vllm-project/vllm/pull/41446
"[Kernel][AMD] Optimize GatedDeltaNet FLA prefill kernels on MI300X"
by @zobinHuang, OPEN as of 2026-05-01.

The PR contains 5 optimizations; THIS patch backports only **pattern (c)**:
scale-fold in `chunk_fwd_kernel_o`. Pattern (a) (fused fwd_h+fwd_o) and
others require deeper kernel rewrites and are out of scope here.

================================================================
WHAT IT DOES
================================================================

In `vllm/model_executor/layers/fla/ops/chunk_o.py`, the
`chunk_fwd_kernel_o` Triton kernel computes the final accumulator update
as:

    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale

This has TWO fp32 multiplications by `scale`. By distributivity:

    b_o * scale + dot * scale  ==  (b_o + dot) * scale

The folded form has ONE multiplication. Triton compiler does NOT
auto-fuse this across the addition boundary (multiply-then-add is a
hardware FMA instruction; multiply-add-multiply is two separate ops).

Per-iter savings: 1 fp32 multiply on a [BT, BV] = [64, 128] tile = 8192
operations. On chunked GDN prefill with hundreds of iterations per layer
× 36 layers, this aggregates to a measurable few-percent reduction in
arithmetic on the GDN prefill path.

Source PR claims hardware-agnostic (works on NVIDIA Triton too), bit-
equivalent on Triton ≥3.2. We add a TDD numerical equivalence test to
verify.

================================================================
APPLICABILITY
================================================================

Hits any model with hybrid GDN layers using `chunk_fwd_kernel_o`. For
Genesis prod stack:

- Qwen3.6-27B-int4-AutoRound (Lorbus) — hybrid GDN, **AFFECTED**.
- Qwen3.6-27B-INT8 (Minachist) — same arch, AFFECTED.
- Qwen3.6-35B-A3B-FP8 — Qwen3MoE, NO GDN layers, NOT triggered.

Expected gain: +1-2% on GDN-heavy workloads (prefill-dominated).

================================================================
SAFETY MODEL
================================================================

- Default OFF (opt-in via `GENESIS_ENABLE_PN29_GDN_SCALE_FOLD=1`).
- Pure text-patch, idempotent via marker.
- Drift-aware: if upstream merges PR #41446, our anchor won't match → no-op.
- Anchor missing → SKIPPED, source stays vanilla. Zero regression risk.
- Worst case: 1-2 ULP fp32 drift per element (within IEEE 754 bounds for
  distributive rearrangement). Numerical equivalence verified by TDD.

Author backport: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Source PR: vllm-project/vllm#41446 by @zobinHuang.
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pN29_gdn_chunk_o_scale_fold")

GENESIS_PN29_MARKER = (
    "Genesis PN29 GDN chunk_o scale-fold (vllm#41446 pattern c) v7.65"
)


# ─── Sub-patch: fold scale multiply in chunk_fwd_kernel_o ──────────────

# Anchor: the exact current upstream line. Indentation: 4 spaces (kernel body).
PN29_ANCHOR = (
    "    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale\n"
)

PN29_REPLACEMENT = (
    "    # [Genesis PN29 vllm#41446 pattern (c) backport]\n"
    "    # Scale-fold: (b_o + dot) * scale instead of b_o*scale + dot*scale.\n"
    "    # One fewer fp32 multiply per inner iteration. Distributive on fp32\n"
    "    # accumulators (drift bounded by 1-2 ULP per element, verified by\n"
    "    # TDD test_pn29_numerical_equivalence_*). Triton compiler does NOT\n"
    "    # auto-fuse across the +/- boundary, so explicit fold = guaranteed\n"
    "    # 1 fewer op per chunk_fwd_kernel_o iteration.\n"
    "    # Source: zobinHuang vllm#41446 GDN MI300X optimization, pattern (c)\n"
    "    # is hardware-agnostic (NVIDIA Triton compatible).\n"
    "    b_o = (b_o + tl.dot(b_A.to(b_v.dtype), b_v)) * scale\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/fla/ops/chunk_o.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN29 model_executor/layers/fla/ops/chunk_o.py — "
            "scale-fold in chunk_fwd_kernel_o (vllm#41446 pattern c)"
        ),
        target_file=str(target),
        marker=GENESIS_PN29_MARKER,
        sub_patches=[
            TextPatch(
                name="pN29_chunk_o_scale_fold",
                anchor=PN29_ANCHOR,
                replacement=PN29_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN29",
            # If upstream PR #41446 lands pattern (c), the line will already
            # have `(b_o + dot) * scale` and our anchor won't match → no-op.
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN29 — GDN chunk_o scale-fold (text-patch)."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN29")
    log_decision("PN29", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file not resolvable"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN29 applied: chunk_fwd_kernel_o now uses scale-fold "
            "(b_o + dot) * scale instead of b_o*scale + dot*scale "
            "(vllm#41446 pattern (c)). 1 fewer fp32 mul per inner iter; "
            "expected +1-2% on GDN-heavy prefill (hybrid Qwen3.5/3.6)."
        ),
        patch_name=patcher.patch_name,
    )
