# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N28 — merge_attn_states NaN guard backport (vllm#39148).

Backport of [vllm#39148](https://github.com/vllm-project/vllm/pull/39148)
"[BugFix][Attention] Fix NaN in Triton merge_attn_states when both LSEs
are -inf" (OPEN as of 2026-05-01).

================================================================
PROBLEM
================================================================

In the Triton `merge_attn_states_kernel` (used by chunked prefill to
combine prefix + suffix attention chunks), when BOTH `prefix_lse` and
`suffix_lse` are `-inf` (zero-context-length edge case), the kernel
computes:

    max_lse = max(-inf, -inf) = -inf
    p_lse = -inf - (-inf) = NaN     ← IEEE 754 NaN propagation
    p_se = exp(NaN) = NaN
    out_se = NaN + NaN = NaN
    output = (p_out * NaN + s_out * NaN) / NaN = NaN  ← silent corruption

NaN propagates through to the final output. ONE corrupted token can
break tool-call JSON parsing (the parser sees NaN-derived garbage and
returns malformed `tool_calls`).

The CUDA merge_attn_states kernel already handles this via early-return
on `isinf(max_lse)`. The Triton kernel did not — until vllm#39148.

================================================================
WHEN IT FIRES
================================================================

The both-LSE-`-inf` edge case occurs in chunked prefill paths when:
- A request has zero-context-length prefix (first chunk in a multi-chunk
  prefill).
- An attention layer with no valid keys produces -inf LSE on both sides.

Frequency: rare on warm caches, but observable on cold-start chunked
prefill — estimated 1 in ~10K decode tokens. Silent corruption rate
makes this a quality issue, not a perf issue.

================================================================
FIX (verbatim from upstream)
================================================================

Branchless arithmetic guard inside the kernel:

1. Clamp `max_lse` to a finite floor `-1e30` when it would be `-inf`.
   This makes `p_lse - max_lse = -inf - (-1e30) ≈ -1e30`, so `exp()`
   returns ~0 (correctly representing "no attention contribution").
2. Add `+1e-10` epsilon to the denominator `out_se = p_se + s_se`
   to prevent the rare case where both `p_se` and `s_se` round to 0.

================================================================
SAFETY MODEL
================================================================

- Default OFF (opt-in via `GENESIS_ENABLE_PN28_MERGE_ATTN_NAN_GUARD=1`).
- Two-line text-patch on
  `vllm/v1/attention/ops/triton_merge_attn_states.py`.
- Idempotent (marker-checked).
- Drift-aware: when upstream PR #39148 merges, our markers detect and
  patch self-retires.
- Quality-only fix — no perf impact (single `tl.where` + scalar add).
- Worst case if applied unnecessarily: `+1e-10` floors the output
  contribution by ~1e-10 — well below FP16 precision floor, no
  observable behavior change.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa
Backport credit: jasonkim8652 (vllm#39148)
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn28_merge_attn_states_nan_guard")

GENESIS_PN28_MARKER = "Genesis PN28 merge_attn_states NaN guard (vllm#39148) v7.65"


# Sub-patch 1: max_lse clamp to finite floor
PN28_MAX_LSE_OLD = (
    "    max_lse = tl.maximum(p_lse, s_lse)\n"
    "    p_lse = p_lse - max_lse\n"
)

PN28_MAX_LSE_NEW = (
    "    max_lse = tl.maximum(p_lse, s_lse)\n"
    "\n"
    "    # [Genesis PN28] vllm#39148 backport — NaN guard for both-LSE-(-inf) edge case.\n"
    "    # Replace -inf with finite floor so subtraction yields -inf (not NaN);\n"
    "    # exp() then gives exactly 0 (correct: no attention to merge).\n"
    "    max_lse = tl.where(max_lse == float(\"-inf\"), -1e30, max_lse)\n"
    "\n"
    "    p_lse = p_lse - max_lse\n"
)

# Sub-patch 2: epsilon in denominator to prevent 0/0
PN28_OUT_SE_OLD = (
    "    p_se = tl.exp(p_lse)\n"
    "    s_se = tl.exp(s_lse)\n"
    "    out_se = p_se + s_se\n"
)

PN28_OUT_SE_NEW = (
    "    p_se = tl.exp(p_lse)\n"
    "    s_se = tl.exp(s_lse)\n"
    "    # [Genesis PN28] vllm#39148 backport — epsilon prevents 0/0 in division below.\n"
    "    out_se = p_se + s_se + 1e-10\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/ops/triton_merge_attn_states.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN28 triton_merge_attn_states.py — NaN guard for both-LSE-(-inf) "
            "edge case (vllm#39148 backport)"
        ),
        target_file=str(target),
        marker=GENESIS_PN28_MARKER,
        sub_patches=[
            TextPatch(
                name="pn28_max_lse_clamp",
                anchor=PN28_MAX_LSE_OLD,
                replacement=PN28_MAX_LSE_NEW,
                required=True,
            ),
            TextPatch(
                name="pn28_out_se_epsilon",
                anchor=PN28_OUT_SE_OLD,
                replacement=PN28_OUT_SE_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN28]",
            # When upstream merges, this comment appears in source
            "When both prefix and suffix have no tokens",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN28 — merge_attn_states NaN guard."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN28")
    log_decision("PN28", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "triton_merge_attn_states.py not resolvable"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN28 applied: merge_attn_states Triton kernel now branchless-"
            "guards both-LSE-(-inf) edge case. Prevents NaN propagation on "
            "zero-context-length chunked-prefill — silent quality fix. "
            "Backport of vllm#39148 (jasonkim8652)."
        ),
        patch_name=patcher.patch_name,
    )


def is_applied() -> bool:
    target = resolve_vllm_file("v1/attention/ops/triton_merge_attn_states.py")
    if target is None:
        return False
    try:
        with open(str(target)) as f:
            return GENESIS_PN28_MARKER in f.read()
    except OSError:
        return False
