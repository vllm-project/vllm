# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N11 — Force a/b contiguity in GatedDeltaNetAttention.

================================================================
Source PR
================================================================
https://github.com/vllm-project/vllm/pull/41142
"[Bugfix][Qwen3-Next] Force a/b contiguity in fix_query_key_value_ordering"
by @Yeuvoir, OPEN as of 2026-04-29.

Fixes upstream issue: https://github.com/vllm-project/vllm/issues/41112

================================================================
WHAT IT DOES
================================================================

In `GatedDeltaNetAttention.fix_query_key_value_ordering`
(`vllm/model_executor/layers/mamba/gdn_linear_attn.py`), the final reshape
of `b` and `a`:

    b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
    a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

returns a NON-CONTIGUOUS view when `num_v_heads == num_k_heads` (i.e.
the GDN config has `np/ng == 1`): the size-1 inner dim from the prior
`torch.split` is squeezed without copying, leaving strides like
`(2K, 2)` on the `[B, num_v_heads]` output.

This breaks `fused_post_conv_prep` (introduced in vllm#37813), whose
Triton kernel indexes `a`/`b` as:

    a_offsets = offs_t * stride_a_tok + i_hv
    b_offsets = offs_t * stride_b_tok + i_hv

— it implicitly assumes the head dim has stride 1. On a non-contiguous
input, this silently mis-indexes and produces corrupted GDN gating
state. Symptom: subtle quality drift, no hard crash.

The fix is a 2-line `.contiguous()` add (with a 3-line comment).
**Zero performance cost** — `.contiguous()` is a no-op when the tensor
is already contiguous (the strict-GQA case). Only triggers a copy when
`num_v_heads == num_k_heads`, which is the buggy path.

================================================================
APPLICABILITY
================================================================

Hits any model that uses `GatedDeltaNetAttention` with
`num_v_heads == num_k_heads` AND a recent enough vllm to ship
`fused_post_conv_prep` (post #37813). For Genesis:

- Qwen3.6-27B-int4-AutoRound (Lorbus) — hybrid GDN, num_v_heads=16,
  num_k_heads=2 → np/ng=8, NOT triggered.
- Qwen3.6-27B-INT8 (Minachist) — same arch.
- Qwen3.6-35B-A3B-FP8 — Qwen3MoE, NO GDN layers, NOT triggered.
- Qwen3-Next family models — variable; some configs hit np/ng=1 and ARE
  affected.

So on our exact prod stack the patch is largely defensive — installs
the contiguity guard so future model swaps or upstream architecture
tweaks can't quietly degrade quality. Cost: zero.

================================================================
SAFETY MODEL
================================================================

- Default OFF (opt-in via `GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS=1`).
- Pure text-patch, idempotent via marker.
- Drift-aware: if upstream merges PR #41142 (or equivalent), the
  marker `.contiguous()` on those exact lines triggers self-retirement.
- Anchor missing → SKIPPED, source stays vanilla. Zero regression risk.
- Worst case: an extra `.contiguous()` call on a path that was already
  contiguous = a single CPU branch + no GPU work.

Author backport: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Source PR: vllm-project/vllm#41142 by @Yeuvoir.
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pN11_gdn_a_b_contiguous")

GENESIS_PN11_MARKER = (
    "Genesis PN11 GDN a/b contiguity (vllm#41142) v7.62.x"
)


# ─── Sub-patch: add .contiguous() to b and a reshape lines ─────────────

# Anchor matches the exact two consecutive lines as they appear today
# (with leading spaces preserved). Indentation: 8 spaces (inside method).
PN11_ANCHOR = (
    "        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)\n"
    "        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)\n"
)

PN11_REPLACEMENT = (
    "        # [Genesis PN11 vllm#41142 backport] Force contiguity:\n"
    "        # when num_v_heads == num_k_heads (np/ng == 1) the reshape\n"
    "        # returns a non-contiguous view, breaking fused_post_conv_prep\n"
    "        # (which assumes head-dim stride 1). Zero cost when already\n"
    "        # contiguous; only copies on the buggy path.\n"
    "        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size).contiguous()\n"
    "        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size).contiguous()\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/mamba/gdn_linear_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN11 model_executor/layers/mamba/gdn_linear_attn.py — "
            "force a/b contiguity in fix_query_key_value_ordering (vllm#41142)"
        ),
        target_file=str(target),
        marker=GENESIS_PN11_MARKER,
        sub_patches=[
            TextPatch(
                name="pN11_a_b_contiguous",
                anchor=PN11_ANCHOR,
                replacement=PN11_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN11",
            # If upstream PR #41142 lands, these lines will already have
            # `.contiguous()` and our anchor won't match → no-op apply.
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN11 — force GDN a/b contiguity (text-patch)."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN11")
    log_decision("PN11", decision, reason)
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
            "PN11 applied: GatedDeltaNetAttention.fix_query_key_value_ordering "
            "now forces .contiguous() on `b` and `a` after reshape (vllm#41142). "
            "Defensive — eliminates silent quality drift class for any future "
            "Qwen3-Next/GDN config with num_v_heads == num_k_heads."
        ),
        patch_name=patcher.patch_name,
    )
