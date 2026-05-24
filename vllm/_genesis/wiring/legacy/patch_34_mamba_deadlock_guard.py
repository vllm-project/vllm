# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 34 — Mamba block-aligned split zero-collapse deadlock fix.

Problem (upstream issue #40707, fixed by open PR #40757 / #40709)
-----------------------------------------------------------------
Hybrid Mamba/DeltaNet models (Qwen3.5-35B-A3B, our prod model) running with
`mamba_cache_mode="align"` can enter a permanent scheduling deadlock when
processing two adjacent large multimodal inputs (e.g. two 3024×4032 images).

Root cause: `_mamba_block_aligned_split` in `v1/core/sched/scheduler.py`
aligns `num_new_tokens` to a multiple of `block_size`. When the encoder
cache cannot hold two images simultaneously, the gap between them can be
smaller than `block_size`, and the alignment truncates `num_new_tokens`
to 0. The scheduler then treats "0 tokens to process" as "cannot
progress" but keeps the request in the waiting queue — a no-progress
loop that never completes.

Fix (3-line change, mirrors PR #40757 verbatim)
-----------------------------------------------
When alignment would collapse `num_new_tokens` to zero, keep the original
sub-block value. The Mamba running state is still correctly maintained
by `preprocess_mamba` via `mamba_state_idx` for sub-block chunks — we
only skip the block-boundary state checkpoint for this chunk, which is
consistent with the existing "simply not cached" exception documented
right above the anchor.

Platform compatibility: vendor-agnostic — Python scheduler logic only.
Model compatibility: hybrid Mamba models with `mamba_cache_mode="align"`
(our Qwen3.6-35B-A3B + Qwen3.5 family). Dense-only models are unaffected
because `need_mamba_block_aligned_split` short-circuits them.

Safety analysis (from the upstream PR body, reproduced here for
reviewers of the Genesis patch):
  - The gated condition `has_mamba_layers and mamba_cache_mode == "align"`
    means this path fires only for the model classes we already ship.
  - The fix only changes behaviour when alignment would produce 0;
    otherwise the original `num_new_tokens // block_size * block_size`
    is kept. No regression path.

Retirement condition: when PR #40757 or its duplicate PR #40709 lands in
upstream main, `aligned = ... // block_size * block_size` + `if aligned > 0`
will appear verbatim in the file — the upstream_drift_markers below
catch that and the patch self-retires as "upstream merged".

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Reference: https://github.com/vllm-project/vllm/pull/40757 (fanghao566)
           https://github.com/vllm-project/vllm/pull/40709 (anishesg)
           https://github.com/vllm-project/vllm/issues/40707 (upstream issue)
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p34_mamba_deadlock_guard")

GENESIS_P34_MARKER = "Genesis P34 Mamba zero-collapse deadlock guard v7.0"

# Upstream landed a fix either via PR #40757 or its duplicate PR #40709.
# Both introduce an `aligned = ...` intermediate with a > 0 check — if
# either appears in the file we self-retire.
UPSTREAM_DRIFT_MARKERS = [
    # Signature of the upstream fix, variant 1 (fanghao566 PR #40757).
    "aligned = num_new_tokens // block_size * block_size",
    # Alternate variable name variant.
    "aligned_num_new_tokens = num_new_tokens // block_size * block_size",
    # Any future fix that replaces the single-line alignment with a
    # `max(..., block_size)` minimum guard.
    "max(num_new_tokens // block_size * block_size",
]

# Anchor is the exact baseline line (v0.19.2rc1.dev134+gfe9c3d6c5) plus
# its two leading context lines (the `if num_computed_tokens_after_sched`
# guard) so the patcher only matches here, not in some other arithmetic
# site that happens to share the alignment idiom.
_OLD = (
    "            if num_computed_tokens_after_sched < last_cache_position:\n"
    "                # align to block_size\n"
    "                num_new_tokens = num_new_tokens // block_size * block_size"
)

_NEW = (
    "            if num_computed_tokens_after_sched < last_cache_position:\n"
    "                # align to block_size\n"
    "                # [Genesis P34] Zero-collapse deadlock guard (upstream PR #40757).\n"
    "                # When two adjacent multimodal inputs can't fit in the encoder\n"
    "                # cache simultaneously, the gap can be < block_size; aligning\n"
    "                # down then collapses to 0 and the scheduler spins forever.\n"
    "                # Keep the sub-block value when alignment would zero-out —\n"
    "                # Mamba state is still maintained by preprocess_mamba via\n"
    "                # mamba_state_idx (\"simply not cached\" exception applies).\n"
    "                aligned = num_new_tokens // block_size * block_size\n"
    "                if aligned > 0:\n"
    "                    num_new_tokens = aligned"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/sched/scheduler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P34 Mamba zero-collapse deadlock guard",
        target_file=target,
        marker=GENESIS_P34_MARKER,
        sub_patches=[
            TextPatch(
                name="p34_deadlock_guard",
                anchor=_OLD,
                replacement=_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def apply() -> tuple[str, str]:
    """Apply P34 wiring. Never raises."""
    # P53 (v7.9): Hybrid-active dispatch gate. Mamba zero-collapse deadlock
    # only hits hybrid schedulers. On dense models the scheduler path
    # never reaches the guarded branch, so the text-patch is harmless
    # but noise — skip to keep dispatch logs clean.
    try:
        from vllm._genesis.model_detect import is_hybrid_model, log_skip
        if not is_hybrid_model():
            log_skip(
                "P34 Mamba zero-collapse deadlock guard",
                "pure-attention model (no hybrid scheduler path)",
            )
            return "skipped", "P53 dispatch: model has no hybrid linear-attention layers"
    except Exception as e:
        log.debug("[Genesis P34] model_detect probe failed (proceeding): %s", e)

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "v1/core/sched/scheduler.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return (
            "applied",
            "zero-collapse deadlock guard inserted "
            "(fixes #40707 for hybrid Mamba + multimodal)",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"
