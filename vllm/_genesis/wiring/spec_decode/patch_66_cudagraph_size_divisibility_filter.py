# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 66 — cudagraph_capture_sizes divisibility filter for spec-decode.

Genesis-original — addresses vLLM bug class identified in:
  - vllm-project/vllm#28015 — "CUDA Graph Capture Issue: Unexpected Prefill
    Branches in Uniform Decode Graphs"
  - vllm-project/vllm#23679 — "refactor: uniform_cudagraph_capture_sizes
    divisible by 1+num_spec_tokens" (CLOSED, stale, never merged)

================================================================
WHAT THIS FIXES

When `uniform_decode_query_len > 1` (e.g., MTP n=3 → q_len=4) and the
cudagraph_capture_sizes set includes sizes NOT divisible by
uniform_decode_query_len, the capture phase produces mixed-q_len batches.

Example with uniform_decode_query_len=4, capture size=10:
  cdiv(10, 4) = 3 reqs
  num_scheduled_tokens_list = [4, 4, 2]   ← last req has q_len=2!

The last req with q_len=2 is classified as PREFILL during capture
(in `split_decodes_and_prefills` with `require_uniform=True`). The
captured "uniform decode" graph silently bakes a PREFILL branch.

At runtime, real decode batches re-enter that prefill branch and read
garbage attention metadata — manifests as illegal memory access OR
silent output corruption (degenerate token cascades).

Filter logic: keep only capture sizes divisible by uniform_decode_query_len.
================================================================

What our prod has WITHOUT this patch
------------------------------------
With `--performance-mode interactivity` + MTP n=3:
  cudagraph_capture_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
  uniform_decode_query_len = 4
  Divisible by 4: only [4, 8, 12, 16]
  Wasted/risky captures: [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15] — 12 of 16!

What this patch does
--------------------
With this patch:
  cudagraph_capture_sizes = [4, 8, 12, 16]  (4 captures instead of 16)

Benefits:
  - Boot time 2-4x faster (less captures during warmup)
  - Less peak GPU memory during capture (avoids OOM at high GMU)
  - No mixed-q_len batches during capture → no prefill branches baked into
    "uniform decode" captures
  - Reduces blast radius for the bug class identified in #28015 / #40880

Status: opt-in via `GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER=1`.

Compatibility
-------------
- Affects ONLY when `speculative_config.num_speculative_tokens > 0`
- For non-spec-decode setups: no change (filter is a no-op when uniform_q_len == 1)
- Idempotent (marker check)
- Auto-no-op once vllm#23679 (or equivalent) lands upstream

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
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

log = logging.getLogger("genesis.wiring.p66_cudagraph_size_divisibility_filter")

GENESIS_P66_MARKER = "Genesis P66 cudagraph_capture_sizes spec-decode divisibility filter v7.13"


# ─── Sub-patch: inject divisibility filter into _set_cudagraph_sizes ────────
# Anchor on the closing of update_sizes_for_sequence_parallelism block
# (lines 1530-1536 in upstream fe9c3d6c5). Insert filter AFTER SP block
# but BEFORE valid_max_size computation.

P66_OLD = (
    "            if (\n"
    "                self.parallel_config.tensor_parallel_size > 1\n"
    "                and self.compilation_config.pass_config.enable_sp\n"
    "            ):\n"
    "                cudagraph_capture_sizes = self.update_sizes_for_sequence_parallelism(\n"
    "                    cudagraph_capture_sizes\n"
    "                )\n"
    "\n"
    "            # user-specific compilation_config.max_cudagraph_capture_size get\n"
    "            # truncated to valid_max_size when they are inconsistent.\n"
)

P66_NEW = (
    "            if (\n"
    "                self.parallel_config.tensor_parallel_size > 1\n"
    "                and self.compilation_config.pass_config.enable_sp\n"
    "            ):\n"
    "                cudagraph_capture_sizes = self.update_sizes_for_sequence_parallelism(\n"
    "                    cudagraph_capture_sizes\n"
    "                )\n"
    "\n"
    "            # [Genesis P66] Filter cudagraph_capture_sizes for spec-decode\n"
    "            # uniform_decode_query_len divisibility. Without this filter,\n"
    "            # capture phase produces mixed-q_len batches (e.g. [4,4,2]) where\n"
    "            # the tail request gets misclassified as prefill, baking a prefill\n"
    "            # branch into the captured uniform decode graph. At runtime real\n"
    "            # decode batches replay that wrong path → degenerate output.\n"
    "            # Mirrors vllm-project/vllm#23679 (closed/stale) + #28015 (bug).\n"
    "            if self.speculative_config is not None and getattr(\n"
    "                self.speculative_config, 'num_speculative_tokens', 0\n"
    "            ):\n"
    "                _p66_uniform_q_len = 1 + self.speculative_config.num_speculative_tokens\n"
    "                if _p66_uniform_q_len > 1:\n"
    "                    _p66_orig = list(cudagraph_capture_sizes)\n"
    "                    cudagraph_capture_sizes = [\n"
    "                        _s for _s in cudagraph_capture_sizes\n"
    "                        if _s % _p66_uniform_q_len == 0\n"
    "                    ]\n"
    "                    # Always retain at least uniform_q_len itself if it fits\n"
    "                    if (\n"
    "                        _p66_uniform_q_len not in cudagraph_capture_sizes\n"
    "                        and _p66_uniform_q_len <= max_num_tokens\n"
    "                    ):\n"
    "                        cudagraph_capture_sizes.append(_p66_uniform_q_len)\n"
    "                    cudagraph_capture_sizes.sort()\n"
    "                    _p66_removed = sorted(\n"
    "                        set(_p66_orig) - set(cudagraph_capture_sizes)\n"
    "                    )\n"
    "                    if _p66_removed:\n"
    "                        logger.info(\n"
    "                            '[Genesis P66] Filtered cudagraph_capture_sizes for '\n"
    "                            'spec-decode uniform_query_len=%d: removed %d '\n"
    "                            'non-divisible sizes %s; kept %s. Prevents mixed-q_len '\n"
    "                            'capture (vllm#28015 mechanism).',\n"
    "                            _p66_uniform_q_len, len(_p66_removed),\n"
    "                            _p66_removed, cudagraph_capture_sizes,\n"
    "                        )\n"
    "\n"
    "            # user-specific compilation_config.max_cudagraph_capture_size get\n"
    "            # truncated to valid_max_size when they are inconsistent.\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("config/vllm.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P66 config/vllm.py — cudagraph_capture_sizes spec-decode filter",
        target_file=str(target),
        marker=GENESIS_P66_MARKER,
        sub_patches=[
            TextPatch(
                name="p66_size_filter",
                anchor=P66_OLD,
                replacement=P66_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P66]",
            "_p66_uniform_q_len",
            "uniform_decode_query_len divisibility",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P66 — cudagraph_capture_sizes divisibility filter."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P66")
    log_decision("P66", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/config/vllm.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        pass  # idempotent
    else:
        for m in patcher.upstream_drift_markers:
            if m in content:
                return (
                    "skipped",
                    f"upstream drift marker {m!r} in {patcher.target_file} — "
                    "size filter likely already merged upstream.",
                )
        if patcher.sub_patches[0].anchor not in content:
            return (
                "skipped",
                "required anchor (sequence-parallelism block + valid_max_size) "
                "not found — P66 cannot apply (upstream code drifted).",
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
        "P66 applied: cudagraph_capture_sizes will be filtered to sizes "
        "divisible by uniform_decode_query_len when spec-decode is active. "
        "Boot 2-4x faster, less peak GPU memory, no mixed-q_len capture risks."
    )
