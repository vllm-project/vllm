# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 27 — Qwen3 reasoning parser BEFORE-THINK fallback.

Problem
-------
Upstream `Qwen3ReasoningParser.extract_reasoning` strips the `<think>` prefix
by partitioning the output and KEEPING ONLY the post-`<think>` side (line 77-79
of qwen3_reasoning_parser.py). Any content the model emits BEFORE it emits
`<think>` is silently dropped.

Concretely:

    model_output = "Here is my answer. <think>Let me check.</think>42"
    #                                   ^^^^^^^^                  ^^
    #                                   stripped                  content=42
    #              ^^^^^^^^^^^^^^^^^^^ DISCARDED (quality regression)

Users report missed summary / scaffolding text in pre-reasoning position
(vLLM issue #40699-class). The streaming path has the same bug: `<think>`
is stripped, any text before it in the delta is lost.

Fix
---
Capture the BEFORE-THINK text and append it to `content` on the
non-streaming path, or emit it as a `DeltaMessage(content=...)` on the
streaming path.

Platform compatibility: vendor-agnostic — pure Python parser logic.

Upstream drift detection: if `before_think` or `pre_think_content` appears
in the file, we assume upstream has merged an equivalent fix and skip.

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

log = logging.getLogger("genesis.wiring.p27_reasoning_before_think")

GENESIS_P27_MARKER = "Genesis P27 Qwen3 BEFORE-THINK fallback v7.0"

UPSTREAM_DRIFT_MARKERS = [
    "before_think",
    "pre_think_content",
    "BEFORE_THINK",
]


# Non-streaming path: extract_reasoning partitions but discards
# model_output_parts[0] (the BEFORE-THINK text). Fix = append before_think
# to content.
_OLD_NONSTREAM = (
    "        # Strip <think> if present in the generated output.\n"
    "        model_output_parts = model_output.partition(self.start_token)\n"
    "        model_output = (\n"
    "            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]\n"
    "        )"
)

_NEW_NONSTREAM = (
    "        # Strip <think> if present in the generated output.\n"
    "        # [Genesis P27] Capture BEFORE-THINK text so it can be prepended\n"
    "        # to content instead of being silently dropped (issue #40699-class).\n"
    "        model_output_parts = model_output.partition(self.start_token)\n"
    "        _genesis_before_think = (\n"
    "            model_output_parts[0] if model_output_parts[1] else \"\"\n"
    "        )\n"
    "        model_output = (\n"
    "            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]\n"
    "        )"
)

# Non-streaming path — original baseline (fe9c3d6c5f and earlier):
# `final_content` aliasing + separate return line.
_OLD_NONSTREAM_RETURN_BASELINE = (
    "        # Extract reasoning content from the model output.\n"
    "        reasoning, _, content = model_output.partition(self.end_token)\n"
    "\n"
    "        final_content = content or None\n"
    "        return reasoning, final_content"
)

_NEW_NONSTREAM_RETURN_BASELINE = (
    "        # Extract reasoning content from the model output.\n"
    "        reasoning, _, content = model_output.partition(self.end_token)\n"
    "\n"
    "        # [Genesis P27] Prepend any BEFORE-THINK text to content so it\n"
    "        # is surfaced to the user instead of silently dropped.\n"
    "        if _genesis_before_think:\n"
    "            content = _genesis_before_think + (content or \"\")\n"
    "\n"
    "        final_content = content or None\n"
    "        return reasoning, final_content"
)

# Non-streaming path — upstream layout AFTER PR #35687 merged (2026-04-24):
# `if self.end_token in model_output:` conditional replaces the old "early
# return on missing end token" dance. Content is inlined into `return`.
# This form also drops BEFORE-THINK text, so our fix is still required.
_OLD_NONSTREAM_RETURN_PR35687 = (
    "        if self.end_token in model_output:\n"
    "            reasoning, _, content = model_output.partition(self.end_token)\n"
    "            return reasoning, content or None"
)

_NEW_NONSTREAM_RETURN_PR35687 = (
    "        if self.end_token in model_output:\n"
    "            reasoning, _, content = model_output.partition(self.end_token)\n"
    "            # [Genesis P27] Prepend BEFORE-THINK text to content so it\n"
    "            # is surfaced to the user instead of silently dropped.\n"
    "            if _genesis_before_think:\n"
    "                content = _genesis_before_think + (content or \"\")\n"
    "            return reasoning, content or None"
)

# Streaming path: when <think> is in the delta, text before it is stripped
# and lost. Emit the stripped BEFORE-THINK portion as a content delta.
_OLD_STREAM_START = (
    "        # Strip <think> from delta if present (old template / edge case\n"
    "        # where the model generates <think> itself).\n"
    "        if self.start_token_id in delta_token_ids:\n"
    "            start_idx = delta_text.find(self.start_token)\n"
    "            if start_idx >= 0:\n"
    "                delta_text = delta_text[start_idx + len(self.start_token) :]"
)

_NEW_STREAM_START = (
    "        # Strip <think> from delta if present (old template / edge case\n"
    "        # where the model generates <think> itself).\n"
    "        # [Genesis P27] Preserve BEFORE-THINK text as a content delta\n"
    "        # (quality fix for #40699-class regressions).\n"
    "        _genesis_pre_think_content = \"\"\n"
    "        if self.start_token_id in delta_token_ids:\n"
    "            start_idx = delta_text.find(self.start_token)\n"
    "            if start_idx > 0:\n"
    "                _genesis_pre_think_content = delta_text[:start_idx]\n"
    "            if start_idx >= 0:\n"
    "                delta_text = delta_text[start_idx + len(self.start_token) :]\n"
    "        if _genesis_pre_think_content:\n"
    "            # Emit the BEFORE-THINK slice as content before reasoning begins.\n"
    "            # Downstream still receives remaining delta_text on the next call.\n"
    "            return DeltaMessage(content=_genesis_pre_think_content)"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("reasoning/qwen3_reasoning_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P27 Qwen3 BEFORE-THINK fallback",
        target_file=target,
        marker=GENESIS_P27_MARKER,
        sub_patches=[
            # Stable anchor: `<think>` strip + partition is unchanged by
            # upstream PR #35687. Hard requirement.
            TextPatch(
                name="p27_nonstream_capture",
                anchor=_OLD_NONSTREAM,
                replacement=_NEW_NONSTREAM,
                required=True,
            ),
            # Two alternate return-block anchors — one matches pre-#35687
            # baseline, one matches post-#35687 upstream layout. EXACTLY
            # ONE is expected to match any given file; both are optional
            # (siblings continue). If both miss, the whole patch yields a
            # no_applicable_sub_patches skip (benign).
            TextPatch(
                name="p27_nonstream_return_baseline",
                anchor=_OLD_NONSTREAM_RETURN_BASELINE,
                replacement=_NEW_NONSTREAM_RETURN_BASELINE,
                required=False,
            ),
            TextPatch(
                name="p27_nonstream_return_pr35687",
                anchor=_OLD_NONSTREAM_RETURN_PR35687,
                replacement=_NEW_NONSTREAM_RETURN_PR35687,
                required=False,
            ),
            TextPatch(
                name="p27_stream_start",
                anchor=_OLD_STREAM_START,
                replacement=_NEW_STREAM_START,
                required=False,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def apply() -> tuple[str, str]:
    """Apply P27 wiring. Never raises."""
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "qwen3_reasoning_parser.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "BEFORE-THINK fallback wired (non-stream + stream)"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"
