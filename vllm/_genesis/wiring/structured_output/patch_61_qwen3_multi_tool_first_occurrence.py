# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 61 — Qwen3 multi-tool first-occurrence fix.

Backport of vllm-project/vllm#40783 (ExtReMLapin), MINIMAL slice.

================================================================
PR #40783 is large (517+/32-, 3 files) — most changes target
streaming path with fragmented tag detection. We backport ONLY the
non-streaming `extract_content_ids` change which fixes multi-tool
requests where multiple `<tool_call>` blocks exist.

The streaming changes are deferred (high anchor-conflict risk with
our P12 / P27 / P59 layers on the same file).
================================================================

What this fixes
---------------
When the model emits MULTIPLE `<tool_call>...</tool_call>` blocks (e.g. a
multi-step plan), the original `extract_content_ids` finds the LAST
occurrence of `<tool_call>`. This means the FIRST tool calls are silently
dropped — only the last is preserved.

PR #40783 changes this to find FIRST occurrence, preserving all tool calls.

For our setup: most requests are single-tool, but agentic flows (sequential
tool use) would have been losing intermediate calls.

Status: opt-in (`GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL=1`).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher, TextPatchResult, TextPatch,
)

log = logging.getLogger("genesis.wiring.p61_qwen3_multi_tool_first_occurrence")

GENESIS_P61_MARKER = "Genesis P61 Qwen3 multi-tool first-occurrence v7.13"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor: P12-modified version of extract_content_ids fallback
LAST_OCCURRENCE_OLD = (
    "        # Fall back: content starts at <tool_call> (implicit reasoning end).\n"
    "        if (\n"
    "            self._tool_call_token_id is not None\n"
    "            and self._tool_call_token_id in input_ids\n"
    "        ):\n"
    "            tool_call_index = (\n"
    "                len(input_ids) - 1 - input_ids[::-1].index(self._tool_call_token_id)\n"
    "            )\n"
    "            return input_ids[tool_call_index:]"
)

FIRST_OCCURRENCE_NEW = (
    "        # [Genesis P61 vllm#40783] Fall back: content starts at FIRST <tool_call>\n"
    "        # (implicit reasoning end). Preserves multi-tool requests where\n"
    "        # multiple <tool_call> blocks exist — original code returned LAST\n"
    "        # occurrence which silently dropped earlier tool calls.\n"
    "        if (\n"
    "            self._tool_call_token_id is not None\n"
    "            and self._tool_call_token_id in input_ids\n"
    "        ):\n"
    "            tool_call_index = input_ids.index(self._tool_call_token_id)\n"
    "            return input_ids[tool_call_index:]"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("reasoning/qwen3_reasoning_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P61 Qwen3 multi-tool first-occurrence",
        target_file=str(target),
        marker=GENESIS_P61_MARKER,
        sub_patches=[
            TextPatch(
                name="p61_first_occurrence",
                anchor=LAST_OCCURRENCE_OLD,
                replacement=FIRST_OCCURRENCE_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # Distinctive upstream phrasing post-PR-40783 (without our
            # [Genesis P61] prefix). The "FIRST" marker suffices — we never
            # use that exact comment shape.
            "Fall back: content starts at the FIRST <tool_call>",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P61")
    log_decision("P61", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "qwen3_reasoning_parser.py not found"

    result, failure = patcher.apply()

    if result == TextPatchResult.APPLIED:
        return (
            "applied",
            "P61 applied: extract_content_ids now finds FIRST <tool_call> "
            "(was LAST). Multi-tool requests preserve all tool calls.",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return "skipped", f"{msg} — likely upstream merged or P12 layout drifted"
    return "failed", failure.reason if failure else "unknown failure"
