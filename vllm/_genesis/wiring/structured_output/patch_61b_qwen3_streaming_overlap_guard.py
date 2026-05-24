# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 61b — Qwen3 streaming `<tool_call>` partial-tag overlap guard.

Backport slice of vllm-project/vllm#40783 (ExtReMLapin), streaming-defensive layer.

================================================================
Companion to P61 (which fixes non-streaming multi-tool first-occurrence).
P61b adds defensive guard against emitting partial `<tool_call>` tag
fragments as reasoning when the tag is being assembled across multiple
streaming deltas.
================================================================

What this fixes
---------------
For Qwen3 with `<tool_call>` as a registered SPECIAL token (single token ID),
P12's existing logic catches `<tool_call>` immediately when its token ID
appears in `delta_token_ids`. So fragmented assembly is rare for Qwen3.

But edge cases exist:
  - Other tokenizers may not treat `<tool_call>` as special → emits character
    fragments
  - Speculative decoding may split tokens differently across delta calls
  - Custom chat templates may emit the tag literal-character-by-character

Without this guard, a delta ending in e.g. `<tool_` may be emitted as
reasoning before the next delta completes the tag — leaking partial XML
into the reasoning channel which clients then double-emit.

What we add
-----------
Single small modification in `extract_reasoning_streaming` (BEFORE the
"still in reasoning" fallback emission):

  - Import `partial_tag_overlap` from vllm.tool_parsers.utils
  - After all `<tool_call>` token-id and explicit-text checks, compute
    `overlap = partial_tag_overlap(current_text, self._tool_call_tag)`
  - If overlap > 0, hold back the `overlap` trailing characters from the
    reasoning emission (return shorter reasoning OR empty DeltaMessage)

The next delta call will see the completed tag and route appropriately.

Risks
-----
- Adds latency per streaming delta (one extra string scan, O(len(tag))).
  Negligible for `<tool_call>` (10 chars).
- For tokenizers where `<tool_call>` is special (Qwen3), this is a no-op
  guard — overlap will always be 0 because the tag arrives as one token.
- May introduce one extra empty DeltaMessage when overlap > 0. Streaming
  clients should handle this gracefully (skip).

Status: opt-in (`GENESIS_ENABLE_P61B_STREAMING_OVERLAP=1`).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher, TextPatchResult, TextPatch,
)

log = logging.getLogger("genesis.wiring.p61b_qwen3_streaming_overlap")

GENESIS_P61B_MARKER = "Genesis P61b Qwen3 streaming partial-tag overlap guard v7.13"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_P61B_STREAMING_OVERLAP", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# Sub-patch 1: import partial_tag_overlap helper
IMPORT_OLD = (
    "from vllm.entrypoints.openai.engine.protocol import DeltaMessage\n"
    "from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser"
)

IMPORT_NEW = (
    "from vllm.entrypoints.openai.engine.protocol import DeltaMessage\n"
    "from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser\n"
    "from vllm.tool_parsers.utils import partial_tag_overlap  # [Genesis P61b]"
)

# Sub-patch 2: insert overlap guard BEFORE the final "still in reasoning"
# emission. Anchor on the P12 fallback layout:
#
#         else:
#             # No end token yet: still in reasoning phase.
#             return DeltaMessage(reasoning=delta_text)
#
# We replace the `else: return DeltaMessage(reasoning=delta_text)` with a
# version that first checks overlap.
FALLBACK_OLD = (
    "        else:\n"
    "            # No end token yet: still in reasoning phase.\n"
    "            return DeltaMessage(reasoning=delta_text)"
)

FALLBACK_NEW = (
    "        else:\n"
    "            # [Genesis P61b vllm#40783] partial-tag overlap guard:\n"
    "            # avoid emitting half-formed <tool_call> as reasoning if the\n"
    "            # tag is being assembled across multiple deltas.\n"
    "            try:\n"
    "                _p61b_overlap = partial_tag_overlap(\n"
    "                    current_text, self._tool_call_tag or \"\"\n"
    "                )\n"
    "            except Exception:\n"
    "                _p61b_overlap = 0\n"
    "            if _p61b_overlap > 0:\n"
    "                _p61b_send_len = len(delta_text) - _p61b_overlap\n"
    "                if _p61b_send_len > 0:\n"
    "                    return DeltaMessage(reasoning=delta_text[:_p61b_send_len])\n"
    "                # Hold back this delta entirely; next delta completes the tag\n"
    "                return DeltaMessage()\n"
    "            # No end token yet: still in reasoning phase.\n"
    "            return DeltaMessage(reasoning=delta_text)"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("reasoning/qwen3_reasoning_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P61b Qwen3 streaming overlap guard",
        target_file=str(target),
        marker=GENESIS_P61B_MARKER,
        sub_patches=[
            TextPatch(name="p61b_import", anchor=IMPORT_OLD,
                      replacement=IMPORT_NEW, required=True),
            TextPatch(name="p61b_overlap_guard", anchor=FALLBACK_OLD,
                      replacement=FALLBACK_NEW, required=True),
        ],
        upstream_drift_markers=[
            "partial_tag_overlap(current_text",  # upstream-merged version
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P61b")
    log_decision("P61b", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "qwen3_reasoning_parser.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "P61b applied: overlap guard active in streaming path"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return "skipped", f"{msg} — likely upstream merged or anchor drifted"
    return "failed", failure.reason if failure else "unknown failure"
