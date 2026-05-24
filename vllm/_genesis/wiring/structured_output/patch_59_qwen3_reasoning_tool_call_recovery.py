# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 59 — Qwen3 reasoning embedded tool_call recovery.

Backport of vllm-project/vllm#39055 (ZenoAFfectionate, OPEN at time of writing).

================================================================
EMPIRICAL CANDIDATE for #40831 / our degenerate-output bug, after
P58 (#40768 backport) was empirically disproven 2026-04-25.
================================================================

What the bug looks like
-----------------------
On Qwen3.5-35B-A3B-FP8 / Qwen3.6-35B-A3B-FP8 with:

  - --reasoning-parser qwen3
  - --tool-call-parser qwen3_coder

tool-calling requests (`tools=[...]`) sometimes return:

  - empty `tool_calls` list with populated `reasoning`
  - OR garbage fragments like `parameter=city`, `<<argname>`, `</parameter`
    leaking into JSON arguments string
  - OR `<tool_call><<parameter name=...>` patterns (extra `<` + tag corruption)

Plain text requests (no `tools`) on the same model are clean.

Why it happens (per PR #39055 design doc)
-----------------------------------------
1. Model emits XML tool-call markup INSIDE the `<think>...</think>` block:

      <think>
      ... reasoning text ...
      <tool_call>
      <function=Finish>
      <parameter=answer>
      204
      </parameter>
      </function>
      </tool_call>
      </think>

2. `qwen3_reasoning_parser.extract_reasoning` partitions on `</think>` and
   puts everything before it (including the embedded `<tool_call>` block)
   into the `reasoning` field.

3. Downstream `qwen3_coder` tool parser only inspects `content`. The valid
   tool_call XML in `reasoning` never reaches it.

4. Result: empty `tool_calls`, OR fragments of incomplete XML elsewhere in
   the model output get mis-parsed as garbage tokens.

Note: our existing P12 already adds awareness of `<tool_call>` as an
implicit reasoning-end marker, but only triggers when `</think>` is
ABSENT from the output. If `</think>` is present and a `<tool_call>`
block is nested before it, P12's branch doesn't help. P59 is additive:
adds extraction of nested tool_call blocks regardless of where `</think>`
sits.

Community confirmations on PR #39055
------------------------------------
- @meitalbensinai: "Also happens for me with the new Qwen 3.6 30b" (our family)
- @epheien: "encountered with both 27b and 397b in streaming"
- @jogoossens: "very hard to get qwen stable on vllm"

Status: opt-in (`GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY=1`).

Compatibility
-------------
- Composes cleanly with P12 (Qwen3 tool_call reasoning fix v2): P12 handles
  the `</think>`-absent case via `<tool_call>` implicit-end; P59 handles
  the `</think>`-present case where `<tool_call>` is nested in reasoning.
- Auto-no-op once #39055 lands upstream (drift marker:
  `_split_embedded_tool_calls` in the file).
- Non-Qwen3 deployments: parser file simply isn't loaded → patcher skips.

Risks acknowledged
------------------
- The PR fix uses a regex to detect `<tool_call>` blocks. Edge cases:
  malformed/truncated XML inside reasoning may be partially extracted.
  PR #39055's tests cover this; we lean on those.
- Streaming path is NOT addressed by this PR (PR author's own caveat).
  Our streaming clients hit a separate bug class (#40816 family).
- This patch is in the parser layer, NOT model generation. Model behavior
  unchanged. Worst case: extraction fails on edge cases → reasoning text
  preserved as-is, parse fallback to original behavior.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Investigation supported by AI tooling for source navigation.
"""
from __future__ import annotations

import logging

# Audit A-19 (2026-05-05): tightly coupled subpatches — both apply
# or both stay un-applied. Shared marker is acceptable here because the
# subpatches together form one logical fix; partial application is not
# desired anyway. _AUDIT_A19_EXEMPT documents this intentional design.
_AUDIT_A19_EXEMPT = True  # tightly coupled subpatches
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.p59_qwen3_reasoning_tool_call_recovery")

GENESIS_P59_MARKER = "Genesis P59 Qwen3 reasoning embedded tool_call recovery v7.13"


def _is_enabled() -> bool:
    """Env-gate. Off by default — opt-in via:
    GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY=1
    """
    return os.environ.get(
        "GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# ─── Sub-patch 1: add `import re` after the existing collections import ─────

IMPORT_OLD = (
    "from collections.abc import Sequence\n"
    "from typing import TYPE_CHECKING"
)

IMPORT_NEW = (
    "import re  # [Genesis P59 vllm#39055]\n"
    "from collections.abc import Sequence\n"
    "from typing import TYPE_CHECKING"
)


# ─── Sub-patch 2: insert _EMBEDDED_TOOL_CALL_RE module-level constant ─────
# Anchor on the blank line just before `class Qwen3ReasoningParser`.

REGEX_OLD = (
    "if TYPE_CHECKING:\n"
    "    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest\n"
    "    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest\n"
    "    from vllm.tokenizers import TokenizerLike\n"
    "\n"
    "\n"
    "class Qwen3ReasoningParser(BaseThinkingReasoningParser):"
)

REGEX_NEW = (
    "if TYPE_CHECKING:\n"
    "    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest\n"
    "    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest\n"
    "    from vllm.tokenizers import TokenizerLike\n"
    "\n"
    "\n"
    "# [Genesis P59 vllm#39055] regex for extracting nested tool_call blocks.\n"
    "_EMBEDDED_TOOL_CALL_RE = re.compile(\n"
    "    r\"<tool_call>(.*?)</tool_call>|<tool_call>.*$\",\n"
    "    re.DOTALL,\n"
    ")\n"
    "\n"
    "\n"
    "class Qwen3ReasoningParser(BaseThinkingReasoningParser):"
)


# ─── Sub-patch 3: insert _split_embedded_tool_calls staticmethod ───────────
# Anchor on the `end_token` property's body so we can insert just after it.

# Anchor only on the end_token property body — the line that follows differs
# between v5.12 monolith P12 (`is_reasoning_end(self, input_ids: Sequence[int])`)
# and modular P12 (`is_reasoning_end(self, input_ids):`). Insert helper right
# after end_token's return statement so it's the FIRST sibling member.
METHOD_OLD = (
    "    @property\n"
    "    def end_token(self) -> str:\n"
    "        \"\"\"The token that ends reasoning content.\"\"\"\n"
    "        return \"</think>\""
)

METHOD_NEW = (
    "    @property\n"
    "    def end_token(self) -> str:\n"
    "        \"\"\"The token that ends reasoning content.\"\"\"\n"
    "        return \"</think>\"\n"
    "\n"
    "    @staticmethod\n"
    "    def _split_embedded_tool_calls(\n"
    "        reasoning,\n"
    "        content,\n"
    "    ):\n"
    "        \"\"\"[Genesis P59 vllm#39055] Promote tool_call XML out of reasoning.\n"
    "\n"
    "        Qwen3.5/3.6 models can emit XML tool calls before </think>. The\n"
    "        downstream tool parser only inspects content, so embedded tool\n"
    "        calls would otherwise be lost. This helper extracts well-formed\n"
    "        <tool_call>...</tool_call> blocks from reasoning and prepends\n"
    "        them to content so qwen3_coder can parse them normally.\n"
    "        \"\"\"\n"
    "        if (\n"
    "            not reasoning\n"
    "            or \"<tool_call>\" not in reasoning\n"
    "            or \"<function=\" not in reasoning\n"
    "        ):\n"
    "            return reasoning, content\n"
    "\n"
    "        extracted_blocks = []\n"
    "\n"
    "        def _collect_or_keep(match):\n"
    "            block = match.group(0)\n"
    "            if \"<function=\" not in block:\n"
    "                return block\n"
    "            extracted_blocks.append(block.strip())\n"
    "            return \"\"\n"
    "\n"
    "        remaining_reasoning = _EMBEDDED_TOOL_CALL_RE.sub(\n"
    "            _collect_or_keep, reasoning\n"
    "        )\n"
    "        remaining_reasoning = remaining_reasoning.strip() or None\n"
    "\n"
    "        if not extracted_blocks:\n"
    "            return reasoning, content\n"
    "\n"
    "        content_parts = [\"\\n\\n\".join(extracted_blocks)]\n"
    "        if content:\n"
    "            content_parts.append(content)\n"
    "        merged_content = \"\\n\\n\".join(\n"
    "            part for part in content_parts if part\n"
    "        ) or None\n"
    "        return remaining_reasoning, merged_content"
)


# ─── Sub-patches 4a/4b: wrap the </think>-present return ─────────────────
# Two variants because P12 (monolith v5.12) and P12+P27 (modular) produce
# different code shapes around the return statement. Both are required=False;
# we expect EXACTLY ONE to match per file layout.

# Variant A: monolith P12 layout (one-line return)
RETURN_THINK_MONOLITH_OLD = (
    "        # [Genesis v5.12] PR #35687: 3-way branch with <tool_call>\n"
    "        if self.end_token in model_output:\n"
    "            reasoning, _, content = model_output.partition(self.end_token)\n"
    "            return reasoning, content or None"
)

RETURN_THINK_MONOLITH_NEW = (
    "        # [Genesis v5.12] PR #35687: 3-way branch with <tool_call>\n"
    "        if self.end_token in model_output:\n"
    "            reasoning, _, content = model_output.partition(self.end_token)\n"
    "            # [Genesis P59 vllm#39055] extract nested tool_call from reasoning\n"
    "            return self._split_embedded_tool_calls(reasoning, content or None)"
)

# Variant B: modular P12+P27 layout (final_content variable + bare return)
RETURN_THINK_MODULAR_OLD = (
    "        final_content = content or None\n"
    "        return reasoning, final_content"
)

RETURN_THINK_MODULAR_NEW = (
    "        final_content = content or None\n"
    "        # [Genesis P59 vllm#39055] extract nested tool_call from reasoning\n"
    "        return self._split_embedded_tool_calls(reasoning, final_content)"
)


# ─── Sub-patch 5: wrap the truncated-output return ───────────────────────
# Same shape in both layouts (P12/P27 don't touch this branch).

RETURN_TRUNC_OLD = (
    "        # Thinking enabled but no </think>: output was truncated.\n"
    "        # Everything generated so far is reasoning.\n"
    "        return model_output, None"
)

RETURN_TRUNC_NEW = (
    "        # Thinking enabled but no </think>: output was truncated.\n"
    "        # Everything generated so far is reasoning.\n"
    "        # [Genesis P59 vllm#39055] still try to extract embedded tool_call\n"
    "        return self._split_embedded_tool_calls(model_output, None)"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("reasoning/qwen3_reasoning_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P59 Qwen3 reasoning embedded tool_call recovery",
        target_file=str(target),
        marker=GENESIS_P59_MARKER,
        sub_patches=[
            TextPatch(
                name="p59_import_re",
                anchor=IMPORT_OLD,
                replacement=IMPORT_NEW,
                required=True,
            ),
            TextPatch(
                name="p59_module_regex",
                anchor=REGEX_OLD,
                replacement=REGEX_NEW,
                required=True,
            ),
            TextPatch(
                name="p59_helper_method",
                anchor=METHOD_OLD,
                replacement=METHOD_NEW,
                required=True,
            ),
            # Variants A/B for the </think>-present return — required=False
            # so whichever layout is present wins; the other soft-skips.
            TextPatch(
                name="p59_wrap_think_return_monolith",
                anchor=RETURN_THINK_MONOLITH_OLD,
                replacement=RETURN_THINK_MONOLITH_NEW,
                required=False,
            ),
            TextPatch(
                name="p59_wrap_think_return_modular",
                anchor=RETURN_THINK_MODULAR_OLD,
                replacement=RETURN_THINK_MODULAR_NEW,
                required=False,
            ),
            TextPatch(
                name="p59_wrap_trunc_return",
                anchor=RETURN_TRUNC_OLD,
                replacement=RETURN_TRUNC_NEW,
                required=False,
            ),
        ],
        upstream_drift_markers=["_split_embedded_tool_calls"],
    )


def apply() -> tuple[str, str]:
    """Apply P59 wiring (5 sub-patches in one file). Never raises.

    All-or-nothing: if any required anchor drifts, abort the whole group.
    Idempotent + auto-no-op once #39055 lands upstream.
    """
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P59")
    log_decision("P59", decision, reason)
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
            "P59 backport applied: 5 sub-patches in qwen3_reasoning_parser.py. "
            "Tool_call XML inside <think>...</think> reasoning now extracted "
            "and routed to content for qwen3_coder parser. Validate with "
            "blue/green reproducer suite before serving traffic.",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        detail = failure.detail if failure else ""
        return (
            "skipped",
            f"{msg} ({detail}) — likely #39055 already merged upstream OR "
            "anchor drifted (P12-modified file changed). Re-anchor needed.",
        )
    return "failed", failure.reason if failure else "unknown failure"
