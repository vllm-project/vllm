# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 12 — Qwen3 `<tool_call>` as implicit reasoning end.

Problem
-------
Qwen3.5/3.6 models occasionally emit `<tool_call>` INSIDE a `<think>` block
without ever closing the thinking with `</think>`. The vLLM serving layer
uses `is_reasoning_end(input_ids)` to decide when reasoning ended; the
baseline parser only looks for `</think>` so the entire tool invocation
stays trapped as reasoning content, never reaching the serving layer as
a real tool call.

Mirrors upstream PR #35687 (still pending merge at v7.0 baseline).

Fix
---
Three orthogonal additions to `Qwen3ReasoningParser`:

  1. `__init__` — resolve `<tool_call>` / `</tool_call>` tag token IDs via
     the tokenizer vocab (mirrors upstream PR #35687).
  2. Inject `is_reasoning_end(input_ids)` method that walks the input
     tokens backwards and returns True if a lone `<tool_call>` (not paired
     with a later `</tool_call>`) precedes any `</think>`.
  3. Inject `is_reasoning_end_streaming(input_ids, delta_ids)` — returns
     True if the base check fires OR a fresh `<tool_call>` token landed
     in this streaming delta.
  4. Inject `extract_content_ids(input_ids)` — fallback to the token
     slice starting at the last `<tool_call>` when the base class returns
     no content IDs.

Scope note: this intentionally does NOT rewrite `extract_reasoning` or the
streaming tail of the parser. Those rewrites would conflict with Patch 27's
BEFORE-THINK fallback (identical source anchors). Doing only the additive
parts keeps both patches coexistable and gets us the serving-layer hook
— which is the behavioral win. The extract_reasoning rewrite is deferred
until upstream PR #35687 lands.

Platform compatibility: vendor-agnostic — pure Python parser logic.
Model compatibility: Qwen3-family only (DeepSeek-V3 uses different parser).

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p12_tool_call_reasoning")

GENESIS_P12_MARKER = "Genesis P12 Qwen3 <tool_call> implicit reasoning end v7.0"

UPSTREAM_DRIFT_MARKERS = [
    "_tool_call_token_id",
    "tool_call_end_token_id",
    "is_reasoning_end_streaming",
]


# Sub 1: Add tool_call token lookups to __init__.
_OLD_INIT = (
    "        chat_kwargs = kwargs.get(\"chat_template_kwargs\", {}) or {}\n"
    "        # Qwen3 defaults to thinking enabled; only treat output as\n"
    "        # pure content when the user explicitly disables it.\n"
    "        self.thinking_enabled = chat_kwargs.get(\"enable_thinking\", True)"
)

_NEW_INIT = (
    "        chat_kwargs = kwargs.get(\"chat_template_kwargs\", {}) or {}\n"
    "        # Qwen3 defaults to thinking enabled; only treat output as\n"
    "        # pure content when the user explicitly disables it.\n"
    "        self.thinking_enabled = chat_kwargs.get(\"enable_thinking\", True)\n"
    "\n"
    "        # [Genesis P12] Implicit reasoning end via <tool_call> (PR #35687).\n"
    "        # Resolve token IDs defensively — BPE tokenizers without these\n"
    "        # special tokens return None and our hooks degrade to no-op.\n"
    "        self._tool_call_tag = \"<tool_call>\"\n"
    "        self._tool_call_end_tag = \"</tool_call>\"\n"
    "        _genesis_vocab = getattr(self, 'vocab', None)\n"
    "        self._tool_call_token_id = (\n"
    "            _genesis_vocab.get(self._tool_call_tag)\n"
    "            if _genesis_vocab is not None else None\n"
    "        )\n"
    "        self._tool_call_end_token_id = (\n"
    "            _genesis_vocab.get(self._tool_call_end_tag)\n"
    "            if _genesis_vocab is not None else None\n"
    "        )\n"
    "        if self._tool_call_token_id is None:\n"
    "            import logging as _genesis_logging\n"
    "            _genesis_logging.getLogger(\"vllm.reasoning.qwen3\").info(\n"
    "                \"[Genesis P12] <tool_call> token not found in tokenizer \"\n"
    "                \"vocab — implicit reasoning-end hooks will be inert \"\n"
    "                \"(tokenizer likely a BPE without this special token)\"\n"
    "            )"
)


# Sub 2: Inject is_reasoning_end / is_reasoning_end_streaming / extract_content_ids
# between the `end_token` property and `extract_reasoning` method. Anchor is
# the boundary between them.
_OLD_METHODS_ANCHOR = (
    "    @property\n"
    "    def end_token(self) -> str:\n"
    "        \"\"\"The token that ends reasoning content.\"\"\"\n"
    "        return \"</think>\"\n"
    "\n"
    "    def extract_reasoning("
)

_NEW_METHODS_BLOCK = (
    "    @property\n"
    "    def end_token(self) -> str:\n"
    "        \"\"\"The token that ends reasoning content.\"\"\"\n"
    "        return \"</think>\"\n"
    "\n"
    "    # [Genesis P12] Serving-layer hooks for <tool_call>-as-reasoning-end.\n"
    "    def is_reasoning_end(self, input_ids):\n"
    "        start_token_id = getattr(self, 'start_token_id', None)\n"
    "        end_token_id = getattr(self, 'end_token_id', None)\n"
    "        tool_call_token_id = self._tool_call_token_id\n"
    "        tool_call_end_token_id = self._tool_call_end_token_id\n"
    "        for i in range(len(input_ids) - 1, -1, -1):\n"
    "            token_id = input_ids[i]\n"
    "            if token_id == start_token_id:\n"
    "                return False\n"
    "            if token_id == end_token_id:\n"
    "                return True\n"
    "            if (\n"
    "                tool_call_token_id is not None\n"
    "                and token_id == tool_call_token_id\n"
    "            ):\n"
    "                if tool_call_end_token_id is not None and any(\n"
    "                    input_ids[j] == tool_call_end_token_id\n"
    "                    for j in range(i + 1, len(input_ids))\n"
    "                ):\n"
    "                    continue  # paired, template example\n"
    "                return True\n"
    "        return False\n"
    "\n"
    "    def is_reasoning_end_streaming(self, input_ids, delta_ids):\n"
    "        base = getattr(super(), 'is_reasoning_end_streaming', None)\n"
    "        if callable(base) and base(input_ids, delta_ids):\n"
    "            return True\n"
    "        if self._tool_call_token_id is not None:\n"
    "            return self._tool_call_token_id in delta_ids\n"
    "        return False\n"
    "\n"
    "    def extract_content_ids(self, input_ids):\n"
    "        base = getattr(super(), 'extract_content_ids', None)\n"
    "        result = base(input_ids) if callable(base) else []\n"
    "        if result:\n"
    "            return result\n"
    "        if (\n"
    "            self._tool_call_token_id is not None\n"
    "            and self._tool_call_token_id in input_ids\n"
    "        ):\n"
    "            # [Genesis P12 v2 / supersedes P61] FIRST occurrence of\n"
    "            # <tool_call> as the content boundary. Multi-tool agentic\n"
    "            # flows emit multiple <tool_call> blocks; the FIRST one\n"
    "            # marks the reasoning -> tool transition. Original LAST-occ\n"
    "            # variant silently dropped earlier tool calls. P61 was meant\n"
    "            # to fix this via post-anchor replacement but its anchor\n"
    "            # 'tool_call_index = ...' does not match this 'idx = ...'\n"
    "            # form, so P61 silent-skips when P12 is also active.\n"
    "            idx = input_ids.index(self._tool_call_token_id)\n"
    "            return list(input_ids[idx:])\n"
    "        return []\n"
    "\n"
    "    def extract_reasoning("
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("reasoning/qwen3_reasoning_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P12 Qwen3 <tool_call> implicit reasoning end",
        target_file=target,
        marker=GENESIS_P12_MARKER,
        sub_patches=[
            TextPatch(
                name="p12_init_tool_call_tokens",
                anchor=_OLD_INIT,
                replacement=_NEW_INIT,
                required=True,
            ),
            TextPatch(
                name="p12_serving_layer_hooks",
                anchor=_OLD_METHODS_ANCHOR,
                replacement=_NEW_METHODS_BLOCK,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def apply() -> tuple[str, str]:
    """Apply P12 wiring. Never raises."""
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "qwen3_reasoning_parser.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "tool_call tokens + serving-layer hooks added"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"
