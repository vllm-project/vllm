# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patches 68 & 69 — long-context tool-call adherence.

Bundle wiring for two complementary mitigations against model-behavior
degradation when calling tools at long context.

================================================================
PROBLEM
- Qwen3-class models (and most LLMs) lose `<tool_call>` format
  adherence at long context (>4K tokens) with significant prefix
  content. Manifests as: emit JSON-text without wrapper, refuse,
  hallucinate, or emit Python-style call.
- Empirically observed in Genesis ladder test 2026-04-25:
  prompts 0-12K chars → 3/3 OK, prompts 16K+ chars → 0/3 OK.
- Same behavior on prod ngram baseline AND on P64+P65+P66 test
  container — confirmed model-level, NOT engine-level bug.

================================================================
SOLUTION

Two independent middleware hooks injected at the top of
`OpenAIServingChat.create_chat_completion`:

  P68 — auto-upgrade `tool_choice` "auto" → "required" when long
        context + tools are detected.
  P69 — append explicit format reminder to last user message
        ("emit <tool_call> markers, do NOT plain-text, ...").

Both are env-flag opt-in:
  GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1
  GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1
  GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS=50000  (default — ~12.5K tokens; raised from 8000 in v7.65 per Issue #9)

================================================================

The hook is a single function call inserted at the very top of
create_chat_completion (before any other processing). The
middleware module
`vllm._genesis.middleware.long_ctx_tool_adherence::apply_hook`
contains the full mutation logic.

Compatibility
-------------
- No-op when both env flags off
- No-op when no tools in request
- No-op when prompt < threshold chars
- No-op when tool_choice already explicit (named function, "none", "required")
- Idempotent (marker check)
- Auto-no-op if upstream adds equivalent functionality

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

log = logging.getLogger("genesis.wiring.p68_69_long_ctx_tool_adherence")

GENESIS_P6869_MARKER = "Genesis P68/P69 long-context tool-call adherence v7.13"


# ─── Sub-patch: insert hook call at top of create_chat_completion ───────────
# Anchor on the docstring closing + "# Streaming response" comment +
# tokenizer fetch. Insert hook call AFTER docstring but BEFORE first action.

P6869_OLD = (
    "    async def create_chat_completion(\n"
    "        self,\n"
    "        request: ChatCompletionRequest,\n"
    "        raw_request: Request | None = None,\n"
    "    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:\n"
    "        \"\"\"\n"
    "        Chat Completion API similar to OpenAI's API.\n"
    "\n"
    "        See https://platform.openai.com/docs/api-reference/chat/create\n"
    "        for the API specification. This API mimics the OpenAI\n"
    "        Chat Completion API.\n"
    "        \"\"\"\n"
    "        # Streaming response\n"
    "        tokenizer = self.renderer.tokenizer\n"
)

P6869_NEW = (
    "    async def create_chat_completion(\n"
    "        self,\n"
    "        request: ChatCompletionRequest,\n"
    "        raw_request: Request | None = None,\n"
    "    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:\n"
    "        \"\"\"\n"
    "        Chat Completion API similar to OpenAI's API.\n"
    "\n"
    "        See https://platform.openai.com/docs/api-reference/chat/create\n"
    "        for the API specification. This API mimics the OpenAI\n"
    "        Chat Completion API.\n"
    "        \"\"\"\n"
    "        # [Genesis P68/P69 long-ctx tool-call adherence] Mutate request\n"
    "        # in-place if conditions met (env-gated). No-op when env flags\n"
    "        # off, when no tools, or when prompt below threshold.\n"
    "        try:\n"
    "            from vllm._genesis.middleware.long_ctx_tool_adherence import (\n"
    "                apply_hook as _genesis_p6869_apply_hook,\n"
    "            )\n"
    "            _genesis_p6869_apply_hook(self, request)\n"
    "        except Exception:\n"
    "            # Hook failure is non-fatal — fall through to standard path.\n"
    "            import logging as _genesis_p6869_logging\n"
    "            _genesis_p6869_logging.getLogger(\n"
    "                'genesis.middleware.long_ctx_tool_adherence'\n"
    "            ).debug('Genesis P68/P69 hook raised; ignored', exc_info=True)\n"
    "        # Streaming response\n"
    "        tokenizer = self.renderer.tokenizer\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("entrypoints/openai/chat_completion/serving.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P68/P69 serving.py — long-ctx tool-call hook injection",
        target_file=str(target),
        marker=GENESIS_P6869_MARKER,
        sub_patches=[
            TextPatch(
                name="p6869_hook_insert",
                anchor=P6869_OLD,
                replacement=P6869_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P68/P69",
            "_genesis_p6869_apply_hook",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P68/P69 hook injection."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    # The wiring is shared — apply if EITHER P68 or P69 is enabled.
    p68_decision, p68_reason = should_apply("P68")
    p69_decision, p69_reason = should_apply("P69")
    log_decision("P68", p68_decision, p68_reason)
    log_decision("P69", p69_decision, p69_reason)

    if not (p68_decision or p69_decision):
        return "skipped", (
            "neither P68 nor P69 enabled; hook injection skipped to keep "
            f"serving.py pristine. P68: {p68_reason} | P69: {p69_reason}"
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/entrypoints/openai/chat_completion/serving.py not found"

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
                    "hook already injected by another tool.",
                )
        if patcher.sub_patches[0].anchor not in content:
            return (
                "skipped",
                "required anchor (create_chat_completion docstring + tokenizer "
                "fetch) not found — P68/P69 cannot apply (upstream drift).",
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

    enabled = []
    if p68_decision:
        enabled.append("P68 (auto force tool_choice=required for long ctx)")
    if p69_decision:
        enabled.append("P69 (append tool-format reminder to last user msg)")
    return "applied", (
        "Hook injected into create_chat_completion. Active mitigations: "
        + "; ".join(enabled)
        + ". Threshold: GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS env (default 50000, raised from 8000 in v7.65 per Issue #9)."
    )
