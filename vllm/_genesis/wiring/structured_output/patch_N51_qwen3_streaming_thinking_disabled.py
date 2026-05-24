# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN51 — Qwen3 streaming `delta.reasoning` vs `delta.content`
routing fix when `enable_thinking=false`.

Backport of upstream issue vllm-project/vllm#40816 (OPEN). The
non-streaming `extract_reasoning` path correctly handles the
`not self.thinking_enabled` short-circuit at line 146-148 of upstream
`vllm/reasoning/qwen3_reasoning_parser.py` (commit a1c0c0c+).

The streaming counterpart `extract_reasoning_streaming` does NOT have
the equivalent short-circuit, so when:

  * server is launched with
    `--default-chat-template-kwargs '{"enable_thinking": false}'`
    (or per-request `chat_template_kwargs.enable_thinking=false`), AND
  * the prompt has the empty `<think>\\n\\n</think>\\n\\n` block pre-baked
    so no `</think>` token ever appears in the generated output, AND
  * the request uses `stream=true`,

then the streaming parser falls through to the final `else:` branch
and emits every model token via `delta.reasoning` instead of
`delta.content`. OpenAI-compatible clients that only read
`delta.content` (Open WebUI, LibreChat, LobeChat, Cline, OpenCode, …)
see "reasoning only" and never receive the answer.

What this fixes
---------------
Insert a single short-circuit at the very start of
`extract_reasoning_streaming`, immediately after the docstring:

  ```python
  if not self.thinking_enabled and self.end_token_id not in current_token_ids:
      if not delta_text:
          return None
      return DeltaMessage(content=delta_text)
  ```

This mirrors the non-streaming logic at line 146-148 of the same file.

Risks
-----
- None for thinking-enabled requests (default, dominant case): the new
  guard's first condition (`not self.thinking_enabled`) is False, so
  the original logic runs unchanged.
- For thinking-disabled requests where </think> DOES appear (legacy
  templates, e.g. Qwen3-235B-A22B-Instruct-2507 generating <think>
  itself): the second condition (`end_token_id not in current_token_ids`)
  is False, so the original branch handling </think> runs unchanged.
- For thinking-disabled requests with NO </think> token (the bug case):
  the guard activates and routes deltas to content. This is the
  intended behavior per the non-streaming path.

Anchor stability
----------------
Anchor is on the docstring closing `\"\"\"` + the next line
`# Strip <think> from delta if present (old template / edge case`.
Both bits are pristine upstream and unchanged by Genesis P27 (which
adds its `_genesis_pre_think_content = \"\"` initialization a few
lines later, AFTER the comment block).

Status: opt-in (`GENESIS_ENABLE_PN51_QWEN3_STREAMING_THINKING_DISABLED=1`).
Default OFF until Open WebUI / LibreChat repro proves the fix in
streaming mode end-to-end.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Backport target: upstream issue #40816 (filed 2026-04-22 by user
"keehawkes"; still OPEN as of 2026-05-04).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn51_qwen3_streaming_thinking_disabled")

GENESIS_PN51_MARKER = (
    "Genesis PN51 Qwen3 streaming thinking-disabled content routing v7.65"
)


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN51_QWEN3_STREAMING_THINKING_DISABLED", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor: docstring tail + next comment. Both lines are pristine in
# upstream and unaffected by P27, P61, P61b runtime mutations.
ANCHOR_OLD = (
    "        prompt_is_reasoning_end and routes deltas as content without\n"
    "        calling this method.\n"
    "        \"\"\"\n"
    "        # Strip <think> from delta if present (old template / edge case\n"
    "        # where the model generates <think> itself)."
)

ANCHOR_NEW = (
    "        prompt_is_reasoning_end and routes deltas as content without\n"
    "        calling this method.\n"
    "        \"\"\"\n"
    "        # [Genesis PN51 vllm#40816] Streaming counterpart to the\n"
    "        # non-streaming `not self.thinking_enabled` short-circuit. When\n"
    "        # the parser was constructed with thinking disabled and no\n"
    "        # </think> token has appeared, all generated tokens are content\n"
    "        # (the prompt has the empty <think>\\n\\n</think>\\n\\n block\n"
    "        # pre-baked, so the serving-layer detection that should have\n"
    "        # bypassed this method missed; we recover here defensively).\n"
    "        if (\n"
    "            not self.thinking_enabled\n"
    "            and self.end_token_id not in current_token_ids\n"
    "        ):\n"
    "            if not delta_text:\n"
    "                return None\n"
    "            return DeltaMessage(content=delta_text)\n"
    "        # Strip <think> from delta if present (old template / edge case\n"
    "        # where the model generates <think> itself)."
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("reasoning/qwen3_reasoning_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN51 Qwen3 streaming thinking-disabled routing",
        target_file=str(target),
        marker=GENESIS_PN51_MARKER,
        sub_patches=[
            TextPatch(
                name="pn51_thinking_disabled_streaming_short_circuit",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # If/when upstream merges the fix, the docstring above the
            # streaming method is likely to be revised. Watch for the
            # serving-layer-bypass NOTE getting trimmed or replaced.
            "if not self.thinking_enabled and self.end_token_id not in current_token_ids",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN51")
    log_decision("PN51", decision, reason)
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
            "PN51 applied: streaming `enable_thinking=false` now emits "
            "delta.content (was delta.reasoning); fixes vllm#40816 for "
            "Open WebUI / LibreChat / LobeChat / Cline / OpenCode clients",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return (
            "skipped",
            f"{msg} — likely upstream merged or anchor drifted "
            "(check qwen3_reasoning_parser.py docstring of "
            "extract_reasoning_streaming)",
        )
    return "failed", failure.reason if failure else "unknown failure"
