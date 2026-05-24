# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN16 — lazy-reasoner request hook.

Injects a hook call at the top of `OpenAIServingChat.create_chat_completion`
that runs `vllm._genesis.middleware.lazy_reasoner.apply_hook(serving, request)`
in a try/except. Anchor is non-overlapping with the P68/P69 hook (we anchor
on the `# Streaming response` + `tokenizer = self.renderer.tokenizer` lines
which sit immediately AFTER the P68/P69 injection point, so PN16 inserts
its hook between the P68/P69 hook and the request body — order is
P68/P69 → PN16 → tokenizer fetch → real work).

PN16 is purely additive and does NOT compete with P68/P69 for the same
text region. Both can be enabled together, neither can be enabled, or
either alone — all four combinations work.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pN16_lazy_reasoner")

GENESIS_PN16_MARKER = "Genesis PN16 lazy-reasoner request hook v7.62.x"


# Anchor: the `# Streaming response\n        tokenizer = self.renderer.tokenizer\n`
# pair. This is stable across upstream + survives P68/P69 application
# (which inserts its hook BEFORE this pair, leaving the pair intact).
PN16_ANCHOR = (
    "        # Streaming response\n"
    "        tokenizer = self.renderer.tokenizer\n"
)

PN16_REPLACEMENT = (
    "        # [Genesis PN16 lazy-reasoner] Per-request decision on whether\n"
    "        # `<think>...</think>` reasoning is likely to add value. Hybrid\n"
    "        # policy: respect explicit client `enable_thinking` (variant 3),\n"
    "        # else for short prompts without tools/schema/reasoning-signals\n"
    "        # force enable_thinking=False (variant 1), else allow with\n"
    "        # optional max-thinking-tokens cap (variant 4, Phase 2 TODO).\n"
    "        try:\n"
    "            from vllm._genesis.middleware.lazy_reasoner import (\n"
    "                apply_hook as _genesis_pN16_apply_hook,\n"
    "            )\n"
    "            _genesis_pN16_apply_hook(self, request)\n"
    "        except Exception:\n"
    "            import logging as _genesis_pN16_logging\n"
    "            _genesis_pN16_logging.getLogger(\n"
    "                'genesis.middleware.lazy_reasoner'\n"
    "            ).debug('Genesis PN16 hook raised; ignored', exc_info=True)\n"
    "        # Streaming response\n"
    "        tokenizer = self.renderer.tokenizer\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("entrypoints/openai/chat_completion/serving.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN16 serving.py — lazy-reasoner request hook (chat_template_kwargs"
            " mutation pre-decision)"
        ),
        target_file=str(target),
        marker=GENESIS_PN16_MARKER,
        sub_patches=[
            TextPatch(
                name="pN16_lazy_reasoner_hook",
                anchor=PN16_ANCHOR,
                replacement=PN16_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN16",
            "_genesis_pN16_apply_hook",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN16 — lazy-reasoner hook injection."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN16")
    log_decision("PN16", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file not resolvable"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN16 applied: lazy-reasoner hook injected at top of "
            "create_chat_completion. Hybrid policy active: respect explicit "
            "client enable_thinking (variant 3), force-off on short prompts "
            "without reasoning signals (variant 1), prompt-engineering soft "
            "cap (variant 5) when GENESIS_PN16_MAX_THINKING_TOKENS > 0. "
            "Variant 4 (LogitsProcessor cap) upstream-blocked by spec-decode "
            "(vllm v1 rejects custom logits processors when "
            "speculative_config is set; see lazy_reasoner.py docstring). "
            "Default "
            "OFF — set GENESIS_ENABLE_PN16_LAZY_REASONER=1 to engage. "
            "Threshold GENESIS_PN16_THRESHOLD_CHARS (default 300)."
        ),
        patch_name=patcher.patch_name,
    )
