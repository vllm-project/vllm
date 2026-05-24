"""Wiring for PN40 sub-D workload classifier — chat-completion entry hook.

Per-request workload classification at chat completion entry. Tags
each request as `code` / `long_ctx` / `short_ctx` / `free_form` and
attaches the classification to a request-scoped attribute that
downstream PN40 sub-kernels can read.

Mirrors the PN16 lazy-reasoner injection pattern: anchor on the
`# Streaming response\n        tokenizer = self.renderer.tokenizer\n`
landmark in `OpenAIServingChat.create_chat_completion`. PN16 inserts
its hook BEFORE this pair; PN40 inserts AFTER it (PN16 → PN40
ordering preserved). Both are additive, neither competes for the same
text region.

What this hook does:
  1. Compute prompt_text (joined messages content) — already cheap as
     vllm has it built for tokenization
  2. Compute prompt_len from request.messages (lazy, no GPU sync)
  3. Call `pn40_dflash_omnibus.classify_workload(text, length)`
  4. Cache result on `request._genesis_pn40_workload_class` for
     downstream observability + future runtime-K-override decisions

Cost: ~1-5 µs per request (cheap string ops). Defensive try/except —
never raises (engine hot path).

Composition (no conflicts):
  - PN16 (lazy-reasoner) — different anchor position, both fire
  - PN40 sub-A (DFlash K-norm) — different file
  - PN40 sub-C+D scheduler observe hook — different file (scheduler.py)

Author: Sandermage (Sander) Barzov Aleksandr — 2026-05-04 v7.71.
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

log = logging.getLogger("genesis.wiring.pN40_workload_classifier_hook")

GENESIS_PN40_HOOK_MARKER = (
    "Genesis PN40 sub-D workload classifier chat-completion hook v1"
)


# Anchor: the `# Streaming response\n        tokenizer = ...\n` pair.
# Stable landmark in OpenAIServingChat.create_chat_completion (vllm
# 0.20.x). PN16 inserts BEFORE this pair; we insert BETWEEN PN16's
# injection and the original pair so both hooks land cleanly.
PN40_HOOK_ANCHOR = (
    "        # Streaming response\n"
    "        tokenizer = self.renderer.tokenizer\n"
)

PN40_HOOK_REPLACEMENT = (
    "        # [Genesis PN40 sub-D] workload classifier hook (per-request).\n"
    "        # Cheap (no GPU sync) — tags request as code/long_ctx/short_ctx/\n"
    "        # free_form for downstream observability + future routing.\n"
    "        try:\n"
    "            from vllm._genesis.kernels.pn40_dflash_omnibus import (\n"
    "                classify_workload as _genesis_pn40_classify,\n"
    "                env_enabled as _genesis_pn40_enabled,\n"
    "                sub_d_enabled as _genesis_pn40_sub_d_enabled,\n"
    "            )\n"
    "            if _genesis_pn40_enabled() and _genesis_pn40_sub_d_enabled():\n"
    "                _genesis_pn40_text = ''\n"
    "                _genesis_pn40_len = 0\n"
    "                _genesis_pn40_msgs = getattr(request, 'messages', None) or []\n"
    "                if _genesis_pn40_msgs:\n"
    "                    # Concatenate last 2 messages' content (cheap, bounded)\n"
    "                    _genesis_pn40_parts = []\n"
    "                    for _m in _genesis_pn40_msgs[-2:]:\n"
    "                        _c = _m.get('content', '') if isinstance(_m, dict) else getattr(_m, 'content', '')\n"
    "                        if isinstance(_c, str):\n"
    "                            _genesis_pn40_parts.append(_c)\n"
    "                        elif isinstance(_c, list):\n"
    "                            for _p in _c:\n"
    "                                if isinstance(_p, dict) and _p.get('type') == 'text':\n"
    "                                    _genesis_pn40_parts.append(_p.get('text', ''))\n"
    "                    _genesis_pn40_text = ' '.join(_genesis_pn40_parts)\n"
    "                    _genesis_pn40_len = len(_genesis_pn40_text)\n"
    "                _genesis_pn40_class = _genesis_pn40_classify(\n"
    "                    _genesis_pn40_text or None,\n"
    "                    _genesis_pn40_len // 4,  # rough char→tok approx\n"
    "                )\n"
    "                # Cache on request for downstream PN40 sub-kernels.\n"
    "                request._genesis_pn40_workload_class = _genesis_pn40_class\n"
    "        except Exception:  # noqa: BLE001 — never raise from hook\n"
    "            pass\n"
    "        # Streaming response\n"
    "        tokenizer = self.renderer.tokenizer\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "entrypoints/openai/chat_completion/serving.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN40 sub-D chat_completion/serving.py — workload classifier hook"
        ),
        target_file=str(target),
        marker=GENESIS_PN40_HOOK_MARKER,
        sub_patches=[
            TextPatch(
                name="pN40_hook_workload_classifier",
                anchor=PN40_HOOK_ANCHOR,
                replacement=PN40_HOOK_REPLACEMENT,
                required=False,  # graceful skip if PN16 changed anchor
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN40 sub-D]",
            "_genesis_pn40_workload_class",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN40 sub-D workload classifier hook."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN40")
    log_decision("PN40-classifier", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "chat_completion/serving.py not resolvable"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN40 sub-D workload classifier hook applied: per-request "
            "classification (code/long_ctx/short_ctx/free_form) cached "
            "on request._genesis_pn40_workload_class. Cheap (~1-5us/req), "
            "no GPU sync. Defensive try/except wrapper. Composes additively "
            "with PN16 (different anchor position in same function)."
        ),
        patch_name=patcher.patch_name,
    )
