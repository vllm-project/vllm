# SPDX-License-Identifier: Apache-2.0
"""Genesis PN16 — lazy reasoner middleware.

Hybrid pre-decision policy that decides per-request whether the model's
`<think>...</think>` reasoning block is likely to add value, and either:

  • Variant 1 (pre-decision): force `enable_thinking=False` for short
    prompts that lack any reasoning signal — saves the entire reasoning
    region's tokens + latency.
  • Variant 3 (client override): if the client explicitly set
    `chat_template_kwargs.enable_thinking` to True or False, respect it
    and do nothing.
  • Variant 4 (LogitsProcessor cap): bound the maximum number of tokens
    spent in `<think>` via a custom LogitsProcessor that boosts the
    `</think>` token's probability after N reasoning tokens.
    **UPSTREAM-BLOCKED in our PROD config (vllm v1 + speculative_config).**
    Per `vllm/v1/sample/logits_processor/__init__.py:200-208`, vllm v1
    explicitly rejects custom logits processors when speculative_config
    is set, AND warns that `logit_bias` "won't work with speculative
    decoding". Genesis prod runs MTP K=3 spec-decode, so this path is
    unavailable until either: (a) we drop spec-decode (would cost ~30%
    TPS), or (b) upstream relaxes the spec-decode + custom-logitsprocs
    restriction. Code path is kept stubbed out + emits a one-time
    warning when the operator sets a non-zero cap. Track upstream
    spec-decode + custom-logitsprocs compatibility under vllm-project/vllm
    issue tracker (search "logits processor speculative decoding").
  • Variant 5 (prompt-engineering soft cap): inject a hint into the
    request's last user message asking the model to keep reasoning
    concise. Works with spec-decode (it's just prompt engineering).
    Soft cap — depends on model compliance — but provides SOME
    bounding on the long-prompt graphomania case. Engaged when
    GENESIS_PN16_MAX_THINKING_TOKENS > 0 AND thinking is allowed.

Why a hybrid: pure variant 1 risks cutting reasoning where needed (e.g.
short math problems); pure variant 5 still wastes tokens when the
model ignores the instruction. Combining gives lower bound on short
trivial prompts + softer upper bound on long-prompt graphomania, with
no retry-induced 2× latency or token doubling.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

log = logging.getLogger("genesis.middleware.lazy_reasoner")


# ─── Operator-tunable thresholds ─────────────────────────────────────────


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "")
    return val.strip().lower() in ("1", "true", "yes", "on") if val else default


def _is_enabled() -> bool:
    """Master gate — same env name as PATCH_REGISTRY['PN16'].env_flag."""
    return _env_bool("GENESIS_ENABLE_PN16_LAZY_REASONER")


def _threshold_chars() -> int:
    """Char count threshold below which thinking is candidate-for-disable."""
    return _env_int("GENESIS_PN16_THRESHOLD_CHARS", 300)


def _max_thinking_tokens() -> int:
    """Phase 2 cap — max reasoning tokens when thinking IS allowed.

    0 disables the cap (default). Non-zero engages the (not-yet-wired)
    LogitsProcessor injection in apply_hook.
    """
    return _env_int("GENESIS_PN16_MAX_THINKING_TOKENS", 0)


# ─── Reasoning-signal detector ───────────────────────────────────────────
# Patterns that suggest the request may need chain-of-thought even on a
# short prompt. Conservative — false POSITIVE here means "we left thinking
# ON when it might have been disposable", which is the safer error.

_REASONING_SIGNAL_PATTERNS = [
    # Math / problem-solving verbs
    r"\b(calculate|compute|solve|prove|derive|integrate|differentiate"
    r"|reason|estimate|optimi[sz]e|simplify|factor|expand)\b",
    # Math / CS nouns
    r"\b(prime|matrix|vector|tensor|equation|theorem|lemma|proof"
    r"|integral|derivative|algorithm|complexity)\b",
    # Code block fence
    r"```",
    # Inline LaTeX-ish math
    r"\$[^$]+\$",
    # Arithmetic operators next to digits (basic math)
    r"[+\-*/=<>%^]\s*\d",
    r"\d\s*[+\-*/=<>%^]",
    # Programming snippet smell — class/function/return on a single line
    r"\b(class|def|function|return|import|from|public|private)\b",
    # Step-by-step request markers
    r"\b(step[- ]by[- ]step|chain[- ]of[- ]thought|explain why|how does)\b",
]
_COMPILED_SIGNAL_PATTERNS: list[re.Pattern[str]] | None = None


def _signal_patterns() -> list[re.Pattern[str]]:
    global _COMPILED_SIGNAL_PATTERNS
    if _COMPILED_SIGNAL_PATTERNS is None:
        _COMPILED_SIGNAL_PATTERNS = [
            re.compile(p, re.IGNORECASE) for p in _REASONING_SIGNAL_PATTERNS
        ]
    return _COMPILED_SIGNAL_PATTERNS


def _has_reasoning_signal(text: str) -> bool:
    """True when text contains any pattern that hints reasoning is useful."""
    for pat in _signal_patterns():
        if pat.search(text):
            return True
    return False


# ─── Prompt-shape inspectors (defensive against schema variation) ────────


def _extract_text_from_message(msg: Any) -> str:
    """Pull plain-text content from a chat message (string or content-parts)."""
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
                elif "text" in p:
                    parts.append(str(p["text"]))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return ""


def _total_chars(request: Any) -> int:
    messages = getattr(request, "messages", None) or []
    return sum(len(_extract_text_from_message(m)) for m in messages)


def _last_user_text(request: Any) -> str:
    messages = getattr(request, "messages", None) or []
    for m in reversed(messages):
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        if role == "user":
            return _extract_text_from_message(m)
    return ""


def _has_tools(request: Any) -> bool:
    tools = getattr(request, "tools", None)
    return bool(tools)


def _has_json_schema_format(request: Any) -> bool:
    """True when response_format constrains output to a JSON schema —
    schema-constrained generation typically benefits from reasoning."""
    rf = getattr(request, "response_format", None)
    if rf is None:
        return False
    type_field = getattr(rf, "type", None) or (rf.get("type") if isinstance(rf, dict) else None)
    return type_field in ("json_schema", "json_object")


def _client_explicit_thinking_choice(request: Any) -> bool | None:
    """Returns True/False if client set chat_template_kwargs.enable_thinking
    explicitly, None otherwise."""
    ctk = getattr(request, "chat_template_kwargs", None)
    if ctk is None:
        return None
    if isinstance(ctk, dict):
        if "enable_thinking" in ctk:
            return bool(ctk["enable_thinking"])
    return None


# ─── Core decision ───────────────────────────────────────────────────────


def _should_disable_thinking(request: Any) -> tuple[bool, str]:
    """Return (decision, reason). True = disable thinking for this request.

    Conservative — ALL of the following must hold to disable:
      1. Total prompt chars below threshold
      2. No tools attached
      3. No JSON-schema response_format
      4. Last user message has no reasoning-signal pattern hits

    Any single failure keeps thinking on. False positives are intentionally
    biased toward "leave thinking on".
    """
    threshold = _threshold_chars()
    char_count = _total_chars(request)
    if char_count >= threshold:
        return False, f"prompt {char_count} chars >= threshold {threshold}"
    if _has_tools(request):
        return False, "tools attached — keep thinking for tool-call planning"
    if _has_json_schema_format(request):
        return False, "json_schema response_format — keep thinking"
    last = _last_user_text(request)
    if last and _has_reasoning_signal(last):
        return False, "reasoning-signal pattern in last user message"
    return True, (
        f"short prompt ({char_count} chars), no tools, no schema, no signal "
        f"— thinking disabled"
    )


# ─── Stats counter ────────────────────────────────────────────────────────


_STATS: dict[str, int] = {
    "total_requests": 0,
    "respect_explicit_on": 0,
    "respect_explicit_off": 0,
    "disabled_by_heuristic": 0,
    "left_on_by_heuristic": 0,
    "soft_cap_hint_injected": 0,  # variant 5
    "errors": 0,
}

# One-shot warning flag — emit the upstream-blocked notice exactly once
# per process even if many requests hit the cap path.
_LOGITS_PROCESSOR_WARNING_EMITTED = False


def get_stats() -> dict[str, int]:
    """Return current counters (for diagnostic/`/v1/genesis/stats` etc.)."""
    return dict(_STATS)


def reset_stats() -> None:
    """Reset counters (for tests)."""
    for k in _STATS:
        _STATS[k] = 0
    # Also reset the one-shot warning flag so each test sees a fresh state
    global _LOGITS_PROCESSOR_WARNING_EMITTED
    _LOGITS_PROCESSOR_WARNING_EMITTED = False


# ─── Variant 5 — prompt-engineering soft cap ──────────────────────────────
# When the LogitsProcessor path (variant 4) is blocked by vllm's spec-decode
# + custom-logitsprocs restriction, fall back to instructing the model
# directly via a system-style hint appended to the last user message.
# Soft cap — model may ignore — but works with all engine configurations.

_SOFT_CAP_TEMPLATE = (
    "\n\n[Genesis hint] Keep your reasoning concise — under {tokens} "
    "tokens of `<think>` — and proceed to the final answer. Be brief in "
    "the thinking block."
)


def _inject_soft_cap_hint(request: Any, max_tokens: int) -> bool:
    """Append a soft-cap reasoning-budget hint to the last user message.

    Returns True if injection succeeded, False otherwise (e.g. no user
    message to append to, or message shape doesn't allow mutation).
    """
    messages = getattr(request, "messages", None) or []
    # Find the LAST user message (working backwards)
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        if role == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return False

    target = messages[last_user_idx]
    content = getattr(target, "content", None)
    if content is None and isinstance(target, dict):
        content = target.get("content")

    hint = _SOFT_CAP_TEMPLATE.format(tokens=max_tokens)

    if isinstance(content, str):
        new_content = content + hint
    elif isinstance(content, list):
        # Append a text part rather than mutating existing parts. Find the
        # last text part and extend it for cleanest tokenization.
        new_content = list(content)
        # Append a fresh text part at end (model concatenates content parts)
        new_content.append({"type": "text", "text": hint})
    else:
        # Unknown content shape — safer to skip than mutate blindly
        return False

    # Try setattr first (pydantic model), fall back to dict mutation.
    try:
        if isinstance(target, dict):
            target["content"] = new_content
        else:
            target.content = new_content
    except Exception:
        try:
            object.__setattr__(target, "content", new_content)
        except Exception:
            return False
    return True


def _emit_logits_processor_blocker_warning() -> None:
    """One-shot warning: explain why variant 4 (LogitsProcessor cap) is
    not engaged when spec-decode is on. Only logs once per process."""
    global _LOGITS_PROCESSOR_WARNING_EMITTED
    if _LOGITS_PROCESSOR_WARNING_EMITTED:
        return
    _LOGITS_PROCESSOR_WARNING_EMITTED = True
    log.warning(
        "[PN16] GENESIS_PN16_MAX_THINKING_TOKENS is set, but vllm v1 "
        "rejects custom LogitsProcessor injection when speculative_config "
        "is active (see vllm/v1/sample/logits_processor/__init__.py:200). "
        "Falling back to variant-5 prompt-engineering soft cap. To enable "
        "the strict LogitsProcessor cap, either disable spec-decode (heavy "
        "TPS cost) or upstream a max-thinking-tokens built-in to vllm "
        "v1 LogitsProcessors."
    )


# ─── Public hook ──────────────────────────────────────────────────────────


def apply_hook(serving: Any, request: Any) -> None:
    """Mutate request in place per PN16 lazy-reasoner policy.

    Called from `OpenAIServingChat.create_chat_completion` near the top
    via the text-patched hook injection in
    `wiring/patch_N16_lazy_reasoner.py`.

    Failure mode: any exception is caught by the wiring's try/except and
    logged at debug; the request continues with its original
    chat_template_kwargs.
    """
    if not _is_enabled():
        return

    _STATS["total_requests"] += 1

    # Variant 3 — respect explicit client choice
    explicit = _client_explicit_thinking_choice(request)
    if explicit is True:
        _STATS["respect_explicit_on"] += 1
        log.debug("PN16: client set enable_thinking=True explicitly — respect")
        return
    if explicit is False:
        _STATS["respect_explicit_off"] += 1
        log.debug("PN16: client set enable_thinking=False explicitly — respect")
        return

    # Variant 1 — pre-decision heuristic
    disable, reason = _should_disable_thinking(request)
    if disable:
        ctk = dict(getattr(request, "chat_template_kwargs", None) or {})
        ctk["enable_thinking"] = False
        try:
            request.chat_template_kwargs = ctk
        except Exception:
            # Pydantic frozen-model fallback — try setattr regardless
            object.__setattr__(request, "chat_template_kwargs", ctk)
        _STATS["disabled_by_heuristic"] += 1
        log.debug("PN16: thinking disabled — %s", reason)
        return

    # Variant 4 (LogitsProcessor cap) is upstream-blocked when vllm has
    # speculative_config (Genesis PROD = MTP K=3). One-shot warning,
    # then fall through to variant 5 (prompt-engineering soft cap).
    cap = _max_thinking_tokens()
    if cap > 0:
        _emit_logits_processor_blocker_warning()
        # Variant 5 — soft cap via prompt hint
        if _inject_soft_cap_hint(request, cap):
            _STATS["soft_cap_hint_injected"] += 1
            log.debug(
                "PN16: soft cap hint injected (max_thinking_tokens=%d)", cap,
            )
    _STATS["left_on_by_heuristic"] += 1
    log.debug("PN16: thinking left on — %s", reason)
