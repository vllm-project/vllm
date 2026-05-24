# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN70 — Tool schema subset filter for combined-schema builders.

Source: noonghunna/club-3090#57 by @lexhoefsloot (2026-05-05). Companion to
Genesis v7.72.1 P68 fix. Where P68's option-1 ("schema-aware skip") refuses
to upgrade `tool_choice "auto" → "required"` whenever ANY tool is xgrammar-
incompatible, PN70 implements lexhoefsloot's option-3 ("subset-only `anyOf`"):
when vLLM builds a combined `anyOf` schema across all tools (the path that
runs whenever `tool_choice="required"`, whether set by P68 or by the user),
**filter out** the incompatible tools and pass only the compileable subset to
xgrammar.

================================================================
WHY THIS MATTERS
================================================================

In stock vLLM with `tool_choice="required"` (or any other path that calls
`_get_json_schema_from_tools`), one tool with `patternProperties` /
`propertyNames` / `$ref` / `oneOf` poisons the entire request — xgrammar
fails to compile the combined schema and the request 400s.

P68's option-1 fix sidesteps this by **not** upgrading at all. Operators who
rely on long-context tool adherence still want the upgrade — they just don't
want the dirty tool to gate everything.

PN70's option-3 fix keeps the upgrade. It filters the tool list before the
combined `anyOf` is built, so:
- xgrammar gets a clean subset → grammar enforcement works
- Model can still see the full tool catalog in context
- Filtered-out tools become un-callable under grammar enforcement (sampling
  is masked to the compat subset). They were also un-callable under stock
  vLLM (the request 400'd). PN70 makes the strict path work on dirty
  catalogs instead of failing.

================================================================
COMPOSITION WITH P68
================================================================

| P68 enabled | PN70 enabled | Behavior on long prompt + incompat tool |
|:-:|:-:|:---|
| OFF | OFF | stock — request fails ONLY if user set tool_choice=required |
| ON  | OFF | option-1 — P68 refuses to upgrade, model is free (no enforcement) |
| OFF | ON  | option-3 path active — applies whenever caller sets `required` (e.g. user, or another patch) |
| ON  | ON  | **recommended** — P68 upgrades, PN70 filters; grammar enforcement on compat subset, no 400 |

================================================================
ENV
================================================================

GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER=1   — master enable (off by default)

================================================================
RISK
================================================================

LOW. Wraps a single internal upstream function (`_get_json_schema_from_tools`
in `vllm.tool_parsers.utils`) and either delegates to original or calls
original with a filtered list. Idempotent via `__pn70_wrapped__` marker. If
upstream signature drifts, NULL-skips. Returns None when the compat subset
is empty — caller already handles None safely (it's the same return value
the public `get_json_schema_from_tools` produces for `tool_choice=None`).

Fallback to original on ANY exception during filtering — never breaks a
request that would have succeeded under stock.

Author: Sandermage 2026-05-05.
Companion to: P68 v7.72.1 (option-1 skip).
Issue: noonghunna/club-3090#57.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.middleware.long_ctx_tool_adherence import (
    _scan_schema_for_unsupported_key,
)

log = logging.getLogger("genesis.wiring.pn70_tool_schema_subset_filter")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in (
        "1", "true", "yes", "y", "on",
    )


def _tool_is_xgrammar_compat(tool: object) -> bool:
    """True iff this tool's parameters schema contains no xgrammar-
    unsupported keys. Reuses the P68 scanner so the unsupported-key set
    is single-sourced — no drift between P68 (skip) and PN70 (filter)."""
    if isinstance(tool, dict):
        fn = tool.get("function") or {}
        params = fn.get("parameters")
    else:
        fn = getattr(tool, "function", None)
        if fn is None:
            return True  # malformed tool — defer to upstream which will error
        params = getattr(fn, "parameters", None)
    if params is None:
        return True  # no schema → nothing to compile → trivially compat
    try:
        return _scan_schema_for_unsupported_key(params) is None
    except Exception:
        # Defensive: if the scanner itself crashes on a weird shape, treat
        # the tool as compat (defer to original behavior — better safe).
        return True


def _tool_name(tool: object) -> str:
    """Best-effort extraction of tool function name for log messages."""
    if isinstance(tool, dict):
        return (tool.get("function") or {}).get("name") or "<anonymous>"
    fn = getattr(tool, "function", None)
    if fn is None:
        return "<anonymous>"
    return getattr(fn, "name", None) or "<anonymous>"


def _wrap_get_json_schema_from_tools(original):
    """Decorator: filter incompat tools from `tools` before calling
    original `_get_json_schema_from_tools`."""
    def wrapped(tools):
        # Env-gate is checked at call time so operators can flip the flag
        # without restarting (env reads cheap; ~µs per request).
        if not _env_flag("GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER"):
            return original(tools)
        if not tools:
            return original(tools)
        try:
            compat = [t for t in tools if _tool_is_xgrammar_compat(t)]
            incompat_count = len(tools) - len(compat)
            if incompat_count == 0:
                # All tools compat — no filter needed, call through cleanly
                return original(tools)
            incompat_names = [
                _tool_name(t) for t in tools if not _tool_is_xgrammar_compat(t)
            ]
            if not compat:
                # Subset is empty — return None so caller skips assigning
                # `request.structured_outputs` entirely. Equivalent to the
                # path stock vLLM takes for `tool_choice="auto"`. Better
                # than calling original([]) which would build an `anyOf: []`
                # array that xgrammar would also reject.
                log.warning(
                    "[Genesis PN70] all %d tools have xgrammar-unsupported "
                    "schema keys — returning None (no grammar enforcement). "
                    "Filtered tools: %s. Set "
                    "GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER=0 to disable.",
                    len(tools), incompat_names,
                )
                return None
            log.warning(
                "[Genesis PN70] filtered %d/%d xgrammar-incompat tools from "
                "combined `anyOf` schema. Compat subset: %d tool(s). "
                "Filtered: %s. Model can still SEE all tools in context but "
                "grammar will only allow calls to the compat subset. Set "
                "GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER=0 to disable.",
                incompat_count, len(tools), len(compat), incompat_names,
            )
            return original(compat)
        except Exception as e:
            # Defensive: any unexpected failure falls through to stock
            # behaviour. We never want PN70 to break a request that would
            # have succeeded under upstream.
            log.warning(
                "[Genesis PN70] filter raised %s: %s — falling back to "
                "stock _get_json_schema_from_tools behavior",
                type(e).__name__, e,
            )
            return original(tools)
    wrapped.__wrapped__ = original
    wrapped.__pn70_wrapped__ = True
    return wrapped


def apply() -> tuple[str, str]:
    """Apply PN70 — wrap `_get_json_schema_from_tools` so vLLM's combined
    schema build path filters out xgrammar-incompat tools instead of
    400-ing the request."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("PN70")
    log_decision("PN70", decision, reason)
    if not decision:
        return "skipped", reason

    try:
        from vllm.tool_parsers import utils as _u
    except Exception:
        return (
            "skipped",
            "vllm.tool_parsers.utils not importable on this pin — PN70 NULL",
        )

    if not hasattr(_u, "_get_json_schema_from_tools"):
        return (
            "skipped",
            "_get_json_schema_from_tools not present in vllm.tool_parsers."
            "utils — upstream API drift, PN70 NULL",
        )

    if getattr(_u._get_json_schema_from_tools, "__pn70_wrapped__", False):
        return "applied", "PN70 already wrapped (idempotent)"

    original = _u._get_json_schema_from_tools
    _u._get_json_schema_from_tools = _wrap_get_json_schema_from_tools(original)
    return (
        "applied",
        "PN70 wrapped vllm.tool_parsers.utils._get_json_schema_from_tools — "
        "combined `anyOf` schema build path now filters xgrammar-incompat "
        "tools instead of failing the request. Set "
        "GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER=1 to activate (off by "
        "default — only takes effect when env flag is set). "
        "Companion to P68 v7.72.1 option-1 skip; closes club-3090#57 "
        "option-3 path."
    )


def is_applied() -> bool:
    try:
        from vllm.tool_parsers import utils as _u
        return getattr(_u._get_json_schema_from_tools, "__pn70_wrapped__", False)
    except Exception:
        return False
