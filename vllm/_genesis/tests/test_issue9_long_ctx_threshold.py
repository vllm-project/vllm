# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Genesis Issue #9 — P68/P69 long-context threshold.

Issue #9 (noonghunna 2026-04-29): the default
GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS=8000 was crossed by every
realistic IDE-agent prompt (Cline / Cursor / OpenCode / Copilot Gateway
typically ship 15-50K-char system prompts). Crossing the threshold:

- P68 silently rewrites `tool_choice: auto -> required`
- P69 appends "must use a tool" reminder to the last user message

Either alone makes the model produce `finish_reason=stop` with empty
content for plain-text user messages — a silent stall every IDE-agent
session sees on every casual question.

Fix: raise default 8000 → 50000 (~12.5K tokens). Long-context tool
adherence still triggers for genuinely long histories; casual flows
unaffected.
"""
from __future__ import annotations



class TestIssue9DefaultThresholdRaised:
    def test_default_threshold_is_50k(self, monkeypatch):
        """The default must be high enough that a realistic IDE-agent
        system-prompt (15-50K chars) does NOT trigger the long-context
        rewrites. 50000 keeps the long-context behavior for genuine
        long histories."""
        monkeypatch.delenv(
            "GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", raising=False
        )
        from vllm._genesis.middleware.long_ctx_tool_adherence import (
            _get_threshold_chars,
        )
        assert _get_threshold_chars() == 50000, (
            "Genesis Issue #9: default threshold must be 50000; lower "
            "values trigger silent finish_reason=stop on casual IDE-agent "
            "messages whose system prompt is 15-50K chars."
        )

    def test_invalid_env_falls_back_to_50k(self, monkeypatch):
        monkeypatch.setenv(
            "GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "not-a-number"
        )
        from vllm._genesis.middleware.long_ctx_tool_adherence import (
            _get_threshold_chars,
        )
        assert _get_threshold_chars() == 50000

    def test_explicit_env_override_respected(self, monkeypatch):
        monkeypatch.setenv(
            "GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "100000"
        )
        from vllm._genesis.middleware.long_ctx_tool_adherence import (
            _get_threshold_chars,
        )
        assert _get_threshold_chars() == 100000

    def test_minimum_floor_at_1000(self, monkeypatch):
        """Tiny values get clamped — protects against `=0` accidents
        that would force every request to take the long-ctx path."""
        monkeypatch.setenv(
            "GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "100"
        )
        from vllm._genesis.middleware.long_ctx_tool_adherence import (
            _get_threshold_chars,
        )
        assert _get_threshold_chars() == 1000


class TestIssue9DocstringMentionsContext:
    def test_docstring_explains_50k_decision(self):
        """The docstring must explain WHY the default was raised, so
        a future maintainer doesn't silently revert the fix."""
        from vllm._genesis.middleware.long_ctx_tool_adherence import (
            _get_threshold_chars,
        )
        assert _get_threshold_chars.__doc__ is not None
        doc = _get_threshold_chars.__doc__
        assert "Issue #9" in doc, "docstring must reference the issue"
        assert "50000" in doc or "12.5K" in doc, "must mention new default"
        assert "8000" in doc or "8K" in doc, (
            "must reference the previous default for context"
        )


class TestIssue9P68LogsAtWarn:
    """Per Issue #9 option 3: when P68 silently rewrites tool_choice the
    operator MUST see it in default log levels. Promote INFO → WARN."""

    def test_log_level_for_rewrite_is_warning(self):
        # Source-level check: scan the middleware module for the WARN log
        # paired with "upgraded tool_choice" to confirm we're not back at
        # log.info.
        from vllm._genesis.middleware import long_ctx_tool_adherence
        import inspect

        src = inspect.getsource(long_ctx_tool_adherence)
        # The relevant log call must be a `log.warning(`, not `log.info(`,
        # paired with the "upgraded tool_choice" message.
        warn_idx = src.find("upgraded tool_choice")
        assert warn_idx > 0, (
            "expected 'upgraded tool_choice' log message — refactor "
            "drift?"
        )
        # Walk backwards from the message to find the log.* call site
        before = src[:warn_idx]
        last_log_call = max(
            before.rfind("log.warning"),
            before.rfind("log.info"),
        )
        assert last_log_call > 0
        assert src[last_log_call:].startswith("log.warning"), (
            "Issue #9 option 3: tool_choice rewrite must log at WARN "
            "(was INFO). Operators need to see this without raising "
            "verbosity."
        )
