# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PN16 — lazy-reasoner middleware policy.

Tests the decision logic in `vllm._genesis.middleware.lazy_reasoner`
without touching the text-patch wiring (covered by the wiring's anchor
invariants in a separate test).

What we cover:
  - Pre-decision heuristic (variant 1): short prompt + no tools + no
    schema + no reasoning signals → disable thinking
  - Client override (variant 3): explicit True/False respected
  - Reasoning signal patterns: math, code fence, CoT keywords
  - Stats counters move correctly per branch
  - Master env gate (default OFF)
  - Defensive against pydantic-frozen request models (object.__setattr__
    fallback)
  - Edge cases: empty messages, content-parts list, missing fields
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm._genesis.middleware import lazy_reasoner as lr


# ─── Helpers to build fake-request objects ──────────────────────────────


def _make_request(*, messages=None, tools=None, chat_template_kwargs=None,
                  response_format=None) -> SimpleNamespace:
    """Mimic ChatCompletionRequest shape with mutable attributes."""
    return SimpleNamespace(
        messages=messages or [],
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
        response_format=response_format,
    )


def _user_msg(text: str) -> SimpleNamespace:
    return SimpleNamespace(role="user", content=text)


def _assistant_msg(text: str) -> SimpleNamespace:
    return SimpleNamespace(role="assistant", content=text)


@pytest.fixture(autouse=True)
def reset_stats():
    """Each test starts with clean counters."""
    lr.reset_stats()
    yield
    lr.reset_stats()


@pytest.fixture
def env_pn16_on(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_PN16_LAZY_REASONER", "1")
    yield


@pytest.fixture
def env_pn16_off(monkeypatch):
    monkeypatch.delenv("GENESIS_ENABLE_PN16_LAZY_REASONER", raising=False)
    yield


# ─── Master gate ────────────────────────────────────────────────────────


class TestMasterGate:
    def test_default_off_no_mutation(self, env_pn16_off):
        req = _make_request(messages=[_user_msg("hi")])
        lr.apply_hook(None, req)
        # No env → no stats movement, no mutation
        assert lr.get_stats()["total_requests"] == 0
        assert req.chat_template_kwargs is None

    def test_env_on_increments_total(self, env_pn16_on):
        req = _make_request(messages=[_user_msg("hi")])
        lr.apply_hook(None, req)
        assert lr.get_stats()["total_requests"] == 1


# ─── Variant 3 — client explicit override ───────────────────────────────


class TestClientOverride:
    def test_explicit_thinking_on_respected(self, env_pn16_on):
        req = _make_request(
            messages=[_user_msg("hi")],
            chat_template_kwargs={"enable_thinking": True},
        )
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs == {"enable_thinking": True}
        assert lr.get_stats()["respect_explicit_on"] == 1
        assert lr.get_stats()["disabled_by_heuristic"] == 0

    def test_explicit_thinking_off_respected(self, env_pn16_on):
        req = _make_request(
            messages=[_user_msg("hi")],
            chat_template_kwargs={"enable_thinking": False},
        )
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs == {"enable_thinking": False}
        assert lr.get_stats()["respect_explicit_off"] == 1
        assert lr.get_stats()["disabled_by_heuristic"] == 0

    def test_no_override_runs_heuristic(self, env_pn16_on):
        req = _make_request(messages=[_user_msg("hi")])
        lr.apply_hook(None, req)
        # short trivial prompt with no override → should disable
        assert req.chat_template_kwargs == {"enable_thinking": False}
        assert lr.get_stats()["disabled_by_heuristic"] == 1


# ─── Variant 1 — pre-decision heuristic ─────────────────────────────────


class TestPreDecisionHeuristic:
    def test_short_trivial_prompt_disables(self, env_pn16_on):
        req = _make_request(messages=[_user_msg("Hello!")])
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_long_prompt_keeps_thinking(self, env_pn16_on, monkeypatch):
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "100")
        long_text = "x" * 200
        req = _make_request(messages=[_user_msg(long_text)])
        lr.apply_hook(None, req)
        # Long prompt — heuristic leaves thinking ALONE (no mutation)
        assert req.chat_template_kwargs is None
        assert lr.get_stats()["left_on_by_heuristic"] == 1

    def test_tools_attached_keeps_thinking(self, env_pn16_on):
        req = _make_request(
            messages=[_user_msg("hi")],
            tools=[{"type": "function", "function": {"name": "x"}}],
        )
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs is None
        assert lr.get_stats()["left_on_by_heuristic"] == 1

    def test_json_schema_response_format_keeps_thinking(self, env_pn16_on):
        rf = SimpleNamespace(type="json_schema")
        req = _make_request(
            messages=[_user_msg("hi")],
            response_format=rf,
        )
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs is None

    def test_math_keyword_keeps_thinking(self, env_pn16_on):
        req = _make_request(messages=[_user_msg("Calculate 7919 prime")])
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs is None
        assert lr.get_stats()["left_on_by_heuristic"] == 1

    def test_code_fence_keeps_thinking(self, env_pn16_on):
        req = _make_request(messages=[_user_msg("fix ```py\nprint(1)\n```")])
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs is None

    def test_arithmetic_keeps_thinking(self, env_pn16_on):
        req = _make_request(messages=[_user_msg("what is 2 + 2")])
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs is None

    def test_step_by_step_keeps_thinking(self, env_pn16_on):
        req = _make_request(messages=[_user_msg("explain step by step")])
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs is None


# ─── Reasoning-signal pattern coverage ──────────────────────────────────


class TestReasoningSignals:
    @pytest.mark.parametrize("text,expected", [
        ("Calculate 5+5", True),
        ("solve x^2 = 4", True),
        ("Is 7 a prime?", True),
        ("```python\nx=1\n```", True),
        ("class Foo:", True),
        ("step-by-step explanation", True),
        ("explain why this happens", True),
        ("Hi how are you", False),
        ("Tell me about cats", False),
        ("Hello world", False),
        ("$x^2 + y^2 = z^2$", True),  # latex
        ("derive the formula", True),
        ("optimize the algorithm", True),
    ])
    def test_signal_detection(self, text, expected):
        assert lr._has_reasoning_signal(text) == expected, (
            f"{text!r} expected signal={expected} but got {not expected}"
        )


# ─── Total chars + content-parts handling ───────────────────────────────


class TestContentExtraction:
    def test_string_content(self):
        req = _make_request(messages=[_user_msg("hello world")])
        assert lr._total_chars(req) == 11

    def test_content_parts_list(self):
        """Multipart content joined with '\n' — char count includes the
        separator (acceptable: 1 char per gap is negligible for threshold)."""
        msg = SimpleNamespace(role="user", content=[
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ])
        req = _make_request(messages=[msg])
        # join("\n") adds one separator between parts → +1 per gap
        expected = len("first") + len("\n") + len("second")
        assert lr._total_chars(req) == expected

    def test_empty_messages(self):
        req = _make_request(messages=[])
        assert lr._total_chars(req) == 0
        last = lr._last_user_text(req)
        assert last == ""

    def test_multiple_messages_summed(self):
        req = _make_request(messages=[
            _user_msg("aaa"),
            _assistant_msg("bbbbb"),
            _user_msg("ccccccc"),
        ])
        assert lr._total_chars(req) == 3 + 5 + 7

    def test_last_user_text_skips_assistant(self):
        req = _make_request(messages=[
            _user_msg("first user"),
            _assistant_msg("assistant reply"),
            _user_msg("second user"),
        ])
        assert lr._last_user_text(req) == "second user"

    def test_dict_message_shape(self):
        """Some clients pass dicts directly instead of pydantic models."""
        req = _make_request(messages=[
            {"role": "user", "content": "from dict"},
        ])
        assert lr._total_chars(req) == len("from dict")
        assert lr._last_user_text(req) == "from dict"


# ─── Threshold env override ─────────────────────────────────────────────


class TestThresholdConfig:
    def test_threshold_default(self, monkeypatch):
        monkeypatch.delenv("GENESIS_PN16_THRESHOLD_CHARS", raising=False)
        assert lr._threshold_chars() == 300

    def test_threshold_env_override(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "1000")
        assert lr._threshold_chars() == 1000

    def test_threshold_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "not a number")
        assert lr._threshold_chars() == 300

    def test_short_under_custom_threshold(self, env_pn16_on, monkeypatch):
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "20")
        # 5-char prompt under 20 threshold → disable
        req = _make_request(messages=[_user_msg("Hi!"
                                                )])
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs == {"enable_thinking": False}


# ─── Variant 5 — prompt-engineering soft cap ───────────────────────────


class TestVariant5SoftCap:
    def test_max_thinking_tokens_default_zero(self, monkeypatch):
        monkeypatch.delenv("GENESIS_PN16_MAX_THINKING_TOKENS", raising=False)
        assert lr._max_thinking_tokens() == 0

    def test_max_thinking_tokens_env_read(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        assert lr._max_thinking_tokens() == 200

    def test_no_cap_no_hint_injection(self, env_pn16_on, monkeypatch):
        """When cap=0 (default), no soft-cap hint is injected."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.delenv("GENESIS_PN16_MAX_THINKING_TOKENS", raising=False)
        long_text = "calculate the sum of digits"
        req = _make_request(messages=[_user_msg(long_text)])
        lr.apply_hook(None, req)
        # Heuristic kept thinking on, but cap=0 → no hint
        assert req.messages[0].content == long_text  # unchanged
        assert lr.get_stats()["soft_cap_hint_injected"] == 0
        assert lr.get_stats()["left_on_by_heuristic"] == 1

    def test_cap_set_hint_injected_into_last_user_msg(
        self, env_pn16_on, monkeypatch,
    ):
        """When cap > 0 AND thinking is allowed, hint is appended to
        last user message."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        long_text = "calculate the sum of digits"
        req = _make_request(messages=[_user_msg(long_text)])
        lr.apply_hook(None, req)
        # Hint should be appended to user message content
        assert long_text in req.messages[0].content
        assert "Genesis hint" in req.messages[0].content
        assert "200" in req.messages[0].content  # token count interpolated
        assert lr.get_stats()["soft_cap_hint_injected"] == 1

    def test_cap_appends_to_LAST_user_message_only(
        self, env_pn16_on, monkeypatch,
    ):
        """If multiple user messages, hint goes to the LAST one."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "150")
        req = _make_request(messages=[
            _user_msg("first user message about reasoning"),
            _assistant_msg("assistant reply"),
            _user_msg("second user message about derive"),
        ])
        lr.apply_hook(None, req)
        # First user msg unchanged
        assert "Genesis hint" not in req.messages[0].content
        # Last user msg has hint
        assert "Genesis hint" in req.messages[2].content
        assert "150" in req.messages[2].content

    def test_cap_with_no_user_message_skips_safely(
        self, env_pn16_on, monkeypatch, caplog,
    ):
        """If there is no user message, hint injection skips gracefully."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        # Only assistant message — no user message
        req = _make_request(messages=[
            _assistant_msg("just an assistant message about derive math"),
        ])
        # No exception, no hint injected
        lr.apply_hook(None, req)
        assert lr.get_stats()["soft_cap_hint_injected"] == 0

    def test_cap_with_dict_message(self, env_pn16_on, monkeypatch):
        """Dict-shaped messages also accept hint injection."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        req = _make_request(messages=[
            {"role": "user", "content": "calculate something here"},
        ])
        lr.apply_hook(None, req)
        assert "Genesis hint" in req.messages[0]["content"]
        assert lr.get_stats()["soft_cap_hint_injected"] == 1

    def test_cap_with_content_parts_list(self, env_pn16_on, monkeypatch):
        """Content-parts lists get a new text part appended."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        msg = SimpleNamespace(role="user", content=[
            {"type": "text", "text": "calculate something here"},
        ])
        req = _make_request(messages=[msg])
        lr.apply_hook(None, req)
        # Original part preserved + hint added as new part
        assert len(msg.content) == 2
        assert msg.content[0]["text"] == "calculate something here"
        assert "Genesis hint" in msg.content[1]["text"]
        assert lr.get_stats()["soft_cap_hint_injected"] == 1

    def test_cap_does_NOT_inject_when_thinking_disabled_by_v1(
        self, env_pn16_on, monkeypatch,
    ):
        """If variant 1 disabled thinking entirely, variant 5 must not
        also inject a hint (would be wasted tokens)."""
        monkeypatch.delenv("GENESIS_PN16_THRESHOLD_CHARS", raising=False)  # default 300
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        # Short trivial prompt — variant 1 disables thinking
        req = _make_request(messages=[_user_msg("Hi!")])
        lr.apply_hook(None, req)
        # variant 1 fired
        assert req.chat_template_kwargs == {"enable_thinking": False}
        # variant 5 did NOT also fire
        assert "Genesis hint" not in req.messages[0].content
        assert lr.get_stats()["soft_cap_hint_injected"] == 0
        assert lr.get_stats()["disabled_by_heuristic"] == 1

    def test_cap_does_NOT_inject_when_explicit_client_choice(
        self, env_pn16_on, monkeypatch,
    ):
        """Variant 3 (explicit client) wins — no variant-5 hint either."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        req = _make_request(
            messages=[_user_msg("calculate something here")],
            chat_template_kwargs={"enable_thinking": True},
        )
        lr.apply_hook(None, req)
        # User message unchanged
        assert "Genesis hint" not in req.messages[0].content
        assert lr.get_stats()["soft_cap_hint_injected"] == 0


class TestUpstreamBlockerWarning:
    def test_warning_emitted_once(self, env_pn16_on, monkeypatch, caplog):
        """The LogitsProcessor-blocker warning fires exactly once per
        process even across many requests."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        # Reset internal one-shot flag so this test starts fresh
        lr.reset_stats()
        with caplog.at_level("WARNING", logger="genesis.middleware.lazy_reasoner"):
            for _ in range(5):
                req = _make_request(messages=[_user_msg("calculate thing")])
                lr.apply_hook(None, req)
        # Filter for our specific warning
        blocker_warnings = [
            r for r in caplog.records
            if r.levelname == "WARNING" and "rejects custom LogitsProcessor" in r.message
        ]
        assert len(blocker_warnings) == 1, (
            f"expected exactly 1 blocker warning across 5 requests, "
            f"got {len(blocker_warnings)}"
        )

    def test_warning_not_emitted_when_cap_zero(
        self, env_pn16_on, monkeypatch, caplog,
    ):
        """When cap=0 (default), no warning fires — there's nothing to warn
        about because the operator hasn't asked for the cap."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.delenv("GENESIS_PN16_MAX_THINKING_TOKENS", raising=False)
        lr.reset_stats()
        with caplog.at_level("WARNING", logger="genesis.middleware.lazy_reasoner"):
            req = _make_request(messages=[_user_msg("calculate")])
            lr.apply_hook(None, req)
        blocker_warnings = [
            r for r in caplog.records
            if "rejects custom LogitsProcessor" in r.message
        ]
        assert blocker_warnings == []


# ─── Stats counters ─────────────────────────────────────────────────────


class TestStatsCounters:
    def test_get_stats_returns_copy(self, env_pn16_on):
        req = _make_request(messages=[_user_msg("hi")])
        lr.apply_hook(None, req)
        stats = lr.get_stats()
        stats["total_requests"] = 99999
        assert lr.get_stats()["total_requests"] == 1, (
            "get_stats() should return a copy, not the underlying dict"
        )

    def test_reset_stats_clears(self, env_pn16_on):
        req = _make_request(messages=[_user_msg("hi")])
        lr.apply_hook(None, req)
        lr.apply_hook(None, _make_request(messages=[_user_msg("hi")]))
        assert lr.get_stats()["total_requests"] == 2
        lr.reset_stats()
        assert lr.get_stats()["total_requests"] == 0


# ─── Dispatcher integration ─────────────────────────────────────────────


class TestDispatcherIntegration:
    def test_pn16_in_registry(self):
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        assert "PN16" in PATCH_REGISTRY
        meta = PATCH_REGISTRY["PN16"]
        assert meta["env_flag"] == "GENESIS_ENABLE_PN16_LAZY_REASONER"
        assert meta["default_on"] is False

    def test_dispatcher_should_apply_default_off(self, monkeypatch):
        monkeypatch.delenv("GENESIS_ENABLE_PN16_LAZY_REASONER", raising=False)
        from vllm._genesis.dispatcher import should_apply
        decision, _ = should_apply("PN16")
        assert decision is False

    def test_dispatcher_should_apply_env_on(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_PN16_LAZY_REASONER", "1")
        from vllm._genesis.dispatcher import should_apply
        decision, _ = should_apply("PN16")
        assert decision is True
