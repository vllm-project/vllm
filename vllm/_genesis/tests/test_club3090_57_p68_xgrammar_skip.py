# SPDX-License-Identifier: Apache-2.0
"""TDD for noonghunna/club-3090#57 — P68 xgrammar-incompat tool skip.

Bug (lexhoefsloot 2026-05-05): when P68 fires (`tool_choice` upgraded
"auto" -> "required") and ANY tool in `request.tools` has a JSON Schema
key xgrammar cannot compile (`patternProperties`, `propertyNames`,
`$ref`, `oneOf`, etc.), vLLM's downstream `_get_json_schema_from_tools`
builds a combined `anyOf` schema across ALL tools, xgrammar tries to
compile it, and 100% of long-prompt requests fail with:

    400 ValueError: The provided JSON schema contains features not
    supported by xgrammar.

Fix: scan `request.tools` for unsupported keys BEFORE upgrading
tool_choice. If any tool is incompatible, skip the upgrade (and let P69
still fire — only the upgrade step is unsafe). Operators on a
non-xgrammar backend can override via GENESIS_P68_FORCE=1.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm._genesis.middleware import long_ctx_tool_adherence as p68p69


# ─── Helpers ──────────────────────────────────────────────────────────


def _long_request(tools, *, prompt_chars: int = 60_000):
    """Build a SimpleNamespace request with a single user message of the
    requested prompt length, and the given tools list. tool_choice="auto"
    so the P68 path runs (caller can override after construction)."""
    msg = "x" * prompt_chars
    return SimpleNamespace(
        messages=[{"role": "user", "content": msg}],
        tools=tools,
        tool_choice="auto",
    )


def _clean_tool(name: str = "get_weather"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }


def _exec_tool_with_pattern_properties():
    """Reproducer from lexhoefsloot's club-3090#57 issue body — the
    OpenClaw `exec.env` tool that triggers the bug."""
    return {
        "type": "function",
        "function": {
            "name": "exec",
            "parameters": {
                "type": "object",
                "properties": {
                    "env": {
                        "type": "object",
                        "patternProperties": {
                            "^[A-Z_][A-Z0-9_]*$": {"type": "string"},
                        },
                    },
                },
            },
        },
    }


# ─── Schema scanner ───────────────────────────────────────────────────


class TestSchemaScanner:
    def test_clean_schema_returns_none(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        assert p68p69._scan_schema_for_unsupported_key(schema) is None

    def test_pattern_properties_caught(self):
        schema = {"type": "object", "patternProperties": {"^X": {"type": "string"}}}
        assert p68p69._scan_schema_for_unsupported_key(schema) == "patternProperties"

    def test_oneOf_caught(self):
        schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
        assert p68p69._scan_schema_for_unsupported_key(schema) == "oneOf"

    def test_ref_caught(self):
        schema = {"$ref": "#/definitions/Foo"}
        assert p68p69._scan_schema_for_unsupported_key(schema) == "$ref"

    def test_propertyNames_caught(self):
        schema = {"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}}
        assert p68p69._scan_schema_for_unsupported_key(schema) == "propertyNames"

    def test_nested_unsupported_key_in_properties_caught(self):
        # Real OpenClaw shape: `properties.env.patternProperties`
        schema = {
            "type": "object",
            "properties": {
                "env": {
                    "type": "object",
                    "patternProperties": {"^X": {"type": "string"}},
                }
            },
        }
        assert p68p69._scan_schema_for_unsupported_key(schema) == "patternProperties"

    def test_nested_in_array_items(self):
        schema = {
            "type": "array",
            "items": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        }
        assert p68p69._scan_schema_for_unsupported_key(schema) == "oneOf"

    def test_depth_limit_does_not_crash_on_cycle(self):
        # Build a self-referential dict (cyclic) — must not RecursionError
        schema: dict = {"type": "object", "properties": {}}
        schema["properties"]["self"] = schema
        # Should return None (no unsupported key) within depth limit
        assert p68p69._scan_schema_for_unsupported_key(schema) is None


# ─── Tool walker ──────────────────────────────────────────────────────


class TestFindIncompatTool:
    def test_empty_list_returns_none(self):
        assert p68p69._find_xgrammar_incompat_tool([]) is None
        assert p68p69._find_xgrammar_incompat_tool(None) is None

    def test_all_clean_tools_returns_none(self):
        assert p68p69._find_xgrammar_incompat_tool(
            [_clean_tool("a"), _clean_tool("b")]
        ) is None

    def test_finds_first_offender(self):
        tools = [_clean_tool("a"), _exec_tool_with_pattern_properties(),
                 _clean_tool("c")]
        result = p68p69._find_xgrammar_incompat_tool(tools)
        assert result == ("exec", "patternProperties")

    def test_handles_pydantic_style_tool_objects(self):
        # Some clients pass pydantic models, not dicts
        from types import SimpleNamespace as N
        bad_tool = N(function=N(name="exec", parameters={
            "type": "object",
            "patternProperties": {"^X": {"type": "string"}},
        }))
        result = p68p69._find_xgrammar_incompat_tool([bad_tool])
        assert result == ("exec", "patternProperties")

    def test_handles_tool_without_parameters(self):
        # Tool with no `parameters` field — must not crash
        tool = {"type": "function", "function": {"name": "no_args"}}
        assert p68p69._find_xgrammar_incompat_tool([tool]) is None


# ─── apply_hook integration: skip path ────────────────────────────────


class TestApplyHookSkipsP68OnIncompat:
    """When tools contain xgrammar-unsupported keys, P68 must NOT upgrade
    tool_choice, even though prompt length + env flag would otherwise
    trigger it."""

    def test_p68_skipped_when_tool_has_patternProperties(
        self, monkeypatch, caplog,
    ):
        monkeypatch.setenv("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL", "1")
        monkeypatch.setenv("GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "1000")
        monkeypatch.delenv("GENESIS_P68_FORCE", raising=False)

        tools = [_clean_tool("get_weather"),
                 _exec_tool_with_pattern_properties()]
        request = _long_request(tools, prompt_chars=60_000)

        result = p68p69.apply_hook(serving_chat=None, request=request)

        # P68 must NOT have fired
        assert result["applied_p68"] is False
        # tool_choice must still be "auto" (untouched)
        assert request.tool_choice == "auto"
        # Reason must mention club-3090#57 + the offending tool
        assert "P68 skipped" in result["reason"]
        assert "exec" in result["reason"]
        assert "patternProperties" in result["reason"]
        assert "club-3090#57" in result["reason"]

    def test_p68_fires_when_all_tools_clean(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL", "1")
        monkeypatch.setenv("GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "1000")
        monkeypatch.delenv("GENESIS_P68_FORCE", raising=False)

        tools = [_clean_tool("a"), _clean_tool("b")]
        request = _long_request(tools, prompt_chars=60_000)

        result = p68p69.apply_hook(serving_chat=None, request=request)

        # P68 should fire — no incompat tools blocked it
        assert result["applied_p68"] is True
        assert request.tool_choice == "required"

    def test_force_env_overrides_skip(self, monkeypatch):
        """Operators on a non-xgrammar backend (guidance / outlines /
        llguidance) can opt out of the skip via GENESIS_P68_FORCE=1."""
        monkeypatch.setenv("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL", "1")
        monkeypatch.setenv("GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "1000")
        monkeypatch.setenv("GENESIS_P68_FORCE", "1")

        tools = [_exec_tool_with_pattern_properties()]
        request = _long_request(tools, prompt_chars=60_000)

        result = p68p69.apply_hook(serving_chat=None, request=request)

        # FORCE override means P68 fires anyway
        assert result["applied_p68"] is True
        assert request.tool_choice == "required"

    def test_p69_still_fires_when_p68_skipped(self, monkeypatch):
        """P69 (the format reminder) doesn't depend on grammar compilation
        — it only mutates the user message text. Must still apply when P68
        is skipped due to xgrammar incompat."""
        monkeypatch.setenv("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL", "1")
        monkeypatch.setenv("GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER", "1")
        monkeypatch.setenv("GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "1000")
        monkeypatch.delenv("GENESIS_P68_FORCE", raising=False)

        tools = [_exec_tool_with_pattern_properties()]
        request = _long_request(tools, prompt_chars=60_000)
        original_content = request.messages[-1]["content"]

        result = p68p69.apply_hook(serving_chat=None, request=request)

        assert result["applied_p68"] is False, "P68 must skip"
        assert result["applied_p69"] is True, "P69 must still fire"
        assert request.messages[-1]["content"] != original_content
        assert "[SYSTEM REMINDER" in request.messages[-1]["content"]

    def test_short_prompt_does_not_trigger_either_path(self, monkeypatch):
        """Sanity: below threshold, neither P68 nor the new skip log fire."""
        monkeypatch.setenv("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL", "1")
        monkeypatch.setenv(
            "GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "100000",
        )

        tools = [_exec_tool_with_pattern_properties()]
        request = _long_request(tools, prompt_chars=5_000)

        result = p68p69.apply_hook(serving_chat=None, request=request)

        assert result["applied_p68"] is False
        assert request.tool_choice == "auto"  # untouched
        # Reason should be the threshold gate, not the schema scanner
        assert "threshold" in result["reason"]
