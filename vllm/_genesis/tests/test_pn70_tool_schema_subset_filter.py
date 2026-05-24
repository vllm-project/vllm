# SPDX-License-Identifier: Apache-2.0
"""TDD for PN70 — Tool schema subset filter (club-3090#57 option-3).

Companion to v7.72.1 P68 fix (option-1). Where P68 refuses to upgrade
tool_choice on dirty catalogs, PN70 keeps the upgrade and filters dirty
tools out of grammar enforcement.

Test strategy:
  1. Pure-function tests for the filter helpers
  2. Wrap-function tests with a stub `original` (no upstream import needed)
  3. apply()-level tests with the env flag matrix
  4. Composability sanity: P68 still works independently
  5. Idempotency
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm._genesis.wiring.structured_output import (
    patch_N70_tool_schema_subset_filter as pn70,
)


# ─── Helpers ──────────────────────────────────────────────────────────


def _clean_tool(name: str = "get_weather"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def _exec_tool_with_pattern_properties(name: str = "exec"):
    return {
        "type": "function",
        "function": {
            "name": name,
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


def _capturing_original():
    """Build a stub `original` impl that records what it was called with."""
    calls = []
    def _orig(tools):
        calls.append(list(tools) if tools else [])
        # Mirror the upstream shape so callers can sanity-check
        return {
            "type": "array",
            "minItems": 1,
            "items": {"type": "object", "anyOf": ["<schema>"] * len(tools or [])},
        }
    return _orig, calls


# ─── Tool compatibility check ─────────────────────────────────────────


class TestToolCompatCheck:
    def test_clean_tool_is_compat(self):
        assert pn70._tool_is_xgrammar_compat(_clean_tool()) is True

    def test_pattern_properties_tool_is_incompat(self):
        assert pn70._tool_is_xgrammar_compat(
            _exec_tool_with_pattern_properties()
        ) is False

    def test_pydantic_style_tool_compat(self):
        bad = SimpleNamespace(function=SimpleNamespace(
            name="exec",
            parameters={
                "type": "object",
                "patternProperties": {"^X": {"type": "string"}},
            },
        ))
        assert pn70._tool_is_xgrammar_compat(bad) is False

    def test_tool_without_function_attr_is_compat(self):
        # Malformed (no function= attr / function= key); treat as compat so
        # upstream gets to handle the malformed shape itself.
        assert pn70._tool_is_xgrammar_compat(SimpleNamespace()) is True

    def test_tool_without_parameters_is_compat(self):
        tool = {"type": "function", "function": {"name": "no_args"}}
        assert pn70._tool_is_xgrammar_compat(tool) is True

    def test_scanner_crash_falls_back_to_compat(self, monkeypatch):
        """If the underlying scanner raises, treat the tool as compat
        (defensive — never break a request PN70 was supposed to help)."""
        def _explode(*a, **kw):
            raise RuntimeError("synthetic scanner crash")
        monkeypatch.setattr(
            "vllm._genesis.wiring.structured_output."
            "patch_N70_tool_schema_subset_filter."
            "_scan_schema_for_unsupported_key",
            _explode,
        )
        assert pn70._tool_is_xgrammar_compat(_clean_tool()) is True


class TestToolNameExtractor:
    def test_dict_tool(self):
        assert pn70._tool_name(_clean_tool("foo")) == "foo"

    def test_pydantic_tool(self):
        t = SimpleNamespace(function=SimpleNamespace(name="bar"))
        assert pn70._tool_name(t) == "bar"

    def test_no_function_falls_back(self):
        assert pn70._tool_name(SimpleNamespace()) == "<anonymous>"

    def test_no_name_falls_back(self):
        t = SimpleNamespace(function=SimpleNamespace())
        assert pn70._tool_name(t) == "<anonymous>"

    def test_empty_dict_function(self):
        assert pn70._tool_name({"type": "function", "function": {}}) == "<anonymous>"


# ─── Wrap function — env-flag matrix + filter behavior ────────────────


class TestWrappedGetJsonSchemaFromTools:
    def test_env_flag_off_passes_through_unchanged(self, monkeypatch):
        monkeypatch.delenv(
            "GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER", raising=False,
        )
        orig, calls = _capturing_original()
        wrapped = pn70._wrap_get_json_schema_from_tools(orig)

        tools = [_clean_tool("a"), _exec_tool_with_pattern_properties()]
        wrapped(tools)

        # original called with FULL list — no filter
        assert len(calls) == 1
        assert len(calls[0]) == 2

    def test_env_flag_on_no_incompat_passes_through(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER", "1")
        orig, calls = _capturing_original()
        wrapped = pn70._wrap_get_json_schema_from_tools(orig)

        tools = [_clean_tool("a"), _clean_tool("b")]
        wrapped(tools)

        # All clean — original called with full list, no filter
        assert len(calls) == 1
        assert len(calls[0]) == 2

    def test_env_flag_on_filters_incompat(self, monkeypatch, caplog):
        import logging
        monkeypatch.setenv("GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER", "1")
        orig, calls = _capturing_original()
        wrapped = pn70._wrap_get_json_schema_from_tools(orig)

        tools = [
            _clean_tool("get_weather"),
            _exec_tool_with_pattern_properties("exec"),
            _clean_tool("get_news"),
        ]
        with caplog.at_level(logging.WARNING,
                             logger="genesis.wiring.pn70_tool_schema_subset_filter"):
            wrapped(tools)

        # original called with filtered (compat-only) list — exec dropped
        assert len(calls) == 1
        assert len(calls[0]) == 2
        names_passed = [
            (t.get("function") or {}).get("name") for t in calls[0]
        ]
        assert "exec" not in names_passed
        assert "get_weather" in names_passed
        assert "get_news" in names_passed
        # WARN log mentions exec + the filter count
        log_text = " ".join(r.message for r in caplog.records)
        assert "exec" in log_text
        assert "filtered 1/3" in log_text or "filtered" in log_text.lower()

    def test_env_flag_on_all_incompat_returns_None(self, monkeypatch, caplog):
        import logging
        monkeypatch.setenv("GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER", "1")
        orig, calls = _capturing_original()
        wrapped = pn70._wrap_get_json_schema_from_tools(orig)

        tools = [
            _exec_tool_with_pattern_properties("exec1"),
            _exec_tool_with_pattern_properties("exec2"),
        ]
        with caplog.at_level(logging.WARNING,
                             logger="genesis.wiring.pn70_tool_schema_subset_filter"):
            result = wrapped(tools)

        # Subset empty — original NOT called, return None directly
        assert result is None
        assert calls == []
        log_text = " ".join(r.message for r in caplog.records)
        assert "all" in log_text.lower() and "incompat" in log_text.lower() \
            or "all 2 tools" in log_text

    def test_empty_tools_passes_through(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER", "1")
        orig, calls = _capturing_original()
        wrapped = pn70._wrap_get_json_schema_from_tools(orig)

        wrapped([])
        wrapped(None)  # type: ignore[arg-type]

        # Both pass through to original (let it handle the empty case)
        assert len(calls) == 2

    def test_filter_exception_falls_back_to_original(
        self, monkeypatch, caplog,
    ):
        """Defensive: if the filter logic itself crashes, call original
        with the unfiltered list. PN70 must NEVER break a request that
        would have succeeded under stock vLLM."""
        import logging
        monkeypatch.setenv("GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER", "1")
        orig, calls = _capturing_original()
        wrapped = pn70._wrap_get_json_schema_from_tools(orig)

        def _explode(*a, **kw):
            raise RuntimeError("synthetic filter crash")
        monkeypatch.setattr(
            "vllm._genesis.wiring.structured_output."
            "patch_N70_tool_schema_subset_filter."
            "_tool_is_xgrammar_compat",
            _explode,
        )

        tools = [_clean_tool("a"), _clean_tool("b")]
        with caplog.at_level(logging.WARNING,
                             logger="genesis.wiring.pn70_tool_schema_subset_filter"):
            wrapped(tools)

        # Original called with full list as fallback
        assert len(calls) == 1
        assert len(calls[0]) == 2
        # Fallback path should have logged the failure
        log_text = " ".join(r.message for r in caplog.records).lower()
        assert "fall" in log_text or "fallback" in log_text or "synthetic" in log_text

    @pytest.mark.parametrize("flag_value", ["1", "true", "Y", "ON", "yes"])
    def test_env_flag_truthy_variants(self, monkeypatch, flag_value):
        monkeypatch.setenv(
            "GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER", flag_value,
        )
        orig, calls = _capturing_original()
        wrapped = pn70._wrap_get_json_schema_from_tools(orig)

        tools = [
            _clean_tool("a"),
            _exec_tool_with_pattern_properties("exec"),
        ]
        wrapped(tools)
        # filter active → only 1 tool passed to original
        assert len(calls[0]) == 1


# ─── apply() — wrapper installation, idempotency, NULL-skip ────────────


class TestApply:
    def test_apply_skipped_when_disabled(self, monkeypatch):
        monkeypatch.delenv("GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER",
                           raising=False)
        status, reason = pn70.apply()
        assert status == "skipped"

    def test_apply_skipped_when_module_missing(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER", "1")
        import sys
        monkeypatch.setitem(sys.modules, "vllm.tool_parsers.utils", None)
        status, reason = pn70.apply()
        assert status == "skipped"
        assert "not importable" in reason.lower() or "null" in reason.lower()

    def test_apply_idempotent(self, monkeypatch):
        """If the upstream symbol is already wrapped, apply must return
        ('applied', 'already wrapped (idempotent)') without re-wrapping."""
        monkeypatch.setenv("GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER", "1")
        # Stub both the parent package AND the submodule (apply() uses
        # `from vllm.tool_parsers import utils as _u` so we need both).
        from types import ModuleType
        import sys
        fake_pkg = sys.modules.get("vllm.tool_parsers") or ModuleType("vllm.tool_parsers")
        fake_u = ModuleType("vllm.tool_parsers.utils")
        def _orig(tools): return None
        fake_u._get_json_schema_from_tools = _orig
        fake_pkg.utils = fake_u
        monkeypatch.setitem(sys.modules, "vllm.tool_parsers", fake_pkg)
        monkeypatch.setitem(sys.modules, "vllm.tool_parsers.utils", fake_u)

        # First apply — wraps
        s1, r1 = pn70.apply()
        assert s1 == "applied", f"first apply: {r1}"
        assert getattr(fake_u._get_json_schema_from_tools,
                       "__pn70_wrapped__", False) is True

        # Second apply — idempotent
        wrapped_first = fake_u._get_json_schema_from_tools
        s2, r2 = pn70.apply()
        assert s2 == "applied"
        assert "idempotent" in r2.lower() or "already" in r2.lower()
        # Verify no double-wrap
        assert fake_u._get_json_schema_from_tools is wrapped_first


# ─── Dispatcher registration sanity ───────────────────────────────────


class TestDispatcherRegistration:
    def test_pn70_in_patch_registry(self):
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        assert "PN70" in PATCH_REGISTRY
        meta = PATCH_REGISTRY["PN70"]
        assert meta["env_flag"] == "GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER"
        assert meta["default_on"] is False
        assert meta["category"] == "structured_output"
        assert "P68" in meta.get("composes_with", [])


# ─── Composability with P68 (regression check) ────────────────────────


class TestComposabilityWithP68:
    def test_p68_still_works_without_pn70(self, monkeypatch):
        """PN70 must NOT break the existing P68 option-1 skip path. When
        only P68 is enabled and PN70 is off, P68 keeps refusing to
        upgrade on dirty catalogs (existing v7.72.1 contract)."""
        from vllm._genesis.middleware import long_ctx_tool_adherence as p68

        monkeypatch.setenv("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL", "1")
        monkeypatch.setenv("GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "1000")
        monkeypatch.delenv("GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER",
                           raising=False)
        monkeypatch.delenv("GENESIS_P68_FORCE", raising=False)

        # Long prompt + dirty catalog — P68 must STILL skip (option-1)
        request = SimpleNamespace(
            messages=[{"role": "user", "content": "x" * 60_000}],
            tools=[_exec_tool_with_pattern_properties()],
            tool_choice="auto",
        )
        result = p68.apply_hook(serving_chat=None, request=request)
        assert result["applied_p68"] is False
        assert request.tool_choice == "auto"

    def test_pn70_does_not_disturb_p68_clean_tools(self, monkeypatch):
        """P68 + clean tools = P68 fires regardless of PN70 state."""
        from vllm._genesis.middleware import long_ctx_tool_adherence as p68
        monkeypatch.setenv("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL", "1")
        monkeypatch.setenv("GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "1000")
        monkeypatch.setenv("GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER", "1")
        monkeypatch.delenv("GENESIS_P68_FORCE", raising=False)

        request = SimpleNamespace(
            messages=[{"role": "user", "content": "x" * 60_000}],
            tools=[_clean_tool("a"), _clean_tool("b")],
            tool_choice="auto",
        )
        result = p68.apply_hook(serving_chat=None, request=request)
        assert result["applied_p68"] is True
        assert request.tool_choice == "required"
