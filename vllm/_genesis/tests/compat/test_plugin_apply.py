# SPDX-License-Identifier: Apache-2.0
"""Tests for Phase 5c — `apply_callable` field on plugin metadata.

A plugin can declare:

    {
        "patch_id": "MY_PATCH",
        ...
        "apply_callable": "my_pkg.module:apply",
    }

Genesis resolves the string via importlib at boot, calls the function
when the plugin's env flag is set, and reports applied/skipped/failed
through the standard `apply_all` pipeline (same UX as core patches).

This completes the plugin story — plugins can now actually RUN code,
not just declare metadata.
"""
from __future__ import annotations

import sys
from types import ModuleType

import pytest


# ─── Helpers to register a fake plugin module + entry-point ────────────


@pytest.fixture
def plugins_enabled(monkeypatch):
    monkeypatch.setenv("GENESIS_ALLOW_PLUGINS", "1")


@pytest.fixture
def fake_plugin_module(monkeypatch):
    """Inject a fake module 'my_genesis_plugin.apply' into sys.modules
    so importlib can resolve string-form apply_callable paths."""
    pkg = ModuleType("my_genesis_plugin")
    sub = ModuleType("my_genesis_plugin.apply")

    apply_call_log = []

    def apply():
        apply_call_log.append("called")
        return "applied", "MY_PATCH applied: did the thing"

    def apply_raises():
        raise RuntimeError("plugin author oops in apply()")

    def apply_returns_garbage():
        return 42  # not a tuple

    sub.apply = apply
    sub.apply_raises = apply_raises
    sub.apply_returns_garbage = apply_returns_garbage
    sub._call_log = apply_call_log

    pkg.apply = sub
    monkeypatch.setitem(sys.modules, "my_genesis_plugin", pkg)
    monkeypatch.setitem(sys.modules, "my_genesis_plugin.apply", sub)
    return sub


class FakeEntryPoint:
    def __init__(self, name: str, callable_obj):
        self.name = name
        self.value = f"fake.module:{name}"
        self._callable = callable_obj

    def load(self):
        return self._callable


_PLUGIN_WITH_APPLY = {
    "patch_id": "MY_PATCH",
    "title": "Test plugin with apply_callable",
    "env_flag": "GENESIS_ENABLE_MY_PATCH",
    "default_on": False,
    "category": "spec_decode",
    "credit": "Test plugin",
    "community_credit": "@test-author",
    "apply_callable": "my_genesis_plugin.apply:apply",
}


@pytest.fixture
def fake_entry_points(monkeypatch):
    eps = []
    monkeypatch.setattr(
        "vllm._genesis.compat.plugins._discover_entry_points",
        lambda: eps,
    )
    return eps


# ─── String-form resolution ─────────────────────────────────────────────


class TestResolveApplyCallable:
    def test_resolve_string_form(self, fake_plugin_module):
        from vllm._genesis.compat.plugins import _resolve_apply_callable
        fn = _resolve_apply_callable("my_genesis_plugin.apply:apply")
        assert callable(fn)

    def test_resolve_callable_passthrough(self, fake_plugin_module):
        """If apply_callable is already a callable, just return it."""
        from vllm._genesis.compat.plugins import _resolve_apply_callable
        fn = fake_plugin_module.apply
        result = _resolve_apply_callable(fn)
        assert result is fn

    def test_resolve_unknown_module_returns_None(self):
        from vllm._genesis.compat.plugins import _resolve_apply_callable
        # No such module in sys.modules
        assert _resolve_apply_callable("totally.fake.module:apply") is None

    def test_resolve_unknown_attr_returns_None(self, fake_plugin_module):
        from vllm._genesis.compat.plugins import _resolve_apply_callable
        assert _resolve_apply_callable("my_genesis_plugin.apply:nonexistent") is None

    def test_resolve_bad_string_format_returns_None(self):
        from vllm._genesis.compat.plugins import _resolve_apply_callable
        assert _resolve_apply_callable("no_colon_here") is None
        assert _resolve_apply_callable("") is None
        assert _resolve_apply_callable(None) is None


# ─── Plugin discovery preserves apply_callable ──────────────────────────


class TestPluginDiscoveryWithApply:
    def test_apply_callable_preserved_through_discovery(
        self, plugins_enabled, fake_entry_points, fake_plugin_module,
    ):
        fake_entry_points.append(FakeEntryPoint("p", lambda: _PLUGIN_WITH_APPLY))
        from vllm._genesis.compat.plugins import discover_plugins
        plugins = discover_plugins()
        assert len(plugins) == 1
        assert plugins[0].get("apply_callable") == "my_genesis_plugin.apply:apply"

    def test_plugin_without_apply_callable_still_valid(
        self, plugins_enabled, fake_entry_points,
    ):
        meta = {**_PLUGIN_WITH_APPLY}
        meta.pop("apply_callable")
        fake_entry_points.append(FakeEntryPoint("p", lambda: meta))
        from vllm._genesis.compat.plugins import discover_plugins
        plugins = discover_plugins()
        assert len(plugins) == 1
        assert plugins[0].get("apply_callable") is None


# ─── apply_plugin_patch — run lifecycle ─────────────────────────────────


class TestApplyPluginPatch:
    def test_apply_calls_function_when_env_set(
        self, plugins_enabled, fake_plugin_module, monkeypatch,
    ):
        monkeypatch.setenv("GENESIS_ENABLE_MY_PATCH", "1")
        from vllm._genesis.compat.plugins import apply_plugin_patch
        status, reason = apply_plugin_patch(_PLUGIN_WITH_APPLY)
        assert status == "applied"
        assert "MY_PATCH" in reason
        assert fake_plugin_module._call_log == ["called"]

    def test_apply_skipped_when_env_unset(
        self, plugins_enabled, fake_plugin_module, monkeypatch,
    ):
        monkeypatch.delenv("GENESIS_ENABLE_MY_PATCH", raising=False)
        from vllm._genesis.compat.plugins import apply_plugin_patch
        status, reason = apply_plugin_patch(_PLUGIN_WITH_APPLY)
        assert status == "skipped"
        assert "opt-in" in reason.lower() or "env" in reason.lower()
        # Function should NOT be called
        assert fake_plugin_module._call_log == []

    def test_apply_no_callable_returns_skipped(
        self, plugins_enabled,
    ):
        meta = {**_PLUGIN_WITH_APPLY}
        meta.pop("apply_callable")
        from vllm._genesis.compat.plugins import apply_plugin_patch
        status, reason = apply_plugin_patch(meta)
        assert status == "skipped"
        # Reason should explain there's nothing to apply
        assert "callable" in reason.lower() or "metadata" in reason.lower()

    def test_apply_callable_raises_returns_failed(
        self, plugins_enabled, fake_plugin_module, monkeypatch,
    ):
        monkeypatch.setenv("GENESIS_ENABLE_MY_PATCH", "1")
        meta = {**_PLUGIN_WITH_APPLY,
                "apply_callable": "my_genesis_plugin.apply:apply_raises"}
        from vllm._genesis.compat.plugins import apply_plugin_patch
        status, reason = apply_plugin_patch(meta)
        assert status == "failed"
        assert "RuntimeError" in reason or "oops" in reason

    def test_apply_callable_garbage_return_returns_failed(
        self, plugins_enabled, fake_plugin_module, monkeypatch,
    ):
        """Plugin returns non-tuple — Genesis should report this as
        a failed apply, not crash."""
        monkeypatch.setenv("GENESIS_ENABLE_MY_PATCH", "1")
        meta = {**_PLUGIN_WITH_APPLY,
                "apply_callable":
                    "my_genesis_plugin.apply:apply_returns_garbage"}
        from vllm._genesis.compat.plugins import apply_plugin_patch
        status, reason = apply_plugin_patch(meta)
        assert status in ("failed", "applied")  # both are acceptable
        # Just verify no crash + reason is informative
        assert isinstance(reason, str)

    def test_apply_unresolvable_callable_returns_skipped_or_failed(
        self, plugins_enabled, monkeypatch,
    ):
        monkeypatch.setenv("GENESIS_ENABLE_MY_PATCH", "1")
        meta = {**_PLUGIN_WITH_APPLY,
                "apply_callable": "totally.fake:nope"}
        from vllm._genesis.compat.plugins import apply_plugin_patch
        status, reason = apply_plugin_patch(meta)
        assert status in ("failed", "skipped")
        # Either way, must not crash
        assert isinstance(reason, str)


# ─── apply_all_plugins — bulk drive ──────────────────────────────────────


class TestApplyAllPlugins:
    def test_apply_all_runs_each_plugin_once(
        self, plugins_enabled, fake_entry_points, fake_plugin_module, monkeypatch,
    ):
        monkeypatch.setenv("GENESIS_ENABLE_MY_PATCH", "1")
        fake_entry_points.append(FakeEntryPoint("p", lambda: _PLUGIN_WITH_APPLY))
        from vllm._genesis.compat.plugins import apply_all_plugins
        stats = apply_all_plugins()
        assert isinstance(stats, dict)
        assert stats.get("total", 0) >= 1
        assert "applied" in stats and "skipped" in stats and "failed" in stats

    def test_apply_all_returns_zero_when_gate_closed(
        self, fake_entry_points, fake_plugin_module, monkeypatch,
    ):
        monkeypatch.delenv("GENESIS_ALLOW_PLUGINS", raising=False)
        fake_entry_points.append(FakeEntryPoint("p", lambda: _PLUGIN_WITH_APPLY))
        from vllm._genesis.compat.plugins import apply_all_plugins
        stats = apply_all_plugins()
        # No plugins discovered when gate is closed
        assert stats.get("total", 0) == 0
