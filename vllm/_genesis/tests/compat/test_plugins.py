# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm._genesis.compat.plugins — community plugin entry-points.

Plugins extend Genesis with community-shipped patches without forking
the core repo. Third-party packages declare entry-points in the
`vllm_genesis_patches` group:

    [project.entry-points."vllm_genesis_patches"]
    my_patch = "my_pkg.genesis_plugin:get_patch_metadata"

The callable returns either a dict (single patch) or list of dicts
(multiple). Each dict is validated against patch_entry.schema.json,
auto-tagged with `lifecycle: community`, and registered into a
parallel `_PLUGIN_REGISTRY` (NOT mixed into core PATCH_REGISTRY by
default — keeps community provenance auditable).

Security: plugins are OPT-IN via `GENESIS_ALLOW_PLUGINS=1`. Default
behavior: zero plugin discovery, zero foreign code loaded. Genesis
must boot identically with or without plugins installed.
"""
from __future__ import annotations


import pytest


_VALID_PLUGIN_DICT = {
    "patch_id": "PLUGIN_TEST_A",
    "title": "Community plugin A — test fixture",
    "env_flag": "GENESIS_ENABLE_PLUGIN_TEST_A",
    "default_on": False,
    "category": "spec_decode",
    "credit": "Test community plugin A.",
    "community_credit": "@test-author on GitHub",
}


# ─── Mock entry-point fixture ───────────────────────────────────────────


class FakeEntryPoint:
    """Minimal mock of importlib.metadata.EntryPoint."""
    def __init__(self, name: str, callable_obj):
        self.name = name
        self.value = f"fake.module:{name}"
        self._callable = callable_obj

    def load(self):
        return self._callable


@pytest.fixture
def fake_entry_points(monkeypatch):
    """Inject synthetic entry points into the discovery path."""
    eps = []
    monkeypatch.setattr(
        "vllm._genesis.compat.plugins._discover_entry_points",
        lambda: eps,
    )
    return eps


@pytest.fixture
def plugins_enabled(monkeypatch):
    monkeypatch.setenv("GENESIS_ALLOW_PLUGINS", "1")


@pytest.fixture
def plugins_disabled(monkeypatch):
    monkeypatch.delenv("GENESIS_ALLOW_PLUGINS", raising=False)


# ─── OPT-IN security gate ────────────────────────────────────────────────


class TestOptInGate:
    def test_default_off_returns_empty(self, plugins_disabled, fake_entry_points):
        """No env → no discovery even if entry points exist."""
        fake_entry_points.append(FakeEntryPoint("p", lambda: _VALID_PLUGIN_DICT))
        from vllm._genesis.compat.plugins import discover_plugins
        plugins = discover_plugins()
        assert plugins == []

    def test_env_on_enables_discovery(self, plugins_enabled, fake_entry_points):
        fake_entry_points.append(FakeEntryPoint("p", lambda: _VALID_PLUGIN_DICT))
        from vllm._genesis.compat.plugins import discover_plugins
        plugins = discover_plugins()
        assert len(plugins) == 1
        assert plugins[0]["patch_id"] == "PLUGIN_TEST_A"


# ─── Discovery shape ─────────────────────────────────────────────────────


class TestDiscovery:
    def test_single_dict_plugin(self, plugins_enabled, fake_entry_points):
        fake_entry_points.append(FakeEntryPoint("p", lambda: _VALID_PLUGIN_DICT))
        from vllm._genesis.compat.plugins import discover_plugins
        plugins = discover_plugins()
        assert len(plugins) == 1
        # Lifecycle auto-tagged community regardless of input
        assert plugins[0]["lifecycle"] == "community"

    def test_list_of_dicts_plugin(self, plugins_enabled, fake_entry_points):
        """A plugin can return a list to ship multiple patches."""
        def two_patches():
            return [
                {**_VALID_PLUGIN_DICT, "patch_id": "PLUGIN_A",
                 "env_flag": "GENESIS_ENABLE_PLUGIN_A"},
                {**_VALID_PLUGIN_DICT, "patch_id": "PLUGIN_B",
                 "env_flag": "GENESIS_ENABLE_PLUGIN_B"},
            ]
        fake_entry_points.append(FakeEntryPoint("multi", two_patches))
        from vllm._genesis.compat.plugins import discover_plugins
        plugins = discover_plugins()
        assert len(plugins) == 2
        ids = {p["patch_id"] for p in plugins}
        assert ids == {"PLUGIN_A", "PLUGIN_B"}

    def test_callable_raises_isolated(self, plugins_enabled, fake_entry_points):
        """One bad plugin must not break discovery of others."""
        def good():
            return {**_VALID_PLUGIN_DICT, "patch_id": "GOOD",
                    "env_flag": "GENESIS_ENABLE_GOOD"}
        def bad():
            raise RuntimeError("plugin author oops")
        fake_entry_points.append(FakeEntryPoint("bad", bad))
        fake_entry_points.append(FakeEntryPoint("good", good))
        from vllm._genesis.compat.plugins import discover_plugins
        plugins = discover_plugins()
        # The good one survives
        ids = {p["patch_id"] for p in plugins}
        assert "GOOD" in ids

    def test_lifecycle_force_community(self, plugins_enabled, fake_entry_points):
        """Plugin can't claim 'stable' or 'experimental' — auto-tagged community."""
        rogue = {**_VALID_PLUGIN_DICT, "lifecycle": "stable"}
        fake_entry_points.append(FakeEntryPoint("rogue", lambda: rogue))
        from vllm._genesis.compat.plugins import discover_plugins
        plugins = discover_plugins()
        assert plugins[0]["lifecycle"] == "community"

    def test_origin_metadata_added(self, plugins_enabled, fake_entry_points):
        """Each discovered plugin gets _plugin_origin attached so doctor
        can show 'this came from <module>:<entrypoint>' provenance."""
        fake_entry_points.append(FakeEntryPoint(
            "my_ep", lambda: _VALID_PLUGIN_DICT,
        ))
        from vllm._genesis.compat.plugins import discover_plugins
        plugins = discover_plugins()
        assert "_plugin_origin" in plugins[0]
        assert "my_ep" in plugins[0]["_plugin_origin"]


# ─── Validation ─────────────────────────────────────────────────────────


class TestValidation:
    def test_plugin_violating_schema_skipped(
        self, plugins_enabled, fake_entry_points,
    ):
        """A plugin returning a malformed dict must be SKIPPED, not
        crash discovery."""
        bad_dict = {"title": "missing required env_flag"}  # no env_flag, default_on
        fake_entry_points.append(FakeEntryPoint("bad", lambda: bad_dict))
        from vllm._genesis.compat.plugins import discover_plugins
        plugins = discover_plugins()
        assert plugins == []

    def test_plugin_collision_with_core_skipped(
        self, plugins_enabled, fake_entry_points,
    ):
        """A plugin can't override a core PATCH_REGISTRY id."""
        # Use a real core patch ID — PN14 is in core registry
        rogue = {**_VALID_PLUGIN_DICT, "patch_id": "PN14",
                  "env_flag": "GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP"}
        fake_entry_points.append(FakeEntryPoint("rogue", lambda: rogue))
        from vllm._genesis.compat.plugins import discover_plugins
        plugins = discover_plugins()
        assert plugins == [], (
            "plugin claiming a core patch_id must be rejected"
        )

    def test_plugin_invalid_callable_return_type(
        self, plugins_enabled, fake_entry_points,
    ):
        """Plugin returning a non-dict, non-list must be rejected."""
        fake_entry_points.append(FakeEntryPoint("bad", lambda: "not a dict"))
        from vllm._genesis.compat.plugins import discover_plugins
        plugins = discover_plugins()
        assert plugins == []


# ─── Registration into PATCH_REGISTRY ────────────────────────────────────


class TestRegistration:
    def test_register_adds_to_registry(self, plugins_enabled, fake_entry_points):
        fake_entry_points.append(FakeEntryPoint("p", lambda: _VALID_PLUGIN_DICT))
        from vllm._genesis.compat.plugins import register_plugins
        from vllm._genesis import dispatcher

        # Snapshot pre
        pre_keys = set(dispatcher.PATCH_REGISTRY.keys())
        register_plugins()
        try:
            assert "PLUGIN_TEST_A" in dispatcher.PATCH_REGISTRY
            entry = dispatcher.PATCH_REGISTRY["PLUGIN_TEST_A"]
            assert entry["lifecycle"] == "community"
        finally:
            # Cleanup so other tests don't see the plugin
            from vllm._genesis.compat.plugins import unregister_plugins
            unregister_plugins()
            assert "PLUGIN_TEST_A" not in dispatcher.PATCH_REGISTRY
            # Side check: we didn't drop core entries
            assert pre_keys.issubset(set(dispatcher.PATCH_REGISTRY.keys()))


# ─── CLI ─────────────────────────────────────────────────────────────────


class TestCLI:
    def test_cli_list_no_plugins(self, plugins_disabled, fake_entry_points, capsys):
        from vllm._genesis.compat.plugins import main
        rc = main(["list"])
        captured = capsys.readouterr()
        assert rc == 0
        joined = captured.out + captured.err
        # When OFF, message about opt-in
        assert "GENESIS_ALLOW_PLUGINS" in joined or "0 plugin" in joined.lower() \
            or "no plugin" in joined.lower()

    def test_cli_list_with_plugins(
        self, plugins_enabled, fake_entry_points, capsys,
    ):
        fake_entry_points.append(FakeEntryPoint("p", lambda: _VALID_PLUGIN_DICT))
        from vllm._genesis.compat.plugins import main
        rc = main(["list"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "PLUGIN_TEST_A" in captured.out

    def test_cli_show_unknown_returns_nonzero(
        self, plugins_enabled, fake_entry_points, capsys,
    ):
        from vllm._genesis.compat.plugins import main
        rc = main(["show", "NOT_A_PLUGIN"])
        assert rc != 0

    def test_cli_show_known(
        self, plugins_enabled, fake_entry_points, capsys,
    ):
        fake_entry_points.append(FakeEntryPoint("p", lambda: _VALID_PLUGIN_DICT))
        from vllm._genesis.compat.plugins import main
        rc = main(["show", "PLUGIN_TEST_A"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "PLUGIN_TEST_A" in captured.out
        assert "community" in captured.out.lower()

    def test_cli_validate_clean(
        self, plugins_enabled, fake_entry_points, capsys,
    ):
        fake_entry_points.append(FakeEntryPoint("p", lambda: _VALID_PLUGIN_DICT))
        from vllm._genesis.compat.plugins import main
        rc = main(["validate"])
        assert rc == 0

    def test_cli_validate_dirty(
        self, plugins_enabled, fake_entry_points, capsys,
    ):
        bad = {"title": "incomplete"}
        fake_entry_points.append(FakeEntryPoint("bad", lambda: bad))
        from vllm._genesis.compat.plugins import main
        rc = main(["validate"])
        # Exit 1 when at least one plugin fails
        assert rc != 0


# ─── Schema validator integration ───────────────────────────────────────


class TestSchemaValidatorIntegration:
    def test_community_lifecycle_passes_schema(self, plugins_enabled, fake_entry_points):
        """Schema validator already accepts 'community' lifecycle as valid."""
        from vllm._genesis.compat.schema_validator import validate_entry
        plugin_meta = {**_VALID_PLUGIN_DICT, "lifecycle": "community"}
        # community lifecycle requires community_credit
        issues = validate_entry("PLUGIN_TEST_A", plugin_meta)
        # Either 0 issues (clean) or just warnings — no errors
        errors = [i for i in issues if i.severity == "ERROR"]
        assert errors == [], f"unexpected errors: {errors}"
