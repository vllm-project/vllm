# SPDX-License-Identifier: Apache-2.0
"""End-to-end test of the tools/examples/genesis-plugin-hello-world reference.

This test:
  1. Adds the example package's source to sys.path
  2. Imports `get_patch_metadata` directly (mimicking what
     setuptools-entry-point loading would do at runtime)
  3. Verifies the plugin's metadata validates against the Genesis
     schema
  4. Verifies apply() returns the expected (status, reason) tuple

Why this matters: the docs/PLUGINS.md guide tells third-party authors
the exact shape Genesis expects. If our reference example doesn't pass
its own validator, the docs are lying. This test pins that contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


# ─── Locate the example package on disk ─────────────────────────────────


# test_plugin_example.py is at <repo>/vllm/_genesis/tests/compat/test_*.py
# parents: [0]=compat [1]=tests [2]=_genesis [3]=vllm [4]=<repo>
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLE_PKG = (
    _REPO_ROOT / "tools" / "examples" / "genesis-plugin-hello-world"
)


@pytest.fixture
def example_on_path():
    """Add the example package source to sys.path so it can be
    imported in-tree without `pip install`."""
    if not _EXAMPLE_PKG.is_dir():
        pytest.skip(f"example package missing at {_EXAMPLE_PKG}")
    src = str(_EXAMPLE_PKG)
    added = src not in sys.path
    if added:
        sys.path.insert(0, src)
    try:
        # Clear any stale import
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("genesis_plugin_hello_world"):
                del sys.modules[mod_name]
        yield
    finally:
        if added:
            sys.path.remove(src)


# ─── Tests ───────────────────────────────────────────────────────────────


class TestExampleStructure:
    def test_pyproject_exists(self):
        assert (_EXAMPLE_PKG / "pyproject.toml").is_file(), \
            "example pyproject.toml missing"

    def test_module_exists(self):
        assert (_EXAMPLE_PKG / "genesis_plugin_hello_world" / "plugin.py").is_file()

    def test_readme_exists(self):
        assert (_EXAMPLE_PKG / "README.md").is_file()

    def test_pyproject_declares_entry_point(self):
        body = (_EXAMPLE_PKG / "pyproject.toml").read_text()
        # The entry-point must be in the standard vllm_genesis_patches group
        assert "vllm_genesis_patches" in body
        assert "get_patch_metadata" in body


class TestExampleMetadata:
    def test_get_patch_metadata_importable(self, example_on_path):
        from genesis_plugin_hello_world.plugin import get_patch_metadata
        meta = get_patch_metadata()
        assert isinstance(meta, dict)

    def test_returns_required_fields(self, example_on_path):
        from genesis_plugin_hello_world.plugin import get_patch_metadata
        meta = get_patch_metadata()
        for key in ("patch_id", "title", "env_flag", "default_on",
                     "community_credit"):
            assert key in meta, f"example metadata missing {key!r}"

    def test_validates_against_genesis_schema(self, example_on_path):
        """The reference must pass the same schema validator core
        patches go through. Otherwise the plugin docs lie."""
        from genesis_plugin_hello_world.plugin import get_patch_metadata
        meta = get_patch_metadata()

        from vllm._genesis.compat.schema_validator import validate_entry
        # Force lifecycle=community since plugin.py declares it
        # (or relies on Genesis to auto-tag); validator should accept
        meta_to_check = dict(meta)
        meta_to_check.setdefault("lifecycle", "community")

        issues = validate_entry(meta["patch_id"], meta_to_check)
        errors = [i for i in issues if i.severity == "ERROR"]
        assert errors == [], (
            f"reference plugin fails schema validation: "
            f"{[i.message for i in errors]}"
        )

    def test_env_flag_is_genesis_prefixed(self, example_on_path):
        from genesis_plugin_hello_world.plugin import get_patch_metadata
        meta = get_patch_metadata()
        assert meta["env_flag"].startswith("GENESIS_")

    def test_default_on_is_false(self, example_on_path):
        """A reference plugin must default OFF (good citizenship —
        operator must explicitly engage)."""
        from genesis_plugin_hello_world.plugin import get_patch_metadata
        meta = get_patch_metadata()
        assert meta["default_on"] is False

    def test_apply_callable_resolves(self, example_on_path):
        from genesis_plugin_hello_world.plugin import get_patch_metadata
        meta = get_patch_metadata()
        spec = meta.get("apply_callable")
        if spec is None:
            pytest.skip("metadata-only plugin (no apply_callable)")
        from vllm._genesis.compat.plugins import _resolve_apply_callable
        fn = _resolve_apply_callable(spec)
        assert callable(fn)


class TestExampleApply:
    def test_apply_returns_tuple(self, example_on_path):
        from genesis_plugin_hello_world.plugin import apply
        result = apply()
        assert isinstance(result, tuple)
        assert len(result) == 2
        status, reason = result
        assert status in ("applied", "skipped", "failed")
        assert isinstance(reason, str)

    def test_apply_returns_applied_status(self, example_on_path):
        from genesis_plugin_hello_world.plugin import apply
        status, _ = apply()
        # Reference plugin is a no-op success path
        assert status == "applied"


class TestExampleGoesThroughDiscoveryPipeline:
    def test_metadata_passes_full_plugin_validation(
        self, example_on_path, monkeypatch,
    ):
        """End-to-end: simulate the entry-point discovery + Genesis
        validation pipeline against the example. If this passes, the
        example is genuinely usable as a real plugin."""
        from genesis_plugin_hello_world.plugin import get_patch_metadata

        class FakeEP:
            name = "hello_world"
            value = "genesis_plugin_hello_world.plugin:get_patch_metadata"
            def load(self):
                return get_patch_metadata

        monkeypatch.setenv("GENESIS_ALLOW_PLUGINS", "1")
        monkeypatch.setattr(
            "vllm._genesis.compat.plugins._discover_entry_points",
            lambda: [FakeEP()],
        )
        from vllm._genesis.compat.plugins import discover_plugins
        plugins = discover_plugins()
        assert len(plugins) == 1
        p = plugins[0]
        assert p["patch_id"] == "HELLO_WORLD"
        assert p["lifecycle"] == "community"
        assert "_plugin_origin" in p
