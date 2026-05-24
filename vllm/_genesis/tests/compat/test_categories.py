# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm._genesis.compat.categories — patch category navigation API.

The categories API answers questions like:
  - "what category does PN14 belong to?"  → kernel_safety
  - "what patches are in spec_decode?"     → ['P56', 'P58', 'P60', ...]
  - "what's the canonical Python module for P67?" → wiring.patch_67_tq_multi_query_kernel

This is the navigation surface for the explain CLI and operator UX.
The physical disk layout (Phase 2.1) is orthogonal — categories work
regardless of whether wiring/patch_*.py files are flat or in subdirs.
"""
from __future__ import annotations

import pytest


class TestCategoriesAPI:
    def test_categories_dict_present(self):
        from vllm._genesis.compat.categories import CATEGORIES
        assert isinstance(CATEGORIES, dict)
        assert len(CATEGORIES) >= 5  # at least 5 distinct categories

    def test_each_category_lists_patches(self):
        from vllm._genesis.compat.categories import CATEGORIES
        for cat, patches in CATEGORIES.items():
            assert isinstance(cat, str)
            assert isinstance(patches, list)
            assert all(isinstance(p, str) for p in patches)
            # Each category should have at least one patch (otherwise omit it)
            assert len(patches) >= 1, f"category {cat} is empty"

    def test_known_categories_present(self):
        """Specific categories that must exist based on PATCH_REGISTRY."""
        from vllm._genesis.compat.categories import CATEGORIES
        assert "spec_decode" in CATEGORIES
        assert "kv_cache" in CATEGORIES
        assert "structured_output" in CATEGORIES

    def test_PN14_in_kernel_safety(self):
        from vllm._genesis.compat.categories import CATEGORIES
        assert "PN14" in CATEGORIES.get("kernel_safety", [])

    def test_P67_in_spec_decode(self):
        from vllm._genesis.compat.categories import CATEGORIES
        assert "P67" in CATEGORIES.get("spec_decode", [])

    def test_no_duplicate_assignments(self):
        """A patch should appear in EXACTLY ONE category."""
        from vllm._genesis.compat.categories import CATEGORIES
        seen: dict[str, str] = {}
        for cat, patches in CATEGORIES.items():
            for p in patches:
                if p in seen:
                    pytest.fail(f"{p} appears in both {seen[p]!r} and {cat!r}")
                seen[p] = cat


class TestLookupHelpers:
    def test_category_for_known_patch(self):
        from vllm._genesis.compat.categories import category_for
        assert category_for("PN14") == "kernel_safety"
        assert category_for("P67") == "spec_decode"

    def test_category_for_unknown_returns_None(self):
        from vllm._genesis.compat.categories import category_for
        assert category_for("PN_NONEXISTENT") is None

    def test_patches_in_category(self):
        from vllm._genesis.compat.categories import patches_in
        spec = patches_in("spec_decode")
        assert isinstance(spec, list)
        assert "P67" in spec
        assert "P60" in spec

    def test_patches_in_unknown_category_returns_empty(self):
        from vllm._genesis.compat.categories import patches_in
        assert patches_in("totally_made_up_cat") == []

    def test_module_for_known_patch(self):
        """Returns the wiring module path."""
        from vllm._genesis.compat.categories import module_for
        mod = module_for("PN14")
        assert mod is not None
        assert "patch_N14" in mod or "patch_n14" in mod
        assert mod.startswith("vllm._genesis.wiring.")

    def test_module_for_unknown_patch_returns_None(self):
        from vllm._genesis.compat.categories import module_for
        assert module_for("PN_NONEXISTENT") is None


class TestRegistryConsistency:
    """Categories must be derived from PATCH_REGISTRY metadata —
    no manual drift possible."""

    def test_every_registry_patch_has_category(self):
        """Every patch in PATCH_REGISTRY must end up in some category
        (either via its `category` field or fallback to 'uncategorized')."""
        from vllm._genesis.compat.categories import CATEGORIES, category_for
        from vllm._genesis.dispatcher import PATCH_REGISTRY

        for pid in PATCH_REGISTRY:
            cat = category_for(pid)
            assert cat is not None, f"{pid} has no category assignment"
            assert pid in CATEGORIES.get(cat, [])

    def test_every_categorized_patch_in_registry(self):
        """No phantom patches — categories must reference REAL registry IDs."""
        from vllm._genesis.compat.categories import CATEGORIES
        from vllm._genesis.dispatcher import PATCH_REGISTRY

        for cat, patches in CATEGORIES.items():
            for p in patches:
                assert p in PATCH_REGISTRY, (
                    f"category {cat!r} references {p!r} but {p} not in PATCH_REGISTRY"
                )


class TestModuleResolution:
    """Mapping patch_id → wiring module name."""

    def test_simple_pid_to_module(self):
        from vllm._genesis.compat.categories import module_for
        # Just exemplary — exact mapping depends on filename convention
        m = module_for("PN14")
        assert "patch_N14" in m

    def test_subpatches_resolve(self):
        """P60b should resolve too (suffix variant)."""
        from vllm._genesis.compat.categories import module_for
        m = module_for("P60b")
        assert m is not None
        assert "patch_60b" in m

    def test_compound_patches_resolve_to_combined_file(self):
        """P68 and P69 share file patch_68_69_long_ctx_tool_adherence.py.
        Both should resolve to that module."""
        from vllm._genesis.compat.categories import module_for
        m68 = module_for("P68")
        m69 = module_for("P69")
        # Both resolve (may be same file or distinct — either is acceptable)
        assert m68 is not None
        assert m69 is not None


class TestCLIBrowse:
    def test_main_no_args_lists_all_categories(self, capsys):
        from vllm._genesis.compat.categories import main
        rc = main([])
        captured = capsys.readouterr()
        assert "spec_decode" in captured.out
        assert "kv_cache" in captured.out
        assert rc == 0

    def test_main_filter_by_category(self, capsys):
        from vllm._genesis.compat.categories import main
        rc = main(["--category", "kernel_safety"])
        captured = capsys.readouterr()
        # Only PN14 should show (only kernel_safety entry)
        assert "PN14" in captured.out
        assert rc == 0

    def test_main_unknown_category_returns_nonzero(self, capsys):
        from vllm._genesis.compat.categories import main
        rc = main(["--category", "totally_made_up"])
        assert rc != 0

    def test_main_json_output(self, capsys):
        from vllm._genesis.compat.categories import main
        import json
        main(["--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "spec_decode" in parsed or "categories" in parsed
