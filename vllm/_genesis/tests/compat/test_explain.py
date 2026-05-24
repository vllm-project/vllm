# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm._genesis.compat.explain — `genesis explain <patch_id>` CLI.

The `explain` tool produces a structured per-patch report (dict for
machine consumers, formatted text for humans) covering:
  - Identity        : title, lifecycle, category, env_flag, default_on
  - Behavior        : credit (what + why), upstream_pr, conflicts/requires
  - Compatibility   : applies_to predicate tree + per-rule eval result
  - Upstream tracking: marker file + symbol from upstream_compat
  - Decision today  : should_apply() output for this process
  - Tests           : count of unit tests covering this patch
"""
from __future__ import annotations

import json

import pytest


# ─── Synthetic registry for hermetic testing ────────────────────────────


_FAKE_REGISTRY = {
    "PN_TEST_STABLE": {
        "title": "Test patch — stable lifecycle",
        "env_flag": "GENESIS_ENABLE_PN_TEST_STABLE",
        "default_on": False,
        "lifecycle": "stable",
        "stable_since": "v7.62.18",
        "category": "kernel_safety",
        "credit": "Genesis-original 2026-04-30 — defensive guard.",
        "upstream_pr": None,
        "applies_to": {"is_turboquant": [True]},
    },
    "PN_TEST_DEPRECATED": {
        "title": "Test patch — deprecated, superseded by something",
        "env_flag": "GENESIS_ENABLE_PN_TEST_DEPRECATED",
        "default_on": False,
        "lifecycle": "deprecated",
        "superseded_by": ["PN_TEST_STABLE"],
        "removal_planned": "v8.0",
        "category": "spec_decode",
        "credit": "Old workaround replaced by PN_TEST_STABLE.",
        "upstream_pr": None,
    },
    "PN_TEST_COMPOUND": {
        "title": "Test patch — uses richer applies_to DSL",
        "env_flag": "GENESIS_ENABLE_PN_TEST_COMPOUND",
        "default_on": False,
        "lifecycle": "stable",
        "category": "kernel_perf",
        "credit": "Compound rule example.",
        "upstream_pr": 99999,
        "applies_to": {
            "all_of": [
                {"is_turboquant": True},
                {"any_of": [
                    {"quant_format": "fp8"},
                    {"quant_format": "autoround_int4"},
                ]},
            ],
        },
        "requires_patches": ["PN_TEST_STABLE"],
        "conflicts_with": ["PN_TEST_DEPRECATED"],
    },
    "PN_TEST_RESEARCH": {
        "title": "Test patch — research lifecycle",
        "env_flag": "GENESIS_ENABLE_PN_TEST_RESEARCH",
        "default_on": False,
        "lifecycle": "research",
        "research_note": "kept as reference for future hardware",
        "category": "memory_pool",
        "credit": "Research artifact.",
        "upstream_pr": None,
    },
}


@pytest.fixture
def fake_registry(monkeypatch):
    """Inject a controlled registry for deterministic tests."""
    from vllm._genesis import dispatcher
    monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", _FAKE_REGISTRY)
    yield _FAKE_REGISTRY


# ─── explain_patch() — dict shape ───────────────────────────────────────


class TestExplainShape:
    def test_returns_dict(self, fake_registry):
        from vllm._genesis.compat.explain import explain_patch
        result = explain_patch("PN_TEST_STABLE")
        assert isinstance(result, dict)
        assert result["patch_id"] == "PN_TEST_STABLE"

    def test_includes_identity_fields(self, fake_registry):
        from vllm._genesis.compat.explain import explain_patch
        r = explain_patch("PN_TEST_STABLE")
        assert r["title"] == "Test patch — stable lifecycle"
        assert r["env_flag"] == "GENESIS_ENABLE_PN_TEST_STABLE"
        assert r["default_on"] is False
        assert r["category"] == "kernel_safety"

    def test_includes_lifecycle_section(self, fake_registry):
        from vllm._genesis.compat.explain import explain_patch
        r = explain_patch("PN_TEST_STABLE")
        assert "lifecycle" in r
        assert r["lifecycle"]["state"] == "stable"

    def test_lifecycle_includes_superseded_by_for_deprecated(self, fake_registry):
        from vllm._genesis.compat.explain import explain_patch
        r = explain_patch("PN_TEST_DEPRECATED")
        assert r["lifecycle"]["state"] == "deprecated"
        assert "PN_TEST_STABLE" in r["lifecycle"]["superseded_by"]
        assert r["lifecycle"]["removal_planned"] == "v8.0"

    def test_includes_dependencies_section(self, fake_registry):
        from vllm._genesis.compat.explain import explain_patch
        r = explain_patch("PN_TEST_COMPOUND")
        assert r["dependencies"]["requires"] == ["PN_TEST_STABLE"]
        assert r["dependencies"]["conflicts_with"] == ["PN_TEST_DEPRECATED"]

    def test_no_dependencies_returns_empty_lists(self, fake_registry):
        from vllm._genesis.compat.explain import explain_patch
        r = explain_patch("PN_TEST_STABLE")
        assert r["dependencies"]["requires"] == []
        assert r["dependencies"]["conflicts_with"] == []

    def test_includes_applies_to_section(self, fake_registry):
        from vllm._genesis.compat.explain import explain_patch
        r = explain_patch("PN_TEST_STABLE")
        assert "applies_to" in r
        # Either declares a rule, or notes none
        assert "rule" in r["applies_to"]

    def test_compound_applies_to_explained_via_predicates(self, fake_registry):
        from vllm._genesis.compat.explain import explain_patch
        r = explain_patch("PN_TEST_COMPOUND")
        # Should call predicates.explain → returns list of indented lines
        lines = r["applies_to"]["explanation"]
        assert isinstance(lines, list)
        joined = "\n".join(lines)
        assert "all_of" in joined
        assert "any_of" in joined

    def test_includes_decision_section(self, fake_registry):
        from vllm._genesis.compat.explain import explain_patch
        r = explain_patch("PN_TEST_STABLE")
        assert "decision" in r
        assert "applied" in r["decision"]
        assert "reason" in r["decision"]
        assert isinstance(r["decision"]["applied"], bool)


class TestExplainErrorPaths:
    def test_unknown_patch_id_raises_or_returns_error(self, fake_registry):
        from vllm._genesis.compat.explain import explain_patch
        # Either raise KeyError or return {"error": ...} — pick one
        try:
            r = explain_patch("PN_NONEXISTENT")
            assert "error" in r
            assert "PN_NONEXISTENT" in r["error"]
        except KeyError as e:
            assert "PN_NONEXISTENT" in str(e)

    def test_empty_patch_id_handled(self, fake_registry):
        from vllm._genesis.compat.explain import explain_patch
        try:
            r = explain_patch("")
            assert "error" in r
        except (KeyError, ValueError):
            pass  # acceptable


# ─── Upstream tracking integration ──────────────────────────────────────


class TestUpstreamTracking:
    def test_includes_upstream_pr_field(self, fake_registry):
        from vllm._genesis.compat.explain import explain_patch
        r = explain_patch("PN_TEST_COMPOUND")
        assert r["upstream"]["pr_number"] == 99999

    def test_no_upstream_pr_returns_None(self, fake_registry):
        from vllm._genesis.compat.explain import explain_patch
        r = explain_patch("PN_TEST_STABLE")
        assert r["upstream"]["pr_number"] is None

    def test_PN14_prefers_PR_40074_over_cross_reference(self):
        """Real-registry regression: PN14 declares upstream_pr=40074. The
        upstream_compat module also has PR_39939_jartx_per_token_head_refactor
        whose `affects_patch` field mentions 'PN14' as a cross-reference.
        explain_patch must prefer the EXACT PR-number match (40074) over
        the substring-match in affects_patch."""
        from vllm._genesis.compat.explain import explain_patch
        r = explain_patch("PN14")
        # Marker must be PN14's actual symbol, not the JartX cross-ref
        assert r["upstream"]["marker"] == "safe_page_idx", (
            f"expected PN14's safe_page_idx marker, got {r['upstream']['marker']!r} "
            f"(probably hit the JartX #39939 cross-reference)"
        )
        assert "40074" in (r["upstream"].get("compat_key") or "")


# ─── Real-registry sanity (no mock) ─────────────────────────────────────


class TestRealRegistry:
    def test_PN14_explain_produces_valid_output(self):
        """Live test against the actual PATCH_REGISTRY — PN14 should
        produce a coherent explain report including the upstream PR."""
        from vllm._genesis.compat.explain import explain_patch
        r = explain_patch("PN14")
        assert r["patch_id"] == "PN14"
        assert "PN14" in r["title"] or "TQ decode" in r["title"]
        assert r["env_flag"] == "GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP"
        assert r["upstream"]["pr_number"] == 40074

    def test_P67_explain_includes_conflicts(self):
        """P67 declares conflicts_with: ['P65'] in real registry."""
        from vllm._genesis.compat.explain import explain_patch
        r = explain_patch("P67")
        assert "P65" in r["dependencies"]["conflicts_with"]

    def test_P85_explain_includes_requires(self):
        """P85 declares requires_patches: ['P84']."""
        from vllm._genesis.compat.explain import explain_patch
        r = explain_patch("P85")
        assert "P84" in r["dependencies"]["requires"]


# ─── Text formatter ─────────────────────────────────────────────────────


class TestTextFormatter:
    def test_format_produces_lines(self, fake_registry):
        from vllm._genesis.compat.explain import explain_patch, format_explain_text
        r = explain_patch("PN_TEST_STABLE")
        lines = format_explain_text(r)
        assert isinstance(lines, list)
        assert len(lines) > 5

    def test_format_contains_section_headers(self, fake_registry):
        from vllm._genesis.compat.explain import explain_patch, format_explain_text
        r = explain_patch("PN_TEST_STABLE")
        joined = "\n".join(format_explain_text(r))
        # Section headers
        assert "PN_TEST_STABLE" in joined
        assert "Lifecycle" in joined or "lifecycle" in joined
        assert "Decision" in joined or "decision" in joined

    def test_format_handles_error_dict(self, fake_registry):
        from vllm._genesis.compat.explain import format_explain_text
        lines = format_explain_text({"error": "unknown patch_id 'X'"})
        joined = "\n".join(lines)
        assert "X" in joined
        assert "error" in joined.lower() or "unknown" in joined.lower()


# ─── CLI smoke ──────────────────────────────────────────────────────────


class TestCLISmoke:
    def test_main_returns_int(self):
        from vllm._genesis.compat.explain import main
        # Real registry entry
        rc = main(["PN14"])
        assert isinstance(rc, int)
        assert rc == 0

    def test_main_unknown_patch_returns_nonzero(self):
        from vllm._genesis.compat.explain import main
        rc = main(["PN_NONEXISTENT_XYZ"])
        assert rc != 0

    def test_main_json_mode(self, capsys):
        from vllm._genesis.compat.explain import main
        main(["PN14", "--json"])
        captured = capsys.readouterr()
        # Verify output is valid JSON
        parsed = json.loads(captured.out)
        assert parsed["patch_id"] == "PN14"

    def test_main_no_args_prints_usage(self, capsys):
        from vllm._genesis.compat.explain import main
        with pytest.raises(SystemExit):
            main([])
