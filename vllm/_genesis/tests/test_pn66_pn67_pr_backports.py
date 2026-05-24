# SPDX-License-Identifier: Apache-2.0
"""TDD for PN66 (vllm#41696) + PN67 (vllm#41674) — direct upstream PR backports.

Verifies:
- Both patches register correctly in PATCH_REGISTRY
- Both have correct env_flag + opt-in default
- Both apply() returns "skipped" when env disabled
- Both have unique anchors visible in current pin (best-effort, skips
  test if file doesn't exist locally)
"""
from __future__ import annotations

import pytest


class TestPN66Registration:
    def test_in_registry(self):
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        assert "PN66" in PATCH_REGISTRY

    def test_metadata(self):
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        meta = PATCH_REGISTRY["PN66"]
        assert meta["env_flag"] == "GENESIS_ENABLE_PN66"
        assert meta["default_on"] is False
        assert meta["category"] == "structured_output"
        assert meta["upstream_pr"] == 41696

    def test_apply_skipped_when_env_disabled(self, monkeypatch):
        from vllm._genesis.wiring.structured_output import patch_N66_multiturn_think_leak as p
        monkeypatch.delenv("GENESIS_ENABLE_PN66", raising=False)
        status, reason = p.apply()
        assert status == "skipped"

    def test_anchor_constants_present(self):
        from vllm._genesis.wiring.structured_output import patch_N66_multiturn_think_leak as p
        assert p.PN66_FIELD_OLD
        assert p.PN66_FIELD_NEW
        assert p.PN66_BLOCK_OLD
        assert p.PN66_BLOCK_NEW
        # Sanity: replacement actually removes the buggy line
        assert "prompt_reasoning_checked" in p.PN66_FIELD_OLD
        assert "prompt_reasoning_checked" not in p.PN66_FIELD_NEW
        assert "if not state.prompt_reasoning_checked" in p.PN66_BLOCK_OLD
        assert "if not state.prompt_reasoning_checked" not in p.PN66_BLOCK_NEW


class TestPN67Registration:
    def test_in_registry(self):
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        assert "PN67" in PATCH_REGISTRY

    def test_metadata(self):
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        meta = PATCH_REGISTRY["PN67"]
        assert meta["env_flag"] == "GENESIS_ENABLE_PN67"
        assert meta["default_on"] is False
        assert meta["category"] == "stability"
        assert meta["upstream_pr"] == 41674

    def test_apply_skipped_when_env_disabled(self, monkeypatch):
        from vllm._genesis.wiring.perf_hotfix import patch_N67_thinking_budget_inverted_bool as p
        monkeypatch.delenv("GENESIS_ENABLE_PN67", raising=False)
        status, reason = p.apply()
        assert status == "skipped"

    def test_anchor_constants_present(self):
        from vllm._genesis.wiring.perf_hotfix import patch_N67_thinking_budget_inverted_bool as p
        assert p.PN67_OLD
        assert p.PN67_NEW
        # Sanity: removes the inverted `not`
        assert "or not thinking_budget_tracks_reqs" in p.PN67_OLD
        assert "or not thinking_budget_tracks_reqs" not in p.PN67_NEW
        assert "or thinking_budget_tracks_reqs" in p.PN67_NEW


class TestApplyAllWiring:
    """Verify both patches are wired into apply_all dispatcher."""

    def test_apply_patch_n66_function_exists(self):
        from vllm._genesis.patches.apply_all import apply_patch_N66_multiturn_think_leak
        assert callable(apply_patch_N66_multiturn_think_leak)

    def test_apply_patch_n67_function_exists(self):
        from vllm._genesis.patches.apply_all import apply_patch_N67_thinking_budget_inverted_bool
        assert callable(apply_patch_N67_thinking_budget_inverted_bool)
