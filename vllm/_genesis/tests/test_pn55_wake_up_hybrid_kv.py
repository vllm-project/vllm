# SPDX-License-Identifier: Apache-2.0
"""TDD for PN55 — vllm#41602 backport: wake_up hybrid KV crash fix."""
from __future__ import annotations

import pytest


def _wiring():
    from vllm._genesis.wiring.perf_hotfix import patch_N55_wake_up_hybrid_kv as M
    return M


def test_anchor_targets_buggy_loop():
    M = _wiring()
    assert "for cache_tensor in kv_caches:" in M.ANCHOR_OLD
    assert "cache_tensor.zero_()" in M.ANCHOR_OLD
    assert "isinstance(cache_entry, list)" in M.ANCHOR_NEW
    assert "for _pn55_t in cache_entry:" in M.ANCHOR_NEW


def test_replacement_carries_pn55_marker():
    M = _wiring()
    assert "PN55" in M.ANCHOR_NEW
    assert "vllm#41602" in M.ANCHOR_NEW


def test_idempotent_on_synthetic(tmp_path):
    from vllm._genesis.wiring.text_patch import (
        TextPatch, TextPatcher, TextPatchResult,
    )
    M = _wiring()
    target = tmp_path / "gpu_model_runner.py"
    target.write_text("# header\n" + M.ANCHOR_OLD + "\n# footer\n")
    patcher = TextPatcher(
        patch_name="PN55 test",
        target_file=str(target),
        marker=M.GENESIS_PN55_MARKER,
        sub_patches=[TextPatch(name="pn55", anchor=M.ANCHOR_OLD,
                                replacement=M.ANCHOR_NEW, required=True)],
    )
    r1, _ = patcher.apply()
    assert r1 == TextPatchResult.APPLIED
    body1 = target.read_text()
    assert "PN55" in body1
    r2, _ = patcher.apply()
    assert r2 == TextPatchResult.IDEMPOTENT
    assert target.read_text() == body1


def test_env_flag_default_off(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.delenv("GENESIS_ENABLE_PN55_WAKE_UP_HYBRID_KV", raising=False)
    decision, _ = should_apply("PN55")
    assert decision is False


def test_env_flag_engages(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.setenv("GENESIS_ENABLE_PN55_WAKE_UP_HYBRID_KV", "1")
    decision, _ = should_apply("PN55")
    assert decision is True


def test_registry_entry_complete():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PN55" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["PN55"]
    assert meta["upstream_pr"] == 41602
    assert "wake_up" in meta["title"].lower()


def test_apply_all_registers_pn55():
    from vllm._genesis.patches import apply_all
    assert hasattr(apply_all, "apply_patch_N55_wake_up_hybrid_kv")
