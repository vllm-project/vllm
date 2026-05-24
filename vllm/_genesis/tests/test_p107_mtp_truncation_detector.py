# SPDX-License-Identifier: Apache-2.0
"""TDD for P107 — MTP truncation detector (vllm#41467)."""
from __future__ import annotations

import pytest


def _wiring():
    from vllm._genesis.wiring.structured_output import (
        patch_107_mtp_truncation_detector as M,
    )
    return M


def test_anchor_targets_finish_reason_block():
    M = _wiring()
    assert "auto_tools_called" in M.ANCHOR_OLD
    assert "tool_choice_function_name" in M.ANCHOR_OLD
    assert "finish_reason_ = \"tool_calls\"" in M.ANCHOR_OLD
    assert "ChatCompletionResponseStreamChoice" in M.ANCHOR_OLD


def test_replacement_adds_p107_guard():
    M = _wiring()
    assert "P107" in M.ANCHOR_NEW
    assert "vllm#41467" in M.ANCHOR_NEW
    assert "MTP truncation detected" in M.ANCHOR_NEW
    assert "MTP speculative decoding truncated" in M.ANCHOR_NEW
    # All 6 AND conditions must be present
    assert "finish_reason_ == \"stop\"" in M.ANCHOR_NEW
    assert "and request.tools" in M.ANCHOR_NEW
    assert "and not tools_streamed[i]" in M.ANCHOR_NEW
    assert "and not auto_tools_called" in M.ANCHOR_NEW
    assert "and reasoning_parser is not None" in M.ANCHOR_NEW
    assert "and not delta_message.content" in M.ANCHOR_NEW
    assert "and not delta_message.tool_calls" in M.ANCHOR_NEW


def test_idempotent_on_synthetic(tmp_path):
    from vllm._genesis.wiring.text_patch import (
        TextPatch, TextPatcher, TextPatchResult,
    )
    M = _wiring()
    target = tmp_path / "serving.py"
    target.write_text("# header\n" + M.ANCHOR_OLD + "\n# tail\n")
    patcher = TextPatcher(
        patch_name="P107 test",
        target_file=str(target),
        marker=M.GENESIS_P107_MARKER,
        sub_patches=[TextPatch(name="p107", anchor=M.ANCHOR_OLD,
                                replacement=M.ANCHOR_NEW, required=True)],
    )
    r1, _ = patcher.apply()
    assert r1 == TextPatchResult.APPLIED
    body1 = target.read_text()
    assert "P107" in body1
    r2, _ = patcher.apply()
    assert r2 == TextPatchResult.IDEMPOTENT


def test_env_flag_default_off(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.delenv("GENESIS_ENABLE_P107_MTP_TRUNCATION_DETECTOR", raising=False)
    decision, _ = should_apply("P107")
    assert decision is False


def test_env_flag_engages(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.setenv("GENESIS_ENABLE_P107_MTP_TRUNCATION_DETECTOR", "1")
    decision, _ = should_apply("P107")
    assert decision is True


def test_registry_entry_complete():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "P107" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["P107"]
    assert meta["upstream_pr"] == 41467


def test_apply_all_registers_p107():
    from vllm._genesis.patches import apply_all
    assert hasattr(apply_all, "apply_patch_107_mtp_truncation_detector")
