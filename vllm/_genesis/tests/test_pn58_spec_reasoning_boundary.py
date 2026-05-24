# SPDX-License-Identifier: Apache-2.0
"""TDD for PN58 — spec-decode reasoning boundary validation (vllm#40962)."""
from __future__ import annotations

import pytest


def _wiring():
    from vllm._genesis.wiring.structured_output import (
        patch_N58_spec_reasoning_boundary as M,
    )
    return M


def test_anchor_envs_targets_lora_dual_stream():
    M = _wiring()
    assert "VLLM_LORA_ENABLE_DUAL_STREAM" in M.ENVS_OLD
    assert "VLLM_SPEC_REASONING_BOUNDARY_VALIDATION" in M.ENVS_NEW


def test_anchor_abs_parser_targets_extract_content_ids():
    M = _wiring()
    assert "@abstractmethod" in M.ABS_PARSER_OLD
    assert "extract_content_ids" in M.ABS_PARSER_OLD
    assert "find_reasoning_end_index" in M.ABS_PARSER_NEW
    assert "may_have_reasoning_end_in_delta" in M.ABS_PARSER_NEW


def test_anchor_basic_parser_overrides_methods():
    M = _wiring()
    assert "is_reasoning_end_streaming" in M.BASIC_PARSER_OLD
    assert "delta_ids.index(end_token_id)" in M.BASIC_PARSER_NEW


def test_anchor_struct_out_inserts_helper():
    M = _wiring()
    assert "class StructuredOutputManager:" in M.STRUCT_OUT_OLD
    assert "validate_spec_tokens_with_reasoning_boundary" in M.STRUCT_OUT_NEW


def test_anchor_sched_replaces_should_advance_block():
    M = _wiring()
    assert "should_advance(request)" in M.SCHED_VALIDATE_OLD
    assert "should_advance" in M.SCHED_VALIDATE_NEW
    assert "_pn58_advanced_with_boundary" in M.SCHED_VALIDATE_NEW
    assert "validate_spec_tokens_with_reasoning_boundary" in M.SCHED_VALIDATE_NEW


def test_replacements_carry_pn58_marker():
    M = _wiring()
    for name, new in [
        ("ENVS_NEW", M.ENVS_NEW),
        ("ABS_PARSER_NEW", M.ABS_PARSER_NEW),
        ("BASIC_PARSER_NEW", M.BASIC_PARSER_NEW),
        ("STRUCT_OUT_NEW", M.STRUCT_OUT_NEW),
        ("SCHED_VALIDATE_NEW", M.SCHED_VALIDATE_NEW),
        ("SCHED_IMPORT_NEW", M.SCHED_IMPORT_NEW),
    ]:
        assert "PN58" in new, f"{name} missing PN58 marker"


def test_idempotent_envs(tmp_path):
    from vllm._genesis.wiring.text_patch import (
        TextPatch, TextPatcher, TextPatchResult,
    )
    M = _wiring()
    target = tmp_path / "envs.py"
    target.write_text("# header\n" + M.ENVS_OLD + "\n")
    patcher = TextPatcher(
        patch_name="PN58 envs test",
        target_file=str(target),
        marker=M.GENESIS_PN58_MARKER + " (envs)",
        sub_patches=[TextPatch(name="pn58_envs",
                                anchor=M.ENVS_OLD,
                                replacement=M.ENVS_NEW, required=True)],
    )
    r1, _ = patcher.apply()
    assert r1 == TextPatchResult.APPLIED
    body1 = target.read_text()
    assert "VLLM_SPEC_REASONING_BOUNDARY_VALIDATION" in body1
    r2, _ = patcher.apply()
    assert r2 == TextPatchResult.IDEMPOTENT


def test_idempotent_sched_validate_block(tmp_path):
    from vllm._genesis.wiring.text_patch import (
        TextPatch, TextPatcher, TextPatchResult,
    )
    M = _wiring()
    target = tmp_path / "scheduler.py"
    target.write_text("# header\n" + M.SCHED_VALIDATE_OLD + "\n# tail\n")
    patcher = TextPatcher(
        patch_name="PN58 sched test",
        target_file=str(target),
        marker=M.GENESIS_PN58_MARKER + " (sched)",
        sub_patches=[TextPatch(name="pn58_sched_validate",
                                anchor=M.SCHED_VALIDATE_OLD,
                                replacement=M.SCHED_VALIDATE_NEW, required=True)],
    )
    r1, _ = patcher.apply()
    assert r1 == TextPatchResult.APPLIED
    r2, _ = patcher.apply()
    assert r2 == TextPatchResult.IDEMPOTENT


def test_mutex_with_p62_skips_when_p62_active(monkeypatch):
    """Apply check must SKIP cleanly when P62 active."""
    monkeypatch.setenv("GENESIS_ENABLE_PN58_SPEC_REASONING_BOUNDARY", "1")
    monkeypatch.setenv("GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING", "1")
    from vllm._genesis.wiring.structured_output import (
        patch_N58_spec_reasoning_boundary as M,
    )
    status, reason = M.apply()
    assert status == "skipped"
    assert "P62" in reason
    assert "MUTUAL" in reason.upper() or "mutual" in reason.lower() or "exclusive" in reason.lower()


def test_env_flag_default_off(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.delenv("GENESIS_ENABLE_PN58_SPEC_REASONING_BOUNDARY", raising=False)
    decision, _ = should_apply("PN58")
    assert decision is False


def test_env_flag_engages(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.setenv("GENESIS_ENABLE_PN58_SPEC_REASONING_BOUNDARY", "1")
    decision, _ = should_apply("PN58")
    assert decision is True


def test_registry_entry_complete():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PN58" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["PN58"]
    assert meta["upstream_pr"] == 40962
    assert "P62" in meta.get("conflicts_with", [])


def test_apply_all_registers_pn58():
    from vllm._genesis.patches import apply_all
    assert hasattr(apply_all, "apply_patch_N58_spec_reasoning_boundary")
