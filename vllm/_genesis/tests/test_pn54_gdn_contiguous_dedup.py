# SPDX-License-Identifier: Apache-2.0
"""TDD for PN54 (plan v3 P0.7) — GDN .contiguous() deduplication."""
from __future__ import annotations

import pytest


def _wiring():
    from vllm._genesis.wiring.hybrid import (
        patch_N54_gdn_contiguous_dedup as M,
    )
    return M


def test_anchor_a_targets_ssm_state_call():
    M = _wiring()
    assert "ssm_state[non_spec_state_indices_tensor].contiguous()" in M.SSM_STATE_OLD
    # Must KEEP the index but DROP .contiguous()
    assert "ssm_state[non_spec_state_indices_tensor]  # type: ignore[index]" in M.SSM_STATE_NEW
    # Must NOT have .contiguous() on the same line
    new_lines = [ln for ln in M.SSM_STATE_NEW.splitlines()
                 if "initial_state =" in ln and not ln.lstrip().startswith("#")]
    assert len(new_lines) == 1
    assert ".contiguous()" not in new_lines[0]


def test_anchor_b_targets_lora_chunk_halves():
    M = _wiring()
    assert "b, a = ba.chunk(2, dim=-1)" in M.LORA_BA_OLD
    assert "b = b.contiguous()" in M.LORA_BA_OLD
    assert "a = a.contiguous()" in M.LORA_BA_OLD
    # Replacement keeps the chunk call, drops the explicit .contiguous() pair
    assert "b, a = ba.chunk(2, dim=-1)" in M.LORA_BA_NEW
    code_lines_new = [ln for ln in M.LORA_BA_NEW.splitlines()
                      if not ln.lstrip().startswith("#")]
    joined = "\n".join(code_lines_new)
    assert "b = b.contiguous()" not in joined
    assert "a = a.contiguous()" not in joined


def test_replacements_carry_pn54_marker():
    M = _wiring()
    for n, new in [("SSM_STATE_NEW", M.SSM_STATE_NEW),
                   ("LORA_BA_NEW", M.LORA_BA_NEW)]:
        assert "PN54" in new, f"{n} missing PN54 marker"
        assert "P0.7" in new, f"{n} missing P0.7 reference"


def test_idempotent_apply(tmp_path):
    from vllm._genesis.wiring.text_patch import (
        TextPatch, TextPatcher, TextPatchResult,
    )
    M = _wiring()

    for fname, old, new, sub_name in [
        ("ssm_state.py", M.SSM_STATE_OLD, M.SSM_STATE_NEW, "pn54_ssm_state"),
        ("lora_ba.py", M.LORA_BA_OLD, M.LORA_BA_NEW, "pn54_lora_ba"),
    ]:
        target = tmp_path / fname
        target.write_text("# header\n" + old + "\n# tail\n")
        patcher = TextPatcher(
            patch_name=fname,
            target_file=str(target),
            marker=M.GENESIS_PN54_MARKER,
            sub_patches=[TextPatch(name=sub_name, anchor=old, replacement=new, required=True)],
        )
        r1, _ = patcher.apply()
        assert r1 == TextPatchResult.APPLIED, f"{fname} 1st apply"
        body1 = target.read_text()
        assert "PN54" in body1
        r2, _ = patcher.apply()
        assert r2 == TextPatchResult.IDEMPOTENT, f"{fname} 2nd apply must be idempotent"
        assert target.read_text() == body1


def test_env_flag_default_off(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.delenv("GENESIS_ENABLE_PN54_GDN_CONTIGUOUS_DEDUP", raising=False)
    decision, reason = should_apply("PN54")
    assert decision is False
    assert "opt-in" in reason.lower() or "off" in reason.lower()


def test_env_flag_engages(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.setenv("GENESIS_ENABLE_PN54_GDN_CONTIGUOUS_DEDUP", "1")
    decision, _ = should_apply("PN54")
    assert decision is True


def test_registry_entry_complete():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PN54" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["PN54"]
    assert meta["env_flag"] == "GENESIS_ENABLE_PN54_GDN_CONTIGUOUS_DEDUP"
    assert meta["default_on"] is False
    assert "GDN" in meta["title"] or "gdn" in meta["title"].lower()


def test_apply_all_registers_pn54():
    from vllm._genesis.patches import apply_all
    assert hasattr(apply_all, "apply_patch_N54_gdn_contiguous_dedup")
