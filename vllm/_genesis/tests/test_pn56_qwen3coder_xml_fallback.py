# SPDX-License-Identifier: Apache-2.0
"""TDD for PN56 — qwen3coder XML parse fallback (vllm#41466)."""
from __future__ import annotations

import pytest


def _wiring():
    from vllm._genesis.wiring.structured_output import (
        patch_N56_qwen3coder_xml_fallback as M,
    )
    return M


def test_anchor_targets_try_block():
    M = _wiring()
    assert "_parse_xml_function_call" in M.ANCHOR_A_OLD
    assert "parsed_tool.function.arguments" in M.ANCHOR_A_OLD
    assert "except Exception:" in M.ANCHOR_A_OLD


def test_replacement_adds_pn56_logic():
    M = _wiring()
    assert "_pn56_parse_succeeded = False" in M.ANCHOR_A_NEW
    assert "_pn56_parse_succeeded = True" in M.ANCHOR_A_NEW
    assert "if (\n                        not _pn56_parse_succeeded" in M.ANCHOR_A_NEW
    # After audit A-14 fix: streamed_args + suffix (computed via rstrip check)
    assert "_pn56_streamed = self.streamed_args_for_tool[" in M.ANCHOR_A_NEW
    assert "_pn56_suffix = \"\" if _pn56_streamed.rstrip().endswith(\"}\") else \"}\"" in M.ANCHOR_A_NEW
    assert "_pn56_streamed + _pn56_suffix" in M.ANCHOR_A_NEW


def test_a14_no_double_close_brace():
    """Audit A-14 invariant: replacement must conditionally append `}` based on
    rstrip().endswith() check — never blind concatenation."""
    M = _wiring()
    # Must NOT have the unconditional + "}" pattern
    assert "self.current_tool_index\n                        ] + \"}\"" not in M.ANCHOR_A_NEW, (
        "A-14 violation: blind `streamed_args + \"}\"` would double-close. "
        "Must use rstrip().endswith(\"}\") guard."
    )
    # Must HAVE the rstrip guard
    assert "rstrip().endswith(\"}\")" in M.ANCHOR_A_NEW


def test_replacement_carries_marker():
    M = _wiring()
    assert "PN56" in M.ANCHOR_A_NEW
    assert "vllm#41466" in M.ANCHOR_A_NEW


def test_idempotent_on_synthetic(tmp_path):
    from vllm._genesis.wiring.text_patch import (
        TextPatch, TextPatcher, TextPatchResult,
    )
    M = _wiring()
    target = tmp_path / "qwen3coder_tool_parser.py"
    target.write_text("# header\n" + M.ANCHOR_A_OLD + "\n# tail\n")
    patcher = TextPatcher(
        patch_name="PN56 test",
        target_file=str(target),
        marker=M.GENESIS_PN56_MARKER,
        sub_patches=[TextPatch(name="pn56", anchor=M.ANCHOR_A_OLD,
                                replacement=M.ANCHOR_A_NEW, required=True)],
    )
    r1, _ = patcher.apply()
    assert r1 == TextPatchResult.APPLIED
    body1 = target.read_text()
    assert "PN56" in body1
    r2, _ = patcher.apply()
    assert r2 == TextPatchResult.IDEMPOTENT


def test_env_flag_default_off(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.delenv("GENESIS_ENABLE_PN56_QWEN3CODER_XML_FALLBACK", raising=False)
    decision, _ = should_apply("PN56")
    assert decision is False


def test_env_flag_engages(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.setenv("GENESIS_ENABLE_PN56_QWEN3CODER_XML_FALLBACK", "1")
    decision, _ = should_apply("PN56")
    assert decision is True


def test_registry_entry_complete():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PN56" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["PN56"]
    assert meta["upstream_pr"] == 41466


def test_apply_all_registers_pn56():
    from vllm._genesis.patches import apply_all
    assert hasattr(apply_all, "apply_patch_N56_qwen3coder_xml_fallback")
