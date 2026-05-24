# SPDX-License-Identifier: Apache-2.0
"""TDD for MoE selection oracle."""
from __future__ import annotations

import pytest


def test_fp8_low_m_routes_to_p81():
    from vllm._genesis.oracle import select_moe_expert_impl
    sel = select_moe_expert_impl("fp8", sm_major=8, num_tokens=4)
    assert "low_m" in sel.impl_name
    assert "P81" in sel.relevant_patches


def test_fp8_high_m_routes_to_standard():
    from vllm._genesis.oracle import select_moe_expert_impl
    sel = select_moe_expert_impl("fp8", sm_major=8, num_tokens=512)
    assert sel.impl_name == "fp8_block_scaled"
    assert "PN8" in sel.relevant_patches


def test_lorbus_27b_int4_g128_marlin_with_p87_p91():
    """27B Lorbus (AutoRound INT4 g=128) → Marlin path → P87 + P91."""
    from vllm._genesis.oracle import select_moe_expert_impl
    sel = select_moe_expert_impl(
        "compressed_tensors_int4_marlin", sm_major=8, num_tokens=128, group_size=128,
    )
    assert sel.impl_name == "marlin"
    assert "P87" in sel.relevant_patches
    assert "P91" in sel.relevant_patches


def test_minachist_27b_int8_no_group_allspark_no_marlin_patches():
    """Memory feedback_v764_swap_findings — Minachist INT8 g=-1 → AllSpark
    → P87/P91 NO-OP. Oracle must explicitly say so."""
    from vllm._genesis.oracle import select_moe_expert_impl
    sel = select_moe_expert_impl("int8_gs-1", sm_major=8, num_tokens=128)
    assert sel.impl_name == "allspark_no_group"
    assert sel.relevant_patches == ()
    assert "NO-OP" in sel.notes
    assert "Marlin" in sel.notes


def test_unknown_quant_falls_back_to_default():
    from vllm._genesis.oracle import select_moe_expert_impl
    sel = select_moe_expert_impl("nvfp4_blackwell", sm_major=12, num_tokens=64)
    assert sel.impl_name == "vllm_default"
    assert sel.relevant_patches == ()


def test_explain_helper_human_readable():
    from vllm._genesis.oracle.moe_select import explain_for_config
    out = explain_for_config("fp8", sm_major=8, num_tokens=4)
    assert "MoE oracle" in out
    assert "P81" in out
    assert "low_m" in out
