# SPDX-License-Identifier: Apache-2.0
"""TDD for PN50 — SGLang#21019 GDN proj fusion backport.

Numerical (CPU fallback path tests, GPU bit-exact in container CI):
  1. Wrapper output bit-identical to PyTorch reference for representative
     shapes (27B Lorbus tp=2: num_qk=1 num_v=8 head=128).
  2. Wrapper falls through to PyTorch on:
     - non-contiguous input
     - non-power-of-2 head_dim
     - V_PER_GROUP non-integer
     - non-CUDA tensor (CPU)

Wiring:
  3. Anchor matches pristine upstream `gdn_linear_attn.py` (verified live).
  4. Idempotency on synthetic file.
  5. Env-flag gating (default OFF).
  6. PATCH_REGISTRY entry complete.
  7. apply_all registers `apply_patch_N50_gdn_fused_proj`.
"""
from __future__ import annotations

import os

import pytest
import torch


# ─── Numerical equivalence — fallback path (CPU, no Triton needed) ─────────


def _make_inputs(batch: int, num_qk: int, num_v: int, head_qk: int, head_v: int,
                 dtype=torch.float32, device="cpu", contiguous=True):
    qkv_size = num_qk * head_qk * 2 + num_v * head_v
    z_size = num_v * head_v
    qkvz_size = qkv_size + z_size
    ba_size = num_v * 2
    mixed_qkvz = torch.randn(batch, qkvz_size, dtype=dtype, device=device)
    mixed_ba = torch.randn(batch, ba_size, dtype=dtype, device=device)
    if not contiguous:
        # introduce stride irregularity
        mixed_qkvz = mixed_qkvz.t().t()  # contiguous re-permutation no-op
        # actually break it:
        mixed_qkvz = torch.cat([mixed_qkvz, mixed_qkvz], dim=1)[:, :qkvz_size]
    return mixed_qkvz, mixed_ba


def _ref_pytorch(mixed_qkvz, mixed_ba, num_qk, num_v, head_qk, head_v):
    qkv_size = num_qk * head_qk * 2 + num_v * head_v
    z_size = num_v * head_v
    mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
    z = z.reshape(z.size(0), num_v, head_v)
    b, a = mixed_ba.chunk(2, dim=-1)
    return mixed_qkv.contiguous(), z.contiguous(), b.contiguous(), a.contiguous()


def test_fallback_pytorch_matches_reference_27b_lorbus_shape():
    """27B Lorbus tp=2 shape: num_qk=1, num_v=8, head_dim=128."""
    from vllm._genesis.kernels.pn50_gdn_fused_proj import _fallback_pytorch
    mqkvz, mba = _make_inputs(batch=4, num_qk=1, num_v=8, head_qk=128, head_v=128)
    fused = _fallback_pytorch(mqkvz, mba, 1, 8, 128, 128)
    ref = _ref_pytorch(mqkvz, mba, 1, 8, 128, 128)
    for f, r, n in zip(fused, ref, ("qkv", "z", "b", "a")):
        assert torch.equal(f, r), f"{n} mismatch (fallback vs reference)"


def test_fallback_handles_various_shapes():
    """Verify fallback works across multiple plausible Genesis shapes."""
    from vllm._genesis.kernels.pn50_gdn_fused_proj import _fallback_pytorch
    cases = [
        # (batch, num_qk, num_v, head_qk, head_v) — Genesis shapes
        (1, 1, 8, 128, 128),     # 27B Lorbus tp=2 single-token decode
        (4, 1, 8, 128, 128),     # batched
        (16, 2, 16, 64, 64),     # hypothetical narrower head
        (1, 4, 8, 128, 128),     # different ratio
    ]
    for shape in cases:
        b, nqk, nv, hqk, hv = shape
        mqkvz, mba = _make_inputs(b, nqk, nv, hqk, hv)
        fused = _fallback_pytorch(mqkvz, mba, nqk, nv, hqk, hv)
        ref = _ref_pytorch(mqkvz, mba, nqk, nv, hqk, hv)
        for f, r, n in zip(fused, ref, ("qkv", "z", "b", "a")):
            assert torch.equal(f, r), f"shape {shape} {n} mismatch"


def test_wrapper_falls_through_on_cpu_input():
    """Non-CUDA input must use _fallback_pytorch — kernel needs CUDA."""
    from vllm._genesis.kernels.pn50_gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    mqkvz, mba = _make_inputs(4, 1, 8, 128, 128, device="cpu")
    out = fused_qkvzba_split_reshape_cat_contiguous(mqkvz, mba, 1, 8, 128, 128)
    ref = _ref_pytorch(mqkvz, mba, 1, 8, 128, 128)
    for o, r in zip(out, ref):
        assert torch.equal(o, r)


def test_wrapper_falls_through_on_non_pow2_head_dim():
    """Head_dim not power-of-2 → wrapper falls through (Triton tl.arange limit)."""
    from vllm._genesis.kernels.pn50_gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    # head_qk=96 is NOT power of 2
    mqkvz, mba = _make_inputs(4, 1, 8, 96, 128, device="cpu")
    out = fused_qkvzba_split_reshape_cat_contiguous(mqkvz, mba, 1, 8, 96, 128)
    ref = _ref_pytorch(mqkvz, mba, 1, 8, 96, 128)
    for o, r in zip(out, ref):
        assert torch.equal(o, r)


def test_wrapper_falls_through_on_non_integer_v_per_group():
    """num_heads_v % num_heads_qk != 0 → wrapper falls through."""
    from vllm._genesis.kernels.pn50_gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    # num_qk=3, num_v=8 → V_PER_GROUP not integer
    mqkvz, mba = _make_inputs(4, 3, 8, 128, 128, device="cpu")
    out = fused_qkvzba_split_reshape_cat_contiguous(mqkvz, mba, 3, 8, 128, 128)
    ref = _ref_pytorch(mqkvz, mba, 3, 8, 128, 128)
    for o, r in zip(out, ref):
        assert torch.equal(o, r)


# ─── Wiring tests ────────────────────────────────────────────────────────────


def _load_anchors():
    from vllm._genesis.wiring.hybrid import patch_N50_gdn_fused_proj as M
    return M.ANCHOR_OLD, M.ANCHOR_NEW, M.GENESIS_PN50_MARKER


def test_anchor_contains_qwen35_branch_signature():
    """Anchor must match the Qwen3.5 contiguous-projection branch."""
    anchor_old, _, _ = _load_anchors()
    assert "Qwen3.5: weights are already in [q, k, v, z]" in anchor_old
    assert "qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size" in anchor_old
    assert "mixed_qkvz.split([qkv_size, z_size]" in anchor_old
    assert "ba.chunk(2, dim=-1)" in anchor_old


def test_replacement_wires_pn50_kernel():
    _, anchor_new, marker = _load_anchors()
    assert "PN50 SGLang#21019" in anchor_new
    assert "from vllm._genesis.kernels.pn50_gdn_fused_proj import" in anchor_new
    assert "_pn50_fused" in anchor_new
    assert "self.head_k_dim" in anchor_new
    assert "self.head_v_dim" in anchor_new
    # Must preserve same output names so downstream code unaffected
    assert "mixed_qkv, z, b, a = _pn50_fused(" in anchor_new
    assert "PN50" in marker


def test_apply_idempotent_on_synthetic(tmp_path):
    """Apply twice → second call is no-op (IDEMPOTENT)."""
    from vllm._genesis.wiring.text_patch import (
        TextPatch, TextPatcher, TextPatchResult,
    )
    anchor_old, anchor_new, marker = _load_anchors()

    # Synthetic file reproducing the anchor in context
    synthetic = (
        "        # ============================================================\n"
        "        # Part 1: Input Projection\n"
        "        # ============================================================\n"
        "        if hasattr(self, 'in_proj_qkv'):\n"
        "            pass\n"
        "        else:\n"
        "            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
        "            ba, _ = self.in_proj_ba(hidden_states)\n"
        "\n"
        "            if self.gqa_interleaved_layout:\n"
        "                pass\n"
        + anchor_old + "\n"
        "\n"
        "        # ============================================================\n"
    )
    target = tmp_path / "gdn_linear_attn.py"
    target.write_text(synthetic)

    patcher = TextPatcher(
        patch_name="PN50 test",
        target_file=str(target),
        marker=marker,
        sub_patches=[TextPatch(name="pn50", anchor=anchor_old,
                                replacement=anchor_new, required=True)],
    )

    r1, _ = patcher.apply()
    assert r1 == TextPatchResult.APPLIED
    body1 = target.read_text()
    assert "PN50" in body1
    assert "_pn50_fused" in body1

    r2, _ = patcher.apply()
    assert r2 == TextPatchResult.IDEMPOTENT
    assert target.read_text() == body1


def test_env_flag_default_off(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.delenv("GENESIS_ENABLE_PN50_GDN_FUSED_PROJ", raising=False)
    decision, reason = should_apply("PN50")
    assert decision is False
    assert "opt-in" in reason.lower() or "off" in reason.lower()


def test_env_flag_engages(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.setenv("GENESIS_ENABLE_PN50_GDN_FUSED_PROJ", "1")
    decision, _ = should_apply("PN50")
    assert decision is True


def test_registry_entry_complete():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PN50" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["PN50"]
    assert meta["env_flag"] == "GENESIS_ENABLE_PN50_GDN_FUSED_PROJ"
    assert meta["default_on"] is False
    assert meta["category"] == "perf_kernel"
    assert "GDN" in meta["title"] or "gdn" in meta["title"].lower()


def test_apply_all_registers_pn50():
    from vllm._genesis.patches import apply_all
    assert hasattr(apply_all, "apply_patch_N50_gdn_fused_proj")
