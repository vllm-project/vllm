# SPDX-License-Identifier: Apache-2.0
"""TDD tests for PN29 — GDN chunk_o scale-fold (vllm#41446 pattern (c) backport).

Test contract:
1. Anchor text matches exact upstream code (line 137 in chunk_o.py)
2. Replacement preserves the math (1 fewer fp32 multiply per inner iter)
3. Marker comment present for drift detection
4. ENV opt-in default OFF
5. Numerical drift bounded for typical attention scales

Mathematical identity:
    b_o * scale + dot * scale  ==  (b_o + dot) * scale     (distributive)

In fp32, this can differ by ≤1 ULP per element. For attention with
scale = 1/sqrt(d_head) (small constant), both forms are numerically
equivalent within rounding error.

Reference: vllm#41446 (zobinHuang, MI300X GDN optimization, pattern (c)).
Hardware-agnostic — Triton compiler can't auto-fuse this pattern across
the `b_o = b_o * scale + dot * scale` boundary, so explicit fold = guaranteed
1 fewer fp32 mul per `chunk_fwd_kernel_o` inner iter.
"""
from __future__ import annotations




def test_pn29_wiring_imports():
    """PN29 wiring module imports cleanly."""
    from vllm._genesis.wiring.hybrid import patch_N29_gdn_chunk_o_scale_fold
    assert hasattr(patch_N29_gdn_chunk_o_scale_fold, "apply")
    assert hasattr(patch_N29_gdn_chunk_o_scale_fold, "GENESIS_PN29_MARKER")


def test_pn29_dispatcher_registry():
    """PN29 registered in PATCH_REGISTRY with correct env flag."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PN29" in PATCH_REGISTRY
    e = PATCH_REGISTRY["PN29"]
    assert e["env_flag"] == "GENESIS_ENABLE_PN29_GDN_SCALE_FOLD"
    assert e["default_on"] is False
    assert e["upstream_pr"] == 41446


def test_pn29_skips_when_env_off(monkeypatch):
    """When env is OFF, apply() returns 'skipped' with opt-in reason."""
    monkeypatch.delenv("GENESIS_ENABLE_PN29_GDN_SCALE_FOLD", raising=False)
    from vllm._genesis.wiring.hybrid.patch_N29_gdn_chunk_o_scale_fold import apply
    status, reason = apply()
    assert status == "skipped"
    assert "opt-in" in reason.lower()


def test_pn29_anchor_text_matches_upstream():
    """PN29 anchor matches exact upstream chunk_o.py:137 line."""
    from vllm._genesis.wiring.hybrid.patch_N29_gdn_chunk_o_scale_fold import (
        PN29_ANCHOR, PN29_REPLACEMENT,
    )
    # Anchor: the EXACT current upstream line
    assert "b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale" in PN29_ANCHOR
    # Replacement: scale-fold form
    assert "b_o = (b_o + tl.dot(b_A.to(b_v.dtype), b_v)) * scale" in PN29_REPLACEMENT
    # Drift marker
    assert "Genesis PN29" in PN29_REPLACEMENT
    assert "vllm#41446" in PN29_REPLACEMENT


def test_pn29_marker_string_unique():
    """PN29 marker is non-trivial for drift detection."""
    from vllm._genesis.wiring.hybrid.patch_N29_gdn_chunk_o_scale_fold import (
        GENESIS_PN29_MARKER,
    )
    assert "PN29" in GENESIS_PN29_MARKER
    assert len(GENESIS_PN29_MARKER) > 30


def test_pn29_register_in_apply_all():
    """PN29 registered via @register_patch in apply_all.py."""
    from vllm._genesis.patches.apply_all import PATCH_REGISTRY as APPLY_REGISTRY
    names = [name for name, _ in APPLY_REGISTRY]
    pn29 = [n for n in names if "PN29" in n]
    assert len(pn29) == 1, f"PN29 not registered, names: {names[:5]}"


# ─────────────────────────────────────────────────────────────────
# Numerical equivalence tests (pure Python, no Triton needed)
# ─────────────────────────────────────────────────────────────────


def test_pn29_numerical_equivalence_typical_attention():
    """Scale-fold preserves attention output within 1 ULP for typical inputs.

    Attention scale = 1/sqrt(d_head) ~= 0.088 for d=128. Magnitudes:
    - b_o accumulator: ~[-10, 10] after several tl.dot accumulations
    - dot result: ~[-100, 100] after Q@K
    - scale * (b_o + dot): ~[-10, 10] expected output range
    """
    import torch

    torch.manual_seed(42)
    BT, BV = 64, 128
    scale = 1.0 / (128 ** 0.5)
    b_o = torch.randn(BT, BV, dtype=torch.float32) * 5.0
    dot = torch.randn(BT, BV, dtype=torch.float32) * 50.0

    original = b_o * scale + dot * scale
    folded = (b_o + dot) * scale

    # Max abs diff per element
    max_abs_diff = (original - folded).abs().max().item()
    # Relative diff
    rel_diff = (original - folded).abs().max() / (original.abs().max() + 1e-10)

    # IEEE 754 fp32: (a*c)+(b*c) vs (a+b)*c can differ by 1-2 ULPs.
    # For our magnitudes (output ~50, scale ~0.088): 1 ULP ~= 6e-6.
    assert max_abs_diff < 1e-4, f"max_abs_diff = {max_abs_diff} too large"
    assert rel_diff.item() < 1e-5, f"rel_diff = {rel_diff.item()} too large"


def test_pn29_numerical_equivalence_extreme_scale():
    """Extreme scale (small d_head, e.g. MLA) — drift still bounded."""
    import torch

    torch.manual_seed(0)
    BT, BV = 64, 64
    scale = 1.0 / (32 ** 0.5)  # d_head=32
    b_o = torch.randn(BT, BV, dtype=torch.float32) * 100.0
    dot = torch.randn(BT, BV, dtype=torch.float32) * 1000.0

    original = b_o * scale + dot * scale
    folded = (b_o + dot) * scale

    rel_diff = (original - folded).abs().max() / (original.abs().max() + 1e-10)
    assert rel_diff.item() < 1e-5, (
        f"Extreme magnitudes: rel_diff = {rel_diff.item()}"
    )


def test_pn29_numerical_equivalence_zero_scale():
    """scale=0 edge case (degenerate): both forms produce zero."""
    import torch

    torch.manual_seed(7)
    b_o = torch.randn(64, 128, dtype=torch.float32)
    dot = torch.randn(64, 128, dtype=torch.float32)
    scale = 0.0

    original = b_o * scale + dot * scale
    folded = (b_o + dot) * scale
    assert torch.equal(original, folded)
    assert original.abs().max().item() == 0.0


def test_pn29_idempotency_via_marker():
    """Re-applying PN29 doesn't double-patch (marker check)."""
    from vllm._genesis.wiring.hybrid.patch_N29_gdn_chunk_o_scale_fold import (
        GENESIS_PN29_MARKER,
    )
    # The TextPatcher uses the marker comment to detect already-applied state.
    # Re-application should be no-op. (Tested in TextPatcher integration; here
    # we just verify the marker has the canonical form.)
    assert GENESIS_PN29_MARKER.startswith("Genesis PN29")
