# SPDX-License-Identifier: Apache-2.0
"""TDD tests for PN37 — DFlash tiny-Q non-causal attention kernel.

Numerical equivalence against torch SDPA (`is_causal=False`) reference,
on the exact shapes Qwen3.6-DFlash drafter produces:

  Qwen3.6-27B-DFlash:
    head_dim = 128, num_heads = 32, num_kv_heads = 8 (HEADS_PER_KV = 4)
    With TP=2: num_heads = 16, num_kv_heads = 4 (HEADS_PER_KV = 4)

  Qwen3.6-35B-A3B-DFlash:
    head_dim = 128, num_heads = 32, num_kv_heads = 4 (HEADS_PER_KV = 8)
    With TP=2: num_heads = 16, num_kv_heads = 2 (HEADS_PER_KV = 8)

Q rows per request range: 4 (35B N=3) ↔ 6 (27B N=5). With max-num-seqs=1
the entire batch is one request → batch=1.

GPU-only. Skip with informative message if no CUDA.

Run via:
    pytest vllm/_genesis/tests/test_pn37_dflash_tiny_q.py -v
"""
from __future__ import annotations

import pytest


requires_cuda = pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="GPU required for PN37 numerical tests",
)


# ─────────────────────────────────────────────────────────────────
# Cheap predicate (no GPU)
# ─────────────────────────────────────────────────────────────────


def test_eligibility_predicate():
    from vllm._genesis.kernels.pn37_dflash_tiny_q_attn import is_eligible_shape
    assert is_eligible_shape(q_len=4, head_dim=128) is True
    assert is_eligible_shape(q_len=6, head_dim=128) is True
    assert is_eligible_shape(q_len=8, head_dim=128) is True
    assert is_eligible_shape(q_len=16, head_dim=128) is True
    # OUT of range
    assert is_eligible_shape(q_len=1, head_dim=128) is False
    assert is_eligible_shape(q_len=17, head_dim=128) is False
    assert is_eligible_shape(q_len=4, head_dim=64) is False
    assert is_eligible_shape(q_len=4, head_dim=256) is False


def test_env_gate(monkeypatch):
    from vllm._genesis.kernels import pn37_dflash_tiny_q_attn as k
    monkeypatch.delenv("GENESIS_ENABLE_PN37_DFLASH_TINY_Q", raising=False)
    assert k.env_enabled() is False
    monkeypatch.setenv("GENESIS_ENABLE_PN37_DFLASH_TINY_Q", "1")
    assert k.env_enabled() is True
    monkeypatch.setenv("GENESIS_ENABLE_PN37_DFLASH_TINY_Q", "false")
    assert k.env_enabled() is False


# ─────────────────────────────────────────────────────────────────
# Numerical equivalence vs torch SDPA (CUDA required)
# ─────────────────────────────────────────────────────────────────


def _ref_sdpa_noncausal(q, k, v, scale):
    """Reference: torch.nn.functional.scaled_dot_product_attention with
    is_causal=False, GQA expansion via repeat_interleave.

    q: [B, Q_LEN, NUM_HEADS, HEAD_DIM]
    k: [B, KV_LEN, NUM_KV_HEADS, HEAD_DIM]
    v: [B, KV_LEN, NUM_KV_HEADS, HEAD_DIM]
    """
    import torch
    import torch.nn.functional as F

    B, Q_LEN, NUM_HEADS, HEAD_DIM = q.shape
    _, KV_LEN, NUM_KV_HEADS, _ = k.shape
    HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS

    # GQA expansion: replicate each kv_head HEADS_PER_KV times
    k_exp = k.repeat_interleave(HEADS_PER_KV, dim=2)  # [B, KV_LEN, NUM_HEADS, D]
    v_exp = v.repeat_interleave(HEADS_PER_KV, dim=2)

    # SDPA expects [B, NUM_HEADS, S, D]
    q_t = q.transpose(1, 2).contiguous()
    k_t = k_exp.transpose(1, 2).contiguous()
    v_t = v_exp.transpose(1, 2).contiguous()

    out = F.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
    )
    return out.transpose(1, 2).contiguous()  # [B, Q_LEN, NUM_HEADS, D]


@requires_cuda
@pytest.mark.parametrize("Q_LEN", [4, 6, 8])
@pytest.mark.parametrize("KV_LEN", [128, 1024, 4096])
@pytest.mark.parametrize("NUM_KV_HEADS,HEADS_PER_KV", [(4, 4), (2, 8)])
def test_numerical_equivalence_27b_35b_dflash_shapes(
    Q_LEN, KV_LEN, NUM_KV_HEADS, HEADS_PER_KV
):
    """PN37 output ≈ torch SDPA output (allclose with BF16 tolerance)."""
    import torch
    from vllm._genesis.kernels.pn37_dflash_tiny_q_attn import (
        pn37_tiny_q_noncausal_attn,
    )

    NUM_HEADS = NUM_KV_HEADS * HEADS_PER_KV
    HEAD_DIM = 128
    B = 1
    DTYPE = torch.bfloat16
    scale = 1.0 / (HEAD_DIM ** 0.5)

    torch.manual_seed(0)
    q = torch.randn(B, Q_LEN, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda") * 0.5
    k = torch.randn(B, KV_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda") * 0.5
    v = torch.randn(B, KV_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda") * 0.5

    out_pn37 = pn37_tiny_q_noncausal_attn(q, k, v, scale=scale)
    out_ref = _ref_sdpa_noncausal(q, k, v, scale=scale)

    # BF16 tolerance — same bar P67 v7.34 uses (rel_avg < 1e-3 there;
    # we relax slightly because we compare two BF16 paths with different
    # accumulator orderings).
    diff = (out_pn37 - out_ref).abs()
    rel = diff / (out_ref.abs() + 1e-3)
    rel_avg = rel.mean().item()
    rel_max = rel.max().item()
    abs_max = diff.max().item()

    print(
        f"\n[PN37 numeric] Q={Q_LEN} KV={KV_LEN} H={NUM_HEADS}/{NUM_KV_HEADS}: "
        f"rel_avg={rel_avg:.4f} rel_max={rel_max:.4f} abs_max={abs_max:.4f}"
    )

    # Strict: rel_avg under 1% (BF16 path)
    assert rel_avg < 1e-2, (
        f"PN37 vs SDPA rel_avg={rel_avg:.4f} > 1e-2 — numerical regression"
    )
    # Worst-case absolute: under 1e-1 (BF16 noise can spike on outliers)
    assert abs_max < 0.1, (
        f"PN37 vs SDPA abs_max={abs_max:.4f} > 0.1 — outlier divergence"
    )


@requires_cuda
def test_short_kv_smoke():
    """Edge case: KV_LEN=8 (very short context). Should still produce
    finite output with no NaN/Inf."""
    import torch
    from vllm._genesis.kernels.pn37_dflash_tiny_q_attn import (
        pn37_tiny_q_noncausal_attn,
    )

    q = torch.randn(1, 4, 4, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16, device="cuda")
    out = pn37_tiny_q_noncausal_attn(q, k, v)
    assert torch.isfinite(out).all(), "NaN/Inf in output on short-KV smoke"
    assert out.shape == q.shape
