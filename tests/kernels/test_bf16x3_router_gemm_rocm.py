# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ROCm BF16x3 router GEMM.

The weight split and the eligibility policy are pure functions, so most of
this runs device-free. Only the GEMM correctness tests need a gfx950 runner.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router import (
    bf16x3_router_gemm_rocm as rocm_bf16x3,
)
from vllm.platforms import current_platform

# (hidden_size, num_experts) for the models that ship fp32 router weights.
#   (6144, 128) -> MiniMax-M3,  (3072, 256) -> MiniMax-M2/M2.5
ROUTER_SHAPES = [(6144, 128), (3072, 256)]
ROUTER_WEIGHT_SCALE = 0.053


def _requires_gfx950():
    if not rocm_bf16x3.platform_supported():
        pytest.skip("ROCm bf16x3 router GEMM requires gfx950")


def _inputs(num_tokens, hidden_size, num_experts, seed=42, device="cuda"):
    torch.manual_seed(seed)
    x = torch.randn(num_tokens, hidden_size, device=device).to(torch.bfloat16)
    w = torch.randn(num_experts, hidden_size, dtype=torch.float32, device=device)
    return x, w * ROUTER_WEIGHT_SCALE


def _top_set(logits, k=4):
    """Sorted top-k indices; topk's order among near-ties is not meaningful."""
    return logits.topk(k, dim=-1).indices.sort(dim=-1).values


@pytest.mark.parametrize(("hidden_size", "num_experts"), ROUTER_SHAPES)
def test_split_reconstructs_exactly(hidden_size: int, num_experts: int):
    torch.manual_seed(42)
    w = torch.randn(num_experts, hidden_size) * ROUTER_WEIGHT_SCALE
    split = rocm_bf16x3.split_bf16x3(w)

    assert split.shape == (rocm_bf16x3.BF16X3_TERMS, num_experts, hidden_size)
    assert split.dtype == torch.bfloat16
    # router_gemm reinterprets this as (TERMS * E, K) without a copy.
    assert split.is_contiguous()
    assert torch.equal(split.float().sum(0), w)


def test_split_is_exact_across_normal_exponents():
    exps = torch.arange(-100, 100, dtype=torch.float32)
    mantissas = torch.linspace(1.0, 1.999, 64, dtype=torch.float32)
    w = (mantissas[:, None] * torch.pow(2.0, exps)[None, :]).contiguous()
    w = torch.cat([w, -w], dim=0)
    assert torch.equal(rocm_bf16x3.split_bf16x3(w).float().sum(0), w)


def test_split_rejects_non_finite():
    """ValueError rather than assert: the caller downgrades to the fp32 path
    instead of failing model load, and this must survive ``python -O``."""
    for bad in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError):
            rocm_bf16x3.split_bf16x3(torch.full((2, 4), bad, dtype=torch.float32))


def test_split_rejects_non_fp32():
    with pytest.raises(ValueError):
        rocm_bf16x3.split_bf16x3(torch.zeros(2, 4, dtype=torch.bfloat16))


def test_split_does_not_mutate_input():
    w = torch.randn(8, 32, dtype=torch.float32)
    original = w.clone()
    rocm_bf16x3.split_bf16x3(w)
    assert torch.equal(w, original)


def test_is_supported_rejects_ineligible_inputs():
    meta = torch.device("meta")
    x = torch.zeros(8192, 6144, dtype=torch.bfloat16, device=meta)
    w = torch.zeros(128, 6144, dtype=torch.float32, device=meta)
    assert rocm_bf16x3.is_supported(x, w)

    assert not rocm_bf16x3.is_supported(x[: rocm_bf16x3.MIN_TOKENS - 1], w)
    assert not rocm_bf16x3.is_supported(x.float(), w)
    assert not rocm_bf16x3.is_supported(x, w.bfloat16())
    assert not rocm_bf16x3.is_supported(x, w[:, :128])
    assert not rocm_bf16x3.is_supported(x.T.contiguous().T, w)


def test_is_supported_rejects_mismatched_devices():
    """The weight can be offloaded while the activations are not."""
    x = torch.zeros(8192, 6144, dtype=torch.bfloat16, device="meta")
    w = torch.zeros(128, 6144, dtype=torch.float32)
    assert not rocm_bf16x3.is_supported(x, w)


def test_is_supported_accepts_odd_expert_counts():
    """Unlike a tl.arange-indexed kernel, torch.mm needs no power-of-two E."""
    meta = torch.device("meta")
    x = torch.zeros(8192, 6144, dtype=torch.bfloat16, device=meta)
    for num_experts in (17, 384, 512):
        w = torch.zeros(num_experts, 6144, dtype=torch.float32, device=meta)
        assert rocm_bf16x3.is_supported(x, w)


# Measured on gfx950: this path lands at 3.4e-7 to 4.9e-7 and varies <0.2%
# across token counts. Fixed bounds rather than a live comparison, because the
# fp32 fallback swings 5.7e-7 to 1.4e-6 as torch picks different hipBLASLt
# kernels at different M. _FP32_FALLBACK_BEST is the best the fallback was ever
# observed to do, so beating it proves the claim without racing a moving target.
_MAX_REL_L2 = 2e-6
_FP32_FALLBACK_BEST = 5.7e-7


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize(("hidden_size", "num_experts"), ROUTER_SHAPES)
@pytest.mark.parametrize("num_tokens", [2048, 8192, 16385])
def test_matches_fp64_reference(hidden_size: int, num_experts: int, num_tokens: int):
    _requires_gfx950()
    x, w = _inputs(num_tokens, hidden_size, num_experts)
    out = rocm_bf16x3.bf16x3_router_gemm(x, rocm_bf16x3.split_bf16x3(w))
    ref = x.double() @ w.double().T

    assert out.shape == (num_tokens, num_experts)
    assert out.dtype == torch.float32
    assert (out.double() - ref).norm() / ref.norm() < _MAX_REL_L2


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize("num_experts", [17, 384])
def test_matches_fp64_for_non_power_of_two_experts(num_experts: int):
    """Exercises the reduction's N mask, which the model shapes never hit."""
    _requires_gfx950()
    x, w = _inputs(8192, 6144, num_experts)
    out = rocm_bf16x3.bf16x3_router_gemm(x, rocm_bf16x3.split_bf16x3(w))
    ref = x.double() @ w.double().T

    assert out.shape == (8192, num_experts)
    assert (out.double() - ref).norm() / ref.norm() < _MAX_REL_L2


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize(("hidden_size", "num_experts"), ROUTER_SHAPES)
def test_more_accurate_than_fp32_fallback(hidden_size: int, num_experts: int):
    """The claim is not just accuracy but beating the fp32 GEMM it replaces."""
    _requires_gfx950()
    x, w = _inputs(16384, hidden_size, num_experts)
    ref = x.double() @ w.double().T
    out = rocm_bf16x3.bf16x3_router_gemm(x, rocm_bf16x3.split_bf16x3(w))
    assert (out.double() - ref).norm() / ref.norm() < _FP32_FALLBACK_BEST


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize(("hidden_size", "num_experts"), ROUTER_SHAPES)
def test_expert_selection_matches_fp64(hidden_size: int, num_experts: int):
    """Relative L2 is a proxy; the expert set is what reaches the model.

    Measures zero misroutes on both shapes, where the fp32 fallback misroutes a
    nonzero number at some token counts.
    """
    _requires_gfx950()
    num_tokens = 16384
    x, w = _inputs(num_tokens, hidden_size, num_experts)
    ref_set = _top_set(x.double() @ w.double().T)
    out = rocm_bf16x3.bf16x3_router_gemm(x, rocm_bf16x3.split_bf16x3(w))

    misrouted = int((_top_set(out) != ref_set).any(-1).sum())
    assert misrouted / num_tokens < 1e-4


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_writes_into_supplied_out():
    _requires_gfx950()
    x, w = _inputs(8192, 6144, 128)
    split = rocm_bf16x3.split_bf16x3(w)
    out = torch.empty(8192, 128, dtype=torch.float32, device="cuda")

    returned = rocm_bf16x3.bf16x3_router_gemm(x, split, out=out)
    assert returned is out
    assert torch.equal(out, rocm_bf16x3.bf16x3_router_gemm(x, split))


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_validates_layout_assumptions():
    """The reduction assumes a contiguous (TERMS, E, K) split and a contiguous
    out; nothing in the kernel enforces either."""
    _requires_gfx950()
    x, w = _inputs(8192, 6144, 128)
    split = rocm_bf16x3.split_bf16x3(w)

    with pytest.raises(AssertionError):
        rocm_bf16x3.bf16x3_router_gemm(x, split.transpose(1, 2))
    with pytest.raises(AssertionError):
        rocm_bf16x3.bf16x3_router_gemm(x.float(), split)
    with pytest.raises(AssertionError):
        rocm_bf16x3.bf16x3_router_gemm(x, split[:2])
    with pytest.raises(AssertionError):
        rocm_bf16x3.bf16x3_router_gemm(x[:, :512], split)
    wide = torch.empty(8192, 256, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError):
        rocm_bf16x3.bf16x3_router_gemm(x, split, out=wide[:, :128])
