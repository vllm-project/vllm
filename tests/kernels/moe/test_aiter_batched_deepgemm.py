# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for AITER batched deepgemm MoE implementation.

These tests compare AiterBatchedExperts against BatchedTritonExperts
to verify numerical correctness of the AITER deepgemm kernel integration.
"""

import pytest
import torch

from vllm.platforms import current_platform

# Skip all tests if not on ROCm
if not current_platform.is_rocm():
    pytest.skip("AITER tests require ROCm", allow_module_level=True)

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedPrepareAndFinalize,
    BatchedTritonExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEModularKernel
from vllm.model_executor.layers.fused_moe.rocm_aiter_batched_moe import (
    AiterBatchedExperts,
)

from .utils import make_dummy_moe_config

BLOCK_SIZE = [128, 128]


def is_aiter_deepgemm_supported() -> bool:
    """Check if AITER deepgemm is supported on current device."""
    if not current_platform.is_rocm():
        return False
    if not rocm_aiter_ops.is_deepgemm_enabled():
        return False
    try:
        gpu_arch = torch.cuda.get_device_properties("cuda").gcnArchName
        return "gfx942" in gpu_arch
    except Exception:
        return False


def make_block_quant_fp8_weights(
    E: int, N: int, K: int, block_size: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create FP8 quantized weights with block scales for testing.

    Args:
        E: Number of experts
        N: Intermediate dimension (2*N for gate+up)
        K: Hidden dimension
        block_size: Block size for quantization [block_k, block_n]

    Returns:
        (w1, w2, w1_scale, w2_scale) tuple of quantized weights and scales
    """
    device = "cuda"
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)

    block_k, block_n = block_size
    num_k_blocks = (K + block_k - 1) // block_k
    num_n_blocks_w1 = (2 * N + block_n - 1) // block_n
    num_n_blocks_w2 = (K + block_n - 1) // block_n

    # Create random weights and clamp to FP8 range
    w1 = torch.randn(E, 2 * N, K, device=device, dtype=torch.bfloat16) / 10.0
    w2 = torch.randn(E, K, N, device=device, dtype=torch.bfloat16) / 10.0

    w1.clamp_(fp8_info.min, fp8_info.max)
    w2.clamp_(fp8_info.min, fp8_info.max)

    # Quantize to FP8
    w1_fp8 = w1.to(fp8_dtype)
    w2_fp8 = w2.to(fp8_dtype)

    # Create per-block scales
    w1_scale = torch.ones(E, num_n_blocks_w1, num_k_blocks, device=device, dtype=torch.float32)
    w2_scale = torch.ones(E, num_n_blocks_w2, num_k_blocks, device=device, dtype=torch.float32)

    return w1_fp8, w2_fp8, w1_scale, w2_scale


def calc_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Calculate relative difference between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    diff = torch.abs(a_flat - b_flat)
    max_val = torch.max(torch.abs(a_flat), torch.abs(b_flat))
    # Avoid division by zero
    max_val = torch.clamp(max_val, min=1e-6)
    rel_diff = diff / max_val
    return rel_diff.mean().item()


@pytest.mark.skipif(
    not is_aiter_deepgemm_supported(),
    reason="Requires AITER deepgemm on MI300X (gfx942)"
)
@pytest.mark.parametrize("E", [8, 16])  # number of experts
@pytest.mark.parametrize("T", [128, 256])  # tokens per expert
@pytest.mark.parametrize("K", [128, 256])  # hidden dim
@pytest.mark.parametrize("N", [256, 512])  # intermediate dim per expert
@pytest.mark.parametrize("topk", [1, 2])
def test_aiter_batched_deepgemm_vs_triton(
    E: int, T: int, K: int, N: int, topk: int
):
    """Compare AiterBatchedExperts to BatchedTritonExperts."""

    device = "cuda"
    w1, w2, w1_s, w2_s = make_block_quant_fp8_weights(E, N, K, BLOCK_SIZE)

    M = E * T  # total tokens
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16) / 10.0
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    a.clamp_(fp8_info.min, fp8_info.max)

    # random router outputs → top-k indices / weights
    router_logits = torch.randn(M, E, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    # token number for each expert
    cnt = torch.bincount(topk_ids.flatten(), minlength=E)
    max_cnt = int(cnt.max().item())
    # next power of 2 for max token number
    max_num_tokens = 1 << (max_cnt - 1).bit_length()

    prep_finalize = BatchedPrepareAndFinalize(
        max_num_tokens=max_num_tokens,
        num_local_experts=E,
        num_dispatchers=1,
        rank=0,
    )

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_s,
        w2_scale=w2_s,
        per_act_token_quant=False,
        block_shape=BLOCK_SIZE,
    )

    moe_config = make_dummy_moe_config()

    # triton (reference)
    triton_experts = BatchedTritonExperts(
        max_num_tokens=max_num_tokens,
        num_dispatchers=1,
        quant_config=quant_config,
        moe_config=moe_config,
    )
    mk_triton = FusedMoEModularKernel(prep_finalize, triton_experts)

    out_triton = mk_triton(
        hidden_states=a,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        global_num_experts=E,
    )

    # AITER batched deepgemm
    aiter_experts = AiterBatchedExperts(
        max_num_tokens=max_num_tokens,
        num_dispatchers=1,
        quant_config=quant_config,
        moe_config=moe_config,
    )
    mk_aiter = FusedMoEModularKernel(prep_finalize, aiter_experts)

    out_aiter = mk_aiter(
        hidden_states=a,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        global_num_experts=E,
    )

    diff = calc_diff(out_aiter, out_triton)
    assert diff < 1e-2, f"Output diff too large: {diff}"


@pytest.mark.skipif(
    not is_aiter_deepgemm_supported(),
    reason="Requires AITER deepgemm on MI300X (gfx942)"
)
def test_aiter_batched_deepgemm_supports_device():
    """Test that _supports_current_device correctly identifies MI300X."""
    assert AiterBatchedExperts._supports_current_device()


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Requires ROCm platform"
)
def test_aiter_batched_deepgemm_activation_format():
    """Test that AiterBatchedExperts uses BatchedExperts format."""
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEActivationFormat,
    )
    assert AiterBatchedExperts.activation_format() == FusedMoEActivationFormat.BatchedExperts


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Requires ROCm platform"
)
def test_aiter_batched_deepgemm_quant_scheme():
    """Test that AiterBatchedExperts supports correct quant schemes."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8Static128BlockSym,
    )
    # Should support FP8 128-block symmetric
    assert AiterBatchedExperts._supports_quant_scheme(
        kFp8Static128BlockSym, kFp8Dynamic128Sym
    )
    # Should not support unquantized
    assert not AiterBatchedExperts._supports_quant_scheme(None, None)
