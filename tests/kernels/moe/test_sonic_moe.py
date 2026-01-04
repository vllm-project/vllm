# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Sonic MoE integration."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.sonic_moe import (
    SonicMoeExperts,
    _check_sonicmoe_available,
    _is_hopper_gpu,
    is_sonic_moe_supported,
    is_valid_sonic_moe,
    permute_weights_for_sonic,
    sonic_moe_forward,
)
from vllm.platforms import current_platform

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA required",
)


def test_check_sonicmoe_available():
    result = _check_sonicmoe_available()
    assert isinstance(result, bool)


def test_is_hopper_gpu():
    result = _is_hopper_gpu()
    assert isinstance(result, bool)

    if current_platform.is_cuda():
        expected = current_platform.has_device_capability(90)
        assert result == expected


def test_is_sonic_moe_supported():
    result = is_sonic_moe_supported()
    assert isinstance(result, bool)

    if not _check_sonicmoe_available():
        assert result is False
    if not _is_hopper_gpu():
        assert result is False


def test_permute_weights_for_sonic():
    """Test weight permutation from vLLM to Sonic format."""
    E, N, K = 8, 512, 256  # 8 experts, 512 intermediate (2*256), 256 hidden
    w = torch.randn(E, N, K)

    w_permuted = permute_weights_for_sonic(w)

    # Shape should be preserved
    assert w_permuted.shape == w.shape
    # Should be contiguous
    assert w_permuted.is_contiguous()

    # Verify the permutation is correct:
    # Original: [first_half, second_half] = [w[:, :N//2, :], w[:, N//2:, :]]
    # Permuted: [interleaved] where even indices = first_half, odd = second_half
    first_half = w[:, : N // 2, :]
    second_half = w[:, N // 2 :, :]

    # After permutation, even columns should be from first_half
    # odd columns should be from second_half
    for i in range(N // 2):
        assert torch.allclose(w_permuted[:, 2 * i, :], first_half[:, i, :])
        assert torch.allclose(w_permuted[:, 2 * i + 1, :], second_half[:, i, :])


def test_sonic_moe_experts_init():
    """Test SonicMoeExperts initialization."""
    experts = SonicMoeExperts(out_dtype=torch.bfloat16)
    assert experts.out_dtype == torch.bfloat16
    assert experts.supports_chunking() is True
    assert experts.supports_expert_map() is False


@requires_cuda
def test_is_valid_sonic_moe_basic():
    M, K, N = 128, 512, 1024
    num_experts, top_k = 8, 2

    hidden_states = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w1 = torch.randn(num_experts, K, N, dtype=torch.float16, device="cuda")
    w2 = torch.randn(num_experts, N, K, dtype=torch.float16, device="cuda")

    result = is_valid_sonic_moe(hidden_states, w1, w2, num_experts, top_k)
    assert isinstance(result, bool)


@requires_cuda
def test_is_valid_sonic_moe_large_topk():
    M, K, N = 128, 512, 1024
    num_experts, top_k = 8, 32

    hidden_states = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w1 = torch.randn(num_experts, K, N, dtype=torch.float16, device="cuda")
    w2 = torch.randn(num_experts, N, K, dtype=torch.float16, device="cuda")

    result = is_valid_sonic_moe(hidden_states, w1, w2, num_experts, top_k)
    # Should be False because top_k > 16, or False because not supported
    assert result is False or not is_sonic_moe_supported()


@requires_cuda
def test_sonic_moe_forward_unsupported():
    """Test that sonic_moe_forward raises RuntimeError on unsupported systems."""
    if is_sonic_moe_supported():
        pytest.skip("Sonic MoE is supported on this system")

    M, K, N = 128, 512, 1024
    num_experts, top_k = 8, 2

    hidden_states = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w1 = torch.randn(num_experts, K, N, dtype=torch.float16, device="cuda")
    w2 = torch.randn(num_experts, N, K, dtype=torch.float16, device="cuda")
    topk_weights = torch.randn(M, top_k, dtype=torch.float16, device="cuda")
    topk_ids = torch.randint(0, num_experts, (M, top_k), device="cuda")

    with pytest.raises(RuntimeError):
        sonic_moe_forward(hidden_states, w1, w2, topk_weights, topk_ids)


def test_import_from_fused_moe():
    from vllm.model_executor.layers.fused_moe import (
        SonicMoeExperts,
        is_sonic_moe_supported,
        is_valid_sonic_moe,
        permute_weights_for_sonic,
        sonic_moe_forward,
    )

    assert callable(is_sonic_moe_supported)
    assert callable(is_valid_sonic_moe)
    assert callable(sonic_moe_forward)
    assert callable(permute_weights_for_sonic)
    assert SonicMoeExperts is not None


SONIC_MNKS = [
    (128, 1024, 256),
    (256, 2048, 512),
    (512, 4096, 1024),
]
SONIC_TOPKS = [2, 4]
SONIC_NUM_EXPERTS = [8, 16]
SONIC_DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize(("m", "n", "k"), SONIC_MNKS)
@pytest.mark.parametrize("topk", SONIC_TOPKS)
@pytest.mark.parametrize("num_experts", SONIC_NUM_EXPERTS)
@pytest.mark.parametrize("dtype", SONIC_DTYPES)
@pytest.mark.skipif(
    not is_sonic_moe_supported(),
    reason="Requires sonicmoe + Hopper GPU",
)
def test_sonic_moe_vs_triton(
    m: int,
    n: int,
    k: int,
    topk: int,
    num_experts: int,
    dtype: torch.dtype,
):
    """Compare Sonic MoE against Triton reference."""
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNoEP,
    )
    from vllm.utils.deep_gemm import calc_diff

    if topk > num_experts:
        pytest.skip(f"topk={topk} > num_experts={num_experts}")

    hidden_states = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    w1 = torch.randn(num_experts, n, k, device="cuda", dtype=dtype) / 10
    w2 = torch.randn(num_experts, k, n // 2, device="cuda", dtype=dtype) / 10

    router_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1).to(dtype)

    triton_kernel = mk.FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(),
        TritonExperts(FUSED_MOE_UNQUANTIZED_CONFIG),
    )
    out_triton = triton_kernel(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        activation="silu",
        global_num_experts=num_experts,
    )

    w1_sonic = permute_weights_for_sonic(w1)
    sonic_kernel = mk.FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(),
        SonicMoeExperts(out_dtype=dtype, weights_prepermuted=True),
    )
    out_sonic = sonic_kernel(
        hidden_states=hidden_states,
        w1=w1_sonic,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        activation="silu",
        global_num_experts=num_experts,
    )

    diff = calc_diff(out_sonic, out_triton)
    assert diff < 0.01, f"Diff exceeded 1%: {diff}"
