# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Sonic MoE integration."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.sonic_moe import (
    SonicMoEExperts,
    _check_sonicmoe_available,
    _is_hopper_gpu,
    is_sonic_moe_supported,
    is_valid_sonic_moe,
    permute_weights_for_sonic,
    sonic_moe_forward,
)
from vllm.platforms import current_platform

# Skip decorator for Hopper-only tests
requires_hopper = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(90)),
    reason="Sonic MoE requires Hopper GPU (SM90+)",
)

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
    """Test SonicMoEExperts initialization."""
    experts = SonicMoEExperts(out_dtype=torch.bfloat16)
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
def test_sonic_moe_forward_not_implemented():
    M, K, N = 128, 512, 1024
    num_experts, top_k = 8, 2

    hidden_states = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w1 = torch.randn(num_experts, K, N, dtype=torch.float16, device="cuda")
    w2 = torch.randn(num_experts, N, K, dtype=torch.float16, device="cuda")
    topk_weights = torch.randn(M, top_k, dtype=torch.float16, device="cuda")
    topk_ids = torch.randint(0, num_experts, (M, top_k), device="cuda")

    if is_sonic_moe_supported():
        with pytest.raises(NotImplementedError):
            sonic_moe_forward(hidden_states, w1, w2, topk_weights, topk_ids)
    else:
        with pytest.raises(RuntimeError):
            sonic_moe_forward(hidden_states, w1, w2, topk_weights, topk_ids)


def test_import_from_fused_moe():
    from vllm.model_executor.layers.fused_moe import (
        SonicMoEExperts,
        is_sonic_moe_supported,
        is_valid_sonic_moe,
        permute_weights_for_sonic,
        sonic_moe_forward,
    )

    assert callable(is_sonic_moe_supported)
    assert callable(is_valid_sonic_moe)
    assert callable(sonic_moe_forward)
    assert callable(permute_weights_for_sonic)
    assert SonicMoEExperts is not None
