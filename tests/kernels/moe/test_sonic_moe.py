# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Sonic MoE integration."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
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
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    is_workspace_manager_initialized,
)

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA required",
)


def _ensure_workspace_initialized() -> None:
    # Many tests rely on tests/conftest.py to init WorkspaceManager.
    # Keep this file self-contained so it can be run under --confcutdir.
    if current_platform.is_cuda() and not is_workspace_manager_initialized():
        init_workspace_manager(torch.device("cuda"))


def make_dummy_moe_config(
    num_experts: int = 1,
    experts_per_token: int = 1,
    hidden_dim: int = 1,
    intermediate_size_per_partition: int = 1,
    in_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda",
    activation: str = "silu",
) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        intermediate_size_per_partition=intermediate_size_per_partition,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=activation,
        in_dtype=in_dtype,
        device=device,
        routing_method=RoutingMethodType.TopK,
    )


def test_check_sonicmoe_available():
    result = _check_sonicmoe_available()
    assert isinstance(result, bool)


def test_is_hopper_gpu():
    result = _is_hopper_gpu()
    assert isinstance(result, bool)

    if current_platform.is_cuda():
        expected = current_platform.is_device_capability(90)
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
    # Original: [gate, up] = [w[:, :N//2, :], w[:, N//2:, :]]
    # Permuted: [interleaved] where even indices = gate, odd = up
    gate = w[:, : N // 2, :]
    up = w[:, N // 2 :, :]

    # After permutation, even columns should be from gate
    # odd columns should be from up
    for i in range(N // 2):
        assert torch.allclose(w_permuted[:, 2 * i, :], gate[:, i, :])
        assert torch.allclose(w_permuted[:, 2 * i + 1, :], up[:, i, :])


def test_sonic_moe_experts_init():
    """Test SonicMoeExperts initialization."""
    moe_config = make_dummy_moe_config(
        num_experts=1,
        experts_per_token=1,
        hidden_dim=1,
        intermediate_size_per_partition=1,
        in_dtype=torch.bfloat16,
    )
    experts = SonicMoeExperts(moe_config=moe_config)
    assert experts.out_dtype == torch.bfloat16
    assert experts.supports_chunking() is True
    assert experts.supports_expert_map() is False


@requires_cuda
def test_is_valid_sonic_moe_basic():
    M, K, two_n = 128, 512, 1024
    num_experts, top_k = 8, 2

    hidden_states = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w1 = torch.randn(num_experts, two_n, K, dtype=torch.float16, device="cuda")
    w2 = torch.randn(num_experts, K, two_n // 2, dtype=torch.float16, device="cuda")

    result = is_valid_sonic_moe(hidden_states, w1, w2, num_experts, top_k)
    assert isinstance(result, bool)


@requires_cuda
def test_is_valid_sonic_moe_large_topk():
    M, K, two_n = 128, 512, 1024
    num_experts, top_k = 8, 32

    hidden_states = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w1 = torch.randn(num_experts, two_n, K, dtype=torch.float16, device="cuda")
    w2 = torch.randn(num_experts, K, two_n // 2, dtype=torch.float16, device="cuda")

    result = is_valid_sonic_moe(hidden_states, w1, w2, num_experts, top_k)
    # Should be False because top_k > 16, or False because not supported
    assert result is False or not is_sonic_moe_supported()


@requires_cuda
def test_sonic_moe_forward_unsupported():
    """Test that sonic_moe_forward raises RuntimeError on unsupported systems."""
    if is_sonic_moe_supported():
        pytest.skip("Sonic MoE is supported on this system")

    M, K, two_n = 128, 512, 1024
    num_experts, top_k = 8, 2

    hidden_states = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w1 = torch.randn(num_experts, two_n, K, dtype=torch.float16, device="cuda")
    w2 = torch.randn(num_experts, K, two_n // 2, dtype=torch.float16, device="cuda")
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
    reason="Requires SonicMoE + Hopper GPU",
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

    _ensure_workspace_initialized()

    hidden_states = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    w1 = torch.randn(num_experts, n, k, device="cuda", dtype=dtype) / 10
    w2 = torch.randn(num_experts, k, n // 2, device="cuda", dtype=dtype) / 10
    if not is_valid_sonic_moe(hidden_states, w1, w2, num_experts, topk):
        pytest.skip("SonicMoE kernels do not support this shape/config.")

    router_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1).to(dtype)

    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=k,
        intermediate_size_per_partition=n // 2,
        in_dtype=dtype,
    )

    triton_kernel = mk.FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(),
        TritonExperts(
            moe_config=moe_config,
            quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
        ),
    )
    out_triton = triton_kernel(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation="silu",
        global_num_experts=num_experts,
    )

    w1_sonic = permute_weights_for_sonic(w1)
    sonic_kernel = mk.FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(),
        SonicMoeExperts(moe_config=moe_config, weights_prepermuted=True),
    )
    out_sonic = sonic_kernel(
        hidden_states=hidden_states,
        w1=w1_sonic,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation="silu",
        global_num_experts=num_experts,
    )

    diff = calc_diff(out_sonic, out_triton)
    assert diff < 0.01, f"Diff exceeded 1%: {diff}"


@pytest.mark.skipif(
    not is_sonic_moe_supported(),
    reason="Requires sonicmoe + Hopper GPU",
)
def test_sonic_moe_apply_router_weight_on_input():
    """Compare Sonic MoE against Triton with apply_router_weight_on_input."""
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNoEP,
    )
    from vllm.utils.deep_gemm import calc_diff

    m, n, k = 128, 2048, 512
    topk = 1
    num_experts = 8
    dtype = torch.float16

    _ensure_workspace_initialized()

    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=k,
        intermediate_size_per_partition=n // 2,
        in_dtype=dtype,
    )

    hidden_states = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    w1 = torch.randn(num_experts, n, k, device="cuda", dtype=dtype) / 10
    w2 = torch.randn(num_experts, k, n // 2, device="cuda", dtype=dtype) / 10
    if not is_valid_sonic_moe(hidden_states, w1, w2, num_experts, topk):
        pytest.skip("SonicMoE kernels do not support this shape/config.")

    topk_ids = torch.randint(0, num_experts, (m, topk), device="cuda")
    topk_weights = torch.rand(m, topk, device="cuda", dtype=dtype) + 0.1

    triton_kernel = mk.FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(),
        TritonExperts(
            moe_config=moe_config,
            quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
        ),
    )
    out_triton = triton_kernel(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation="silu",
        apply_router_weight_on_input=True,
        global_num_experts=num_experts,
    )

    w1_sonic = permute_weights_for_sonic(w1)
    sonic_kernel = mk.FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(),
        SonicMoeExperts(moe_config=moe_config, weights_prepermuted=True),
    )
    out_sonic = sonic_kernel(
        hidden_states=hidden_states,
        w1=w1_sonic,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation="silu",
        apply_router_weight_on_input=True,
        global_num_experts=num_experts,
    )

    diff = calc_diff(out_sonic, out_triton)
    assert diff < 0.01, f"Diff exceeded 1%: {diff}"
