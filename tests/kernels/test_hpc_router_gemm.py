# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm._custom_ops import fp32_router_gemm
from vllm.model_executor.layers.fused_moe.router.gate_linear import (
    hpc_gemm_bf16xfp32_dispatch_impl,
)
from vllm.utils.hpc import has_hpc_bf16xfp32_gemm

SHAPES = [
    (4096, 192),
    (3072, 256),
    (6144, 128),
]
FP32_ROUTER_SHAPES = [
    (3072, 256),
    (6144, 128),
    (6144, 256),
]
TOKEN_COUNTS = [1, 2, 8, 16, 32, 128, 512]

ATOL = 1e-1
RTOL = 8e-2
MAX_RELATIVE_L2 = 2e-3


def _requires_hpc_sm90() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("HPC BF16xFP32 router GEMM requires SM90")
    if not has_hpc_bf16xfp32_gemm():
        pytest.skip("HPC BF16xFP32 GEMM is unavailable")


@pytest.mark.parametrize("hidden_size,num_experts", SHAPES)
@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("seed", [1, 17, 42])
def test_hpc_router_gemm_numerics(
    hidden_size: int,
    num_experts: int,
    num_tokens: int,
    seed: int,
) -> None:
    _requires_hpc_sm90()
    torch.manual_seed(seed)

    x = torch.randn(
        num_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = (
        torch.randn(
            num_experts,
            hidden_size,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    )

    output = hpc_gemm_bf16xfp32_dispatch_impl(x, weight)

    # FP64 reference avoids TF32/cuBLAS FP32 reference uncertainty.
    reference = torch.nn.functional.linear(
        x.double(),
        weight.double(),
    ).float()

    assert output.dtype == torch.float32
    assert output.shape == reference.shape
    assert torch.isfinite(output).all()

    torch.testing.assert_close(
        output,
        reference,
        atol=ATOL,
        rtol=RTOL,
    )

    error_l2 = torch.linalg.vector_norm((output - reference).double())
    reference_l2 = torch.linalg.vector_norm(reference.double())
    relative_l2 = error_l2 / reference_l2.clamp_min(1e-12)
    assert relative_l2.item() < MAX_RELATIVE_L2


@pytest.mark.parametrize("hidden_size,num_experts", FP32_ROUTER_SHAPES)
@pytest.mark.parametrize("num_tokens", [1, 8, 32, 128, 512])
@pytest.mark.parametrize("seed", range(5))
def test_hpc_vs_native_router(
    hidden_size: int,
    num_experts: int,
    num_tokens: int,
    seed: int,
) -> None:
    _requires_hpc_sm90()
    torch.manual_seed(1000 + seed)

    x = torch.randn(
        num_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = (
        torch.randn(
            num_experts,
            hidden_size,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    )

    hpc_output = hpc_gemm_bf16xfp32_dispatch_impl(x, weight)
    reference = torch.nn.functional.linear(
        x.double(),
        weight.double(),
    )

    hpc_error = torch.linalg.vector_norm((hpc_output - reference).double())
    reference_norm = torch.linalg.vector_norm(reference.double()).clamp_min(1e-12)

    assert (hpc_error / reference_norm).item() < MAX_RELATIVE_L2

    if num_tokens <= 32:
        native_output = fp32_router_gemm(x, weight)
        native_error = torch.linalg.vector_norm((native_output - reference).double())
        assert (native_error / reference_norm).item() < MAX_RELATIVE_L2
