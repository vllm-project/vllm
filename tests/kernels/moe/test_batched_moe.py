# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch
import triton.language as tl

from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    invoke_moe_batched_triton_kernel)


@dataclass
class BatchedMMConfig:
    dtype: torch.dtype
    num_experts: int
    max_tokens_per_expert: int
    K: int
    N: int


@dataclass
class BatchedMMTensors:
    A: torch.Tensor  # [E, max_tokens, K]
    B: torch.Tensor  # [E, K, N] - column major
    C: torch.Tensor  # [E, max_tokens, N]
    num_expert_tokens: torch.Tensor  # [E]

    @staticmethod
    def make_tensors(config: BatchedMMConfig):
        A = torch.randn(
            (config.num_experts, config.max_tokens_per_expert, config.K),
            device="cuda",
            dtype=config.dtype) / 10
        B = torch.randn((config.num_experts, config.N, config.K),
                        device="cuda",
                        dtype=config.dtype)
        C = torch.zeros(
            (config.num_experts, config.max_tokens_per_expert, config.N),
            device="cuda",
            dtype=config.dtype)
        num_expert_tokens = torch.randint(low=0,
                                          high=config.max_tokens_per_expert,
                                          size=(config.num_experts, ),
                                          device="cuda",
                                          dtype=torch.int32)
        return BatchedMMTensors(A, B, C, num_expert_tokens)


def ref_impl(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
             num_expert_tokens: torch.Tensor) -> torch.Tensor:

    num_expert_tokens_cpu = num_expert_tokens.clone()
    num_expert_tokens_cpu = num_expert_tokens_cpu.to(device="cpu")
    num_experts = num_expert_tokens.size(0)

    for e in range(num_experts):
        num_tokens = num_expert_tokens_cpu[e]
        C[e, :num_tokens, :] = A[e, :num_tokens, :] @ B[e].transpose(0, 1)

    return C


@pytest.mark.parametrize("num_experts", [16, 32])
@pytest.mark.parametrize("max_tokens_per_expert",
                         [32, 64, 128, 192, 224, 256, 512])
@pytest.mark.parametrize("K", [128, 256, 1024])
@pytest.mark.parametrize("N", [128, 256, 512, 1024])
@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
def test_batched_mm(num_experts: int, max_tokens_per_expert: int, K: int,
                    N: int, dtype: torch.dtype):

    config = BatchedMMConfig(dtype, num_experts, max_tokens_per_expert, K, N)
    tensors = BatchedMMTensors.make_tensors(config)

    test_output = tensors.C
    ref_output = test_output.clone()

    compute_tl_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32
    }[test_output.dtype]
    invoke_moe_batched_triton_kernel(
        tensors.A,
        tensors.B,
        test_output,
        tensors.num_expert_tokens,
        compute_tl_dtype,
        # Quantization data
        None,
        None,
        None,
        # Quantization schemes
        False,
        False,
        False,
        config={
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 16
        })

    ref_output = ref_impl(tensors.A, tensors.B, ref_output,
                          tensors.num_expert_tokens)

    rtol, atol = {
        torch.float16: (6e-2, 6e-2),
        torch.bfloat16: (6e-2, 6e-2),
        torch.float32: (1e-2, 1e-2),
    }[test_output.dtype]

    torch.testing.assert_close(test_output, ref_output, atol=atol, rtol=rtol)
