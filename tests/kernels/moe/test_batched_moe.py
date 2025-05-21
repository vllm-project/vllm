# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest
import torch
import triton.language as tl
from typing import Optional

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
        if config.dtype == torch.torch.float8_e4m3fn:
            config_dtype = torch.bfloat16
        else:
            config_dtype = config.dtype

        A = torch.randn(
            (config.num_experts, config.max_tokens_per_expert, config.K),
            device="cuda",
            dtype=config_dtype) / 10
        B = torch.randn((config.num_experts, config.N, config.K),
                        device="cuda",
                        dtype=config_dtype)
        C = torch.zeros(
            (config.num_experts, config.max_tokens_per_expert, config.N),
            device="cuda",
            dtype=config_dtype)

        A = A.to(config.dtype)
        B = B.to(config.dtype)
        C = C.to(config.dtype)

        num_expert_tokens = torch.randint(low=0,
                                          high=config.max_tokens_per_expert,
                                          size=(config.num_experts, ),
                                          device="cuda",
                                          dtype=torch.int32)
        return BatchedMMTensors(A, B, C, num_expert_tokens)


def native_w8a8_block_matmul(A: torch.Tensor, B: torch.Tensor,
                             As: torch.Tensor, Bs: torch.Tensor,
                             block_size,
                             output_dtype = torch.bfloat16):
    """This function performs matrix multiplication with block-wise
    quantization using native torch.
    It is agnostic to the input data type and can be used for both int8 and
    fp8 data types.

    It takes two input tensors `A` and `B` (int8) with scales `As` and
    `Bs` (float32).
    The output is returned in the specified `output_dtype`.
    """
    A = A.to(torch.float32)
    B = B.to(torch.float32).contiguous()
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1], (
        f"{(A.shape[-1] + block_k - 1) // block_k} == {As.shape[-1]}")
    assert A.shape[:-1] == As.shape[:-1], f"{A.shape} == {As.shape}"

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N, )
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0]
    assert k_tiles == Bs.shape[1]

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=torch.float32, device=A.device)

    A_tiles = [
        A[:, i * block_k:min((i + 1) * block_k, K)] for i in range(k_tiles)
    ]
    B_tiles = [[
        B[
            j * block_n:min((j + 1) * block_n, N),
            i * block_k:min((i + 1) * block_k, K),
        ] for i in range(k_tiles)
    ] for j in range(n_tiles)]
    C_tiles = [
        C[:, j * block_n:min((j + 1) * block_n, N)] for j in range(n_tiles)
    ]
    As_tiles = [As[:, i:i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C


def ref_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    num_expert_tokens: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    block_shape: Optional[list[int]],
) -> torch.Tensor:
    num_expert_tokens_cpu = num_expert_tokens.clone()
    num_expert_tokens_cpu = num_expert_tokens_cpu.to(device="cpu")
    num_experts = num_expert_tokens.size(0)

    for e in range(num_experts):
        num_tokens = num_expert_tokens_cpu[e]
        if A.dtype == torch.torch.float8_e4m3fn:
            C[e, :, :] = native_w8a8_block_matmul(A[e, :, :],
                                                  B[e].transpose(0, 1),
                                                  A_scale,
                                                  B_scale,
                                                  [1,1])#block_shape)
        else:
            C[e, :num_tokens, :] = A[e, :num_tokens, :] @ B[e].transpose(0, 1)

    return C

@pytest.mark.parametrize("num_experts", [16, 32])
@pytest.mark.parametrize("max_tokens_per_expert",
                         [32, 64, 128, 192, 224, 256, 512])
@pytest.mark.parametrize("K", [128, 256, 1024])
@pytest.mark.parametrize("N", [128, 256, 512, 1024])
@pytest.mark.parametrize(
    "dtype",
    [torch.torch.float8_e4m3fn, torch.float32, torch.float16, torch.bfloat16])
def test_batched_mm(num_experts: int, max_tokens_per_expert: int, K: int,
                    N: int, dtype: torch.dtype):

    config = BatchedMMConfig(dtype, num_experts, max_tokens_per_expert, K, N)
    tensors = BatchedMMTensors.make_tensors(config)

    test_output = tensors.C
    ref_output = test_output.clone()

    compute_tl_dtype = {
        torch.torch.float8_e4m3fn: tl.bfloat16,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32
    }[test_output.dtype]

    use_fp8_w8a8 = dtype == torch.torch.float8_e4m3fn
    block_shape = [16, 16, 32] # 16 for k if not fp8

    print(f"tensors.A {tensors.A.shape}")
    print(f"tensors.B {tensors.B.shape}")

    if use_fp8_w8a8:
        A_scale = torch.ones((max_tokens_per_expert,K), dtype=torch.float32, device=tensors.A.device)
        B_scale = torch.ones((N, K), dtype=torch.float32, device=tensors.A.device)
    else:
        A_scale = None
        B_scale = None

    invoke_moe_batched_triton_kernel(
        tensors.A,
        tensors.B,
        test_output,
        tensors.num_expert_tokens,
        compute_tl_dtype,
        # Quantization data
        A_scale,
        B_scale,
        None,
        # Quantization schemes
        use_fp8_w8a8,
        False,
        False,
        config={
            "BLOCK_SIZE_M": block_shape[0],
            "BLOCK_SIZE_N": block_shape[1],
            "BLOCK_SIZE_K": block_shape[2],
        })

    ref_output = ref_impl(tensors.A,
                          tensors.B,
                          ref_output,
                          tensors.num_expert_tokens,
                          A_scale,
                          B_scale,
                          block_shape[-2:])

    rtol, atol = {
        torch.torch.float8_e4m3fn: (6e-2, 6e-2),
        torch.float16: (6e-2, 6e-2),
        torch.bfloat16: (6e-2, 6e-2),
        torch.float32: (1e-2, 1e-2),
    }[test_output.dtype]

    torch.testing.assert_close(test_output, ref_output, atol=atol, rtol=rtol)
