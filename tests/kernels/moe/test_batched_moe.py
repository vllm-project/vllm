# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import triton.language as tl

import vllm._custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedPrepareAndFinalize, BatchedTritonExperts,
    invoke_moe_batched_triton_kernel)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.platforms import current_platform
from vllm.utils import round_up

NUM_EXPERTS = [8, 64]
TOP_KS = [1, 2, 6]

vllm_config = VllmConfig()
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192


@dataclass
class BatchedMMConfig:
    in_dtype: torch.dtype
    out_dtype: torch.dtype
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
        if config.in_dtype == torch.torch.float8_e4m3fn:
            config_in_dtype = torch.bfloat16
        else:
            config_in_dtype = config.in_dtype

        A = torch.randn(
            (config.num_experts, config.max_tokens_per_expert, config.K),
            device="cuda",
            dtype=config_in_dtype) / 10
        B = torch.randn((config.num_experts, config.N, config.K),
                        device="cuda",
                        dtype=config_in_dtype)
        C = torch.zeros(
            (config.num_experts, config.max_tokens_per_expert, config.N),
            device="cuda",
            dtype=config.out_dtype)

        A = A.to(config.in_dtype)
        B = B.to(config.in_dtype)

        num_expert_tokens = torch.randint(low=0,
                                          high=config.max_tokens_per_expert,
                                          size=(config.num_experts, ),
                                          device="cuda",
                                          dtype=torch.int32)
        return BatchedMMTensors(A, B, C, num_expert_tokens)


def native_w8a8_block_matmul(A: torch.Tensor,
                             B: torch.Tensor,
                             As: torch.Tensor,
                             Bs: torch.Tensor,
                             block_size,
                             output_dtype=torch.bfloat16):
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
            if False:
                tmp = native_w8a8_block_matmul(A[e, :, :],
                                               B[e].transpose(0, 1), A_scale,
                                               B_scale, block_shape)
            else:
                tmp = ops.cutlass_scaled_mm(A[e, :, :], B[e].transpose(0, 1),
                                            A_scale, B_scale, torch.bfloat16)
            C[e, :num_tokens, :] = tmp[:num_tokens, :]
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

    if dtype == torch.torch.float8_e4m3fn:
        in_dtype = dtype
        out_dtype = torch.bfloat16
    else:
        in_dtype = dtype
        out_dtype = dtype

    config = BatchedMMConfig(in_dtype, out_dtype, num_experts,
                             max_tokens_per_expert, K, N)
    tensors = BatchedMMTensors.make_tensors(config)

    test_output = tensors.C
    ref_output = test_output.clone()
    ref_output2 = test_output.clone()

    compute_tl_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32
    }[test_output.dtype]

    use_fp8_w8a8 = dtype == torch.torch.float8_e4m3fn
    block_shape = [16, 16, 32]  # 16 for k if not fp8

    if use_fp8_w8a8:
        A_scale = torch.ones(1, dtype=torch.float32, device=tensors.A.device)
        B_scale = torch.ones(1, dtype=torch.float32, device=tensors.B.device)
        quant_block_shape = [1, 1]
    else:
        A_scale = None
        B_scale = None
        quant_block_shape = None

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
        },
        block_shape=quant_block_shape,
    )

    ref_output = ref_output.to(dtype=out_dtype)
    ref_output = ref_impl(tensors.A.to(dtype=out_dtype),
                          tensors.B.to(dtype=out_dtype), ref_output,
                          tensors.num_expert_tokens, A_scale, B_scale,
                          block_shape[-2:])

    ref_output2 = ref_impl(tensors.A, tensors.B, ref_output2,
                           tensors.num_expert_tokens, A_scale, B_scale,
                           block_shape[-2:])

    rtol, atol = {
        torch.float16: (6e-2, 6e-2),
        torch.bfloat16: (6e-2, 6e-2),
        torch.float32: (1e-2, 1e-2),
    }[test_output.dtype]

    torch.testing.assert_close(ref_output, ref_output2, atol=atol, rtol=rtol)
    torch.testing.assert_close(test_output, ref_output2, atol=atol, rtol=rtol)


def batched_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    qtype: Optional[torch.dtype] = None,
    block_shape: Optional[list[int]] = None,
    per_act_token: bool = False,
) -> torch.Tensor:
    max_num_tokens = round_up(a.shape[0], 64)  # ?
    fused_experts = FusedMoEModularKernel(
        BatchedPrepareAndFinalize(max_num_tokens,
                                  world_size=1,
                                  dp_size=1,
                                  rank=0,
                                  qtype=qtype,
                                  block_shape=block_shape,
                                  per_act_token=False),
        BatchedTritonExperts(max_num_tokens=max_num_tokens,
                             dp_size=1,
                             world_size=1,
                             use_fp8_w8a8=qtype == torch.float8_e4m3fn,
                             block_shape=block_shape))

    return fused_experts(a,
                         w1,
                         w2,
                         topk_weight,
                         topk_ids,
                         w1_scale=w1_scale,
                         w2_scale=w2_scale)


# Note: same as torch_moe but with fused_topk factored out.
def torch_moe2(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    M, K = a.shape
    topk = topk_ids.shape[1]

    a = a.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K)

    if use_fp8_w8a8:
        a, a_scale = per_token_group_quant_fp8(a, block_shape[1])
    else:
        a_scale = None

    out = torch.zeros(M * topk,
                      w2.shape[1],
                      dtype=torch.bfloat16,
                      device=a.device)
    num_experts = w1.shape[0]
    for i in range(num_experts):
        mask = (topk_ids == i).view(-1)
        if mask.sum():
            if not use_fp8_w8a8:
                tmp1 = a[mask] @ w1[i].transpose(0, 1)
                tmp2 = SiluAndMul()(tmp1)
                out[mask] = tmp2 @ w2[i].transpose(0, 1)
            else:
                #tmp1 = ops.cutlass_scaled_mm(a[mask],
                #                             w1[i].transpose(0, 1),
                #                             a_scale[mask],
                #                             w1_scale[i],
                #                             torch.bfloat16)
                tmp1 = native_w8a8_block_matmul(a[mask], w1[i], a_scale[mask],
                                                w1_scale[i], block_shape,
                                                torch.bfloat16)
                tmp2 = SiluAndMul()(tmp1)
                tmp2, b_scale = per_token_group_quant_fp8(tmp2, block_shape[1])

                # out[mask] = ops.cutlass_scaled_mm(tmp2,
                #                                   w2[i].transpose(0, 1),
                #                                   b_scale,
                #                                   w2_scale[i],
                #                                   torch.bfloat16)
                out[mask] = native_w8a8_block_matmul(tmp2, w2[i], b_scale,
                                                     w2_scale[i], block_shape,
                                                     torch.bfloat16)

    return (out.view(M, -1, w2.shape[1]) *
            topk_weight.view(M, -1, 1).to(out.dtype)).sum(dim=1)


@pytest.mark.parametrize("m", [1, 33, 64, 222])
@pytest.mark.parametrize("n", [128, 1024, 2048])
@pytest.mark.parametrize("k", [128, 512, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_fused_moe_batched_experts(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    current_platform.seed_everything(7)
    block_shape = [128, 128]

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=torch.bfloat16) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=torch.bfloat16) / 10
    score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)

    use_fp8_w8a8 = dtype == torch.torch.float8_e4m3fn
    qtype = dtype if dtype == torch.torch.float8_e4m3fn else None

    if use_fp8_w8a8:
        block_n, block_k = block_shape[0], block_shape[1]
        n_tiles_w1 = (2 * n + block_n - 1) // block_n
        n_tiles_w2 = (k + block_n - 1) // block_n
        k_tiles_w1 = (k + block_k - 1) // block_k
        k_tiles_w2 = (n + block_k - 1) // block_k

        finfo = torch.finfo(dtype)
        fp8_min = finfo.min
        fp8_max = finfo.max

        w1 = w1.clamp(min=fp8_min, max=fp8_max).to(dtype)
        w2 = w2.clamp(min=fp8_min, max=fp8_max).to(dtype)

        factor_for_scale = 1e-2
        w1_s = torch.rand(
            (e, n_tiles_w1, k_tiles_w1), dtype=torch.float32,
            device="cuda") * factor_for_scale
        w2_s = torch.rand(
            (e, n_tiles_w2, k_tiles_w2), dtype=torch.float32,
            device="cuda") * factor_for_scale
    else:
        w1_s = None
        w2_s = None

    with set_current_vllm_config(vllm_config):
        topk_weight, topk_ids, _ = fused_topk(a, score, topk, False)
        baseline_output = torch_moe2(a, w1, w2, topk_weight, topk_ids, w1_s,
                                     w2_s, use_fp8_w8a8, block_shape)
        batched_output = batched_moe(a, w1, w2, topk_weight, topk_ids, w1_s,
                                     w2_s, qtype, block_shape)

    torch.testing.assert_close(baseline_output,
                               batched_output,
                               atol=2e-2,
                               rtol=0)
