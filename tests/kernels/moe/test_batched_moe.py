# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import triton.language as tl

from tests.kernels.moe.utils import (
    batched_moe,
    make_test_weights,
    make_quantized_test_activations,
    torch_moe2,
    triton_moe)
from tests.kernels.quant_utils import native_w8a8_block_matmul
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    invoke_moe_batched_triton_kernel)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.platforms import current_platform

NUM_EXPERTS = [8, 64]
TOP_KS = [1, 2, 6]

vllm_config = VllmConfig()
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192


@dataclass
class BatchedMMConfig:
    in_dtype: torch.dtype
    quant_dtype: Optional[torch.dtype]
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
        A = torch.randn(
            (config.num_experts, config.max_tokens_per_expert, config.K),
            device="cuda",
            dtype=config.in_dtype) / 10
        B = torch.randn((config.num_experts, config.N, config.K),
                        device="cuda",
                        dtype=config.in_dtype)
        C = torch.zeros(
            (config.num_experts, config.max_tokens_per_expert, config.N),
            device="cuda",
            dtype=config.out_dtype)

        num_expert_tokens = torch.randint(low=0,
                                          high=config.max_tokens_per_expert,
                                          size=(config.num_experts, ),
                                          device="cuda",
                                          dtype=torch.int32)



        return BatchedMMTensors(A, B, C, num_expert_tokens)


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

    f32 = torch.float32
    bf16 = torch.bfloat16

    for e in range(num_experts):
        num_tokens = num_expert_tokens_cpu[e]
        if A.dtype.itemsize == 1 and block_shape is not None:
            tmp = native_w8a8_block_matmul(A[e], B[e], A_scale[e], B_scale[e],
                                           block_shape, C.dtype)
            C[e, :num_tokens, :] = tmp[:num_tokens, :]
        elif A.dtype.itemsize == 1 and block_shape is None:
            C[e, :num_tokens, :] = (
                (A[e, :num_tokens, :].to(f32) * A_scale[e]).to(bf16)
                @ (B[e].transpose(0, 1).to(f32) * B_scale[e]).to(bf16))
        else:
            assert A_scale is None
            assert B_scale is None
            C[e, :num_tokens, :] = A[e, :num_tokens, :] @ B[e].transpose(0, 1)

    return C


@pytest.mark.parametrize("num_experts", [8, 16, 32])
@pytest.mark.parametrize("max_tokens_per_expert",
                         [32, 64, 128, 192, 224, 256, 512])
@pytest.mark.parametrize("K", [128, 256, 1024])
@pytest.mark.parametrize("N", [128, 256, 512, 1024])
@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_shape", [None])
@pytest.mark.parametrize("per_act_token_quant", [False])
def test_batched_mm(num_experts: int, max_tokens_per_expert: int, K: int,
                    N: int, dtype: torch.dtype, block_shape: Optional[list[int]],
                    per_act_token_quant: bool):
    current_platform.seed_everything(7)

    use_fp8_w8a8 = dtype == torch.float8_e4m3fn

    if block_shape is not None and not use_fp8_w8a8:
        pytest.skip("Don't test blocking for non-quantized types.")

    if dtype.itemsize == 1:
        act_dtype = torch.bfloat16
        quant_dtype = dtype
    else:
        act_dtype = dtype
        quant_dtype = None

    #print(f"TYPES {dtype}, {act_dtype}, {quant_dtype}")

    num_expert_tokens = torch.randint(low=0,
                                      high=max_tokens_per_expert,
                                      size=(num_experts, ),
                                      device="cuda",
                                      dtype=torch.int32)

    A, A_q, A_scale = make_quantized_test_activations(
        num_experts,
        max_tokens_per_expert,
        K,
        in_dtype=act_dtype,
        quant_dtype=quant_dtype,
        block_shape=block_shape,
        per_act_token_quant=per_act_token_quant
    )

    B, B_q, B_scale, _, _, _ = make_test_weights(
        num_experts,
        N // 2,
        K,
        in_dtype=act_dtype,
        quant_dtype=quant_dtype,
        block_shape=block_shape,
    )

    out_shape = (num_experts, max_tokens_per_expert, N)
    test_output = torch.zeros(out_shape, dtype=act_dtype, device="cuda")
    ref_output = torch.zeros(out_shape, dtype=act_dtype, device="cuda")
    q_ref_output = torch.zeros(out_shape, dtype=act_dtype, device="cuda")

    compute_tl_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32
    }[test_output.dtype]

    assert A_q.dtype == B_q.dtype

    invoke_moe_batched_triton_kernel(
        A_q,
        B_q,
        test_output,
        num_expert_tokens,
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
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 16 if dtype.itemsize > 1 else 32
        },
        block_shape=block_shape,
    )

    ref_output = ref_impl(
        A,
        B,
        ref_output,
        num_expert_tokens,
        None,
        None,
        None,
    )

    q_ref_output = ref_impl(A_q, B_q, q_ref_output, num_expert_tokens, A_scale,
                            B_scale, block_shape)

    rtol, atol = {
        torch.float16: (6e-2, 6e-2),
        torch.bfloat16: (6e-2, 6e-2),
        torch.float32: (1e-2, 1e-2),
    }[test_output.dtype]

    torch.testing.assert_close(ref_output, test_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(test_output, q_ref_output, atol=atol, rtol=rtol)


@pytest.mark.parametrize("m", [1, 32, 45, 64, 222])
@pytest.mark.parametrize("n", [128, 512, 1024, 2048])
@pytest.mark.parametrize("k", [128, 512, 1024, 2048])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("per_act_token_quant", [False])
@pytest.mark.parametrize("block_shape", [None])
def test_fused_moe_batched_experts(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    per_act_token_quant: bool,
    block_shape: Optional[list[int]],
):
    current_platform.seed_everything(7)

    use_fp8_w8a8 = dtype == torch.float8_e4m3fn

    if not use_fp8_w8a8 and per_act_token_quant and block_shape is not None:
        pytest.skip("Skip quantization test for non-quantized type")

    if per_act_token_quant and block_shape is not None or topk > e:
        pytest.skip("Skip illegal quantization test")

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
    score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)

    if dtype.itemsize == 1:
        act_dtype = torch.bfloat16
        quant_dtype = dtype
    else:
        act_dtype = dtype
        quant_dtype = None

    _, w1, w1_s, _, w2, w2_s = make_test_weights(e, n, k, block_shape=block_shape,
                                                 in_dtype=act_dtype,
                                                 quant_dtype=quant_dtype)

    torch.set_printoptions(profile="full")

    with set_current_vllm_config(vllm_config):
        topk_weight, topk_ids, _ = fused_topk(a, score, topk, False)
        batched_output = batched_moe(a, w1, w2, topk_weight, topk_ids, w1_s,
                                     w2_s, quant_dtype, per_act_token_quant,
                                     block_shape)
        baseline_output = torch_moe2(a, w1, w2, topk_weight, topk_ids, w1_s,
                                     w2_s, quant_dtype, per_act_token_quant,
                                     block_shape)
        triton_output = triton_moe(a, w1, w2, topk_weight, topk_ids, w1_s,
                                   w2_s, quant_dtype, per_act_token_quant,
                                   block_shape)

    torch.testing.assert_close(triton_output,
                               baseline_output,
                               atol=2e-2,
                               rtol=2e-2)

    torch.testing.assert_close(triton_output,
                               batched_output,
                               atol=2e-2,
                               rtol=2e-2)
