# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import triton.language as tl

import vllm._custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedPrepareAndFinalize, BatchedTritonExperts,
    invoke_moe_batched_triton_kernel)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.platforms import current_platform
from tests.kernels.moe.utils import (
    native_w8a8_block_matmul,
    torch_moe2,
    triton_moe,
    batched_moe,
)

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


@pytest.mark.parametrize("m", [32, 45, 64])  #[1, 33, 64, 222])
@pytest.mark.parametrize("n", [128, 512, 1024, 2048])
@pytest.mark.parametrize("k", [128, 512, 1024, 2048])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
#@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
#@pytest.mark.parametrize("per_act_token_quant", [False, True])
#@pytest.mark.parametrize("block_shape", [None, [128, 128]])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("per_act_token_quant", [False, True])
@pytest.mark.parametrize("block_shape", [[128, 128]])
#@pytest.mark.parametrize("per_act_token_quant", [False])
#@pytest.mark.parametrize("block_shape", [[128, 128]])
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

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
    w1_16 = torch.randn((e, 2 * n, k), device="cuda", dtype=torch.bfloat16) / 15
    w2_16 = torch.randn((e, k, n), device="cuda", dtype=torch.bfloat16) / 15
    score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)

    use_fp8_w8a8 = dtype == torch.torch.float8_e4m3fn

    if not use_fp8_w8a8 and per_act_token_quant and block_shape is not None:
        pytest.skip("Skip quantization test for non-quantized type")

    if per_act_token_quant and block_shape is not None:
        pytest.skip("Skip illegal quantization test")

    # TODO (bnell): scale setup for different quant strategies?
    if use_fp8_w8a8:
        quant_type = torch.float8_e4m3fn

        #finfo = torch.finfo(dtype)
        #fp8_min = finfo.min
        #fp8_max = finfo.max
        #w1 = w1.clamp(min=fp8_min, max=fp8_max).to(dtype)
        #w2 = w2.clamp(min=fp8_min, max=fp8_max).to(dtype)
        # block_n, block_k = block_shape[0], block_shape[1]
        # n_tiles_w1 = (2 * n + block_n - 1) // block_n
        # k_tiles_w1 = (k + block_k - 1) // block_k
        #
        # n_tiles_w2 = (k + block_n - 1) // block_n
        # k_tiles_w2 = (n + block_k - 1) // block_k
        # factor_for_scale = 1e-2
        # w1_s = torch.rand(
        #     (e, n_tiles_w1, k_tiles_w1), dtype=torch.float32,
        #     device="cuda") * factor_for_scale
        # w2_s = torch.rand(
        #     (e, n_tiles_w2, k_tiles_w2), dtype=torch.float32,
        #     device="cuda") * factor_for_scale
        w1_l = [None] * e
        w2_l = [None] * e
        w1_s = [None] * e
        w2_s = [None] * e
        for idx in range(e):
            w1_l[idx], w1_s[idx] = moe_kernel_quantize_input(
                w1_16[idx],
                None,
                quant_type,
                per_act_token_quant,
                block_shape
            )
            #print(w1_s[idx].shape)
            w2_l[idx], w2_s[idx] = moe_kernel_quantize_input(
                w2_16[idx],
                None,
                quant_type,
                per_act_token_quant,
                block_shape,
            )
        w1 = torch.stack(w1_l)
        w2 = torch.stack(w2_l)
        w1_s = torch.stack(w1_s)
        w2_s = torch.stack(w2_s)
        if w1_s.ndim == 2:
            assert w1_s.shape[-1] == 1
            w1_s = w1_s.view(-1, 1, 1)
            w2_s = w2_s.view(-1, 1, 1)

        #block_n, block_k = block_shape[0], block_shape[1]
        #n_tiles_w1 = (2 * n + block_n - 1) // block_n
        #n_tiles_w2 = (k + block_n - 1) // block_n
        #k_tiles_w1 = (k + block_k - 1) // block_k
        #k_tiles_w2 = (n + block_k - 1) // block_k
        #print(f"BLOCK_SHAPE {block_shape}")
        #print(f"w1_s = {w1_s.shape} {(e, n_tiles_w1, k_tiles_w1)}")
        #print(f"w2_s = {w2_s.shape} {(e, n_tiles_w2, k_tiles_w2)}")
    else:
        quant_type = None
        w1 = w1_16
        w2 = w2_16
        w1_s = None
        w2_s = None

    with set_current_vllm_config(vllm_config):
        topk_weight, topk_ids, _ = fused_topk(a, score, topk, False)
        batched_output = batched_moe(a, w1, w2, topk_weight, topk_ids, w1_s,
                                     w2_s, quant_type, per_act_token_quant, block_shape)
        baseline_output = torch_moe2(a, w1, w2, topk_weight, topk_ids, w1_s,
                                     w2_s, quant_type, per_act_token_quant, block_shape)
        triton_output = triton_moe(a, w1, w2, topk_weight, topk_ids, w1_s,
                                   w2_s, quant_type, per_act_token_quant, block_shape)

    torch.testing.assert_close(triton_output,
                               baseline_output,
                               atol=2e-2,
                               rtol=0)

    torch.testing.assert_close(baseline_output,
                               batched_output,
                               atol=3e-2,
                               rtol=0)
