# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from tests.kernels.moe.utils import (batched_moe,
                                     make_quantized_test_activations,
                                     make_test_weights, naive_batched_moe)
from tests.kernels.quant_utils import native_batched_masked_quant_matmul
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    invoke_moe_batched_triton_kernel)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.platforms import current_platform
from vllm.triton_utils import tl

MNK_FACTORS = [
    (1, 128, 128),
    (1, 128, 2048),
    (1, 512, 512),
    (1, 1024, 128),
    (1, 1024, 2048),
    (32, 128, 128),
    (32, 512, 512),
    (32, 1024, 2048),
    (45, 128, 128),
    (45, 128, 2048),
    (45, 512, 512),
    (45, 1024, 128),
    (45, 1024, 2048),
    (64, 512, 512),
    (64, 1024, 2048),
    (222, 128, 128),
    (222, 128, 2048),
    (222, 1024, 128),
    (222, 1024, 2048),
]
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


@pytest.mark.parametrize("num_experts", [8, 16, 32])
@pytest.mark.parametrize("max_tokens_per_expert",
                         [32, 64, 128, 192, 224, 256, 512])
@pytest.mark.parametrize("K", [128, 256, 1024])
@pytest.mark.parametrize("N", [128, 256, 1024])
@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_shape", [None, [128, 128]])
@pytest.mark.parametrize("per_act_token_quant", [False, True])
def test_batched_mm(num_experts: int, max_tokens_per_expert: int, K: int,
                    N: int, dtype: torch.dtype,
                    block_shape: Optional[list[int]],
                    per_act_token_quant: bool):
    current_platform.seed_everything(7)

    use_fp8_w8a8 = dtype == torch.float8_e4m3fn

    if (per_act_token_quant or block_shape is not None) and not use_fp8_w8a8:
        pytest.skip("Don't test blocking for non-quantized types.")

    if per_act_token_quant and block_shape is not None:
        pytest.skip("Skip illegal quantization test.")

    if dtype.itemsize == 1:
        act_dtype = torch.bfloat16
        quant_dtype = dtype
    else:
        act_dtype = dtype
        quant_dtype = None

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
        per_act_token_quant=per_act_token_quant,
    )

    B, B_q, B_scale, _, _, _ = make_test_weights(
        num_experts,
        N // 2,
        K,
        in_dtype=act_dtype,
        quant_dtype=quant_dtype,
        block_shape=block_shape,
        per_act_token_quant=per_act_token_quant,
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
        per_act_token_quant=per_act_token_quant,
        block_shape=block_shape,
    )

    ref_output = native_batched_masked_quant_matmul(
        A,
        B,
        ref_output,
        num_expert_tokens,
    )

    q_ref_output = native_batched_masked_quant_matmul(A_q, B_q, q_ref_output,
                                                      num_expert_tokens,
                                                      A_scale, B_scale,
                                                      block_shape,
                                                      per_act_token_quant)

    rtol, atol = {
        torch.float16: (6e-2, 6e-2),
        torch.bfloat16: (6e-2, 6e-2),
        torch.float32: (1e-2, 1e-2),
    }[test_output.dtype]

    torch.testing.assert_close(ref_output, q_ref_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(test_output, q_ref_output, atol=atol, rtol=rtol)


@pytest.mark.parametrize(("m", "n", "k"), MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("per_act_token_quant", [False, True])
@pytest.mark.parametrize("block_shape", [None, [128, 128]])
@pytest.mark.parametrize("input_scales", [False])
def test_fused_moe_batched_experts(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    per_act_token_quant: bool,
    block_shape: Optional[list[int]],
    input_scales: bool,
):
    current_platform.seed_everything(7)

    use_fp8_w8a8 = dtype == torch.float8_e4m3fn

    if topk > e:
        pytest.skip("topk > e")

    if not use_fp8_w8a8 and (per_act_token_quant or block_shape is not None):
        pytest.skip("Skip quantization test for non-quantized type")

    if per_act_token_quant and block_shape is not None:
        pytest.skip("Skip illegal quantization test.")

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
    score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)

    if dtype.itemsize == 1:
        act_dtype = torch.bfloat16
        quant_dtype = dtype
    else:
        act_dtype = dtype
        quant_dtype = None

    w1_16, w1, w1_s, w2_16, w2, w2_s = make_test_weights(
        e,
        n,
        k,
        block_shape=block_shape,
        in_dtype=act_dtype,
        quant_dtype=quant_dtype,
        per_act_token_quant=per_act_token_quant,
    )

    if input_scales and quant_dtype is not None:
        a1_scale = torch.tensor(1, device="cuda", dtype=torch.float32)
        a2_scale = torch.tensor(1, device="cuda", dtype=torch.float32)
    else:
        a1_scale = None
        a2_scale = None

    with set_current_vllm_config(vllm_config):
        topk_weight, topk_ids, _ = fused_topk(a, score, topk, False)

        baseline_output = torch_experts(
            a,
            w1,
            w2,
            topk_weight,
            topk_ids,
            w1_scale=w1_s,
            w2_scale=w2_s,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_act_token_quant,
            block_shape=block_shape,
        )

        batched_output = naive_batched_moe(
            a,
            w1,
            w2,
            topk_weight,
            topk_ids,
            w1_scale=w1_s,
            w2_scale=w2_s,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_act_token_quant,
            block_shape=block_shape,
        )

        triton_output = batched_moe(
            a,
            w1,
            w2,
            topk_weight,
            topk_ids,
            w1_scale=w1_s,
            w2_scale=w2_s,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_act_token_quant,
            block_shape=block_shape,
        )

    torch.testing.assert_close(batched_output,
                               baseline_output,
                               atol=3e-2,
                               rtol=2e-2)

    torch.testing.assert_close(triton_output,
                               batched_output,
                               atol=2e-2,
                               rtol=2e-2)
