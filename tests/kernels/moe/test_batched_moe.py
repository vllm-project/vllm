# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import triton.language as tl

from tests.kernels.moe.utils import (batched_moe, make_test_weights,
                                     per_block_cast_to_fp8, torch_moe2,
                                     triton_moe)
from tests.kernels.quant_utils import native_w8a8_block_matmul
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    invoke_moe_batched_triton_kernel)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.platforms import current_platform

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
        if config.in_dtype == torch.float8_e4m3fn:
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


def make_quantized_test_activations(E, m, k, dtype, block_shape,
                                    per_act_token):
    assert not per_act_token, "NYI"

    a_type = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype

    a = torch.randn((E, m, k), device="cuda", dtype=a_type) / 10
    a_q = a
    a_scale = None

    if dtype == torch.float8_e4m3fn:
        a_q = torch.zeros_like(a, dtype=dtype)
        a_scale = [None] * E
        for e in range(E):
            if block_shape is not None:
                a_q[e], a_scale[e] = per_token_group_quant_fp8(
                    a[e], block_shape[1])
            else:
                a_tmp, a_scale[e] = per_token_group_quant_fp8(
                    a[e].view(1, -1), a[e].numel())
                a_q[e] = a_tmp.view(*a[e].shape)
        a_scale = torch.stack(a_scale)

    return a, a_q, a_scale


@pytest.mark.parametrize("num_experts", [8, 16, 32])
@pytest.mark.parametrize("max_tokens_per_expert",
                         [32, 64, 128, 192, 224, 256, 512])
@pytest.mark.parametrize("K", [128, 256, 1024])
@pytest.mark.parametrize("N", [128, 256, 512, 1024])
@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_shape", [None, [128, 128]])
def test_batched_mm(num_experts: int, max_tokens_per_expert: int, K: int,
                    N: int, dtype: torch.dtype, block_shape: list[int]):
    current_platform.seed_everything(7)

    use_fp8_w8a8 = dtype == torch.float8_e4m3fn

    per_act_token_quant = False

    if block_shape is not None and not use_fp8_w8a8:
        pytest.skip("Don't test blocking for non-quantized types.")

    num_expert_tokens = torch.randint(low=0,
                                      high=max_tokens_per_expert,
                                      size=(num_experts, ),
                                      device="cuda",
                                      dtype=torch.int32)

    A, A_q, A_scale = make_quantized_test_activations(num_experts,
                                                      max_tokens_per_expert, K,
                                                      dtype, block_shape,
                                                      per_act_token_quant)

    B_q, _, B_scale, _, B, _ = make_test_weights(
        num_experts,
        N // 2,
        K,
        block_shape,
        dtype,
    )

    out_dtype = torch.bfloat16 if use_fp8_w8a8 else dtype
    out_shape = (num_experts, max_tokens_per_expert, N)
    test_output = torch.zeros(out_shape, dtype=out_dtype, device="cuda")
    ref_output = torch.zeros(out_shape, dtype=out_dtype, device="cuda")
    q_ref_output = torch.zeros(out_shape, dtype=out_dtype, device="cuda")

    compute_tl_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32
    }[test_output.dtype]

    config_block_shape = [16, 16, 32]  # 16 for k if not fp8

    #print(f"A {use_fp8_w8a8} {A_q.dtype} {B_q.dtype} {A_scale.shape} {B_scale.shape}")
    if False:
        from vllm.model_executor.layers.fused_moe.batched_moe2 import (
            fused_moe_kernel2)
        fused_moe_kernel2(
            A_q,
            B_q,
            test_output,
            A_scale,
            B_scale,
            num_expert_tokens,
            N,
            K,
            max_tokens_per_expert,
            max_tokens_per_expert,
            A_q.stride(0),
            A_q.stride(1),
            A_q.stride(2),
            B_q.stride(0),
            B_q.stride(1),
            B_q.stride(2),
            test_output.stride(0),
            test_output.stride(1),
            A_scale.stride(0),
            A_scale.stride(1),
            A_scale.stride(2),
            B_scale.stride(0),
            B_scale.stride(1),
            B_scale.stride(2),
            block_shape[0] if block_shape is not None else 0,
            block_shape[1] if block_shape is not None else 0,
            config_block_shape[0],
            config_block_shape[1],
            config_block_shape[2],
            1,
            1,  # topk hack
            compute_tl_dtype,
            use_fp8_w8a8,
            False,
            False,
            per_channel_quant=False,
        )
    else:
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
                "BLOCK_SIZE_M": config_block_shape[0],
                "BLOCK_SIZE_N": config_block_shape[1],
                "BLOCK_SIZE_K": config_block_shape[2],
            },
            per_act_token_quant=False,
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

    torch.set_printoptions(profile="full")
    #print(f"REF {q_ref_output.shape}\n{q_ref_output}")
    #print(f"TRI {test_output.shape}\n{test_output}")

    torch.testing.assert_close(ref_output, q_ref_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(test_output, q_ref_output, atol=atol, rtol=rtol)


def _make_test_weights(
    e: int,
    n: int,
    k: int,
    block_size: list[int],
    dtype=torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return weights w1, w2, w1q, w2q, w1_scale, w2_scale
    """
    dtype = torch.bfloat16

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    w1_bf16 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w1_bf16 = w1_bf16.clamp(min=fp8_min, max=fp8_max).to(dtype=dtype)

    w2_bf16 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    w2_bf16 = w2_bf16.clamp(min=fp8_min, max=fp8_max).to(dtype=dtype)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles_w1 = ((2 * n) + block_n - 1) // block_n
    k_tiles_w1 = (k + block_k - 1) // block_k
    n_tiles_w2 = (k + block_n - 1) // block_n
    k_tiles_w2 = (n + block_k - 1) // block_k

    w1 = torch.empty_like(w1_bf16, dtype=torch.float8_e4m3fn)
    w2 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn)

    w1_s = torch.empty((e, n_tiles_w1, k_tiles_w1),
                       device="cuda",
                       dtype=torch.float32)
    w2_s = torch.empty((e, n_tiles_w2, k_tiles_w2),
                       device="cuda",
                       dtype=torch.float32)

    assert w1_s.shape == (e, (2 * n + 127) // 128, (k + 127) // 128)
    assert (w2.shape[-2] + block_n - 1) // block_n == w2_s.shape[-2]

    for i in range(e):
        w1[i], w1_s[i] = per_block_cast_to_fp8(w1_bf16[i])
        w2[i], w2_s[i] = per_block_cast_to_fp8(w2_bf16[i])

    return w1, w2, w1_s, w2_s, w1_bf16, w2_bf16


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

    use_fp8_w8a8 = dtype == torch.float8_e4m3fn
    quant_type = torch.float8_e4m3fn if use_fp8_w8a8 else None

    if not use_fp8_w8a8 and per_act_token_quant and block_shape is not None:
        pytest.skip("Skip quantization test for non-quantized type")

    if per_act_token_quant and block_shape is not None or topk > e:
        pytest.skip("Skip illegal quantization test")

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
    score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
    w1, w2, w1_s, w2_s, _, _ = make_test_weights(e, n, k, block_shape, dtype)

    torch.set_printoptions(profile="full")

    with set_current_vllm_config(vllm_config):
        topk_weight, topk_ids, _ = fused_topk(a, score, topk, False)
        batched_output = batched_moe(a, w1, w2, topk_weight, topk_ids, w1_s,
                                     w2_s, quant_type, per_act_token_quant,
                                     block_shape)
        baseline_output = torch_moe2(a, w1, w2, topk_weight, topk_ids, w1_s,
                                     w2_s, quant_type, per_act_token_quant,
                                     block_shape)
        triton_output = triton_moe(a, w1, w2, topk_weight, topk_ids, w1_s,
                                   w2_s, quant_type, per_act_token_quant,
                                   block_shape)

    torch.testing.assert_close(triton_output,
                               baseline_output,
                               atol=2e-2,
                               rtol=2e-2)

    #print(f"TORCH {baseline_output.shape}\n{baseline_output}")
    #print(f"TRITON {triton_output.shape}\n{triton_output}")
    #print(f"BATCHED {batched_output.shape}\n{batched_output}")

    torch.testing.assert_close(triton_output,
                               batched_output,
                               atol=2e-2,
                               rtol=2e-2)  # 0
