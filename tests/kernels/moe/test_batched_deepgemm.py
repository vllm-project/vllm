# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedPrepareAndFinalize,
    BatchedTritonExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEModularKernel
from vllm.utils.deep_gemm import calc_diff, is_deep_gemm_supported

from .test_deepgemm import make_block_quant_fp8_weights

BLOCK_SIZE = [128, 128]


@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
@pytest.mark.parametrize("E", [16, 32])  # number of experts
@pytest.mark.parametrize("T", [256, 512])  # tokens per expert
@pytest.mark.parametrize("K", [128, 256])  # hidden dim
@pytest.mark.parametrize("N", [512, 1024])  # intermediate dim per expert
@pytest.mark.parametrize("topk", [2, 4])
def test_batched_deepgemm_vs_triton(
    E: int, T: int, K: int, N: int, topk: int, monkeypatch, workspace_init
):
    """Compare BatchedDeepGemmExperts to BatchedTritonExperts."""

    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")

    device = "cuda"
    w1, w2, w1_s, w2_s = make_block_quant_fp8_weights(E, N, K, BLOCK_SIZE)

    M = E * T  # total tokens
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16) / 10.0
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    a.clamp_(fp8_info.min, fp8_info.max)

    # random router outputs → top-k indices / weights
    router_logits = torch.randn(M, E, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    # token number for each expert
    cnt = torch.bincount(topk_ids.flatten(), minlength=E)
    max_cnt = int(cnt.max().item())
    # next power of 2 for max token number
    max_num_tokens = 1 << (max_cnt - 1).bit_length()

    prep_finalize = BatchedPrepareAndFinalize(
        max_num_tokens=max_num_tokens,
        num_local_experts=E,
        num_dispatchers=1,
        rank=0,
    )

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_s,
        w2_scale=w2_s,
        per_act_token_quant=False,
        block_shape=BLOCK_SIZE,
    )

    # triton (reference)
    triton_experts = BatchedTritonExperts(
        max_num_tokens=max_num_tokens,
        num_dispatchers=1,
        quant_config=quant_config,
    )
    mk_triton = FusedMoEModularKernel(prep_finalize, triton_experts)

    out_triton = mk_triton(
        hidden_states=a,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        global_num_experts=E,
    )

    # deepgemm
    deepgemm_experts = BatchedDeepGemmExperts(
        max_num_tokens=max_num_tokens,
        num_dispatchers=1,
        quant_config=quant_config,
    )
    mk_deepgemm = FusedMoEModularKernel(prep_finalize, deepgemm_experts)

    out_deepgemm = mk_deepgemm(
        hidden_states=a,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        global_num_experts=E,
    )

    diff = calc_diff(out_deepgemm, out_triton)
    assert diff < 1e-3, f"Output diff too large: {diff}"
