# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit-test DeepGEMM FP8 kernels (no DeepEP).
Compare DeepGEMM path against the Triton fallback inside vLLM's fused_experts.
"""

import importlib
import math

import pytest
import torch

# vLLM fused-expert reference (Triton fallback + DeepGEMM option)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.utils import has_deep_gemm
from vllm.utils.deep_gemm import calc_diff, per_block_cast_to_fp8

BLOCK_SIZE = [128, 128]

requires_deep_gemm = pytest.mark.skipif(
    not has_deep_gemm(),
    reason="Requires deep_gemm kernels",
)


def make_block_quant_fp8_weights(
    e: int,
    n: int,
    k: int,
    block_size: list[int],
):
    """
    Generate (w1, w2) expert weights and their per-block scale tensors
    in FP8 block-quantized format.

      w1 shape: (E, 2N, K)
      w2 shape: (E, K, N)
    """
    dtype = torch.bfloat16
    fp8_max, fp8_min = torch.finfo(torch.float8_e4m3fn).max, torch.finfo(
        torch.float8_e4m3fn).min

    # bf16 reference weights
    w1_bf16 = torch.randn(e, 2 * n, k, device="cuda", dtype=dtype) / 10
    w2_bf16 = torch.randn(e, k, n, device="cuda", dtype=dtype) / 10
    w1_bf16.clamp_(fp8_min, fp8_max)
    w2_bf16.clamp_(fp8_min, fp8_max)

    block_n, block_k = block_size
    n_tiles_w1 = math.ceil((2 * n) / block_n)
    k_tiles_w1 = math.ceil(k / block_k)
    n_tiles_w2 = math.ceil(k / block_n)
    k_tiles_w2 = math.ceil(n / block_k)

    w1 = torch.empty_like(w1_bf16, dtype=torch.float8_e4m3fn)
    w2 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn)
    w1_s = torch.empty(e,
                       n_tiles_w1,
                       k_tiles_w1,
                       device="cuda",
                       dtype=torch.float32)
    w2_s = torch.empty(e,
                       n_tiles_w2,
                       k_tiles_w2,
                       device="cuda",
                       dtype=torch.float32)

    for i in range(e):
        w1[i], w1_s[i] = per_block_cast_to_fp8(w1_bf16[i])
        w2[i], w2_s[i] = per_block_cast_to_fp8(w2_bf16[i])

    return w1, w2, w1_s, w2_s


def run_single_case(m, n, k, topk, num_experts, block_size):
    """
    Run one (M,N,K) configuration on a single GPU and assert DeepGEMM ==
    Triton baseline within tolerance.
    """
    tokens_bf16 = torch.randn(
        m, k, device="cuda", dtype=torch.bfloat16).clamp_min_(-1).clamp_max_(1)
    _, a1_scale = per_token_group_quant_fp8(tokens_bf16, block_size[1])

    # expert weight tensors
    w1, w2, w1_s, w2_s = make_block_quant_fp8_weights(num_experts, n, k,
                                                      block_size)

    router_logits = torch.randn(m,
                                num_experts,
                                device="cuda",
                                dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    # triton reference
    out_triton = fused_experts(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        use_fp8_w8a8=True,
        w1_scale=w1_s,
        w2_scale=w2_s,
        a1_scale=a1_scale,
        block_shape=block_size,
        allow_deep_gemm=False,
    )

    # DeepGemm
    out_deepgemm = fused_experts(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        use_fp8_w8a8=True,
        w1_scale=w1_s,
        w2_scale=w2_s,
        a1_scale=a1_scale,
        block_shape=block_size,
        allow_deep_gemm=True,
    )
    diff = calc_diff(out_deepgemm, out_triton)
    assert diff < 0.001, f"Diff exceeded 1%: {diff}"


# Note: W1 has shape (E, 2N, K), so N = 512
# can trigger the deepgemm path.
MNKs = [
    (1024, 512, 128),
    (1024, 512, 512),
    (2048, 512, 512),
    (512, 1024, 1024),
    (512, 2048, 2048),
    (4096, 4096, 1024),
]

TOPKS = [2, 6]
NUM_EXPERTS = [32]


@pytest.mark.parametrize("mnk", MNKs)
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@requires_deep_gemm
def test_deepgemm_vs_triton(mnk, topk, num_experts, monkeypatch):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_DEEP_GEMM", "1")

        _fused_moe_mod = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.fused_moe")

        call_counter = {"cnt": 0}

        orig_fn = _fused_moe_mod.deep_gemm_moe_fp8

        def _spy_deep_gemm_moe_fp8(*args, **kwargs):
            call_counter["cnt"] += 1
            return orig_fn(*args, **kwargs)

        monkeypatch.setattr(_fused_moe_mod, "deep_gemm_moe_fp8",
                            _spy_deep_gemm_moe_fp8)

        m, n, k = mnk

        if topk > num_experts:
            pytest.skip(f"topk={topk} > num_experts={num_experts}")

        run_single_case(
            m=m,
            n=n,
            k=k,
            topk=topk,
            num_experts=num_experts,
            block_size=BLOCK_SIZE,
        )

        # ensure that the DeepGEMM path was indeed taken.
        assert call_counter["cnt"] == 1, \
            f"DeepGEMM path was not executed during the test. " \
            f"Call counter: {call_counter['cnt']}"
