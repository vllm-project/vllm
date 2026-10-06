# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.platforms import current_platform


def test_contiguous_deepgemm_forwards_clamp_to_bi_kernel(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts import deep_gemm_moe
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    calls = []

    def fused(value, **kwargs):
        calls.append(kwargs)
        return kwargs["output_q"], torch.ones(1)

    monkeypatch.setattr(
        deep_gemm_moe, "is_batch_invariant_quant_kernel_enabled", lambda: True
    )
    monkeypatch.setattr(
        DeepGemmQuantScaleFMT,
        "from_oracle",
        staticmethod(lambda: DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0),
    )
    monkeypatch.setattr(
        deep_gemm_moe, "fused_silu_mul_per_token_group_quant_fp8", fused
    )
    expert = SimpleNamespace(
        block_shape=[128, 128],
        gemm1_clamp_limit=10.0,
        gemm1_alpha=1.0,
        gemm1_beta=0.0,
        adjust_N_for_activation=lambda n, _activation: n // 2,
    )
    value = torch.randn(2, 256, dtype=torch.bfloat16)
    output = torch.empty(2, 128, dtype=torch.float8_e4m3fn)

    quantized, _scales = deep_gemm_moe.DeepGemmExperts._act_mul_quant(
        expert, value, output, MoEActivation.SILU
    )

    assert quantized is output
    assert len(calls) == 1
    assert calls[0]["output_q"] is output
    assert calls[0]["use_ue8m0"] is False
    assert calls[0]["round_scale"] is True
    assert calls[0]["clamp_limit"] == 10.0
    assert calls[0]["masked_m"] is None
    assert calls[0]["group_size"] == 128


def test_masked_deepgemm_forwards_clamp_to_bi_kernel(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts import batched_deep_gemm_moe
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    calls = []

    def fused(value, **kwargs):
        calls.append(kwargs)
        return torch.empty(1), torch.empty(1)

    monkeypatch.setattr(
        batched_deep_gemm_moe,
        "is_batch_invariant_quant_kernel_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        batched_deep_gemm_moe,
        "fused_silu_mul_per_token_group_quant_fp8",
        fused,
    )
    value = torch.randn(2, 3, 256, dtype=torch.bfloat16)
    counts = torch.tensor([2, 3], dtype=torch.int32)

    batched_deep_gemm_moe.persistent_masked_m_silu_mul_quant(
        value,
        counts,
        quant_scale_fmt=DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
        clamp_limit=10.0,
    )

    assert calls[0]["round_scale"] is True
    assert calls[0]["clamp_limit"] == 10.0
    assert calls[0]["masked_m"] is counts


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("use_ue8m0", [False, True])
@pytest.mark.parametrize("clamp_limit", [None, 7.0])
def test_fused_quant_is_independent_of_cobatched_tokens(use_ue8m0, clamp_limit):
    assert fp8_utils.is_batch_invariant_quant_kernel_enabled()
    torch.manual_seed(42)
    x = torch.randn(5, 1024, device="cuda", dtype=torch.bfloat16) * 3
    results = [
        fp8_utils.fused_silu_mul_per_token_group_quant_fp8(
            inp, use_ue8m0=use_ue8m0, clamp_limit=clamp_limit, masked_m=None
        )
        for inp in (x[:2].contiguous(), x.contiguous())
    ]
    (small_q, small_s), (large_q, large_s) = results
    assert torch.equal(small_q.view(torch.uint8), large_q[:2].view(torch.uint8))
    assert torch.equal(small_s, large_s[:2])
