# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import torch

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    mxfp4_mxfp8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_weight_to_mxfp4_moe_kernel_format,
)


def _make_moe_config() -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=16,
        intermediate_size_per_partition=16,
        num_local_experts=2,
        num_logical_experts=2,
        activation=MoEActivation.SILU,
        device="cpu",
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=16,
    )


def _make_experts(
    *,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    gemm1_clamp_limit: float | None = None,
) -> FlashInferExperts:
    quant_config = mxfp4_mxfp8_moe_quant_config(
        w1_scale=torch.ones((2, 32, 1), dtype=torch.float8_e4m3fn),
        w2_scale=torch.ones((2, 16, 1), dtype=torch.float8_e4m3fn),
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        return FlashInferExperts(
            moe_config=_make_moe_config(),
            quant_config=quant_config,
        )


def test_mxfp4_swiglu_parameters_stay_unset_without_quant_config() -> None:
    experts = _make_experts()

    assert experts.gemm1_alpha is None
    assert experts.gemm1_beta is None
    assert experts.gemm1_clamp_limit is None


def test_mxfp4_swiglu_parameters_follow_quant_config() -> None:
    experts = _make_experts(
        gemm1_alpha=1.25,
        gemm1_beta=0.75,
        gemm1_clamp_limit=5.5,
    )

    torch.testing.assert_close(experts.gemm1_alpha, torch.tensor([1.25, 1.25]))
    torch.testing.assert_close(experts.gemm1_beta, torch.tensor([0.75, 0.75]))
    torch.testing.assert_close(
        experts.gemm1_clamp_limit,
        torch.tensor([5.5, 5.5]),
    )


def test_cutlass_mxfp8_kernel_format_converts_gate_up_layout(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        types.SimpleNamespace(block_scale_interleave=lambda x: x.contiguous()),
    )

    num_experts = 1
    intermediate_size = 64
    hidden_size = 64
    packed_hidden_size = hidden_size // 2
    sf_block_size = 32

    w13_weight = torch.arange(
        num_experts * 2 * intermediate_size * packed_hidden_size,
        dtype=torch.uint8,
    ).reshape(num_experts, 2 * intermediate_size, packed_hidden_size)
    w2_weight = torch.arange(
        num_experts * hidden_size * (intermediate_size // 2),
        dtype=torch.uint8,
    ).reshape(num_experts, hidden_size, intermediate_size // 2)
    w13_scale_u8 = torch.arange(
        num_experts * 2 * intermediate_size * (hidden_size // sf_block_size),
        dtype=torch.uint8,
    ).reshape(num_experts, 2 * intermediate_size, hidden_size // sf_block_size)
    w2_scale_u8 = torch.arange(
        num_experts * hidden_size * (intermediate_size // sf_block_size),
        dtype=torch.uint8,
    ).reshape(num_experts, hidden_size, intermediate_size // sf_block_size)
    w13_bias = torch.arange(
        num_experts * 2 * intermediate_size,
        dtype=torch.bfloat16,
    ).reshape(num_experts, 2 * intermediate_size)
    w2_bias = torch.arange(
        num_experts * hidden_size,
        dtype=torch.bfloat16,
    ).reshape(num_experts, hidden_size)

    (
        out_w13,
        out_w2,
        out_w13_scale,
        out_w2_scale,
        out_w13_bias,
        out_w2_bias,
    ) = convert_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
        layer=torch.nn.Module(),
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        w13_weight_scale=w13_scale_u8.view(torch.float8_e4m3fn),
        w2_weight_scale=w2_scale_u8.view(torch.float8_e4m3fn),
        w13_bias=w13_bias,
        w2_bias=w2_bias,
    )

    expected_w13 = torch.cat(
        [
            w13_weight[:, intermediate_size:, :],
            w13_weight[:, :intermediate_size, :],
        ],
        dim=1,
    )
    expected_w13_scale = torch.cat(
        [
            w13_scale_u8[:, intermediate_size:, :],
            w13_scale_u8[:, :intermediate_size, :],
        ],
        dim=1,
    )
    expected_w13_bias = torch.cat(
        [
            w13_bias[:, intermediate_size:],
            w13_bias[:, :intermediate_size],
        ],
        dim=1,
    )

    assert out_w13.is_contiguous()
    assert out_w2.is_contiguous()
    torch.testing.assert_close(out_w13, expected_w13)
    torch.testing.assert_close(out_w2, w2_weight)
    torch.testing.assert_close(out_w13_scale, expected_w13_scale)
    torch.testing.assert_close(out_w2_scale, w2_scale_u8)
    torch.testing.assert_close(out_w13_bias, expected_w13_bias)
    torch.testing.assert_close(out_w2_bias, w2_bias)
