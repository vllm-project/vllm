# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.platforms import current_platform


def _make_moe_config() -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=2,
        num_logical_experts=2,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        device="cuda",
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.TopK,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        swiglu_limit=7.0,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(
    "experts_cls,quant_kwargs",
    [
        pytest.param(
            FlashInferExperts,
            {"quant_dtype": "nvfp4", "weight_dtype": "nvfp4"},
            id="flashinfer-cutlass",
        ),
        pytest.param(
            MarlinExperts,
            {"weight_dtype": "nvfp4"},
            id="marlin",
        ),
    ],
)
@pytest.mark.parametrize(
    "quant_params,expected",
    [
        pytest.param({}, (1.702, 1.0, 7.0), id="model-config"),
        pytest.param(
            {
                "gemm1_alpha": 2.0,
                "gemm1_beta": 0.5,
                "gemm1_clamp_limit": 6.0,
            },
            (2.0, 0.5, 6.0),
            id="quant-config",
        ),
    ],
)
def test_swigluoai_params(experts_cls, quant_kwargs, quant_params, expected):
    quant_config = FusedMoEQuantConfig.make(**quant_kwargs, **quant_params)
    experts = experts_cls(moe_config=_make_moe_config(), quant_config=quant_config)

    assert experts._supports_activation(MoEActivation.SWIGLUOAI_UNINTERLEAVE)
    for name, value in zip(
        ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"), expected
    ):
        actual = getattr(experts, name)
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(
                actual, torch.full_like(actual, value, dtype=torch.float32)
            )
        else:
            assert actual == pytest.approx(value)
