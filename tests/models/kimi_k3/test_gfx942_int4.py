# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`int4_per_group_32` MoE config resolution, as used for Kimi-K3 on gfx942."""

import os
from unittest.mock import patch

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.int4 import (
    Int4PerGroup32OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.requantization import (
    skip_checkpoint_backend_selection,
)

pytestmark = pytest.mark.cpu_test

_PREFIX = "model.layers.0.mlp.experts"


def _online_config(**args) -> OnlineQuantizationConfig:
    return OnlineQuantizationConfig(QuantizationConfigArgs(**args))


def test_int4_per_group_32_is_an_online_moe_method():
    config = _online_config(moe={"weight": "int4_per_group_32"})
    resolved = config.resolve_quant_method_cls(RoutedExperts, _PREFIX)
    assert resolved is not None
    assert resolved[-1] is Int4PerGroup32OnlineMoEMethod


def test_int4_per_group_32_is_rejected_for_linear_layers():
    config = _online_config(linear={"weight": "int4_per_group_32"})
    with pytest.raises(ValueError, match="online quantization"):
        config.resolve_quant_method_cls(LinearBase, "model.layers.0.mlp.down_proj")


def test_unknown_online_moe_weight_still_raises():
    config = _online_config(moe={"weight": "fp8_per_token"})
    with pytest.raises(ValueError, match="online quantization"):
        config.resolve_quant_method_cls(RoutedExperts, _PREFIX)


def test_mxfp4_requantization_source_skips_k3_aiter_setup(monkeypatch):
    """As a requantization source, Kimi-K3's MXFP4 experts only dequantize: the
    gfx950 AITER route, and its AITER_BF16_FP8_MOE_BOUND override, stay off."""
    monkeypatch.delenv("AITER_BF16_FP8_MOE_BOUND", raising=False)
    moe = make_dummy_moe_config(
        num_experts=8,
        hidden_dim=256,
        intermediate_size=256,
        activation=MoEActivation.SITU,
    )
    with (
        patch(
            "vllm.model_executor.layers.quantization.mxfp4._use_k3_situ_aiter",
            return_value=True,
        ),
        skip_checkpoint_backend_selection(),
    ):
        method = Mxfp4MoEMethod(moe)
    assert not method.is_k3_situ_aiter
    assert method.mxfp4_backend == Mxfp4MoeBackend.NONE
    assert "AITER_BF16_FP8_MOE_BOUND" not in os.environ
