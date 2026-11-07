# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which MegaMoE expert format a vLLM config resolves to.

Online MXFP8 quantization is what lets a BF16 checkpoint run with MXFP8
experts; every other checkpoint keeps its own expert layout.
"""

import pytest

from vllm.config import VllmConfig
from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    MEGA_MOE_MXFP8,
)
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Static,
    kMxfp8Dynamic,
)
from vllm.models.deepseek_v4.nvidia.model import resolve_mega_moe_weight_format

pytestmark = pytest.mark.cpu_test


def _online_quant_config(moe_weight) -> VllmConfig:
    args = QuantizationConfigArgs(moe=QuantSpec(weight=moe_weight))
    return VllmConfig(quant_config=OnlineQuantizationConfig(args))


def test_dynamic_mxfp8_experts_are_quantized_online() -> None:
    """A BF16 checkpoint quantizes its routed experts while loading."""
    config = _online_quant_config(kMxfp8Dynamic)

    assert resolve_mega_moe_weight_format(config) == (MEGA_MOE_MXFP8, True)


def test_pre_quantized_experts_keep_their_offline_layout() -> None:
    config = _online_quant_config(kMxfp4Static)

    assert resolve_mega_moe_weight_format(config) == (MEGA_MOE_MXFP4, False)


def test_online_quantization_without_a_moe_spec_keeps_offline_experts() -> None:
    """Quantizing only the linear layers leaves the routed experts alone."""
    args = QuantizationConfigArgs(linear=QuantSpec(weight=kMxfp8Dynamic))
    config = VllmConfig(quant_config=OnlineQuantizationConfig(args))

    assert resolve_mega_moe_weight_format(config) == (MEGA_MOE_MXFP4, False)


def test_unquantized_config_keeps_offline_experts() -> None:
    assert resolve_mega_moe_weight_format(VllmConfig()) == (MEGA_MOE_MXFP4, False)
