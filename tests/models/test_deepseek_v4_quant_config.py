# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from torch import nn

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config
from vllm.platforms import current_platform


def test_deepseek_v4_fp4_guard_uses_moe_shard_tp_size():
    quant_config = DeepseekV4FP8Config.__new__(DeepseekV4FP8Config)
    quant_config.ignored_layers = []
    quant_config.packed_modules_mapping = {}
    quant_config._resolved_expert_dtype = "fp4"
    quant_config._resolved_moe_quant_algo = "MXFP4"

    layer = RoutedExperts.__new__(RoutedExperts)
    nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(tp_size=2),
    )

    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        pytest.raises(ValueError, match="not supported with tensor parallelism"),
    ):
        quant_config.get_quant_method(layer, "experts")


def test_deepseek_v4_fp4_guard_keeps_rocm_tp_available(monkeypatch):
    quant_config = DeepseekV4FP8Config.__new__(DeepseekV4FP8Config)
    quant_config.ignored_layers = []
    quant_config.packed_modules_mapping = {}
    quant_config._resolved_expert_dtype = "fp4"
    quant_config._resolved_moe_quant_algo = "MXFP4"

    layer = RoutedExperts.__new__(RoutedExperts)
    nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(tp_size=2),
    )

    expected_method = object()
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.quant_config.Mxfp4MoEMethod",
        lambda moe_config: expected_method,
    )

    assert quant_config.get_quant_method(layer, "experts") is expected_method
