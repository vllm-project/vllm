# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from torch import nn

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config


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

    with pytest.raises(ValueError, match="not supported with tensor parallelism"):
        quant_config.get_quant_method(layer, "experts")
