# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import AriaTextConfig
from transformers.models.aria.modeling_aria import AriaTextMoELayer as HFMoE

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.models.aria import AriaTextModel, AriaTextMoELayer
from vllm.model_executor.models.utils import AutoWeightsLoader


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
@pytest.mark.parametrize(
    "checkpoint_name,loaded_name,param_name",
    [
        (
            "experts.fc1.weight",
            "experts.w13_weight",
            "experts.routed_experts.w13_weight",
        ),
        (
            "experts.fc2.weight",
            "experts.w2_weight",
            "experts.routed_experts.w2_weight",
        ),
        (
            "shared_experts.down_proj.weight",
            "shared_experts.down_proj.weight",
            "shared_experts.down_proj.weight",
        ),
    ],
)
def test_aria_expert_weights_load_with_checkpoint_layout(
    checkpoint_name: str, loaded_name: str, param_name: str
):
    """Real Aria checkpoint weights must reach the right parameter and layout."""
    config = AriaTextConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_num_experts=2,
        moe_topk=1,
        moe_num_shared_experts=1,
    )
    checkpoint_weight = HFMoE(config).state_dict()[checkpoint_name]
    checkpoint_weight.copy_(
        torch.arange(
            checkpoint_weight.numel(), dtype=checkpoint_weight.dtype
        ).reshape_as(checkpoint_weight)
    )
    expected = checkpoint_weight.clone()
    if checkpoint_name.startswith("experts."):
        expected = expected.transpose(-1, -2)

    layer = AriaTextMoELayer(config, quant_config=None, prefix="mlp")
    loaded = AutoWeightsLoader(layer).load_weights(
        [(checkpoint_name, checkpoint_weight)], mapper=AriaTextModel.hf_to_vllm_mapper
    )

    assert loaded == {loaded_name}
    torch.testing.assert_close(
        dict(layer.named_parameters())[param_name], expected, rtol=0, atol=0
    )


@pytest.mark.cpu_test
def test_aria_quant_config_renames_expert_modules():
    quant_config = CompressedTensorsConfig(
        target_scheme_map={},
        ignore=[
            "model.layers.0.mlp.experts.fc1",
            "model.layers.0.mlp.experts.fc2",
        ],
        quant_format="pack-quantized",
    )
    AriaTextModel.__new__(
        AriaTextModel, vllm_config=VllmConfig(quant_config=quant_config)
    )

    assert quant_config.ignore == [
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    ]


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "name,expected", [("fc1", "gate_up_proj"), ("fc2", "down_proj")]
)
def test_aria_expert_scale_names(name: str, expected: str):
    suffixes = ["weight_scale", "input_scale"]
    names = [f"model.layers.0.mlp.experts.{name}.{suffix}" for suffix in suffixes]
    assert AriaTextModel.hf_to_vllm_mapper.apply_list(names) == [
        f"model.layers.0.mlp.experts.{expected}.{suffix}" for suffix in suffixes
    ]
