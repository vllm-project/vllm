# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.models.gemma4 import Gemma4Model


def _make_expert_params_mapping(num_experts=2):
    mapping = []
    for expert_id in range(num_experts):
        for shard_id, proj_name in [
            ("w1", "gate_proj"),
            ("w2", "down_proj"),
            ("w3", "up_proj"),
        ]:
            param_name = (
                "experts.w13_"
                if proj_name in ["gate_proj", "up_proj"]
                else "experts.w2_"
            )
            weight_name = f"experts.{expert_id}.{proj_name}."
            mapping.append((param_name, weight_name, expert_id, shard_id))
    return mapping


def _make_params_dict(num_experts=2, layer_indices=None):
    if layer_indices is None:
        layer_indices = [0]
    mapping = _make_expert_params_mapping(num_experts)
    params = OrderedDict()
    suffixes = ["weight", "weight_scale", "weight_scale_2", "input_scale"]
    for layer_idx in layer_indices:
        for param_name, _, _, _ in mapping:
            for suffix in suffixes:
                key = f"layers.{layer_idx}.moe.{param_name}{suffix}"
                mock_param = MagicMock()
                mock_param.weight_loader = MagicMock()
                params[key] = mock_param
    return params


def _make_model(params_dict, num_experts=2):
    with patch.object(Gemma4Model, "__init__", lambda self, **kw: None):
        model = Gemma4Model.__new__(Gemma4Model)
    model.quant_config = None
    model.config = MagicMock()
    model.config.num_experts = num_experts
    return model


def _extract_layer_index(checkpoint_key):
    parts = checkpoint_key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    return 0


MAPPING_CASES = [
    (
        "layers.0.moe.experts.0.gate_proj.weight",
        "layers.0.moe.experts.w13_weight",
        "experts.0.gate_proj.weight",
        0,
        "w1",
    ),
    (
        "layers.0.moe.experts.0.gate_proj.weight_scale",
        "layers.0.moe.experts.w13_weight_scale",
        "experts.0.gate_proj.weight_scale",
        0,
        "w1",
    ),
    (
        "layers.0.moe.experts.0.gate_proj.weight_scale_2",
        "layers.0.moe.experts.w13_weight_scale_2",
        "experts.0.gate_proj.weight_scale_2",
        0,
        "w1",
    ),
    (
        "layers.0.moe.experts.0.gate_proj.input_scale",
        "layers.0.moe.experts.w13_input_scale",
        "experts.0.gate_proj.input_scale",
        0,
        "w1",
    ),
    (
        "layers.0.moe.experts.1.up_proj.weight",
        "layers.0.moe.experts.w13_weight",
        "experts.1.up_proj.weight",
        1,
        "w3",
    ),
    (
        "layers.0.moe.experts.1.up_proj.weight_scale",
        "layers.0.moe.experts.w13_weight_scale",
        "experts.1.up_proj.weight_scale",
        1,
        "w3",
    ),
    (
        "layers.0.moe.experts.0.down_proj.weight",
        "layers.0.moe.experts.w2_weight",
        "experts.0.down_proj.weight",
        0,
        "w2",
    ),
    (
        "layers.0.moe.experts.0.down_proj.weight_scale",
        "layers.0.moe.experts.w2_weight_scale",
        "experts.0.down_proj.weight_scale",
        0,
        "w2",
    ),
    (
        "layers.0.moe.experts.1.down_proj.weight_scale_2",
        "layers.0.moe.experts.w2_weight_scale_2",
        "experts.1.down_proj.weight_scale_2",
        1,
        "w2",
    ),
    (
        "layers.7.moe.experts.0.gate_proj.weight_scale",
        "layers.7.moe.experts.w13_weight_scale",
        "experts.0.gate_proj.weight_scale",
        0,
        "w1",
    ),
    (
        "layers.15.moe.experts.1.down_proj.input_scale",
        "layers.15.moe.experts.w2_input_scale",
        "experts.1.down_proj.input_scale",
        1,
        "w2",
    ),
    (
        "layers.0.moe.experts.0.gate_proj",
        "layers.0.moe.experts.w13_weight",
        "experts.0.gate_proj",
        0,
        "w1",
    ),
    (
        "layers.0.moe.experts.0.down_proj",
        "layers.0.moe.experts.w2_weight",
        "experts.0.down_proj",
        0,
        "w2",
    ),
]


class TestGemma4ExpertWeightMapping:
    @pytest.mark.parametrize(
        "checkpoint_key, expected_moe_name, expected_wl_name, expert_id, shard_id",
        MAPPING_CASES,
    )
    def test_suffix_mapping(
        self,
        checkpoint_key,
        expected_moe_name,
        expected_wl_name,
        expert_id,
        shard_id,
    ):
        layer_idx = _extract_layer_index(checkpoint_key)
        params_dict = _make_params_dict(num_experts=2, layer_indices=[layer_idx])
        model = _make_model(params_dict, num_experts=2)
        fake_weight = torch.randn(4, 4)

        with (
            patch.object(model, "named_parameters", return_value=params_dict.items()),
            patch.object(model, "named_buffers", return_value=[]),
            patch(
                "vllm.model_executor.models.gemma4.is_pp_missing_parameter",
                return_value=False,
            ),
        ):
            loaded = model.load_weights([(checkpoint_key, fake_weight)])

        assert expected_moe_name in loaded, (
            f"Expected '{expected_moe_name}' in loaded params, got {sorted(loaded)}"
        )

        param = params_dict[expected_moe_name]
        param.weight_loader.assert_called_once()
        call_args = param.weight_loader.call_args
        assert call_args[0][2] == expected_wl_name
        assert call_args[1]["shard_id"] == shard_id
        assert call_args[1]["expert_id"] == expert_id

    @pytest.mark.parametrize(
        "checkpoint_key",
        [
            "layers.0.moe.experts.0.gate_proj.unknown_suffix",
            "layers.0.moe.experts.0.gate_proj.bias",
            "layers.0.mlp.gate_proj.weight_scale",
        ],
    )
    def test_unrecognized_suffix_rejected(self, checkpoint_key):
        params_dict = _make_params_dict(num_experts=2, layer_indices=[0])
        model = _make_model(params_dict, num_experts=2)
        fake_weight = torch.randn(4, 4)

        with (
            patch.object(model, "named_parameters", return_value=params_dict.items()),
            patch.object(model, "named_buffers", return_value=[]),
            patch(
                "vllm.model_executor.models.gemma4.is_pp_missing_parameter",
                return_value=False,
            ),
        ):
            loaded = model.load_weights([(checkpoint_key, fake_weight)])

        expert_keys = [k for k in loaded if "experts.w13_" in k or "experts.w2_" in k]
        assert len(expert_keys) == 0

        for param in params_dict.values():
            param.weight_loader.assert_not_called()

    @pytest.mark.parametrize("layer_idx", [0, 3, 7, 15])
    @pytest.mark.parametrize("expert_id", [0, 1])
    @pytest.mark.parametrize(
        "proj,expected_prefix",
        [("gate_proj", "w13"), ("up_proj", "w13"), ("down_proj", "w2")],
    )
    @pytest.mark.parametrize(
        "suffix",
        ["weight", "weight_scale", "weight_scale_2", "input_scale"],
    )
    def test_prefix_preservation_across_layers(
        self, layer_idx, expert_id, proj, expected_prefix, suffix
    ):
        checkpoint_key = f"layers.{layer_idx}.moe.experts.{expert_id}.{proj}.{suffix}"
        expected_moe_name = f"layers.{layer_idx}.moe.experts.{expected_prefix}_{suffix}"
        expected_wl_name = f"experts.{expert_id}.{proj}.{suffix}"

        params_dict = _make_params_dict(num_experts=2, layer_indices=[layer_idx])
        model = _make_model(params_dict, num_experts=2)
        fake_weight = torch.randn(4, 4)

        with (
            patch.object(model, "named_parameters", return_value=params_dict.items()),
            patch.object(model, "named_buffers", return_value=[]),
            patch(
                "vllm.model_executor.models.gemma4.is_pp_missing_parameter",
                return_value=False,
            ),
        ):
            loaded = model.load_weights([(checkpoint_key, fake_weight)])

        assert expected_moe_name in loaded
        param = params_dict[expected_moe_name]
        param.weight_loader.assert_called_once()
        assert param.weight_loader.call_args[0][2] == expected_wl_name
