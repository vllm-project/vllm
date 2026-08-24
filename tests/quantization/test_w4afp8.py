# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.w4afp8 import (
    W4AFP8Config,
    W4AFP8MoEMethod,
    _convert_signed_int4_to_uint4b8,
)


def test_w4afp8_config_resolves_checkpoint_quant_method() -> None:
    config_cls = get_quantization_config("w4afp8")
    config = config_cls.from_config({"quant_method": "w4afp8"})

    assert config_cls is W4AFP8Config
    assert config.group_size == 128
    assert config.linear_quant_config.is_checkpoint_fp8_serialized
    assert config.linear_quant_config.weight_block_size == [128, 128]
    assert config.get_supported_act_dtypes() == [torch.bfloat16]


def test_w4afp8_config_rejects_unsupported_group_size() -> None:
    with pytest.raises(ValueError, match="group_size=128"):
        W4AFP8Config.from_config({"quant_method": "w4afp8", "group_size": 64})


def test_w4afp8_converts_signed_nibbles_to_humming_uint4b8() -> None:
    signed_nibbles = torch.arange(-8, 8, dtype=torch.int8)
    encoded = signed_nibbles.to(torch.uint8) & 0xF
    packed = (encoded[0::2] | (encoded[1::2] << 4)).view(torch.int8)

    converted = _convert_signed_int4_to_uint4b8(packed)
    converted_bytes = converted.view(torch.uint8)
    low = converted_bytes & 0xF
    high = converted_bytes >> 4
    unpacked = torch.stack((low, high), dim=1).flatten()

    torch.testing.assert_close(unpacked, torch.arange(16, dtype=torch.uint8))


def test_w4afp8_rejects_invalid_packed_weight_storage() -> None:
    with pytest.raises(ValueError, match="int8 storage"):
        _convert_signed_int4_to_uint4b8(torch.zeros(4, dtype=torch.uint8))
    with pytest.raises(ValueError, match="full int32 words"):
        _convert_signed_int4_to_uint4b8(torch.zeros(2, dtype=torch.int8))


def test_w4afp8_expert_parameter_shapes_match_checkpoint_layout() -> None:
    method = object.__new__(W4AFP8MoEMethod)
    method.group_size = 128
    method.moe = SimpleNamespace(w13_num_shards=2)
    layer = torch.nn.Module()

    method.create_weights(
        layer=layer,
        num_experts=3,
        hidden_size=512,
        intermediate_size_per_partition=256,
        params_dtype=torch.bfloat16,
    )

    assert layer.w13_weight.shape == (3, 512, 256)
    assert layer.w2_weight.shape == (3, 512, 128)
    assert layer.w13_weight.dtype == torch.int8
    assert layer.w2_weight.dtype == torch.int8
    assert layer.w13_weight_scale_inv.shape == (3, 512, 4)
    assert layer.w2_weight_scale_inv.shape == (3, 512, 2)
    assert layer.w13_weight_scale_inv.dtype == torch.bfloat16
    assert layer.w2_weight_scale_inv.dtype == torch.bfloat16
    assert layer.w13_input_scale.shape == (3, 2)
    assert layer.w2_input_scale.shape == (3,)


def _make_input_scale_param(shape: tuple[int, ...]) -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.ones(shape), requires_grad=False)
    param.is_split_input_scale = True
    param.is_w4afp8_input_scale = True
    return param


def test_w4afp8_mapping_adds_checkpoint_input_scales() -> None:
    model = torch.nn.Module()
    model.register_parameter(
        "w13_input_scale",
        _make_input_scale_param((2, 2)),
    )

    mapping = RoutedExperts.make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=2,
    )

    assert (
        "experts.routed_experts.w13_",
        "experts.0.w1.",
        0,
        "w1",
    ) in mapping
    assert (
        "experts.routed_experts.w2_",
        "experts.1.w2.",
        1,
        "w2",
    ) in mapping
    assert (
        "experts.routed_experts.w13_",
        "experts.1.w3.",
        1,
        "w3",
    ) in mapping


def test_w4afp8_w1_w3_input_scales_do_not_overwrite_each_other() -> None:
    loader = object.__new__(RoutedExperts)
    loader.quant_config = W4AFP8Config()
    loader.quant_method = SimpleNamespace()
    loader._map_global_expert_id_to_local_expert_id = lambda expert_id: expert_id
    param = _make_input_scale_param((2, 2))

    for shard_id, value in (("w1", 0.25), ("w3", 0.75)):
        loaded = loader.weight_loader(
            param=param,
            loaded_weight=torch.tensor(value),
            weight_name=f"experts.1.{shard_id}.input_scale",
            shard_id=shard_id,
            expert_id=1,
            return_success=True,
        )
        assert loaded

    torch.testing.assert_close(param[1], torch.tensor([0.25, 0.75]))


def test_w4afp8_reduces_checkpoint_input_scales_like_sglang() -> None:
    layer = SimpleNamespace(
        w13_input_scale=torch.tensor(
            [[0.25, 0.5], [0.75, 0.125]],
            dtype=torch.bfloat16,
        ),
        w2_input_scale=torch.tensor([0.25, 0.5], dtype=torch.bfloat16),
    )
    a1_scale, a2_scale = W4AFP8MoEMethod._prepare_input_scales(layer)

    assert a1_scale.dtype == torch.float32
    assert a2_scale.dtype == torch.float32
    torch.testing.assert_close(a1_scale, torch.tensor([0.75]))
    torch.testing.assert_close(a2_scale, torch.tensor([0.5]))


@pytest.mark.parametrize(
    (
        "ep_size",
        "use_batched_activation_format",
        "apply_router_weight_on_input",
        "match",
    ),
    [
        (2, False, False, "expert parallel size 1"),
        (1, True, False, "batched-expert activation format"),
        (1, False, True, "router weights on input"),
    ],
)
def test_w4afp8_rejects_unsupported_moe_formats(
    ep_size: int,
    use_batched_activation_format: bool,
    apply_router_weight_on_input: bool,
    match: str,
) -> None:
    method = object.__new__(W4AFP8MoEMethod)
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            ep_size=ep_size, use_batched_activation_format=use_batched_activation_format
        )
    )
    layer = SimpleNamespace(apply_router_weight_on_input=apply_router_weight_on_input)

    with pytest.raises(NotImplementedError, match=match):
        method.process_weights_after_loading(layer)
