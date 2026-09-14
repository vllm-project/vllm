# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.models import afmoe as afmoe_module
from vllm.model_executor.models.afmoe import AfmoeForCausalLM


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("is_cuda", "tp_size", "dtype", "quant_config", "lora_config", "expected"),
    [
        (True, 1, torch.bfloat16, None, None, True),
        (False, 1, torch.bfloat16, None, None, False),
        (True, 2, torch.bfloat16, None, None, False),
        (True, 1, torch.float16, None, None, False),
        (True, 1, torch.float32, None, None, False),
        (True, 1, torch.bfloat16, object(), None, False),
        (True, 1, torch.bfloat16, None, object(), False),
    ],
)
def test_afmoe_fused_qkv_gate_enablement(
    monkeypatch,
    is_cuda,
    tp_size,
    dtype,
    quant_config,
    lora_config,
    expected,
):
    monkeypatch.setattr(afmoe_module.current_platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(
        afmoe_module,
        "get_tensor_model_parallel_world_size",
        lambda: tp_size,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=dtype),
        quant_config=quant_config,
        lora_config=lora_config,
    )
    assert afmoe_module._should_use_fused_qkv_gate(vllm_config) is expected


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "model.layers.0.self_attn.q_proj.weight",
            ("model.layers.0.self_attn.qkv_gate_proj.weight", 0),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            ("model.layers.0.self_attn.qkv_gate_proj.weight", 1),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            ("model.layers.0.self_attn.qkv_gate_proj.weight", 2),
        ),
        (
            "model.layers.0.self_attn.gate_proj.weight",
            ("model.layers.0.self_attn.qkv_gate_proj.weight", 3),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            ("model.layers.0.mlp.gate_up_proj.weight", 0),
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            ("model.layers.0.mlp.gate_up_proj.weight", 1),
        ),
    ],
)
def test_afmoe_fused_qkv_gate_weight_mapping(source, expected):
    assert (
        AfmoeForCausalLM.hf_to_vllm_fused_mapper._map_name_with_shard(source)
        == expected
    )


@pytest.mark.cpu_test
def test_afmoe_fused_qkv_gate_shard_loading_and_projection(dist_init):
    input_size = 8
    output_sizes = [8, 2, 2, 8]
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        layer = MergedColumnParallelLinear(
            input_size,
            output_sizes,
            bias=False,
            params_dtype=torch.float32,
            prefix="model.layers.0.self_attn.qkv_gate_proj",
            disable_tp=True,
        )

    shards = [
        torch.arange(size * input_size, dtype=torch.float32).reshape(size, input_size)
        + shard_id * 1000
        for shard_id, size in enumerate(output_sizes)
    ]
    for shard_id, shard in enumerate(shards):
        layer.weight.weight_loader(layer.weight, shard, shard_id)

    expected_weight = torch.cat(shards, dim=0)
    torch.testing.assert_close(layer.weight, expected_weight)

    hidden_states = torch.arange(3 * input_size, dtype=torch.float32).reshape(
        3, input_size
    )
    actual = torch.nn.functional.linear(hidden_states, layer.weight)
    expected = torch.cat(
        [torch.nn.functional.linear(hidden_states, shard) for shard in shards], dim=-1
    )
    torch.testing.assert_close(actual, expected)
