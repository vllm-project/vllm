# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest

from vllm.platforms import current_platform


def test_qwen3_5_models_use_platform_specific_implementation():
    from vllm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MTP

    # NVIDIA, XPU and ROCm variants drop torch.compile; other platforms
    # keep the previous torch.compile implementation in
    # vllm/model_executor/models/qwen3_5{,_mtp}.py.
    if current_platform.is_cuda():
        model_module = "vllm.models.qwen3_5.nvidia.model"
        mtp_module = "vllm.models.qwen3_5.nvidia.mtp"
    elif current_platform.is_xpu():
        model_module = "vllm.models.qwen3_5.xpu.model"
        mtp_module = "vllm.models.qwen3_5.xpu.mtp"
    elif current_platform.is_rocm():
        model_module = "vllm.models.qwen3_5.amd.model"
        mtp_module = "vllm.models.qwen3_5.amd.mtp"
    else:
        model_module = "vllm.model_executor.models.qwen3_5"
        mtp_module = "vllm.model_executor.models.qwen3_5_mtp"
    assert Qwen3_5ForCausalLM.__module__ == model_module
    assert Qwen3_5MTP.__module__ == mtp_module


@pytest.mark.parametrize("platform", ["nvidia", "xpu", "amd"])
def test_qwen3_5_platform_models_do_not_use_torch_compile_wrapper(platform: str):
    # Import the platform modules directly so the assertion is independent of
    # the platform this test runs on.
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    model_module = importlib.import_module(f"vllm.models.qwen3_5.{platform}.model")
    mtp_module = importlib.import_module(f"vllm.models.qwen3_5.{platform}.mtp")

    assert not issubclass(model_module.Qwen3_5Model, TorchCompileWithNoGuardsWrapper)
    assert not issubclass(
        mtp_module.Qwen3_5MultiTokenPredictor, TorchCompileWithNoGuardsWrapper
    )
    assert not issubclass(mtp_module.Qwen3_5MTP, TorchCompileWithNoGuardsWrapper)


def test_qwen3_5_legacy_models_keep_torch_compile_wrapper():
    # The legacy implementation in vllm/model_executor/models/ serves the
    # other platforms; it must keep the torch.compile wrapper.
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.model_executor.models import qwen3_5 as model_module
    from vllm.model_executor.models import qwen3_5_mtp as mtp_module

    assert issubclass(model_module.Qwen3_5Model, TorchCompileWithNoGuardsWrapper)
    assert issubclass(
        mtp_module.Qwen3_5MultiTokenPredictor, TorchCompileWithNoGuardsWrapper
    )
    assert issubclass(mtp_module.Qwen3_5MTP, TorchCompileWithNoGuardsWrapper)


@pytest.mark.usefixtures("dist_init")
def test_packed_gate_weights_across_layers():
    import torch
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    from torch import nn

    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors as ct,
    )
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLMBase,
        load_qwen3_5_weights,
    )
    from vllm.model_executor.models.utils import PPMissingLayer

    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module(), nn.Module(), PPMissingLayer()])
    groups = {}
    packed = []
    expected = {}
    for i, group_size in enumerate((128, 64)):
        layer = model.model.layers[i]
        layer.mlp = nn.Module()
        layer.mlp.shared_expert_gate = ReplicatedLinear(
            256, 1, bias=False, params_dtype=torch.float32
        )
        gate_name = f"model.layers.{i}.mlp.shared_expert_gate"
        groups[str(i)] = {
            "targets": [gate_name],
            "format": "pack-quantized",
            "weights": {
                "num_bits": 4,
                "type": "int",
                "strategy": "group",
                "group_size": group_size,
                "symmetric": True,
            },
        }
        q = (torch.arange(256).reshape(1, -1) % 16 - 8).to(torch.int8)
        scale = torch.arange(1, 1 + 256 // group_size).reshape(1, -1).float() / 64
        expected[gate_name + ".weight"] = q * scale.repeat_interleave(group_size, -1)
        packed.append(
            [
                (gate_name + ".weight_packed", pack_to_int32(q, 4)),
                (gate_name + ".weight_scale", scale),
                (gate_name + ".weight_shape", torch.tensor(q.shape)),
            ]
        )

    model.quant_config = ct.CompressedTensorsConfig.from_config(
        {
            "format": "pack-quantized",
            "quantization_status": "compressed",
            "config_groups": groups,
        }
    )

    mapper = Qwen3_5ForCausalLMBase.hf_to_vllm_mapper
    # Use conditional checkpoint names through the text-only entry point's mapper.
    weights = [
        (
            "model.language_model.layers.2.mlp.shared_expert_gate.weight_packed",
            packed[0][0][1],
        )
    ]
    for first, second in zip(packed[0], reversed(packed[1]), strict=True):
        for name, tensor in (first, second):
            weights.append((name.replace("model.", "model.language_model.", 1), tensor))

    assert load_qwen3_5_weights(model, weights, mapper) == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(model.get_parameter(name), value, rtol=0, atol=0)
