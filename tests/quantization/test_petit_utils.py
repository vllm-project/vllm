# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization import petit as petit_quant
from vllm.model_executor.layers.quantization.petit import (
    PetitMxFp4Config,
    PetitMxFp4LinearMethod,
)
from vllm.model_executor.layers.quantization.utils import petit_utils


def test_get_quant_config_petit_mxfp4():
    assert get_quantization_config("petit_mxfp4") is PetitMxFp4Config


def test_verify_petit_mxfp4_supported():
    petit_utils.verify_petit_mxfp4_supported("MXFP4", 32)
    petit_utils.verify_petit_mxfp4_supported("MXFP4", None)

    with pytest.raises(ValueError, match="MXFP4"):
        petit_utils.verify_petit_mxfp4_supported("NVFP4", 32)

    with pytest.raises(ValueError, match="group_size=32"):
        petit_utils.verify_petit_mxfp4_supported("MXFP4", 16)


def test_petit_mxfp4_override_quantization_method(monkeypatch):
    monkeypatch.setattr(petit_quant.current_platform, "is_rocm", lambda: True)

    assert (
        PetitMxFp4Config.override_quantization_method(
            {"quantization": {"quant_algo": "MXFP4"}}, None
        )
        == "petit_mxfp4"
    )
    assert (
        PetitMxFp4Config.override_quantization_method({"quant_algo": "mxfp4"}, None)
        == "petit_mxfp4"
    )
    assert (
        PetitMxFp4Config.override_quantization_method({"quant_method": "mxfp4"}, None)
        is None
    )


def test_petit_mxfp4_from_config_supports_root_quant_algo():
    cfg = PetitMxFp4Config.from_config(
        {
            "quant_algo": "mxfp4",
            "group_size": 32,
            "exclude_modules": ["model.layers.*.mlp.gate_up_proj"],
        }
    )
    assert cfg.group_size == 32
    assert cfg.require_exclude_modules() == ["model.layers.*.mlp.gate_up_proj"]


def test_prepare_and_apply_petit_mxfp4_linear(monkeypatch):
    calls: dict[str, tuple] = {}

    class FakePetitKernel:
        def repack_mxfp4(self, qw, size_n, size_k):
            calls["repack_mxfp4"] = (tuple(qw.shape), size_n, size_k, qw.dtype)
            return qw

        def process_mxfp4_scales(self, scales, size_n, size_k):
            calls["process_mxfp4_scales"] = (
                tuple(scales.shape),
                size_n,
                size_k,
                scales.dtype,
            )
            return scales

        def mul_mxfp4_a16(
            self,
            a,
            b,
            s,
            global_scale,
            size_m,
            size_n,
            size_k,
            solution_id,
        ):
            calls["mul_mxfp4_a16"] = (
                tuple(a.shape),
                tuple(b.shape),
                tuple(s.shape),
                tuple(global_scale.shape),
                size_m,
                size_n,
                size_k,
                solution_id,
            )
            return torch.zeros((size_m, size_n), dtype=a.dtype, device=a.device)

    monkeypatch.setattr(petit_utils, "_petit_kernel", FakePetitKernel())

    layer = torch.nn.Module()
    layer.output_size_per_partition = 32
    layer.input_size_per_partition = 64
    layer.weight = torch.nn.Parameter(
        torch.randint(
            0,
            256,
            (layer.output_size_per_partition, layer.input_size_per_partition // 2),
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.randint(
            1,
            128,
            (layer.output_size_per_partition, layer.input_size_per_partition // 32),
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )

    petit_utils.prepare_mxfp4_layer_for_petit(layer)
    assert "repack_mxfp4" in calls
    assert "process_mxfp4_scales" in calls

    x = torch.randn((3, layer.input_size_per_partition), dtype=torch.bfloat16)
    bias = torch.ones(layer.output_size_per_partition, dtype=torch.bfloat16)
    y = petit_utils.apply_petit_mxfp4_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_scale_2=torch.ones(1, dtype=torch.float32),
        size_n=layer.output_size_per_partition,
        size_k=layer.input_size_per_partition,
        bias=bias,
    )
    assert y.shape == (3, layer.output_size_per_partition)
    assert torch.allclose(y, torch.ones_like(y))
    assert "mul_mxfp4_a16" in calls


def test_petit_mxfp4_weight_scale_2_uses_loaded_value(monkeypatch, caplog):
    monkeypatch.setattr(
        petit_quant, "prepare_mxfp4_layer_for_petit", lambda layer: None
    )
    from vllm.model_executor import parameter as param_mod

    monkeypatch.setattr(param_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(param_mod, "get_tensor_model_parallel_world_size", lambda: 1)

    method = PetitMxFp4LinearMethod(PetitMxFp4Config(group_size=32))
    layer = torch.nn.Module()

    def _weight_loader(param, loaded_weight):
        param.data.copy_(loaded_weight)

    method.create_weights(
        layer=layer,
        input_size_per_partition=64,
        output_partition_sizes=[16, 16],
        input_size=64,
        output_size=32,
        params_dtype=torch.bfloat16,
        weight_loader=_weight_loader,
    )

    assert torch.allclose(layer.weight_scale_2, torch.ones_like(layer.weight_scale_2))
    layer.weight_scale_2.data.copy_(torch.tensor([0.5, 1.75], dtype=torch.float32))

    with caplog.at_level("WARNING", logger=petit_quant.__name__):
        method.process_weights_after_loading(layer)

    assert layer.weight_scale_2.shape == (1,)
    assert torch.allclose(layer.weight_scale_2, torch.tensor([1.75]))
    assert "falls back to max(weight_scale_2)" in caplog.text
