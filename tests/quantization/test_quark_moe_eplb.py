# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.quantization.quark import quark_moe
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
)


class _MockLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter(
            "w13_weight",
            torch.nn.Parameter(
                torch.zeros(2, 4, 2, dtype=torch.uint8), requires_grad=False
            ),
        )
        self.register_parameter(
            "w2_weight",
            torch.nn.Parameter(
                torch.zeros(2, 3, 2, dtype=torch.uint8), requires_grad=False
            ),
        )
        self.register_parameter(
            "w13_weight_scale",
            torch.nn.Parameter(
                torch.ones(2, 4, 1, dtype=torch.uint8), requires_grad=False
            ),
        )
        self.register_parameter(
            "w2_weight_scale",
            torch.nn.Parameter(
                torch.ones(2, 3, 1, dtype=torch.uint8), requires_grad=False
            ),
        )
        self.w13_input_scale = torch.ones(2)
        self.w2_input_scale = torch.ones(2)
        self.w13_bias = None
        self.w2_bias = None


class _MockStorage:
    def __init__(self, data: torch.Tensor):
        self.data = data


class _MockWrappedTensor:
    def __init__(self, data: torch.Tensor):
        self.storage = _MockStorage(data)
        self.shape = data.shape


class _MockPrecisionConfig:
    def __init__(self, weight_scale: _MockWrappedTensor):
        self.weight_scale = weight_scale


def test_quark_ocp_mx_aiter_fp8_registers_kernel_storage_for_eplb(
    monkeypatch,
):
    layer = _MockLayer()
    method = QuarkOCP_MX_MoEMethod.__new__(QuarkOCP_MX_MoEMethod)
    method.mxfp4_backend = Mxfp4MoeBackend.AITER_MXFP4_FP8
    method.experts_cls = None
    method.moe_kernel = None
    method.moe = SimpleNamespace(moe_parallel_config=SimpleNamespace(enable_eplb=True))
    method.get_fused_moe_quant_config = lambda _layer: None

    w13_storage = torch.full_like(layer.w13_weight, 1)
    w2_storage = torch.full_like(layer.w2_weight, 2)
    w13_scale_storage = torch.full_like(layer.w13_weight_scale, 3)
    w2_scale_storage = torch.full_like(layer.w2_weight_scale, 4)

    def fake_convert(**kwargs):
        del layer.w13_weight
        del layer.w2_weight
        return (
            _MockWrappedTensor(w13_storage),
            _MockWrappedTensor(w2_storage),
            _MockPrecisionConfig(_MockWrappedTensor(w13_scale_storage)),
            _MockPrecisionConfig(_MockWrappedTensor(w2_scale_storage)),
            None,
            None,
        )

    monkeypatch.setattr(
        quark_moe,
        "convert_gpt_oss_weight_to_mxfp4_moe_kernel_format",
        fake_convert,
    )
    monkeypatch.setattr(quark_moe.torch.accelerator, "empty_cache", lambda: None)

    method._setup_kernel(layer)

    params = dict(layer.named_parameters())
    assert torch.equal(params["w13_weight_eplb"], w13_storage)
    assert torch.equal(params["w2_weight_eplb"], w2_storage)
    assert torch.equal(params["w13_weight_scale"], w13_scale_storage)
    assert torch.equal(params["w2_weight_scale"], w2_scale_storage)
    assert (
        layer.w13_weight.storage.data.data_ptr() == params["w13_weight_eplb"].data_ptr()
    )
    assert (
        layer.w2_weight.storage.data.data_ptr() == params["w2_weight_eplb"].data_ptr()
    )
    assert (
        method.w13_precision_config.weight_scale.storage.data.data_ptr()
        == params["w13_weight_scale"].data_ptr()
    )
    assert (
        method.w2_precision_config.weight_scale.storage.data.data_ptr()
        == params["w2_weight_scale"].data_ptr()
    )
