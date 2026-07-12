# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
    CompressedTensorsWNA16MoEMethod,
)
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test on CUDA")
def test_moe_wna16_apply_passes_layer_activation(monkeypatch):
    captured_kwargs = {}

    def fake_fused_experts(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return torch.empty(1, 2)

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.fused_experts",
        fake_fused_experts,
    )

    method = object.__new__(MoeWNA16Method)
    method.moe = SimpleNamespace(disable_inplace=False)
    method.moe_quant_config = object()
    layer = SimpleNamespace(
        w13_qweight=torch.empty(1, 2),
        w2_qweight=torch.empty(1, 2),
        activation=MoEActivation.GELU_TANH,
        apply_router_weight_on_input=False,
        global_num_experts=1,
        expert_map=None,
    )

    output = method.apply(
        layer,
        x=torch.empty(1, 2),
        topk_weights=torch.empty(1, 1),
        topk_ids=torch.empty(1, 1, dtype=torch.int32),
        shared_experts=None,
        shared_experts_input=None,
    )

    assert output.shape == (1, 2)
    assert captured_kwargs["activation"] is MoEActivation.GELU_TANH


def test_compressed_tensors_wna16_binds_checkpoint_views_to_runtime_storage():
    layer = torch.nn.Module()
    checkpoint_shapes = {
        "w13_weight_packed": ((2, 4, 6), torch.int32),
        "w2_weight_packed": ((2, 3, 8), torch.int32),
        "w13_weight_scale": ((2, 5, 6), torch.float16),
        "w2_weight_scale": ((2, 3, 8), torch.float16),
        "w13_weight_shape": ((2, 2), torch.float32),
        "w2_weight_shape": ((2, 2), torch.float32),
    }
    for name, (shape, dtype) in checkpoint_shapes.items():
        layer.register_parameter(
            name,
            torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False),
        )

    runtime_params = {
        "w13_weight_packed": torch.nn.Parameter(
            torch.empty((2, 6, 16), dtype=torch.uint8), requires_grad=False
        ),
        "w2_weight_packed": torch.nn.Parameter(
            torch.empty((2, 8, 12), dtype=torch.uint8), requires_grad=False
        ),
        "w13_weight_scale": torch.nn.Parameter(
            torch.empty((2, 6, 5), dtype=torch.float16), requires_grad=False
        ),
        "w2_weight_scale": torch.nn.Parameter(
            torch.empty((2, 8, 3), dtype=torch.float16), requires_grad=False
        ),
        "w13_weight_shape": torch.nn.Parameter(
            torch.empty((2, 2)), requires_grad=False
        ),
        "w2_weight_shape": torch.nn.Parameter(torch.empty((2, 2)), requires_grad=False),
    }
    runtime_ptrs = {name: param.data_ptr() for name, param in runtime_params.items()}
    method = object.__new__(CompressedTensorsWNA16MoEMethod)

    assert method.bind_runtime_weight_reload(layer, runtime_params)
    assert layer.w13_weight_packed.shape == (2, 6, 4)
    assert layer.w2_weight_packed.shape == (2, 8, 3)
    assert layer.w13_weight_scale.shape == (2, 6, 5)
    assert layer.w2_weight_scale.shape == (2, 8, 3)
    assert not layer.w13_weight_packed.is_transposed
    assert not layer.w13_weight_scale.is_transposed
    bound_ptrs = {name: param.data_ptr() for name, param in layer.named_parameters()}
    assert bound_ptrs == runtime_ptrs
