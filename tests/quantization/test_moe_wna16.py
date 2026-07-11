# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import WNA16MoEBackend
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


@pytest.mark.parametrize(
    ("actorder", "expected_g_idx_size"),
    [(None, 0), ("group", 256)],
)
def test_compressed_tensors_wna16_only_allocates_loaded_g_idx(
    actorder, expected_g_idx_size
):
    module = importlib.import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_wna16_marlin"
    )
    method = object.__new__(module.CompressedTensorsWNA16MarlinMoEMethod)
    method.moe = SimpleNamespace(is_act_and_mul=True, has_bias=False)
    method.packed_factor = 8
    method.wna16_backend = WNA16MoEBackend.MARLIN
    method.actorder = actorder
    method.group_size = 128
    method.strategy = "group"
    method.symmetric = True
    layer = torch.nn.Module()

    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=256,
        intermediate_size_per_partition=256,
        intermediate_size_full=256,
        params_dtype=torch.float16,
    )

    assert layer.w13_weight_g_idx.shape == (2, expected_g_idx_size)
    assert layer.w2_weight_g_idx.shape == (2, expected_g_idx_size)
    assert layer.w13_g_idx_sort_indices.shape == (2, 0)
    assert layer.w2_g_idx_sort_indices.shape == (2, 0)
