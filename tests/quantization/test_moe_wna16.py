# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
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
