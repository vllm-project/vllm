# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization.utils.humming_utils import (
    apply_humming_linear,
)


def test_humming_linear_passes_explicit_tensors(monkeypatch):
    """Guard the backend boundary against receiving a vLLM layer."""
    import vllm.utils.humming as humming

    calls = {}

    def fake_forward(config, **kwargs):
        calls["config"] = config
        calls.update(kwargs)
        return torch.empty(2, 4)

    monkeypatch.setattr(humming, "humming_forward", fake_forward)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.empty(4, 8))
    layer.weight_scale = torch.nn.Parameter(torch.empty(4, 1))
    layer.zero_point = torch.nn.Parameter(torch.empty(4, 1))
    layer.weight_scale_2 = torch.nn.Parameter(torch.empty(1))
    layer.bias = torch.nn.Parameter(torch.empty(4))

    config = object()
    locks = torch.empty(1024, dtype=torch.int32)
    inputs = torch.empty(2, 8)

    output = apply_humming_linear(
        config,
        "compute-config",
        locks,
        layer,
        inputs,
    )

    assert output.shape == (2, 4)
    assert calls["config"] is config
    assert calls["inputs"] is inputs
    assert calls["weight"] is layer.weight
    assert calls["weight_scale"] is layer.weight_scale
    assert calls["zero_point"] is layer.zero_point
    assert calls["bias"] is layer.bias
    assert calls["weight_scale_2"] is layer.weight_scale_2
    assert calls["locks"] is locks
    assert calls["compute_config"] == "compute-config"
    assert "layer" not in calls


def test_humming_moe_passes_modular_kernel_weight(monkeypatch):
    """Guard MoE execution against reading weights from a captured layer."""
    import vllm.utils.humming as humming
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingExpertsBase,
    )

    calls = {}

    def fake_forward(config, **kwargs):
        calls["config"] = config
        calls.update(kwargs)
        return kwargs["outputs"]

    monkeypatch.setattr(humming, "humming_forward", fake_forward)

    experts = object.__new__(HummingExpertsBase)
    experts.humming_configs = {"w13": object()}
    experts.locks = torch.empty(1024, dtype=torch.int32)
    experts.quant_config = SimpleNamespace(
        w1_scale=torch.empty(1),
        w2_scale=None,
        w1_zp=torch.empty(1),
        w2_zp=None,
        w1_bias=torch.empty(1),
        w2_bias=None,
        g1_alphas=torch.empty(1),
        g2_alphas=None,
    )
    inputs = torch.empty(2, 8)
    weight = torch.empty(4, 8)
    input_scale = torch.empty(2, 1)
    outputs = torch.empty(2, 4)

    result = experts.humming_forward(
        "w13",
        inputs=inputs,
        weight=weight,
        input_scale=input_scale,
        outputs=outputs,
    )

    assert result is outputs
    assert calls["weight"] is weight
    assert calls["weight_scale"] is experts.quant_config.w1_scale
    assert calls["zero_point"] is experts.quant_config.w1_zp
    assert calls["bias"] is experts.quant_config.w1_bias
    assert calls["weight_scale_2"] is experts.quant_config.g1_alphas
    assert "layer" not in calls
