# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn


def bias_only_model(monkeypatch):
    from vllm import platforms
    from vllm.platforms.cpu import CpuPlatform

    monkeypatch.setattr(platforms, "_current_platform", CpuPlatform())
    from vllm.model_executor.models import minicpm5_moe as modeling

    model = modeling.MiniCPM5Model.__new__(modeling.MiniCPM5Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(n_routed_experts=160)
    model.mlp = nn.Module()
    model.mlp.gate = nn.Module()
    model.mlp.gate.e_score_correction_bias = nn.Parameter(torch.empty(4))
    monkeypatch.setattr(
        modeling, "fused_moe_make_expert_params_mapping", lambda *a, **k: []
    )
    return model


def test_correction_bias_preserves_all_fp32_bits(monkeypatch):
    model = bias_only_model(monkeypatch)
    source = torch.tensor(
        [0, -2147483648, 0x3F800001, 0x7FC00123], dtype=torch.int32
    ).view(torch.float32)
    loaded = model.load_weights([("mlp.gate.e_score_correction_bias", source)])
    assert loaded == {"mlp.gate.e_score_correction_bias"}
    assert torch.equal(
        model.mlp.gate.e_score_correction_bias.view(torch.int32),
        source.view(torch.int32),
    )
