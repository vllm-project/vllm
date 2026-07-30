# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

import vllm.models.kimi_k3.amd.kda as amd_kda
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention as CommonKimiGatedDeltaNetAttention,
)
from vllm.models.kimi_k3.amd.kda import KimiGatedDeltaNetAttention
from vllm.models.kimi_k3.amd.ops.kda_input_projection import (
    kda_input_projection,
)


class _Projection(nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        return hidden_states + 1, None


def _make_amd_layer() -> KimiGatedDeltaNetAttention:
    layer = object.__new__(KimiGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.in_proj_qkvgfab = _Projection()
    layer.register_buffer("_kda_group64_weight", None, persistent=False)
    layer.register_buffer("_kda_group64_scale", None, persistent=False)
    return layer


def test_default_projection_seam_preserves_exact_result() -> None:
    layer = object.__new__(CommonKimiGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.in_proj_qkvgfab = _Projection()
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(1, 8)

    actual = CommonKimiGatedDeltaNetAttention._project_qkvgfab(layer, hidden_states)

    assert torch.equal(actual, layer.in_proj_qkvgfab(hidden_states)[0])


def test_group64_dispatch_falls_back_without_prepacked_weight() -> None:
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    fallback_calls = 0

    def fallback(value: torch.Tensor) -> torch.Tensor:
        nonlocal fallback_calls
        fallback_calls += 1
        return value + 1

    actual = kda_input_projection(hidden_states, fallback, None, None)

    assert fallback_calls == 1
    assert torch.equal(actual, hidden_states + 1)


def test_amd_projection_seam_preserves_fallback_result() -> None:
    layer = _make_amd_layer()
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(1, 8)

    actual = layer._project_qkvgfab(hidden_states)

    assert torch.equal(actual, hidden_states + 1)


def test_post_load_hook_installs_prepacked_buffers(monkeypatch) -> None:
    layer = _make_amd_layer()
    layer.in_proj_qkvgfab = nn.Linear(8, 8, bias=False)
    packed_weight = torch.empty(2, 2)
    packed_scale = torch.empty(2)
    monkeypatch.setattr(
        amd_kda,
        "prepack_kda_input_group64",
        lambda weight: (packed_weight, packed_scale),
    )

    layer.process_weights_after_loading(torch.bfloat16)

    assert layer._kda_group64_weight is packed_weight
    assert layer._kda_group64_scale is packed_scale
