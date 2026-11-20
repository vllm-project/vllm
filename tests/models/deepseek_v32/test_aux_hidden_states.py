# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v32.amd import model as amd_model
from vllm.models.deepseek_v32.nvidia import model as nvidia_model


class _AddOffsetLayer(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        attn_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = torch.full_like(hidden_states, self.offset)
        else:
            residual = hidden_states + residual
            hidden_states = torch.full_like(hidden_states, self.offset)
        return hidden_states, residual


class _ScaleNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_residual = hidden_states + residual
        return new_residual * 0.5, new_residual


def _make_nvidia_model(
    *, use_sequence_parallel: bool, aux_layers: tuple[int, ...]
) -> nvidia_model.DeepseekV32Model:
    model = object.__new__(nvidia_model.DeepseekV32Model)
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 2
    model.layers = nn.ModuleList([_AddOffsetLayer(1.0), _AddOffsetLayer(2.0)])
    model.norm = _ScaleNorm()
    model.use_sequence_parallel = use_sequence_parallel
    model.shardq_sequence_parallel = False
    model.shardq_min_tokens = 0
    model.aux_hidden_state_layers = aux_layers
    model.pp_buffers = {}
    return model


def _make_amd_model(*, aux_layers: tuple[int, ...]) -> amd_model.DeepseekV32Model:
    model = object.__new__(amd_model.DeepseekV32Model)
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 2
    model.layers = nn.ModuleList([_AddOffsetLayer(1.0), _AddOffsetLayer(2.0)])
    model.norm = _ScaleNorm()
    model.aux_hidden_state_layers = aux_layers
    model.pp_buffers = {}
    return model


@pytest.mark.parametrize("aux_layers", [(0, 2), (2,)])
def test_nvidia_captures_final_layer_aux_hidden_state(
    monkeypatch, aux_layers: tuple[int, ...]
) -> None:
    model = _make_nvidia_model(use_sequence_parallel=False, aux_layers=aux_layers)
    pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(nvidia_model, "get_pp_group", lambda: pp)
    monkeypatch.setattr(
        nvidia_model,
        "fused_allreduce_rms_norm",
        lambda h, r, norm: norm(h * 10.0, r),
    )

    inputs_embeds = torch.arange(6, dtype=torch.float32).view(3, 2)
    hidden_states, aux = model(
        input_ids=None,
        positions=torch.arange(3),
        inputs_embeds=inputs_embeds,
    )

    # Layer 0 output: h=1, r=inputs_embeds
    # Layer 1 output: h=2, r=inputs_embeds + 1
    # fused_allreduce_rms_norm multiplies partial h by 10 -> new_residual = inputs_embeds + 21
    expected_final_aux = inputs_embeds + 21.0
    torch.testing.assert_close(hidden_states, expected_final_aux * 0.5)
    assert len(aux) == len(aux_layers)
    if 0 in aux_layers:
        torch.testing.assert_close(aux[0], inputs_embeds)
    torch.testing.assert_close(aux[-1], expected_final_aux)


@pytest.mark.parametrize("aux_layers", [(0, 2), (2,)])
def test_nvidia_sequence_parallel_captures_final_layer_aux_hidden_state(
    monkeypatch, aux_layers: tuple[int, ...]
) -> None:
    model = _make_nvidia_model(use_sequence_parallel=True, aux_layers=aux_layers)
    pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(nvidia_model, "get_pp_group", lambda: pp)
    monkeypatch.setattr(nvidia_model, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(
        nvidia_model,
        "sp_shard",
        lambda t: torch.nn.functional.pad(t, (0, 0, 0, 1))[:2],
    )
    monkeypatch.setattr(
        nvidia_model,
        "sp_all_gather",
        lambda t: torch.cat([t, t], dim=0),
    )

    inputs_embeds = torch.arange(6, dtype=torch.float32).view(3, 2)
    hidden_states, aux = model(
        input_ids=None,
        positions=torch.arange(3),
        inputs_embeds=inputs_embeds,
    )

    sharded = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, 1))[:2]
    expected_final_aux = torch.cat([sharded + 3.0, sharded + 3.0], dim=0)[:3]
    torch.testing.assert_close(hidden_states, expected_final_aux * 0.5)
    assert len(aux) == len(aux_layers)
    assert all(t.shape == (3, 2) for t in aux)
    torch.testing.assert_close(aux[-1], expected_final_aux)


@pytest.mark.parametrize("aux_layers", [(0, 2), (2,)])
def test_amd_captures_final_layer_aux_hidden_state(
    monkeypatch, aux_layers: tuple[int, ...]
) -> None:
    model = _make_amd_model(aux_layers=aux_layers)
    pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(amd_model, "get_pp_group", lambda: pp)
    monkeypatch.setattr(
        amd_model,
        "fused_allreduce_rms_norm",
        lambda h, r, norm: norm(h * 10.0, r),
    )

    inputs_embeds = torch.arange(6, dtype=torch.float32).view(3, 2)
    hidden_states, aux = model(
        input_ids=None,
        positions=torch.arange(3),
        inputs_embeds=inputs_embeds,
    )

    expected_final_aux = inputs_embeds + 21.0
    torch.testing.assert_close(hidden_states, expected_final_aux * 0.5)
    assert len(aux) == len(aux_layers)
    if 0 in aux_layers:
        torch.testing.assert_close(aux[0], inputs_embeds)
    torch.testing.assert_close(aux[-1], expected_final_aux)
