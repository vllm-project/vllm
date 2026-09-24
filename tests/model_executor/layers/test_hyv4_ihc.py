# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HY V4 cross-layer iHC scheduling."""

import pytest
import torch
from torch import nn


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


class _Scale(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.factor


class _Attention(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(
        self, *, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        del positions
        return hidden_states * self.factor


class _HC(nn.Module):
    def __init__(self, hc_mult: int, gate: float) -> None:
        super().__init__()
        self.hc_mult = hc_mult
        self.gate = gate

    def prepare_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == 2:
            return hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        return hidden_states

    def pre(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        post_gates = hidden_states.new_full(hidden_states.shape[:2], self.gate)
        return hidden_states.mean(dim=1), post_gates, hidden_states

    def post(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_gates: torch.Tensor,
    ) -> torch.Tensor:
        return residual + post_gates.unsqueeze(-1) * hidden_states.unsqueeze(1)


class _FusedPreNorm(nn.Module):
    def __init__(self, owner: _HC, norm: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "norm", norm)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, post_gates, _ = self.owner.pre(residual)
        return self.norm(hidden_states), post_gates


class _FusedPostPre(nn.Module):
    def __init__(self, pre_owner: _HC, norm: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "pre_owner", pre_owner)
        object.__setattr__(self, "norm", norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_gates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = residual + post_gates.unsqueeze(-1) * hidden_states.unsqueeze(1)
        hidden_states, post_gates, _ = self.pre_owner.pre(residual)
        return residual, self.norm(hidden_states), post_gates


def _make_decoder_layer(layer_index: int):
    from vllm.models.hy_v4.nvidia.model import HYV4DecoderLayer

    layer = HYV4DecoderLayer.__new__(HYV4DecoderLayer)
    nn.Module.__init__(layer)
    layer.enable_ihc = True
    layer.hc_attn_layer = _HC(hc_mult=4, gate=0.2 + layer_index * 0.03)
    layer.hc_mlp_layer = _HC(hc_mult=4, gate=0.4 + layer_index * 0.02)
    layer.input_layernorm = _Scale(1.1 + layer_index * 0.01)
    layer.post_attention_layernorm = _Scale(0.9 + layer_index * 0.01)
    layer.self_attn = _Attention(0.7 + layer_index * 0.02)
    layer.mlp = _Scale(1.3 + layer_index * 0.03)
    layer.hpc_attn_pre_norm = _FusedPreNorm(layer.hc_attn_layer, layer.input_layernorm)
    layer.hpc_mlp_post_pre = _FusedPostPre(
        layer.hc_mlp_layer, layer.post_attention_layernorm
    )
    layer.hpc_attn_post_pre = _FusedPostPre(layer.hc_attn_layer, layer.input_layernorm)
    return layer


def test_cross_layer_ihc_matches_layer_by_layer_schedule() -> None:
    """The carried post boundary must be finalized exactly once per layer."""
    from vllm.models.hy_v4.nvidia.model import IHCCarry

    positions = torch.arange(3)
    inputs = torch.arange(18, dtype=torch.float32).reshape(3, 6) / 10
    reference_layers = [_make_decoder_layer(index) for index in range(3)]
    fused_layers = [_make_decoder_layer(index) for index in range(3)]

    reference = inputs
    for layer in reference_layers:
        reference, _ = layer._forward_ihc(positions, reference)

    actual = inputs
    carry: IHCCarry | None = None
    for index, layer in enumerate(fused_layers):
        actual, carry = layer._forward_ihc_fused(
            positions,
            actual,
            carry,
            is_last_on_rank=index == len(fused_layers) - 1,
        )

    assert carry is None
    torch.testing.assert_close(actual, reference)
