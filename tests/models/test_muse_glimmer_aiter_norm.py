"""Muse's AITER sandwich norm across decoder-layer boundaries."""

import pytest
import torch
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.models.muse_glimmer import MuseGlimmerDecoderLayer


class _IdentityAttention(nn.Module):
    def forward(self, positions, hidden_states):
        return hidden_states


def _make_layer(width: int, fused: bool) -> MuseGlimmerDecoderLayer:
    layer = MuseGlimmerDecoderLayer.__new__(MuseGlimmerDecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = _IdentityAttention()
    layer.mlp = nn.Identity()
    with set_current_vllm_config(VllmConfig()):
        layer.input_layernorm = GemmaRMSNorm(width, eps=1e-5)
        layer.post_attention_layernorm = GemmaRMSNorm(width, eps=1e-8)
        layer.pre_feedforward_layernorm = GemmaRMSNorm(width, eps=1e-5)
        layer.post_feedforward_layernorm = GemmaRMSNorm(width, eps=1e-8)
    layer.use_amd_fused_norm = True
    layer.use_aiter_sandwich_norm = fused
    if fused:
        from aiter.ops.triton.normalization.fused_rmsnorm_add_rmsnorm import (
            fused_rmsnorm_add_rmsnorm,
        )

        layer.fused_rmsnorm_add_rmsnorm = fused_rmsnorm_add_rmsnorm
    return layer.to(device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("rows", [1, 64, 65])
@torch.inference_mode()
def test_muse_sandwich_norm_across_layers(rows: int) -> None:
    if torch.version.hip is None:
        pytest.skip("AITER sandwich norm requires ROCm")

    torch.manual_seed(1064)
    width = 6656
    baseline_layers = [_make_layer(width, False) for _ in range(2)]
    for layer in baseline_layers:
        for norm in (
            layer.input_layernorm,
            layer.post_attention_layernorm,
            layer.pre_feedforward_layernorm,
            layer.post_feedforward_layernorm,
        ):
            norm.weight.data.normal_(mean=0.0, std=0.1)

    fused_layers = [_make_layer(width, True) for _ in range(2)]
    for fused, baseline in zip(fused_layers, baseline_layers, strict=True):
        fused.load_state_dict(baseline.state_dict())

    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(rows, device="cuda")
    baseline_first, baseline_residual = baseline_layers[0](positions, x, None)
    baseline_last, _ = baseline_layers[1](
        positions, baseline_first, baseline_residual
    )
    fused_first, fused_residual = fused_layers[0](
        positions, x, None, fused_layers[1].input_layernorm, False
    )
    fused_last, _ = fused_layers[1](
        positions, fused_first, fused_residual, None, True
    )

    torch.testing.assert_close(fused_residual, baseline_first, atol=0.05, rtol=0.05)
    torch.testing.assert_close(fused_last, baseline_last, atol=0.05, rtol=0.05)
