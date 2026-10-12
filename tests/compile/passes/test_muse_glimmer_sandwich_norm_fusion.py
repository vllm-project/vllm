# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Muse's sandwich norm fusion, including the decoder-layer boundary."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.compile.backend import TestBackend as CompileTestBackend
from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.compilation.passes.fusion.muse_glimmer_sandwich_norm_fusion import (
    MuseSandwichNormFusionPass,
)
from vllm.compilation.passes.pass_manager import PostGradPassManager
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.models.muse_glimmer import MuseGlimmerDecoderLayer
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not is_aiter_found_and_supported(),
    reason="Requires ROCm and AITER",
)


class _IdentityAttention(nn.Module):
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        return hidden_states


def _make_layer(width: int) -> MuseGlimmerDecoderLayer:
    layer = MuseGlimmerDecoderLayer.__new__(MuseGlimmerDecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = _IdentityAttention()
    layer.mlp = nn.Identity()
    layer.use_amd_fused_norm = True
    layer.input_layernorm = GemmaRMSNorm(width, eps=1e-5)
    layer.post_attention_layernorm = GemmaRMSNorm(width, eps=1e-8)
    layer.pre_feedforward_layernorm = GemmaRMSNorm(width, eps=1e-5)
    layer.post_feedforward_layernorm = GemmaRMSNorm(width, eps=1e-8)
    return layer


class _TwoMuseLayers(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.ModuleList(_make_layer(width) for _ in range(2))

    def forward(self, positions: torch.Tensor, x: torch.Tensor):
        first, residual = self.layers[0](positions, x, None)
        return self.layers[1](positions, first, residual)


def test_muse_sandwich_norm_pass_enabled_by_default() -> None:
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("Muse AITER sandwich norm requires gfx950")

    text_config = SimpleNamespace(
        hidden_size=6656, post_norm_eps=1e-8, rms_norm_eps=1e-5
    )
    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    config.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_config=text_config,
        architecture="MuseGlimmerForCausalLM",
        using_transformers_backend=lambda: False,
    )
    manager = PostGradPassManager()
    manager.configure(config)
    assert any(isinstance(p, MuseSandwichNormFusionPass) for p in manager.passes)


@pytest.mark.parametrize("rows", [1, 64, 257])
@torch.inference_mode()
def test_muse_sandwich_norm_fusion_across_layers(rows: int) -> None:
    torch.manual_seed(1064)
    width = 6656
    text_config = SimpleNamespace(
        hidden_size=width, post_norm_eps=1e-8, rms_norm_eps=1e-5
    )
    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    config.model_config = SimpleNamespace(dtype=torch.bfloat16, hf_config=text_config)
    with set_current_vllm_config(config):
        model = _TwoMuseLayers(width).to(device="cuda", dtype=torch.bfloat16)
        for layer in model.layers:
            for norm in (
                layer.input_layernorm,
                layer.post_attention_layernorm,
                layer.pre_feedforward_layernorm,
                layer.post_feedforward_layernorm,
            ):
                norm.weight.data.normal_(mean=0.0, std=0.1)

        positions = torch.arange(rows, device="cuda")
        x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
        expected = model(positions, x)

        fusion_pass = MuseSandwichNormFusionPass(config)
        backend = CompileTestBackend(
            NoOpEliminationPass(config), fusion_pass, PostCleanupPass(config)
        )
        actual = torch.compile(model, backend=backend)(positions, x)

    # Each post-attention norm pair matches, and the first layer's post-FFN
    # norm matches the input norm of the second layer.
    if fusion_pass.matched_count != 3:
        backend.print_graphs()
    assert fusion_pass.matched_count == 3
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0.05, rtol=0.05)
    backend.check_after_ops([torch.ops.aiter.fused_rmsnorm_add_rmsnorm])
