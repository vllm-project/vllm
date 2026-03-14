# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for mamba attention backend selectors."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.model_executor.models.minimax_text_01 import MiniMaxText01LinearAttention
from vllm.v1.attention.backends.linear_attn import LinearAttentionBackend
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionBackend
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionBackend
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionBackend


@pytest.mark.parametrize(
    "layer_class, init_kwargs, expected_backend, expected_mamba_type",
    [
        (
            MambaMixer,
            dict(
                hidden_size=128,
                ssm_state_size=16,
                conv_kernel_size=4,
                intermediate_size=256,
                time_step_rank=8,
                use_conv_bias=True,
                use_bias=False,
                use_rms_norm=True,
            ),
            Mamba1AttentionBackend,
            "mamba1",
        ),
        (
            MambaMixer2,
            dict(
                hidden_size=128,
                ssm_state_size=16,
                conv_kernel_size=4,
                intermediate_size=256,
                use_conv_bias=True,
                use_bias=False,
                n_groups=1,
                num_heads=8,
                head_dim=32,
            ),
            Mamba2AttentionBackend,
            "mamba2",
        ),
        (
            MiniMaxText01LinearAttention,
            dict(
                hidden_size=128,
                hidden_inner_size=256,
                num_heads=8,
                head_dim=32,
                max_position=2048,
                block_size=64,
                num_hidden_layer=12,
                layer_idx=0,
                linear_layer_idx=0,
            ),
            LinearAttentionBackend,
            "linear_attention",
        ),
        (
            ShortConv,
            dict(
                config=SimpleNamespace(conv_L_cache=32, conv_bias=True),
                dim=128,
                layer_idx=0,
            ),
            ShortConvAttentionBackend,
            "short_conv",
        ),
    ],
)
def test_mamba_layers_get_attn_backend(
    default_vllm_config,
    dist_init,
    layer_class,
    init_kwargs,
    expected_backend,
    expected_mamba_type,
):
    """Test that Mamba-like layers return the correct attention backend."""
    layer = layer_class(**init_kwargs)

    backend_class = layer.get_attn_backend()
    assert backend_class is expected_backend
    assert layer.mamba_type == expected_mamba_type


@pytest.mark.parametrize(
    "layer_class,expected_backend,expected_mamba_type",
    [
        (MambaMixer, Mamba1AttentionBackend, "mamba1"),
        (MambaMixer2, Mamba2AttentionBackend, "mamba2"),
        (MiniMaxText01LinearAttention, LinearAttentionBackend, "linear_attention"),
        (ShortConv, ShortConvAttentionBackend, "short_conv"),
    ],
)
def test_mamba_layers_have_unified_interface(
    layer_class, expected_backend, expected_mamba_type
):
    """Test that all Mamba layers have the unified get_attn_backend
    interface."""
    assert hasattr(layer_class, "get_attn_backend"), (
        f"{layer_class.__name__} should have get_attn_backend method"
    )
    assert hasattr(layer_class, "mamba_type"), (
        f"{layer_class.__name__} should have mamba_type property"
    )
