# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for mamba attention backend selectors."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.linear.minimax_linear_attn import (
    MiniMaxText01LinearAttention,
)
from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.model_executor.models.falcon_h1 import FalconH1ForCausalLM
from vllm.model_executor.models.granitemoehybrid import GraniteMoeHybridForCausalLM
from vllm.model_executor.models.zamba2 import Zamba2ForCausalLM
from vllm.v1.attention.backends.linear_attn import LinearAttentionBackend
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionBackend
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionBackend


@pytest.mark.parametrize(("use_replayssm", "expected_blocks"), [(False, 3), (True, 0)])
def test_replayssm_does_not_reserve_speculative_state_blocks(
    use_replayssm, expected_blocks
):
    layer = SimpleNamespace(
        get_state_shape=lambda: ((2,),),
        get_state_dtype=lambda: (torch.float32,),
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
        is_kv_cache_tp_replicated=False,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_block_size=1,
            mamba_page_size_padded=None,
            mamba_cache_mode="none",
            use_replayssm=use_replayssm,
        ),
        num_speculative_tokens=3,
    )

    spec = MambaBase.get_kv_cache_spec(layer, vllm_config)

    assert spec is not None
    assert spec.num_speculative_blocks == expected_blocks


@pytest.mark.parametrize(
    "model_cls", [FalconH1ForCausalLM, GraniteMoeHybridForCausalLM, Zamba2ForCausalLM]
)
def test_mamba2_config_state_shape_includes_speculative_tokens(model_cls):
    """The config-time shape sizes the padded Mamba page, so it must match
    MambaMixer2.get_state_shape, whose conv state grows with spec tokens."""
    hf_config = SimpleNamespace(
        hidden_size=128,
        mamba_expand=2,
        mamba_d_ssm=None,
        mamba_n_groups=1,
        mamba_ngroups=1,
        mamba_n_heads=8,
        n_mamba_heads=8,
        mamba_d_head=32,
        mamba_headdim=32,
        mamba_d_state=16,
        mamba_d_conv=4,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        num_speculative_tokens=2,
    )

    assert model_cls.get_mamba_state_shape_from_config(
        vllm_config
    ) == MambaStateShapeCalculator.mamba2_state_shape(
        tp_world_size=1,
        intermediate_size=256,
        n_groups=1,
        num_heads=8,
        head_dim=32,
        state_size=16,
        conv_kernel=4,
        num_spec=2,
    )


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
            MambaAttentionBackendEnum.MAMBA1,
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
            MambaAttentionBackendEnum.MAMBA2,
        ),
        (
            MiniMaxText01LinearAttention,
            dict(
                config=SimpleNamespace(
                    hidden_size=256,
                    num_attention_heads=8,
                    head_dim=32,
                    num_hidden_layers=12,
                    block=64,
                ),
                prefix="layers.0.self_attn",
            ),
            LinearAttentionBackend,
            MambaAttentionBackendEnum.LINEAR,
        ),
        (
            ShortConv,
            dict(
                config=SimpleNamespace(conv_L_cache=32, conv_bias=True),
                dim=128,
                layer_idx=0,
            ),
            ShortConvAttentionBackend,
            MambaAttentionBackendEnum.SHORT_CONV,
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
    if layer_class is MiniMaxText01LinearAttention:
        init_kwargs["vllm_config"] = default_vllm_config
    layer = layer_class(**init_kwargs)

    backend_class = layer.get_attn_backend()
    assert backend_class is expected_backend
    assert layer.mamba_type == expected_mamba_type


@pytest.mark.parametrize(
    "layer_class,expected_backend,expected_mamba_type",
    [
        (MambaMixer, Mamba1AttentionBackend, MambaAttentionBackendEnum.MAMBA1),
        (MambaMixer2, Mamba2AttentionBackend, MambaAttentionBackendEnum.MAMBA2),
        (
            MiniMaxText01LinearAttention,
            LinearAttentionBackend,
            MambaAttentionBackendEnum.LINEAR,
        ),
        (ShortConv, ShortConvAttentionBackend, MambaAttentionBackendEnum.SHORT_CONV),
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
