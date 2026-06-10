# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import torch
from torch import nn


def set_unspecified_platform():
    import vllm.platforms as platforms
    from vllm.platforms.interface import UnspecifiedPlatform

    platforms._current_platform = UnspecifiedPlatform()


def test_lfm2_short_conv_layer_receives_quant_config():
    set_unspecified_platform()
    from vllm.model_executor.models.lfm2 import Lfm2ShortConvDecoderLayer

    mock_quant_config = Mock()
    mock_config = Mock()
    mock_config.conv_dim = 64
    mock_config.hidden_size = 64
    mock_config.block_dim = 64
    mock_config.intermediate_size = 128
    mock_config.block_multiple_of = 16
    mock_config.block_auto_adjust_ff_dim = False
    mock_config.block_ffn_dim_multiplier = None
    mock_config.norm_eps = 1e-5

    with (
        patch("vllm.model_executor.models.lfm2.ShortConv") as mock_short_conv,
        patch("vllm.model_executor.models.lfm2.Lfm2MLP"),
        patch("vllm.model_executor.models.lfm2.RMSNorm"),
    ):
        Lfm2ShortConvDecoderLayer(
            config=mock_config,
            layer_idx=0,
            quant_config=mock_quant_config,
            prefix="model.layers.0",
        )

    mock_short_conv.assert_called_once()
    assert mock_short_conv.call_args.kwargs["quant_config"] is mock_quant_config


def test_lfm2_moe_short_conv_layer_receives_quant_config():
    set_unspecified_platform()
    from vllm.model_executor.models.lfm2_moe import Lfm2MoeShortConvDecoderLayer

    mock_quant_config = Mock()
    mock_config = Mock()
    mock_config.hidden_size = 64
    mock_config.num_dense_layers = 1
    mock_config.intermediate_size = 128
    mock_config.norm_eps = 1e-5

    with (
        patch("vllm.model_executor.models.lfm2_moe.ShortConv") as mock_short_conv,
        patch("vllm.model_executor.models.lfm2_moe.Lfm2MoeMlp"),
        patch("vllm.model_executor.models.lfm2_moe.RMSNorm"),
    ):
        Lfm2MoeShortConvDecoderLayer(
            config=mock_config,
            layer_idx=0,
            quant_config=mock_quant_config,
            prefix="model.layers.0",
        )

    mock_short_conv.assert_called_once()
    assert mock_short_conv.call_args.kwargs["quant_config"] is mock_quant_config


def test_short_conv_projection_linears_receive_quant_config():
    set_unspecified_platform()
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba.short_conv import ShortConv

    mock_quant_config = Mock()
    mock_config = Mock()
    mock_config.conv_L_cache = 3
    mock_config.conv_bias = False

    column_calls = []
    merged_column_calls = []
    row_calls = []

    def make_fake_linear(*args, **kwargs):
        module = nn.Module()
        module.weight = nn.Parameter(torch.empty(1, 1))
        module.bias = None
        return module

    def make_fake_column_linear(*args, **kwargs):
        column_calls.append(kwargs)
        return make_fake_linear(*args, **kwargs)

    def make_fake_merged_column_linear(*args, **kwargs):
        merged_column_calls.append(kwargs)
        return make_fake_linear(*args, **kwargs)

    def make_fake_row_linear(*args, **kwargs):
        row_calls.append(kwargs)
        return make_fake_linear(*args, **kwargs)

    with (
        patch(
            "vllm.model_executor.layers.mamba.short_conv.ColumnParallelLinear",
            side_effect=make_fake_column_linear,
        ),
        patch(
            "vllm.model_executor.layers.mamba.short_conv.MergedColumnParallelLinear",
            side_effect=make_fake_merged_column_linear,
        ),
        patch(
            "vllm.model_executor.layers.mamba.short_conv.RowParallelLinear",
            side_effect=make_fake_row_linear,
        ),
        set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))),
    ):
        ShortConv(
            config=mock_config,
            dim=64,
            layer_idx=0,
            quant_config=mock_quant_config,
            prefix="model.layers.0.short_conv",
        )

    assert "quant_config" not in column_calls[0]
    assert merged_column_calls[0]["quant_config"] is mock_quant_config
    assert row_calls[0]["quant_config"] is mock_quant_config
