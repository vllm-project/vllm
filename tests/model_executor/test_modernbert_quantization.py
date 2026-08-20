# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
from torch import nn

pytestmark = pytest.mark.skip_global_cleanup


def test_modernbert_encoder_linears_receive_quant_config():
    from vllm.model_executor.models.modernbert import ModernBertEncoderLayer

    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=24,
        num_attention_heads=4,
        num_hidden_layers=1,
        deterministic_flash_attn=False,
        attention_bias=False,
        layer_types=["full_attention"],
        rope_parameters={
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 160_000.0,
            }
        },
        max_position_embeddings=128,
        mlp_bias=False,
        hidden_activation="gelu",
        norm_eps=1e-5,
        norm_bias=False,
    )
    quant_config = Mock()
    vllm_config = Mock()
    vllm_config.model_config.hf_config = config
    vllm_config.model_config.dtype = None
    vllm_config.quant_config = quant_config

    with (
        patch(
            "vllm.model_executor.models.modernbert."
            "get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.model_executor.models.modernbert.QKVParallelLinear",
            return_value=nn.Identity(),
        ) as qkv_linear,
        patch(
            "vllm.model_executor.models.modernbert.MergedColumnParallelLinear",
            return_value=nn.Identity(),
        ) as merged_linear,
        patch(
            "vllm.model_executor.models.modernbert.RowParallelLinear",
            side_effect=[nn.Identity(), nn.Identity()],
        ) as row_linear,
        patch(
            "vllm.model_executor.models.modernbert.get_rope",
            return_value=nn.Identity(),
        ),
        patch(
            "vllm.model_executor.models.modernbert.EncoderOnlyAttention",
            return_value=nn.Identity(),
        ),
    ):
        ModernBertEncoderLayer(vllm_config, prefix="encoder")

    qkv_linear.assert_called_once_with(
        16,
        4,
        4,
        bias=False,
        quant_config=quant_config,
        prefix="encoder.layers.0.attn.Wqkv",
    )
    merged_linear.assert_called_once_with(
        16,
        [24, 24],
        bias=False,
        quant_config=quant_config,
        prefix="encoder.layers.0.mlp.Wi",
    )
    row_linear.assert_has_calls(
        [
            call(
                16,
                16,
                bias=False,
                quant_config=quant_config,
                prefix="encoder.layers.0.attn.Wo",
            ),
            call(
                24,
                16,
                bias=False,
                quant_config=quant_config,
                prefix="encoder.layers.0.mlp.Wo",
            ),
        ]
    )
