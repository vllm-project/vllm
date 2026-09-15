# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.models.nemotron as nemotron
from vllm.model_executor.models.nemotron import NemotronDecoderLayer


@pytest.mark.parametrize(
    ("quant_name", "expected_offset"),
    [
        (None, 1.0),
        ("gguf", 0.0),
    ],
)
def test_nemotron_layernorm_matches_checkpoint_weight_semantics(
    quant_name,
    expected_offset,
):
    config = Mock(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=8,
        hidden_act="relu",
        norm_eps=1e-5,
        max_position_embeddings=128,
        attention_bias=False,
        bias=False,
        mlp_bias=False,
    )

    quant_config = None
    if quant_name is not None:
        quant_config = Mock()
        quant_config.get_name.return_value = quant_name

    with (
        patch.object(nemotron, "NemotronAttention"),
        patch.object(nemotron, "NemotronMLP"),
    ):
        layer = NemotronDecoderLayer(
            config=config,
            quant_config=quant_config,
        )

    x = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    stored_weight = torch.tensor([0.1, 0.2, 0.3, 0.4])

    for norm in (layer.input_layernorm, layer.post_attention_layernorm):
        with torch.no_grad():
            norm.weight.copy_(stored_weight)
            norm.bias.zero_()

        expected = F.layer_norm(
            x,
            norm.normalized_shape,
            stored_weight + expected_offset,
            norm.bias,
            norm.eps,
        )

        torch.testing.assert_close(norm(x), expected)
