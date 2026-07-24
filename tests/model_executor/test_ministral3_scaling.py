# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import torch
from transformers import Ministral3Config

from vllm.model_executor.models.mistral import (
    MistralAttention,
    _get_llama_4_scaling_config,
)


def test_llama_4_scaling_uses_top_level_config():
    config = SimpleNamespace(
        llama_4_scaling={
            "beta": 0.2,
            "original_max_position_embeddings": 8192,
        },
        rope_parameters={
            "llama_4_scaling_beta": 0.1,
            "original_max_position_embeddings": 16384,
        },
    )

    assert _get_llama_4_scaling_config(config) == config.llama_4_scaling


def test_llama_4_scaling_uses_transformers_rope_parameters():
    config = Ministral3Config()

    assert _get_llama_4_scaling_config(config) == {
        "beta": 0.1,
        "original_max_position_embeddings": 16384,
    }


def test_llama_4_scaling_requires_complete_rope_parameters():
    config = SimpleNamespace(rope_parameters={"llama_4_scaling_beta": 0.1})

    assert _get_llama_4_scaling_config(config) is None


def test_llama_4_scaling_boundary():
    attention = SimpleNamespace(
        llama_4_scaling_beta=0.1,
        llama_4_scaling_original_max_position_embeddings=16384,
    )
    positions = torch.tensor([16383, 16384, 32768])

    actual = MistralAttention._get_llama_4_attn_scale(attention, positions)
    expected = torch.tensor([1.0, 1.0 + 0.1 * math.log(2), 1.0 + 0.1 * math.log(3)])

    torch.testing.assert_close(actual.squeeze(-1), expected)
