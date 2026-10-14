# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for treating dynamic rope as default rope where it can never rescale."""

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from vllm.model_executor.models.transformers.utils import (
    can_enable_torch_compile,
    disable_noop_dynamic_rope,
)

MAX_POSITION_EMBEDDINGS = 128
HIDDEN_SIZE = 64


def make_config() -> LlamaConfig:
    return LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        num_hidden_layers=1,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        rope_parameters={"rope_type": "dynamic", "factor": 4.0, "rope_theta": 1e4},
    )


@pytest.mark.parametrize(
    "max_model_len,rope_type,compilable",
    [
        (MAX_POSITION_EMBEDDINGS, "default", True),
        (MAX_POSITION_EMBEDDINGS + 1, "dynamic", False),
    ],
)
def test_disable_noop_dynamic_rope(max_model_len, rope_type, compilable):
    """Dynamic rope only blocks compilation when it can still rescale."""
    config = make_config()
    disable_noop_dynamic_rope(config, max_model_len)
    assert config.rope_parameters["rope_type"] == rope_type

    model_config = SimpleNamespace(hf_config=config, max_model_len=max_model_len)
    vllm_config = SimpleNamespace(model_config=model_config)
    assert can_enable_torch_compile(vllm_config) is compilable


def test_disabled_dynamic_rope_is_unchanged():
    """The rewrite is exact within the pretrained context."""
    inputs = torch.zeros(1, 8, HIDDEN_SIZE), torch.arange(8)[None, :]
    config = make_config()
    expected = LlamaRotaryEmbedding(config)(*inputs)

    disable_noop_dynamic_rope(config, MAX_POSITION_EMBEDDINGS)
    actual = LlamaRotaryEmbedding(config)(*inputs)
    assert all(torch.equal(e, a) for e, a in zip(expected, actual))
