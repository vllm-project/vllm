# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for skipping dynamic rope's device sync where it can never rescale."""

from types import SimpleNamespace

import pytest
import torch
from transformers import HunYuanDenseV1Config, LlamaConfig
from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
    HunYuanDenseV1RotaryEmbedding,
)
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from vllm.model_executor.models.transformers.utils import (
    can_enable_torch_compile,
    disable_noop_dynamic_rope,
)

MAX_POSITION_EMBEDDINGS = 128
CONFIG_KWARGS = dict(
    hidden_size=64,
    num_attention_heads=4,
    num_hidden_layers=1,
    max_position_embeddings=MAX_POSITION_EMBEDDINGS,
)
DYNAMIC = {"rope_type": "dynamic", "factor": 4.0, "rope_theta": 1e4}


def make_vllm_config(hf_config, max_model_len: int) -> SimpleNamespace:
    model_config = SimpleNamespace(hf_config=hf_config, max_model_len=max_model_len)
    return SimpleNamespace(model_config=model_config)


@pytest.mark.parametrize(
    "rope_parameters,max_model_len,compilable",
    [
        (DYNAMIC, MAX_POSITION_EMBEDDINGS, True),
        (DYNAMIC, MAX_POSITION_EMBEDDINGS + 1, False),
        # The longrope layer type still syncs
        (
            {"full_attention": DYNAMIC, "sliding_attention": {"rope_type": "longrope"}},
            MAX_POSITION_EMBEDDINGS,
            False,
        ),
    ],
)
def test_can_enable_torch_compile(rope_parameters, max_model_len, compilable):
    """Dynamic rope only blocks compilation when it can still rescale."""
    config = SimpleNamespace(
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        rope_parameters=rope_parameters,
    )
    config.get_text_config = lambda: config
    vllm_config = make_vllm_config(config, max_model_len)
    assert can_enable_torch_compile(vllm_config) is compilable


@pytest.mark.parametrize(
    "config,rotary_cls",
    [
        (LlamaConfig(**CONFIG_KWARGS, rope_parameters=DYNAMIC), LlamaRotaryEmbedding),
        # Hunyuan's dynamic init also applies `alpha`, which must be kept
        (
            HunYuanDenseV1Config(
                **CONFIG_KWARGS, head_dim=16, rope_parameters=DYNAMIC | {"alpha": 1e3}
            ),
            HunYuanDenseV1RotaryEmbedding,
        ),
    ],
)
def test_disable_noop_dynamic_rope(monkeypatch, config, rotary_cls):
    """The update no longer syncs and the frequencies are unchanged."""
    inputs = torch.zeros(1, 8, 64), torch.arange(8)[None, :]
    rotary_emb = rotary_cls(config)
    expected = rotary_emb(*inputs)

    vllm_config = make_vllm_config(config, MAX_POSITION_EMBEDDINGS)
    disable_noop_dynamic_rope(rotary_emb, vllm_config)

    def no_sync(*args, **kwargs):
        raise AssertionError("dynamic rope update still runs")

    monkeypatch.setattr(torch, "max", no_sync)
    actual = rotary_emb(*inputs)
    assert all(torch.equal(e, a) for e, a in zip(expected, actual))
