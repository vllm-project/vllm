# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for how the backend makes Transformers' rope updates static."""

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, Phi3Config
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding

from vllm.model_executor.models.transformers.utils import (
    can_enable_torch_compile,
    maybe_disable_dynamic_rope_update,
)

MAX_POSITION_EMBEDDINGS = 128
CONFIG_KWARGS = dict(
    hidden_size=64,
    num_attention_heads=4,
    num_hidden_layers=1,
    max_position_embeddings=MAX_POSITION_EMBEDDINGS,
)
DYNAMIC = {"rope_type": "dynamic", "factor": 4.0, "rope_theta": 1e4}
ORIGINAL = MAX_POSITION_EMBEDDINGS // 2

LLAMA_CONFIG = LlamaConfig(**CONFIG_KWARGS, rope_parameters=DYNAMIC)
PHI3_CONFIG = Phi3Config(
    **CONFIG_KWARGS,
    original_max_position_embeddings=ORIGINAL,
    rope_parameters={
        "rope_type": "longrope",
        "short_factor": [1.0] * 8,
        "long_factor": [4.0] * 8,
        "rope_theta": 1e4,
    },
)


def make_vllm_config(hf_config, max_model_len: int) -> SimpleNamespace:
    model_config = SimpleNamespace(hf_config=hf_config, max_model_len=max_model_len)
    return SimpleNamespace(model_config=model_config)


def forbid_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if `@dynamic_rope_update` still runs, since it syncs on `torch.max`."""

    def fail(*args, **kwargs):
        raise AssertionError("rope update still runs")

    monkeypatch.setattr(torch, "max", fail)


@pytest.mark.parametrize(
    "rope_parameters,max_model_len,compilable",
    [
        (DYNAMIC, MAX_POSITION_EMBEDDINGS, True),
        (DYNAMIC, MAX_POSITION_EMBEDDINGS + 1, False),
        # Longrope is frozen at init, so only dynamic rope matters
        (
            {"full_attention": DYNAMIC, "sliding_attention": {"rope_type": "longrope"}},
            MAX_POSITION_EMBEDDINGS,
            True,
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
    "config,rotary_cls,max_model_len,ref_len",
    [
        (LLAMA_CONFIG, LlamaRotaryEmbedding, MAX_POSITION_EMBEDDINGS, 8),
        # Longrope uses the long factors iff `max_model_len` exceeds the original
        (PHI3_CONFIG, Phi3RotaryEmbedding, ORIGINAL, 8),
        (PHI3_CONFIG, Phi3RotaryEmbedding, ORIGINAL + 1, ORIGINAL + 1),
    ],
)
def test_disable_dynamic_rope_update(
    monkeypatch, config, rotary_cls, max_model_len, ref_len
):
    """No sync, and short positions get HF's frequencies for `ref_len` positions."""
    x, short = torch.zeros(1), torch.arange(8)[None]
    expected = rotary_cls(config)(x, torch.arange(ref_len)[None])

    rotary_emb = rotary_cls(config)
    vllm_config = make_vllm_config(config, max_model_len)
    maybe_disable_dynamic_rope_update(rotary_emb, vllm_config)

    forbid_sync(monkeypatch)
    actual = rotary_emb(x, short)
    assert all(torch.equal(e[:, :8], a) for e, a in zip(expected, actual))
