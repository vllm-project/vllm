# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for dropping Transformers' no-op `@dynamic_rope_update`.

The decorator reads `torch.max(position_ids)` back to the host to decide whether
to rescale `inv_freq`. That sync is invisible to `torch.compile` and illegal
during CUDA graph capture, so the backend removes the decorator from the RoPE
modules whose rescaling `max_model_len` puts out of reach, and only those.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from vllm.model_executor.models.transformers.utils import (
    can_enable_torch_compile,
    dynamic_rope_update_is_noop,
    remove_noop_dynamic_rope_update,
)

MAX_POSITION_EMBEDDINGS = 128
HIDDEN_SIZE = 64

DEFAULT = {"rope_type": "default", "rope_theta": 10000.0}
DYNAMIC = {"rope_type": "dynamic", "factor": 4.0, "rope_theta": 10000.0}
LONGROPE = {
    "rope_type": "longrope",
    "rope_theta": 10000.0,
    "original_max_position_embeddings": MAX_POSITION_EMBEDDINGS,
    "short_factor": [1.0] * 8,
    "long_factor": [2.0] * 8,
}


def make_config(rope_parameters: dict) -> LlamaConfig:
    return LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        num_hidden_layers=1,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        rope_parameters=rope_parameters,
    )


def make_vllm_config(hf_config, max_model_len: int) -> SimpleNamespace:
    """Enough of a `VllmConfig` for the checks under test."""
    model_config = SimpleNamespace(hf_config=hf_config, max_model_len=max_model_len)
    return SimpleNamespace(model_config=model_config)


def rescales_inv_freq(rotary_embedding: LlamaRotaryEmbedding) -> bool:
    """Whether the module still rewrites `inv_freq` for an over-long position.

    Only the frequency update does that, so this says whether the decorator is
    still in place without reaching into how it is attached.
    """
    positions = torch.arange(MAX_POSITION_EMBEDDINGS * 2)[None, :]
    rotary_embedding(torch.zeros(1, 1, HIDDEN_SIZE), positions)
    original = rotary_embedding.original_inv_freq
    return not torch.equal(rotary_embedding.inv_freq, original)


@pytest.fixture
def rope_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(1, 8, HIDDEN_SIZE), torch.arange(8)[None, :]


@pytest.mark.parametrize(
    "max_model_len", [MAX_POSITION_EMBEDDINGS - 1, MAX_POSITION_EMBEDDINGS]
)
def test_update_is_noop_within_pretrained_context(max_model_len):
    """Positions vLLM can emit never reach the cached length, so nothing rescales."""
    vllm_config = make_vllm_config(make_config(DYNAMIC), max_model_len)
    assert dynamic_rope_update_is_noop(vllm_config) is True


def test_update_is_not_noop_beyond_pretrained_context():
    """A context longer than the pretrained one is what dynamic rope scales for."""
    vllm_config = make_vllm_config(make_config(DYNAMIC), MAX_POSITION_EMBEDDINGS + 1)
    assert dynamic_rope_update_is_noop(vllm_config) is False


@pytest.mark.parametrize(
    "rope_parameters,max_model_len,expected",
    [
        (DEFAULT, MAX_POSITION_EMBEDDINGS * 2, True),
        (DYNAMIC, MAX_POSITION_EMBEDDINGS, True),
        (DYNAMIC, MAX_POSITION_EMBEDDINGS + 1, False),
    ],
)
def test_can_enable_torch_compile(rope_parameters, max_model_len, expected):
    """Compilation is only withheld from rope that can still rescale itself."""
    vllm_config = make_vllm_config(make_config(rope_parameters), max_model_len)
    assert can_enable_torch_compile(vllm_config) is expected


@pytest.mark.parametrize(
    "max_model_len,expected",
    [(MAX_POSITION_EMBEDDINGS, True), (MAX_POSITION_EMBEDDINGS + 1, False)],
)
def test_can_enable_torch_compile_per_layer_rope(max_model_len, expected):
    """A dynamic layer type decides for the whole model, however few there are."""
    hf_config = SimpleNamespace(
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        rope_parameters={"full_attention": DYNAMIC, "sliding_attention": DEFAULT},
    )
    hf_config.get_text_config = lambda: hf_config
    vllm_config = make_vllm_config(hf_config, max_model_len)
    assert can_enable_torch_compile(vllm_config) is expected


def test_removes_update_that_cannot_fire():
    """Within the pretrained context the update is dead code, so it goes away."""
    rotary_embedding = LlamaRotaryEmbedding(make_config(DYNAMIC))
    assert rescales_inv_freq(rotary_embedding)

    rotary_embedding = LlamaRotaryEmbedding(make_config(DYNAMIC))
    remove_noop_dynamic_rope_update(rotary_embedding, MAX_POSITION_EMBEDDINGS)
    assert not rescales_inv_freq(rotary_embedding)


def test_removal_does_not_change_frequencies(rope_inputs):
    """The forward left behind returns exactly what the decorated one returned."""
    rotary_embedding = LlamaRotaryEmbedding(make_config(DYNAMIC))
    decorated = rotary_embedding(*rope_inputs)

    remove_noop_dynamic_rope_update(rotary_embedding, MAX_POSITION_EMBEDDINGS)

    undecorated = rotary_embedding(*rope_inputs)
    assert all(torch.equal(a, b) for a, b in zip(decorated, undecorated))


def test_remaining_forward_keeps_no_grad(rope_inputs):
    """Transformers stacks `@torch.no_grad()` on top; unwrapping must not lose it."""
    rotary_embedding = LlamaRotaryEmbedding(make_config(DYNAMIC))
    remove_noop_dynamic_rope_update(rotary_embedding, MAX_POSITION_EMBEDDINGS)

    with torch.enable_grad():
        cos, sin = rotary_embedding(*rope_inputs)

    assert not cos.requires_grad
    assert not sin.requires_grad


@pytest.mark.parametrize(
    "rope_parameters,max_model_len",
    [
        # Rescaling is reachable, so the update still has work to do
        (DYNAMIC, MAX_POSITION_EMBEDDINGS + 1),
        # longrope swaps `inv_freq` at any context length
        (LONGROPE, MAX_POSITION_EMBEDDINGS),
    ],
)
def test_keeps_update_that_can_still_fire(rope_parameters, max_model_len):
    rotary_embedding = LlamaRotaryEmbedding(make_config(rope_parameters))
    remove_noop_dynamic_rope_update(rotary_embedding, max_model_len)
    assert rescales_inv_freq(rotary_embedding)
