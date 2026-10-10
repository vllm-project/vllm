# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.model_executor.models.granite_speech import (
    GraniteSpeechConformerAttention,
)


def _make_config() -> PretrainedConfig:
    return PretrainedConfig(
        hidden_dim=16,
        context_size=8,
        num_heads=2,
        dim_head=8,
        max_pos_emb=16,
    )


def _make_attention_dists(config: PretrainedConfig) -> torch.Tensor:
    seq = torch.arange(config.context_size)
    return (
        torch.clamp(
            seq[:, None] - seq[None, :],
            -config.context_size,
            config.context_size,
        )
        + config.max_pos_emb
    )


def _reference_forward(
    module: GraniteSpeechConformerAttention,
    hidden_states: torch.Tensor,
    attention_dists: torch.Tensor,
) -> torch.Tensor:
    hidden_states = module.pre_norm(hidden_states)
    bsz, num_features, _ = hidden_states.shape
    num_blocks = math.ceil(num_features / module.context_size)
    remainder = num_features % module.context_size
    if remainder > 0:
        hidden_states = F.pad(hidden_states, (0, 0, 0, module.context_size - remainder))

    query_states = module.to_q(hidden_states)
    key_states, value_states = module.to_kv(hidden_states).chunk(2, dim=-1)
    query_states = query_states.reshape(
        bsz, num_blocks, module.context_size, module.num_heads, -1
    ).transpose(2, 3)
    key_states = key_states.reshape(
        bsz, num_blocks, module.context_size, module.num_heads, -1
    ).transpose(2, 3)
    value_states = value_states.reshape(
        bsz, num_blocks, module.context_size, module.num_heads, -1
    ).transpose(2, 3)

    rel_pos_emb = module.rel_pos_emb(attention_dists)
    pos_attn = (
        torch.einsum("bnhid,ijd->bnhij", query_states, rel_pos_emb) * module.scale
    )
    if remainder > 0:
        mask = torch.ones(
            module.context_size,
            module.context_size,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        mask[:remainder, :remainder] = False
        mask_value = -torch.finfo(pos_attn.dtype).max
        pos_attn[:, -1, :].masked_fill_(mask, mask_value)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=pos_attn,
            scale=module.scale,
        )
    out = out.transpose(2, 3).reshape(bsz, hidden_states.shape[1], -1)
    return module.to_out(out[:, :num_features, :])


@pytest.mark.parametrize("num_features", [16, 13])
def test_attention_matches_math_reference(num_features: int) -> None:
    torch.manual_seed(17)
    config = _make_config()
    module = GraniteSpeechConformerAttention(config)
    hidden_states = torch.randn(2, num_features, config.hidden_dim)
    attention_dists = _make_attention_dists(config)

    expected = _reference_forward(module, hidden_states, attention_dists)
    actual = module(hidden_states, attention_dists)

    torch.testing.assert_close(actual, expected)


def test_attention_requests_efficient_backend_before_math(monkeypatch) -> None:
    requested_backends = []
    original_sdpa_kernel = torch.nn.attention.sdpa_kernel

    @contextmanager
    def record_sdpa_kernel(backends, *args, **kwargs):
        requested_backends.append(backends)
        with original_sdpa_kernel(backends, *args, **kwargs):
            yield

    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", record_sdpa_kernel)
    config = _make_config()
    module = GraniteSpeechConformerAttention(config)
    hidden_states = torch.randn(1, config.context_size, config.hidden_dim)

    module(hidden_states, _make_attention_dists(config))

    assert requested_backends == [
        [
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ]
    ]
