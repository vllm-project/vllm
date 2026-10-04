# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FoPE rotary embedding cache construction."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.device import DeviceConfig
from vllm.model_executor.layers.rotary_embedding.fope import FourierRotaryEmbedding

HEAD_SIZE = 64
ROTARY_DIM = 32
MAX_POSITION = 128
NUM_KV_HEADS = 2


@pytest.fixture
def cpu_vllm_config():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        yield config


def make_fope(fope_sep_head: bool) -> FourierRotaryEmbedding:
    rope = FourierRotaryEmbedding(
        head_size=HEAD_SIZE,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=MAX_POSITION,
        base=10000.0,
        is_neox_style=True,
        dtype=torch.float32,
        init_cache=False,
        num_key_value_heads=NUM_KV_HEADS,
        num_inv_freq=ROTARY_DIM // 2,
        fope_sep_head=fope_sep_head,
        fope_init_factor=1.0,
    )
    # Identity mixing coefficients reduce FoPE to plain RoPE over the
    # retained frequencies, so any channel change beyond them is padding.
    eye = torch.eye(rope.input_dim).unsqueeze(0).expand(NUM_KV_HEADS, -1, -1)
    with torch.no_grad():
        rope.cos_coef.copy_(eye)
        rope.sin_coef.copy_(eye)
    return rope


@pytest.mark.parametrize("fope_sep_head", [True, False])
def test_fope_leaves_unrotated_channels_untouched(cpu_vllm_config, fope_sep_head):
    """Channels past the retained frequencies must pass through unchanged.

    Padding sin with 1 instead of 0 turned them into x[j] - x[j + head/2].
    """
    rope = make_fope(fope_sep_head)

    torch.manual_seed(0)
    positions = torch.arange(4)
    query = torch.randn(4, NUM_KV_HEADS * HEAD_SIZE)
    key = torch.randn_like(query)

    query_out, key_out = rope.forward_native(positions, query.clone(), key.clone())

    n_rotated = rope.input_dim
    for out, inp in ((query_out, query), (key_out, key)):
        out = out.view(4, NUM_KV_HEADS, HEAD_SIZE)
        inp = inp.view(4, NUM_KV_HEADS, HEAD_SIZE)
        for half in range(2):
            offset = half * (HEAD_SIZE // 2)
            unrotated = slice(offset + n_rotated, offset + HEAD_SIZE // 2)
            torch.testing.assert_close(out[..., unrotated], inp[..., unrotated])
