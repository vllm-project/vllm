# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for miscellaneous utilities
"""

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding)
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "max_position,is_neox_style,rotary_dim,head_size,seq_len,use_key", [
        (16, False, 32, 32, 1024, True),
        (16, False, 32, 128, 1024, True),
        (16, True, 32, 32, 1024, True),
        (16, True, 32, 128, 1024, True),
        (16, False, 32, 128, 1024, False),
        (16, True, 32, 128, 1024, False),
    ])
def test_rotary_embedding_opcheck(max_position, is_neox_style, rotary_dim,
                                  head_size, seq_len, use_key):
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    current_platform.seed_everything(0)
    torch.set_default_device("cpu")

    batch_size = 1
    base = 10000
    num_heads = 8

    rot = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                          is_neox_style, torch.float32)

    positions = torch.randint(0,
                              max_position, (batch_size, seq_len),
                              device="cpu")
    query = torch.randn(batch_size,
                        seq_len,
                        num_heads * head_size,
                        dtype=torch.float32,
                        device="cpu")
    key = torch.randn_like(query) if use_key else None
    assert positions.is_cpu, \
        "reference input tensor is expected to be CPU tensor."
    ref_query, ref_key = rot.to(device="cpu").forward_native(
        positions, query, key)
    out_query, out_key = rot.to(device=device).forward_neuron(
        positions.to(device=device), query.to(device=device),
        key.to(device=device) if key is not None else None)
    if use_key:
        assert out_query.is_xla and out_key.is_xla, \
            "output tensor is expected to be XLA tensor"
        torch.testing.assert_close(out_key.cpu(),
                                   ref_key,
                                   atol=1e-2,
                                   rtol=1e-2)
    else:
        assert out_key is None, "expected returned key to be None"
        assert out_query.is_xla, \
            "output tensor is expected to be XLA tensor"
    torch.testing.assert_close(out_query.cpu(),
                               ref_query,
                               atol=1e-2,
                               rtol=1e-2)


def test_deepseek_rotary_embedding():
    device = torch.device("cuda:0")
    current_platform.seed_everything(0)
    torch.set_default_device("cuda:0")
    batch_size = 10
    base = 10000
    num_heads = 8
    max_position = 4096
    is_neox_style = False
    rotary_dim = 32
    head_size = 64
    scaling_factor = 40.0

    rot = DeepseekScalingRotaryEmbedding(head_size,
                                         rotary_dim,
                                         max_position,
                                         base,
                                         is_neox_style,
                                         scaling_factor,
                                         torch.float32,
                                         reference=False).to(device)

    rot_ref = DeepseekScalingRotaryEmbedding(head_size,
                                             rotary_dim,
                                             max_position,
                                             base,
                                             is_neox_style,
                                             scaling_factor,
                                             torch.float32,
                                             reference=True).to(device)

    positions = torch.randint(0, max_position, (batch_size, ), device=device)
    # query is [batch, num_heads, head_size]
    # key is [batch, 1, head_size]
    # cos_sin is [batch, head_size]
    query = torch.randn(batch_size,
                        num_heads,
                        head_size,
                        dtype=torch.float32,
                        device=device)
    key = torch.randn(batch_size,
                      1,
                      head_size,
                      dtype=torch.float32,
                      device=device)
    ref_query, ref_key = rot_ref.forward(positions, query, key)
    out_query, out_key = rot.forward(positions, query, key)
    torch.testing.assert_close(out_key.cpu(),
                               ref_key.cpu(),
                               atol=1e-4,
                               rtol=1e-4)
    torch.testing.assert_close(out_query.cpu(),
                               ref_query.cpu(),
                               atol=1e-4,
                               rtol=1e-4)
