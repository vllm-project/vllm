# SPDX-License-Identifier: Apache-2.0
"""
Tests for miscellaneous utilities
"""

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "max_position,is_neox_style,rotary_dim,head_size,seq_len", [
        (16, False, 32, 32, 1024),
        (16, False, 32, 128, 1024),
        (16, True, 32, 32, 1024),
        (16, True, 32, 128, 1024),
    ])
def test_rotary_embedding_opcheck(max_position, is_neox_style, rotary_dim,
                                  head_size, seq_len):
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
    key = torch.randn_like(query)

    assert positions.is_cpu, \
        "reference input tensor is expected to be CPU tensor."
    ref_query, ref_key = rot.to(device="cpu").forward_native(
        positions, query, key)
    out_query, out_key = rot.to(device=device).forward_neuron(
        positions.to(device=device), query.to(device=device),
        key.to(device=device))
    assert out_query.is_xla and out_key.is_xla, \
        "output tensor is expected to be XLA tensor"
    torch.testing.assert_close(out_query.cpu(),
                               ref_query,
                               atol=1e-2,
                               rtol=1e-2)
    torch.testing.assert_close(out_key.cpu(), ref_key, atol=1e-2, rtol=1e-2)
