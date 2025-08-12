# SPDX-License-Identifier: Apache-2.0
"""
Tests for miscellaneous utilities
"""

from typing import Optional

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding


def rotary_embedding_opcheck(rot,
                             positions: torch.Tensor,
                             query: torch.Tensor,
                             key: Optional[torch.Tensor] = None,
                             offsets: Optional[torch.Tensor] = None):
    cos_sin_cache = rot.cos_sin_cache.to(query.device, dtype=query.dtype)

    # ops.rotary_embedding()/batched_rotary_embedding()
    # are in-place operations that update the query and key tensors.
    if offsets is not None:
        opcheck(torch.ops._C.batched_rotary_embedding,
                (positions, query, key, rot.head_size, cos_sin_cache,
                 rot.is_neox_style, rot.rotary_dim, offsets))
    else:
        opcheck(torch.ops._C.rotary_embedding,
                (positions, query, key, rot.head_size, cos_sin_cache,
                 rot.is_neox_style))


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("max_position", [11, 4096, 32768])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("rotary_dim", [32])
@pytest.mark.parametrize("head_size", [32, 108])
@pytest.mark.parametrize("seq_len", [11, 1024])
@pytest.mark.parametrize("use_key", [True, False])
@pytest.mark.parametrize("head_stride_is_contingous", [True, False])
def test_rotary_embedding_opcheck(dist_init, device, max_position,
                                  is_neox_style, rotary_dim, head_size,
                                  seq_len, use_key, head_stride_is_contingous):
    batch_size = 1
    base = 10000
    num_heads = 7
    rot = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                          is_neox_style, torch.float32)

    positions = torch.randint(0,
                              max_position, (batch_size, seq_len),
                              device=device)
    head_stride = head_size + (64 if head_stride_is_contingous else 0)

    query = torch.randn(batch_size,
                        seq_len,
                        num_heads,
                        head_stride,
                        dtype=torch.float32,
                        device=device)
    key = torch.randn_like(query) if use_key else None
    query = query[..., :head_size]
    key = key[..., :head_size] if use_key else None

    rotary_embedding_opcheck(rot, positions, query, key)
    offsets = torch.zeros(batch_size * seq_len,
                          device=device,
                          dtype=torch.long)
    rotary_embedding_opcheck(rot, positions, query, key, offsets)

    # if we have a contiguous head stride, test the alternate
    # [..., num_heads * head_dim] shape/layout
    if head_stride_is_contingous:
        rotary_embedding_opcheck(
            rot, positions, query.flatten(start_dim=-2),
            key.flatten(start_dim=-2) if use_key else None)
