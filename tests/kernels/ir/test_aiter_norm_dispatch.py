# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the AITER norm support predicates.

The AITER norm wrappers flatten >2-D activations with ``Tensor.reshape``, which
silently copies when the flattened shape is not expressible with the input's
strides. The support predicates must reject those inputs so dispatch falls
through to a provider that handles arbitrary strides.
"""

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.kernels.aiter_ops import flatten_to_2d_is_free

pytestmark = pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)


def _qkv_slice_by_head(num_tokens, num_q_heads, num_kv_heads, head_dim):
    """q viewed per-head from a fused QKV projection, as Qwen3-style QK-norm does."""
    q_size, kv_size = num_q_heads * head_dim, num_kv_heads * head_dim
    qkv = torch.empty(num_tokens, q_size + 2 * kv_size)
    q = qkv.split([q_size, kv_size, kv_size], dim=-1)[0]
    return q.view(*q.shape[:-1], num_q_heads, head_dim)


@pytest.mark.parametrize(
    "x",
    [
        torch.empty(8, 16),
        torch.empty(16),
        torch.empty(2, 4, 16),
        torch.empty(2, 1, 16),
        # A contiguous tensor stays flattenable after a leading-dim slice.
        torch.empty(8, 4, 16)[2:6],
    ],
)
def test_flattenable(x):
    assert flatten_to_2d_is_free(x)
    assert x.reshape(-1, x.shape[-1]).data_ptr() == x.data_ptr()


@pytest.mark.parametrize(
    "x",
    [
        _qkv_slice_by_head(8, 32, 8, 128),
        # Non-unit last-dim stride.
        torch.empty(4, 8, 32).transpose(-1, -2),
        # Leading dims cannot be merged: a slice along the middle dim.
        torch.empty(4, 8, 32)[:, :4],
    ],
)
def test_not_flattenable(x):
    assert not flatten_to_2d_is_free(x)
    # reshape has to copy, which is exactly what the predicate guards against.
    assert x.reshape(-1, x.shape[-1]).data_ptr() != x.data_ptr()
