# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.kv_cache_shape_utils import (
    maybe_adjust_kv_cache_shape_for_padded_page_size,
)


def test_maybe_adjust_kv_cache_shape_noop_without_padding():
    shape = (4, 2, 16, 2, 260)
    out = maybe_adjust_kv_cache_shape_for_padded_page_size(
        kv_cache_shape=shape,
        num_blocks=4,
        padded_page_size_bytes=None,
        dtype=torch.int8,
    )
    assert out == shape


def test_maybe_adjust_kv_cache_shape_recomputes_last_dim_for_padding():
    shape = (4, 2, 16, 2, 260)
    padded_page_size_bytes = 2 * 16 * 2 * 264
    out = maybe_adjust_kv_cache_shape_for_padded_page_size(
        kv_cache_shape=shape,
        num_blocks=4,
        padded_page_size_bytes=padded_page_size_bytes,
        dtype=torch.int8,
    )
    assert out == (4, 2, 16, 2, 264)


def test_maybe_adjust_kv_cache_shape_raises_on_invalid_padding_ratio():
    shape = (4, 2, 16, 2, 260)
    with pytest.raises(ValueError, match="integer last dimension"):
        maybe_adjust_kv_cache_shape_for_padded_page_size(
            kv_cache_shape=shape,
            num_blocks=4,
            padded_page_size_bytes=(2 * 16 * 2 * 264) + 1,
            dtype=torch.int8,
        )
