# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.kv_cache_shape_utils import (
    adjust_kv_cache_shape_for_padded_page_size,
    get_padded_page_size_for_kernel_block_size,
)


def test_get_padded_page_size_noop_without_padding():
    assert (
        get_padded_page_size_for_kernel_block_size(
            padded_page_size_bytes=None,
            kv_block_size=16,
            kernel_block_size=16,
        )
        is None
    )


def test_get_padded_page_size_scales_for_smaller_kernel_block():
    assert (
        get_padded_page_size_for_kernel_block_size(
            padded_page_size_bytes=1024,
            kv_block_size=16,
            kernel_block_size=8,
        )
        == 512
    )


def test_get_padded_page_size_raises_when_not_divisible():
    with pytest.raises(ValueError, match="num_blocks_per_kv_block"):
        get_padded_page_size_for_kernel_block_size(
            padded_page_size_bytes=1025,
            kv_block_size=16,
            kernel_block_size=8,
        )


def test_adjust_kv_cache_shape_noop_without_padding():
    shape = (4, 2, 16, 2, 260)
    out = adjust_kv_cache_shape_for_padded_page_size(
        kv_cache_shape=shape,
        num_blocks=4,
        padded_page_size_bytes=None,
        dtype=torch.int8,
    )
    assert out == shape


def test_adjust_kv_cache_shape_recomputes_last_dim_for_padding():
    shape = (4, 2, 16, 2, 260)
    padded_page_size_bytes = 2 * 16 * 2 * 264
    out = adjust_kv_cache_shape_for_padded_page_size(
        kv_cache_shape=shape,
        num_blocks=4,
        padded_page_size_bytes=padded_page_size_bytes,
        dtype=torch.int8,
    )
    assert out == (4, 2, 16, 2, 264)


def test_adjust_kv_cache_shape_raises_on_invalid_padding_ratio():
    shape = (4, 2, 16, 2, 260)
    with pytest.raises(ValueError, match="integer last dimension"):
        adjust_kv_cache_shape_for_padded_page_size(
            kv_cache_shape=shape,
            num_blocks=4,
            padded_page_size_bytes=(2 * 16 * 2 * 264) + 1,
            dtype=torch.int8,
        )
