# SPDX-License-Identifier: Apache-2.0
"""Regression test for the NVFP4 KV-cache layout on SM120.

#44455 ("[2/N][KV-Cache Layout Refactor]") folded K/V into the content dim for
all dtypes. That is a pure reindex when the last dim is `head_size`, but NVFP4
stores `full_dim = [fp4 data | fp8 block scales]`, so a `2 * num_kv_heads`
head-group slice no longer yields the contiguous `[data-block | scale-block]`
region that `nvfp4_split_data_scale()` reconstructs its views from. On SM120
(fa2 prefill / XQA decode) attention then dequantises against the wrong scales
and every generation collapses to token id 0.

These are pure-CPU shape/stride contract checks — no GPU required — and they
would have failed the moment the shape changed without the NVFP4 consumers
being updated.
"""
import pytest

from vllm.v1.attention.backends.flashinfer import FlashInferBackend

NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE = 8, 16, 4, 256


def _shape(dtype):
    return FlashInferBackend.get_kv_cache_shape(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, cache_dtype_str=dtype
    )


def _stride_order(dtype):
    try:
        return FlashInferBackend.get_kv_cache_stride_order(cache_dtype_str=dtype)
    except TypeError:  # backend predating the cache_dtype_str kwarg
        return FlashInferBackend.get_kv_cache_stride_order()


@pytest.mark.parametrize("dtype", ["auto", "fp8_e4m3", "nvfp4"])
def test_stride_order_rank_matches_shape_rank(dtype):
    """The allocator asserts these agree; a mismatch aborts engine start."""
    assert len(_stride_order(dtype)) == len(_shape(dtype)), (
        f"stride order rank {len(_stride_order(dtype))} != shape rank "
        f"{len(_shape(dtype))} for cache_dtype_str={dtype!r}"
    )


@pytest.mark.parametrize("dtype", ["auto", "fp8_e4m3", "nvfp4"])
def test_stride_order_is_a_permutation(dtype):
    order = _stride_order(dtype)
    assert sorted(order) == list(range(len(order))), (
        f"stride order {order} is not a permutation for cache_dtype_str={dtype!r}"
    )


def test_nvfp4_last_dim_carries_data_and_scale():
    """full_dim must be data + scale, i.e. 9/8 of the fp4 data width."""
    shape = _shape("nvfp4")
    full_dim = shape[-1]
    assert full_dim == HEAD_SIZE // 2 + HEAD_SIZE // 16
    assert full_dim * 8 % 9 == 0, "full_dim must split cleanly into data|scale"


def test_nvfp4_kv_sides_are_separable_without_head_group_slicing():
    """
    The NVFP4 consumers (`nvfp4_split_data_scale`) require each K/V side to be a
    single contiguous [data | scale] region. On SM120 that means the layout must
    expose K and V on a dedicated leading axis rather than folding them into a
    2*num_kv_heads head-group dim.
    """
    from vllm.platforms import current_platform

    shape = _shape("nvfp4")
    if current_platform.is_device_capability_family(120):
        assert len(shape) == 5 and shape[1] == 2, (
            "SM120 NVFP4 must keep the (num_blocks, 2, block_size, num_kv_heads,"
            f" full_dim) layout; got {shape}"
        )
    else:
        # SM100/trtllm-gen keeps the packed layout introduced by #44455.
        assert len(shape) == 4 and shape[1] == 2 * NUM_KV_HEADS, (
            f"non-SM120 NVFP4 should keep the packed layout; got {shape}"
        )
