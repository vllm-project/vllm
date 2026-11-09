# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the 64-token paged prefix helpers in the ROCm AITER backend."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm AITER backend only", allow_module_level=True)

from vllm.v1.attention.backends.rocm_aiter_fa import (  # noqa: E402
    paged_prefix_blocks_per_page,
    paged_prefix_cache_views,
    paged_prefix_page_ids,
)


def _kernel_block_table(manager_blocks: list[int], ratio: int) -> torch.Tensor:
    # vLLM splits each attention block into `ratio` kernel blocks numbered
    # manager_block * ratio + j, the same as BlockTable.map_to_kernel_blocks.
    ids = [m * ratio + j for m in manager_blocks for j in range(ratio)]
    return torch.tensor(ids, dtype=torch.int32)


def test_page_ids_from_pairs_of_32_token_blocks():
    # A 2112-token attention block is 66 kernel blocks of 32 tokens, which
    # is 33 pages of 64 tokens.
    ratio = 66
    manager_blocks = [7, 3, 12]
    row = _kernel_block_table(manager_blocks, ratio)
    context_len = 2 * 2112 + 100
    num_pages = -(-context_len // 64)

    page_ids = paged_prefix_page_ids(row, num_pages, blocks_per_page=2)

    expected = [m * (ratio // 2) + i for m in manager_blocks for i in range(33)]
    assert page_ids.dtype == torch.int32
    assert page_ids.is_contiguous()
    assert page_ids.tolist() == expected[:num_pages]


def test_page_ids_from_64_token_blocks():
    row = _kernel_block_table([5, 1], ratio=33)
    page_ids = paged_prefix_page_ids(row, num_pages=40, blocks_per_page=1)
    assert page_ids.tolist() == row[:40].tolist()


def _interleaved_cache(num_blocks: int, num_heads: int, head_size: int):
    # Same layout as the backend: [blocks, heads, tokens, 2 * head_size],
    # with K in the first half of the content dimension and V in the second.
    kv = torch.arange(
        num_blocks * num_heads * 32 * 2 * head_size, dtype=torch.int32
    ).reshape(num_blocks, num_heads, 32, 2 * head_size)
    return kv.transpose(1, 2).split(head_size, dim=-1)


def _assert_aiter_accepts_linear_layout(pages: torch.Tensor, head_size: int):
    # The same stride checks that AITER mha_batch_prefill applies to a 4D
    # [pages, page_size, heads, head_size] K or V tensor before any kernel
    # runs.
    page_stride, token_stride, head_stride, element_stride = pages.stride()
    num_heads = pages.shape[2]
    assert element_stride == 1
    assert head_stride >= head_size
    assert token_stride >= num_heads * head_stride
    assert page_stride >= pages.shape[1] * token_stride


def test_cache_views_join_adjacent_blocks_into_64_token_pages():
    key_cache, value_cache = _interleaved_cache(8, 1, 16)

    key_pages, value_pages = paged_prefix_cache_views(
        key_cache, value_cache, blocks_per_page=2
    )

    assert key_pages.shape == (4, 64, 1, 16)
    assert key_pages.stride() == (2 * 32 * 32, 32, 16, 1)
    _assert_aiter_accepts_linear_layout(key_pages, 16)
    _assert_aiter_accepts_linear_layout(value_pages, 16)
    for page in range(4):
        for token in (0, 31, 32, 63):
            block = 2 * page + token // 32
            offset = token % 32
            torch.testing.assert_close(key_pages[page, token], key_cache[block, offset])
            torch.testing.assert_close(
                value_pages[page, token], value_cache[block, offset]
            )


def test_cache_views_keep_64_token_blocks_and_fix_head_stride():
    key_cache, value_cache = _interleaved_cache(8, 1, 16)

    key_pages, value_pages = paged_prefix_cache_views(
        key_cache, value_cache, blocks_per_page=1
    )

    assert key_pages.shape == key_cache.shape
    _assert_aiter_accepts_linear_layout(key_pages, 16)
    _assert_aiter_accepts_linear_layout(value_pages, 16)
    torch.testing.assert_close(key_pages, key_cache)
    torch.testing.assert_close(value_pages, value_cache)


def test_cache_views_reject_more_than_one_kv_head():
    key_cache, value_cache = _interleaved_cache(8, 2, 16)
    with pytest.raises(ValueError, match="one KV head"):
        paged_prefix_cache_views(key_cache, value_cache, blocks_per_page=2)


def _blocks_per_page(**overrides):
    # The Qwen3.8-2.4T TP8 case with --block-size 64: prequantized FP8 QKV,
    # one KV head per rank, 32-token kernel blocks, 2112-token attention
    # blocks and an AITER build with the paged kernel.
    args = dict(
        direct_fp8_context_gather=True,
        num_kv_heads=1,
        kernel_block_size=32,
        attention_block_size=2112,
        has_paged_kernel=True,
    )
    args.update(overrides)
    return paged_prefix_blocks_per_page(**args)


def test_paged_prefix_turns_on_without_a_flag():
    assert _blocks_per_page() == (2, [])
    assert _blocks_per_page(kernel_block_size=16) == (4, [])


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"direct_fp8_context_gather": False}, "prequantized FP8 QKV"),
        ({"num_kv_heads": 2}, "needs 1"),
        ({"has_paged_kernel": False}, "aiter#4971"),
        # The default attention block for this model without --block-size 64.
        ({"attention_block_size": 2080}, "--block-size 64"),
    ],
)
def test_paged_prefix_stays_off_and_says_why(overrides, reason):
    blocks_per_page, reasons = _blocks_per_page(**overrides)
    assert blocks_per_page is None
    assert len(reasons) == 1
    assert reason in reasons[0]
