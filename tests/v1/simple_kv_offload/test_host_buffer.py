# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import mmap
import sys

import pytest

from vllm.utils.host_memory import host_registration_ranges, host_tensor_offsets
from vllm.utils.math_utils import round_up
from vllm.v1.simple_kv_offload.host_buffer import (
    alloc_host_buffer,
    parse_hugepage_size,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux mm APIs")


@pytest.mark.parametrize(
    "value,expected",
    [(None, None), ("none", None), ("2MB", 2 << 20), ("1gb", 1 << 30)],
)
def test_parse_hugepage_size(value, expected):
    assert parse_hugepage_size(value) == expected


def test_parse_hugepage_size_rejects_unknown():
    with pytest.raises(ValueError, match="cpu_hugepage_block_size"):
        parse_hugepage_size("4KB")


def test_buffer_rounds_up_to_whole_pages():
    buf = alloc_host_buffer(mmap.PAGESIZE + 1)
    assert buf.nbytes == 2 * mmap.PAGESIZE


@pytest.mark.parametrize("page", [2 << 11, 2 << 20])
def test_pool_over_one_chunk_splits_on_aligned_tensor_starts(page):
    num_blocks = 6
    sizes = [num_blocks * page // 4] + [num_blocks * page] * 8
    max_chunk = 16 * page
    with pytest.raises(ValueError, match="No block and page boundary"):
        host_registration_ranges(
            sizes, num_blocks, round_up(sum(sizes), page), max_chunk, page_size=page
        )

    offsets = host_tensor_offsets(sizes, max_chunk, page_size=page)
    total = round_up(offsets[-1] + sizes[-1], page)
    ranges = host_registration_ranges(
        sizes, num_blocks, total, max_chunk, page_size=page, tensor_offsets=offsets
    )

    assert ranges[0][0] == 0 and ranges[-1][1] == total
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    assert all(end - start <= max_chunk for start, end in ranges)
    for _, split in ranges[:-1]:
        assert split % page == 0
        for offset, size in zip(offsets, sizes):
            if offset < split < offset + size:
                assert (split - offset) % (size // num_blocks) == 0
    assert total - round_up(sum(sizes), page) < page * len(ranges)


def test_hugepage_buffer_takes_pages_from_the_pool(free_hugepages, hugepage_with_free):
    page = hugepage_with_free(2)
    free = free_hugepages(page)
    buf = alloc_host_buffer(page + 1, page)
    assert buf.nbytes == 2 * page
    assert free_hugepages(page) <= free - 2


@pytest.mark.parametrize("page", [2 << 20, 1 << 30])
def test_hugepage_buffer_fails_closed_when_pool_is_short(free_hugepages, page):
    free = free_hugepages(page)
    if free < 0:
        pytest.skip(f"no {page >> 20} MiB hugepage pool")
    with pytest.raises(RuntimeError, match="Cannot map"):
        alloc_host_buffer((free + 1) * page, page)
