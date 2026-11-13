# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import mmap
import sys

import pytest

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
