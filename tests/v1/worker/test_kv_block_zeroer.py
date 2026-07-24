# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.utils import KVBlockZeroer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_ids_are_not_overwritten_while_copy_is_in_flight():
    device = torch.device("cuda")
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    # Build the minimal zeroer state directly so the test can focus on the
    # in-flight copy behavior without constructing model attention groups.
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._metas = [
        (
            torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
            page_size_el,
            page_size_el,
            1,
        )
    ]

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Keep the first nonblocking H2D copy pending while the host submits the
        # second call. Each call must stage from its own pinned source so the
        # first copy is not corrupted before it runs.
        torch.cuda._sleep(10_000_000)
        zeroer.zero_block_ids([1])
        zeroer.zero_block_ids([2])
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)
    assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zeroing_across_non_uniform_page_sizes():
    # Layers within one KV-cache group can share a block-id space while having
    # different page sizes (e.g. DeepSeek-V3.2's main MLA cache and sparse
    # indexer cache). Each page size must be zeroed against its own storage.
    device = torch.device("cuda")
    num_blocks = 4
    small_page_el = 4
    large_page_el = 16
    small = torch.ones((num_blocks, small_page_el), dtype=torch.int32, device=device)
    large = torch.ones((num_blocks, large_page_el), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._metas = [
        (
            torch.tensor([small.data_ptr()], dtype=torch.uint64, device=device),
            small_page_el,
            small_page_el,
            1,
        ),
        (
            torch.tensor([large.data_ptr()], dtype=torch.uint64, device=device),
            large_page_el,
            large_page_el,
            1,
        ),
    ]

    zeroer.zero_block_ids([1, 3])
    torch.accelerator.synchronize()

    for storage in (small, large):
        assert torch.all(storage[0] == 1)
        assert torch.all(storage[1] == 0)
        assert torch.all(storage[2] == 1)
        assert torch.all(storage[3] == 0)
