# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.worker.utils import KVBlockZeroer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_ids_are_not_overwritten_while_copy_is_in_flight():
    device = torch.device("cuda")
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    # Build the minimal zeroer state directly so the test can focus on ID-buffer
    # lifetime without constructing model attention groups.
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer.pin_memory = True
    zeroer.max_concurrency = 2
    zeroer._id_cap = 8
    zeroer._allocate_id_buffers()
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        page_size_el,
        page_size_el,
        1,
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Keep the first nonblocking H2D copy pending while the host submits the
        # second call. A single shared pinned source would be overwritten here.
        torch.cuda._sleep(10_000_000)
        zeroer.zero_block_ids([1])
        zeroer.zero_block_ids([2])
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)
    assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_cache_zeroes_each_physical_block_once():
    device = torch.device("cuda")
    num_blocks = 3
    block_stride = 16
    backing = torch.ones((num_blocks, block_stride), dtype=torch.uint8, device=device)
    contexts = {
        "layer_0": SimpleNamespace(kv_cache=backing[:, :8]),
        "layer_1": SimpleNamespace(kv_cache=backing[:, 8:]),
    }
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=backing.numel(),
                shared_by=["layer_0"],
                offset=0,
                block_stride=block_stride,
            ),
            KVCacheTensor(
                size=backing.numel(),
                shared_by=["layer_1"],
                offset=8,
                block_stride=block_stride,
            ),
        ],
        kv_cache_groups=[],
    )
    zeroer = KVBlockZeroer(
        device,
        pin_memory=True,
        attn_groups_iter=iter(()),
        kernel_block_sizes=[],
        cache_dtype="auto",
        static_forward_context=contexts,
        kv_cache_config=kv_cache_config,
    )

    zeroer.zero_block_ids([1])
    torch.accelerator.synchronize()

    assert torch.all(backing[0] == 1)
    assert torch.all(backing[1] == 0)
    assert torch.all(backing[2] == 1)
