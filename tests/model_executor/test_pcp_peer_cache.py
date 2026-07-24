# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.pcp_peer_cache import (
    PCPPeerCacheFence,
    make_rank_major_block_tensor_view,
    make_rank_major_tensor_view,
)


def test_rank_major_tensor_view_preserves_packed_offset_and_stride() -> None:
    world_size = 2
    bytes_per_rank = 64
    global_view = torch.arange(world_size * bytes_per_rank, dtype=torch.int8)
    local_view = global_view[bytes_per_rank:]
    allocation = SimpleNamespace(
        global_view=global_view,
        local_view=local_view,
        bytes_per_rank=bytes_per_rank,
        world_size=world_size,
    )
    local_tensor = local_view.view(4, 16)[:, 4:12]

    peer_tensor = make_rank_major_tensor_view(allocation, local_tensor)

    assert peer_tensor.shape == (world_size, 4, 8)
    assert peer_tensor.stride() == (bytes_per_rank, 16, 1)
    torch.testing.assert_close(peer_tensor[1], local_tensor)
    torch.testing.assert_close(
        peer_tensor[0], global_view[:bytes_per_rank].view(4, 16)[:, 4:12]
    )


def test_rank_major_block_tensor_view_preserves_padded_peer_stride() -> None:
    world_size = 3
    num_local_blocks = 5
    peer_block_stride = 8
    block_shape = (2, 4)
    block_stride = 8
    rank_stride = peer_block_stride * block_stride
    storage_offset = 3
    storage = torch.arange(
        storage_offset
        + (world_size - 1) * rank_stride
        + num_local_blocks * block_stride,
        dtype=torch.int32,
    )
    peer_tensor = torch.as_strided(
        storage,
        size=(world_size, num_local_blocks, *block_shape),
        stride=(rank_stride, block_stride, 4, 1),
        storage_offset=storage_offset,
    )

    flat, actual_peer_block_stride = make_rank_major_block_tensor_view(peer_tensor)

    assert actual_peer_block_stride == peer_block_stride
    assert flat.shape == (
        (world_size - 1) * peer_block_stride + num_local_blocks,
        *block_shape,
    )
    assert flat.stride() == (block_stride, 4, 1)
    assert flat.data_ptr() == peer_tensor.data_ptr()
    for owner in range(world_size):
        for physical_block in range(num_local_blocks):
            torch.testing.assert_close(
                flat[owner * peer_block_stride + physical_block],
                peer_tensor[owner, physical_block],
            )

    flat[peer_block_stride + 2].fill_(-1)
    torch.testing.assert_close(
        peer_tensor[1, 2], torch.full(block_shape, -1, dtype=torch.int32)
    )


def test_rank_major_block_tensor_view_rejects_non_integral_padding() -> None:
    storage = torch.empty(256, dtype=torch.int32)
    peer_tensor = torch.as_strided(
        storage,
        size=(2, 5, 2, 4),
        stride=(65, 8, 4, 1),
    )

    with pytest.raises(ValueError, match="must be divisible"):
        make_rank_major_block_tensor_view(peer_tensor)


def test_rank_major_block_tensor_view_rejects_overlapping_peers() -> None:
    storage = torch.empty(128, dtype=torch.int32)
    peer_tensor = torch.as_strided(
        storage,
        size=(2, 5, 2, 4),
        stride=(32, 8, 4, 1),
    )

    with pytest.raises(ValueError, match="segments overlap"):
        make_rank_major_block_tensor_view(peer_tensor)


def test_rank_major_block_tensor_view_rejects_overlapping_blocks() -> None:
    storage = torch.empty(128, dtype=torch.int32)
    peer_tensor = torch.as_strided(
        storage,
        size=(2, 5, 2, 4),
        stride=(64, 4, 8, 1),
    )

    with pytest.raises(ValueError, match="Cache blocks overlap"):
        make_rank_major_block_tensor_view(peer_tensor)


def test_peer_cache_fence_close_quiesces_group_before_unmap(monkeypatch) -> None:
    events: list[object] = []
    allocation = SimpleNamespace(
        device=torch.device("cuda:2"),
        close=lambda: events.append("close"),
    )
    fence = object.__new__(PCPPeerCacheFence)
    fence._group = "pcp-cpu-group"
    fence._allocation = allocation

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: events.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.pcp_peer_cache.dist.barrier",
        lambda *, group: events.append(("barrier", group)),
    )

    fence.close()

    assert events == [
        ("synchronize", torch.device("cuda:2")),
        ("barrier", "pcp-cpu-group"),
        "close",
    ]
