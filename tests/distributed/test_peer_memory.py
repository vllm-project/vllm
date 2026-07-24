# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.device_communicators.peer_memory import (
    PeerMemoryFence,
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


def test_peer_memory_fence_close_quiesces_group_before_unmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    allocation = SimpleNamespace(close=lambda: events.append("close"))
    fence = object.__new__(PeerMemoryFence)
    fence._group = "pcp-cpu-group"
    fence._allocation = allocation
    fence._peer_signals = torch.empty(0)
    fence._closed = False

    monkeypatch.setattr(
        torch.accelerator,
        "synchronize",
        lambda: events.append("synchronize"),
    )
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.peer_memory.dist.barrier",
        lambda *, group: events.append(("barrier", group)),
    )

    fence.close()
    fence.close()

    assert events == [
        "synchronize",
        ("barrier", "pcp-cpu-group"),
        "close",
    ]
    with pytest.raises(RuntimeError, match="is closed"):
        fence()
