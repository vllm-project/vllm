# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.attention.pcp_peer_cache import (
    PCPPeerCacheFence,
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
