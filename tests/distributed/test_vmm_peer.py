# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer aliases preserve padded segment storage through growth and remapping."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, port, world_size):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.accelerator.set_device_index(rank)
    torch.manual_seed(1234 + rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    from vllm.utils.extensible_tensor import (
        ExtensibleTensor,
        uint8_tensor_from_ptr,
    )

    granule = 2 << 20
    owner = ExtensibleTensor(
        5 * granule,
        device=torch.device("cuda", rank),
        segment_capacities=[2 * granule, 3 * granule],
        exportable=True,
    )
    owner.resize_segments_([512, 1024], zero_new=True)
    physical = owner.physical_bytes
    peers = owner.share_with(dist.group.WORLD)
    assert owner.physical_bytes == physical
    addresses = tuple(peers.bases)
    table = peers.pointers.data_ptr()
    aliases = [
        uint8_tensor_from_ptr(base, owner.capacity_bytes, rank) for base in addresses
    ]

    def write():
        for alias in aliases:
            for offset in owner.segment_offsets:
                alias[offset + rank * 16 : offset + (rank + 1) * 16].fill_(rank + 1)

    write()
    torch.accelerator.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write()
    for phase in range(3):
        if phase == 1:
            owner.resize_segments_([granule + 128, 2 * granule + 128], zero_new=True)
        elif phase == 2:
            owner.release_physical()
            owner.resize_segments_([granule + 128, 2 * granule + 128], zero_new=True)
        graph.replay()
        torch.accelerator.synchronize()
        dist.barrier()
        expected = torch.arange(1, world_size + 1, device="cuda", dtype=torch.uint8)
        expected = expected.repeat_interleave(16)
        for segment in range(owner.num_segments):
            torch.testing.assert_close(
                owner.segment_view(segment)[: world_size * 16], expected, rtol=0, atol=0
            )
        assert tuple(peers.bases) == addresses
        assert peers.pointers.data_ptr() == table
    del graph
    owner.free()
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.skipif(
    torch.accelerator.device_count() < 4, reason="requires four CUDA GPUs"
)
def test_padded_unequal_peer_segments(world_size):
    from vllm.utils.network_utils import get_open_port

    mp.spawn(_worker, args=(get_open_port(), world_size), nprocs=world_size, join=True)
