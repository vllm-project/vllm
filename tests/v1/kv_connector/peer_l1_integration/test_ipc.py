# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Three processes batch IPC reads from two owners and release their grants."""

import threading
import time
from uuid import uuid4

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.native import (
    CleanupQueue,
    NativeOwner,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
    NativeOwnerChannel,
    peer_key,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.worker import (
    IPCSession,
    LoadStoreMover,
)


def _copy_rank(rank, rendezvous, endpoint_prefix, group_bytes):
    device = rank % 2
    torch.accelerator.set_device_index(device)
    world = 3
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world)
    cleanup = CleanupQueue()
    owner = NativeOwner(f"{endpoint_prefix}/{rank}", rank + 1, 32, window=16)
    rows = torch.arange(32, dtype=torch.int64, device=f"cuda:{device}")[:, None]
    row_bytes = max(group_bytes)
    columns = torch.arange(row_bytes // 8, dtype=torch.int64, device=f"cuda:{device}")
    source = ((rank << 50) + (rows << 32) + columns).contiguous()
    torch.accelerator.synchronize()
    storage = source.untyped_storage()
    exports = [None] * world
    dist.all_gather_object(
        exports, {peer: storage._share_cuda_() for peer in range(world) if peer != rank}
    )
    peers = [(rank + offset) % world for offset in (1, 2)]
    imported = [
        torch.UntypedStorage._new_shared_cuda(device, *exports[peer][rank][1:])
        for peer in peers
    ]
    stream = torch.cuda.Stream(priority=-1)
    session = IPCSession(storage, imported[0], cleanup, stream)
    keys = [peer_key(bytes([i])) for i in range(1, 9)]
    owner.publish(keys, list(range(1, 9)))
    dist.barrier()
    channels = [
        NativeOwnerChannel(f"{endpoint_prefix}/{peer}", peer + 1, rank + 10, cleanup)
        for peer in peers
    ]
    try:
        first = channels[0].acquire_exact(keys[:4])
        second = channels[1].acquire_exact(keys[4:])
        assert first is not None and second is not None
        grants = [first, second]
        source[17:25].fill_(-1)
        transfer = session.mover.submit(
            grants,
            [
                (
                    imported[(row - 1) // 4].data_ptr() + row * row_bytes,
                    storage.data_ptr() + (row + 16) * row_bytes,
                    group_bytes[row % len(group_bytes)],
                )
                for row in range(1, 9)
            ],
        )
        for row in range(1, 9):
            peer = peers[(row - 1) // 4]
            words = group_bytes[row % len(group_bytes)] // 8
            expected = (peer << 50) + (row << 32) + columns[:words]
            assert torch.equal(source[row + 16, :words], expected)
            assert torch.all(source[row + 16, words:] == -1)
        deadline = time.monotonic() + 5
        while not transfer.done() or owner.refs or cleanup.pending:
            assert time.monotonic() < deadline
            threading.Event().wait(0.001)
        assert isinstance(session.mover, LoadStoreMover)
        stats = session.mover.get_stats().reduce()
        assert stats["worker_copy_bytes"] == sum(
            group_bytes[row % len(group_bytes)] for row in range(1, 9)
        )
        assert stats["worker_copy_descriptors"] == 8
        assert stats["worker_load_steps"] == 1
        assert stats["worker_copy_gpu_seconds_count"] == 1
        assert stats["worker_main_wait_seconds_count"] == 1
        dist.barrier()
    finally:
        stream.synchronize()
        cleanup.close()
        owner.close()
        del session, imported
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.parametrize(
    "group_bytes",
    [
        (1160640, 1160640, 1315008, 1017792, 1244160),
        (210240,),
    ],
    ids=["hybrid", "full-attention"],
)
def test_three_rank_ipc_copies_and_callback_release(tmp_path, group_bytes):
    rendezvous = f"file://{tmp_path}/gloo"
    mp.spawn(
        _copy_rank,
        args=(rendezvous, f"vllm-peer-l1/bytes/{uuid4().hex}", group_bytes),
        nprocs=3,
        join=True,
    )


@pytest.mark.distributed
def test_descriptor_limit_splits_copy_and_releases_grant():
    """A batch above HIP's 8192-descriptor ceiling still completes exactly."""
    cleanup = CleanupQueue()
    owner = NativeOwner(f"vllm-peer-l1/chunks/{uuid4().hex}", 1, 8, window=4)
    stream = torch.cuda.Stream(priority=-1)
    try:
        key = peer_key(b"chunked")
        owner.publish([key], [1])
        channel = NativeOwnerChannel(owner.endpoint, 1, 42, cleanup)
        grant = channel.acquire_exact([key])
        assert grant is not None
        source = torch.arange(8193, device="cuda", dtype=torch.int64)
        target = torch.full_like(source, -1)
        mover = LoadStoreMover(cleanup, stream)
        transfer = mover.submit(
            [grant],
            [
                (source.data_ptr() + i * 8, target.data_ptr() + i * 8, 8)
                for i in range(source.numel())
            ],
        )
        assert torch.equal(source, target)
        deadline = time.monotonic() + 5
        while not transfer.done() or owner.refs or cleanup.pending:
            assert time.monotonic() < deadline
            threading.Event().wait(0.001)
        stats = mover.get_stats().reduce()
        assert stats["worker_copy_bytes"] == 8193 * 8
        assert stats["worker_copy_descriptors"] == 8193
        assert stats["worker_main_wait_seconds_count"] == 1
    finally:
        stream.synchronize()
        cleanup.close()
        owner.close()
