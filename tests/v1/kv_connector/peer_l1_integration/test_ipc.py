# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Three processes batch IPC reads from two owners and release their grants."""

import secrets
import threading
import time
from uuid import uuid4

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import PeerMemory
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.native import (
    CleanupQueue,
    NativeOwner,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
    NativeOwnerChannel,
    peer_key,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.worker import (
    LoadStoreMover,
    exchange_handshake,
)


def _copy_rank(rank, rendezvous, endpoint_prefix, group_bytes):
    device = rank % 2
    torch.accelerator.set_device_index(device)
    world = 3
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world)
    rows = torch.arange(32, dtype=torch.int64, device=f"cuda:{device}")[:, None]
    row_bytes = max(group_bytes)
    columns = torch.arange(row_bytes // 8, dtype=torch.int64, device=f"cuda:{device}")
    source = ((rank << 50) + (rows << 32) + columns).contiguous()
    torch.accelerator.synchronize()
    storage = source.untyped_storage()
    local = PeerMemory(
        rank,
        f"{endpoint_prefix}/{rank}",
        rank + 1,
        "local",
        device,
        "ipc-test",
        32,
        ((),),
        {peer: storage._share_cuda_() for peer in range(world) if peer != rank},
    )
    handshake = exchange_handshake(local, b"", implicit=True, group=dist.group.WORLD)
    cleanup = CleanupQueue(credential=handshake.credential)
    owner = NativeOwner(
        local.endpoint, local.epoch, 32, window=16, credential=handshake.credential
    )
    peers = [(rank + offset) % world for offset in (1, 2)]
    imported = [
        torch.UntypedStorage._new_shared_cuda(
            device, *handshake.members[peer].handles[rank][1:]
        )
        for peer in peers
    ]
    stream = torch.cuda.Stream(priority=-1)
    mover = LoadStoreMover(cleanup, stream)
    keys = [peer_key(bytes([i])) for i in range(1, 9)]
    owner.publish(keys, list(range(1, 9)))
    outsider = CleanupQueue()
    try:
        assert (
            NativeOwnerChannel(
                owner.endpoint, owner.epoch, 999, outsider
            ).acquire_exact(keys[:1])
            is None
        )
        assert owner.refs == 0
    finally:
        outsider.close()
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
        transfer = mover.submit(
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
        while not transfer.query() or owner.refs or cleanup.pending:
            assert time.monotonic() < deadline
            threading.Event().wait(0.001)
        dist.barrier()
    finally:
        stream.synchronize()
        cleanup.close()
        owner.close()
        del mover, imported
        dist.destroy_process_group()


@pytest.mark.distributed
def test_three_rank_ipc_copies_and_callback_release(tmp_path):
    rendezvous = f"file://{tmp_path}/gloo"
    mp.spawn(
        _copy_rank,
        args=(
            rendezvous,
            f"vllm-peer-l1/bytes/{uuid4().hex}",
            (1160640, 1160640, 1315008, 1017792, 1244160),
        ),
        nprocs=3,
        join=True,
    )


def _descriptor_rank(rank):
    torch.accelerator.set_device_index(rank)
    credential = secrets.token_bytes(32)
    cleanup = CleanupQueue(credential=credential)
    owner = NativeOwner(
        f"vllm-peer-l1/chunks/{uuid4().hex}", 1, 8, window=4, credential=credential
    )
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
        while not transfer.query() or owner.refs or cleanup.pending:
            assert time.monotonic() < deadline
            threading.Event().wait(0.001)
    finally:
        stream.synchronize()
        cleanup.close()
        owner.close()


@pytest.mark.distributed
def test_descriptor_limit_splits_copy_and_releases_grant(monkeypatch):
    """Use a fresh process so CUDA and HIP resolve the cap before their first copy."""
    monkeypatch.setenv("VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS", "8192")
    mp.spawn(_descriptor_rank, nprocs=1, join=True)
