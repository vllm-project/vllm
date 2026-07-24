# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import array
import importlib.util
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators import cuda_vmm
from vllm.distributed.device_communicators.cuda_vmm import (
    RankMajorPeerView,
    create_rank_major_peer_view,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


class _FakeDriver:
    CUresult = SimpleNamespace(CUDA_SUCCESS=0)
    CUmemAllocationHandleType = SimpleNamespace(
        CU_MEM_HANDLE_TYPE_FABRIC=1,
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR=2,
    )

    def __init__(self) -> None:
        self.unmapped: list[tuple[int, int]] = []
        self.freed: list[tuple[int, int]] = []

    def cuMemUnmap(self, address: int, size: int):
        self.unmapped.append((address, size))
        return (0,)

    def cuMemAddressFree(self, address: int, size: int):
        self.freed.append((address, size))
        return (0,)


def test_allocation_property_preserves_typed_handle() -> None:
    driver = SimpleNamespace(
        CUmemAllocationProp=lambda: SimpleNamespace(),
        CUmemAllocationType=SimpleNamespace(CU_MEM_ALLOCATION_TYPE_PINNED=1),
        CUmemLocation=lambda: SimpleNamespace(),
        CUmemLocationType=SimpleNamespace(CU_MEM_LOCATION_TYPE_DEVICE=2),
    )

    handle_type = SimpleNamespace(value=1)
    prop = cuda_vmm._make_allocation_property(
        driver, device_id=3, handle_types=handle_type
    )

    assert prop.requestedHandleTypes is handle_type
    assert prop.location.id == 3


def test_aligned_local_shape_honors_granularity_and_row_multiple() -> None:
    shape, size = cuda_vmm._aligned_local_shape(
        (7, 3), torch.float16, granularity=64, first_dim_multiple=5
    )

    assert shape == (160, 3)
    assert size == 960
    assert size % 64 == 0
    assert shape[0] % 5 == 0


def test_handle_export_uses_posix_collectively(monkeypatch) -> None:
    driver = _FakeDriver()
    local_fd = os.open(os.devnull, os.O_RDONLY)

    def export(_handle, handle_type, _flags):
        assert (
            handle_type
            == driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        return (0, local_fd)

    driver.cuMemExportToShareableHandle = export
    monkeypatch.setattr(cuda_vmm, "_driver", driver)
    monkeypatch.setattr(cuda_vmm.dist, "get_world_size", lambda _group: 2)

    def all_gather(output, value, *, group):
        assert group == "cpu-group"
        output[:] = [value, value]

    monkeypatch.setattr(cuda_vmm.dist, "all_gather_object", all_gather)

    try:
        transport, exported = cuda_vmm._export_local_handle(
            driver, "cpu-group", rank=0, local_handle=19
        )
        assert transport == "posix_fd"
        assert exported == local_fd
    finally:
        os.close(local_fd)


def test_fd_message_transfers_descriptor_ownership() -> None:
    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    local_fd = os.open(os.devnull, os.O_RDONLY)
    received_fd = None
    try:
        cuda_vmm._send_fd(sender, rank=3, fd=local_fd)
        message = cuda_vmm._recv_fd(receiver)
        assert message is not None
        rank, received_fd = message
        assert rank == 3
        assert os.fstat(received_fd).st_mode == os.fstat(local_fd).st_mode
    finally:
        sender.close()
        receiver.close()
        os.close(local_fd)
        if received_fd is not None:
            os.close(received_fd)


def test_fd_message_accepts_partial_stream_header() -> None:
    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    local_fd = os.open(os.devnull, os.O_RDONLY)
    received_fd = None
    try:
        payload = cuda_vmm._FD_HEADER.pack(5)
        fds = array.array("i", [local_fd])
        assert (
            sender.sendmsg(
                [payload[:1]],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())],
            )
            == 1
        )
        sender.sendall(payload[1:])

        message = cuda_vmm._recv_fd(receiver)
        assert message is not None
        rank, received_fd = message
        assert rank == 5
        assert os.fstat(received_fd).st_mode == os.fstat(local_fd).st_mode
    finally:
        sender.close()
        receiver.close()
        os.close(local_fd)
        if received_fd is not None:
            os.close(received_fd)


def test_fd_exchange_timeout_joins_receiver_before_cleanup(monkeypatch) -> None:
    local_fd = os.open(os.devnull, os.O_RDONLY)
    gather_count = 0

    def all_gather(output, value, *, group):
        nonlocal gather_count
        assert group == "cpu-group"
        if gather_count == 0:  # socket setup errors
            output[:] = [None, None]
        elif gather_count == 1:  # socket paths; peer path is unreachable
            output[:] = [value, "/missing/vllm-vmm-peer.sock"]
        else:  # transfer errors
            output[:] = [value, value]
        gather_count += 1

    monkeypatch.setattr(cuda_vmm, "_FD_EXCHANGE_TIMEOUT_S", 0.01)
    monkeypatch.setattr(cuda_vmm.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(cuda_vmm.dist, "all_gather_object", all_gather)
    try:
        with pytest.raises(RuntimeError, match="POSIX FD transfer failed"):
            cuda_vmm._exchange_posix_fds(
                "cpu-group", rank=0, world_size=2, local_fd=local_fd
            )
    finally:
        os.close(local_fd)

    assert not any(
        thread.name == "vllm-vmm-fd-rank-0" and thread.is_alive()
        for thread in cuda_vmm.threading.enumerate()
    )


def test_request_rejects_a_multi_host_group(monkeypatch) -> None:
    group = MagicMock(spec=dist.ProcessGroup)
    request = ((8, 4), torch.float16, 1, False, False)
    gathered = [[None, None], [request, request], ["host-a", "host-b"]]

    monkeypatch.setattr(cuda_vmm.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(cuda_vmm.dist, "get_backend", lambda _group: "gloo")
    monkeypatch.setattr(cuda_vmm.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(cuda_vmm.dist, "get_world_size", lambda _group: 2)

    def all_gather(output, _value, *, group):
        assert group is not None
        output[:] = gathered.pop(0)

    monkeypatch.setattr(cuda_vmm.dist, "all_gather_object", all_gather)

    with pytest.raises(ValueError, match="every group rank on one host"):
        cuda_vmm._validate_request(group, *request)


def test_peer_capability_rejects_missing_native_atomics(monkeypatch) -> None:
    driver = SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        CUdevice_P2PAttribute=SimpleNamespace(
            CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED=7
        ),
        cuDeviceGet=lambda device_id: (0, device_id),
        cuDeviceCanAccessPeer=lambda _source, _peer: (0, 1),
        cuDeviceGetP2PAttribute=lambda _attribute, _source, _peer: (0, 0),
    )
    monkeypatch.setattr(cuda_vmm, "_driver", driver)
    monkeypatch.setattr(cuda_vmm.dist, "get_world_size", lambda _group: 2)

    gather_count = 0

    def all_gather(output, value, *, group):
        nonlocal gather_count
        assert group == "cpu-group"
        if gather_count == 0:
            output[:] = [0, 1]
        else:
            output[:] = [value, value]
        gather_count += 1

    monkeypatch.setattr(cuda_vmm.dist, "all_gather_object", all_gather)

    with pytest.raises(RuntimeError, match="requires native peer atomics"):
        cuda_vmm._validate_peer_capabilities(
            driver,
            "cpu-group",
            rank=0,
            device_id=0,
            require_native_atomics=True,
        )


def test_mapping_rollback_unmaps_in_reverse_order(monkeypatch) -> None:
    driver = _FakeDriver()
    monkeypatch.setattr(cuda_vmm, "_driver", driver)
    mappings = [100, 164, 228]

    cuda_vmm._rollback_mapping(driver, 100, 192, mappings, 64)

    assert mappings == []
    assert driver.unmapped == [(228, 64), (164, 64), (100, 64)]
    assert driver.freed == [(100, 192)]


def test_close_is_idempotent_and_keeps_dlpack_callbacks_alive(monkeypatch) -> None:
    driver = _FakeDriver()
    monkeypatch.setattr(cuda_vmm, "_driver", driver)
    synchronize = MagicMock()
    monkeypatch.setattr(cuda_vmm.torch.cuda, "synchronize", synchronize)
    refs = [object()]
    retired_count = len(cuda_vmm._RETIRED_DLPACK_REFS)
    view = RankMajorPeerView(
        global_view=MagicMock(),
        rank_local_view=MagicMock(),
        local_view=MagicMock(),
        requested_shape=(2, 4),
        aligned_local_shape=(8, 4),
        rows_per_rank=8,
        bytes_per_rank=64,
        rank=0,
        world_size=2,
        device=torch.device("cuda:0"),
        transport="fabric",
        _global_base_va=100,
        _rank_local_base_va=300,
        _total_bytes=128,
        _global_mappings=[100, 164],
        _rank_local_mappings=[300, 364],
        _dlpack_refs=refs,
    )

    view.close()
    view.close()

    synchronize.assert_called_once_with(torch.device("cuda:0"))
    assert driver.unmapped == [(164, 64), (100, 64), (364, 64), (300, 64)]
    assert driver.freed == [(100, 128), (300, 128)]
    assert view.closed
    assert view.global_view is None
    assert view.rank_local_view is None
    assert view.local_view is None
    assert cuda_vmm._RETIRED_DLPACK_REFS[retired_count] is refs
    cuda_vmm._RETIRED_DLPACK_REFS.pop()


def test_close_can_retry_a_failed_unmap(monkeypatch) -> None:
    driver = _FakeDriver()
    attempts = 0

    def unmap(address: int, size: int):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return (1,)
        driver.unmapped.append((address, size))
        return (0,)

    driver.cuMemUnmap = unmap
    monkeypatch.setattr(cuda_vmm, "_driver", driver)
    monkeypatch.setattr(cuda_vmm.torch.cuda, "synchronize", lambda _device: None)
    view = RankMajorPeerView(
        global_view=MagicMock(),
        rank_local_view=None,
        local_view=MagicMock(),
        requested_shape=(1,),
        aligned_local_shape=(64,),
        rows_per_rank=64,
        bytes_per_rank=64,
        rank=0,
        world_size=1,
        device=torch.device("cuda:0"),
        transport="posix_fd",
        _global_base_va=100,
        _rank_local_base_va=None,
        _total_bytes=64,
        _global_mappings=[100],
        _rank_local_mappings=[],
        _dlpack_refs=[],
    )

    with pytest.raises(RuntimeError, match="CUDA VMM close failed"):
        view.close()
    assert not view.closed

    view.close()
    assert view.closed
    assert driver.unmapped == [(100, 64)]
    assert driver.freed == [(100, 64)]
    cuda_vmm._RETIRED_DLPACK_REFS.pop()


def _rank_major_peer_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        module_path = (
            Path(__file__).parents[2]
            / "vllm/distributed/device_communicators/peer_memory.py"
        )
        spec = importlib.util.spec_from_file_location("_test_peer_memory", module_path)
        assert spec is not None and spec.loader is not None
        peer_memory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(peer_memory)

        view = create_rank_major_peer_view(
            (7, 4),
            dtype=torch.uint8,
            group=dist.group.WORLD,
            first_dim_multiple=8,
            map_rank_local=True,
            require_native_atomics=True,
        )
        fence = peer_memory.PeerMemoryFence(dist.group.WORLD, torch.device("cuda"))
        assert view.global_view is not None
        assert view.local_view is not None
        assert view.rank_local_view is not None
        assert (
            view.local_view.untyped_storage().data_ptr() == view.local_view.data_ptr()
        )
        assert view.local_view.untyped_storage().nbytes() == view.bytes_per_rank
        view.local_view.fill_(rank + 1)
        fence()

        for owner_rank in range(world_size):
            start = owner_rank * view.rows_per_rank
            canonical = view.global_view.narrow(0, start, 7)
            assert torch.all(canonical == owner_rank + 1).item(), (
                rank,
                owner_rank,
                canonical[:, 0].tolist(),
            )

            local_segment = (owner_rank - rank) % world_size
            start = local_segment * view.rows_per_rank
            rotated = view.rank_local_view.narrow(0, start, 7)
            assert torch.all(rotated == owner_rank + 1).item(), (
                rank,
                owner_rank,
                rotated[:, 0].tolist(),
            )
        dist.barrier()

        # Model producer kernels use the canonical mapping to store directly
        # into every rank's cache. Give each producer a disjoint row in every
        # owner segment so all directed peer-store paths are covered.
        for owner_rank in range(world_size):
            start = owner_rank * view.rows_per_rank
            view.global_view[start + rank].fill_(101 + rank)
        fence()

        expected = torch.arange(
            101,
            101 + world_size,
            dtype=torch.uint8,
            device=view.local_view.device,
        )
        assert torch.equal(view.local_view[:world_size, 0], expected)
        dist.barrier()

        # Exercise alternating epochs and skewed producers. Each rank writes a
        # disjoint byte in every owner segment, then the system-scope fence
        # must make all producers visible in every local allocation.
        for epoch in range(1_000 if world_size == 4 else 10):
            if epoch % 97 == rank:
                time.sleep(rank * 0.001)
            for owner_rank in range(world_size):
                start = owner_rank * view.rows_per_rank
                view.global_view[start + rank, 1].fill_((epoch + rank) % 251)
            fence()
            expected = torch.tensor(
                [(epoch + source_rank) % 251 for source_rank in range(world_size)],
                dtype=torch.uint8,
                device=view.local_view.device,
            )
            assert torch.equal(view.local_view[:world_size, 1], expected)
            dist.barrier()

        # Closing an owner allocation while a peer is still reading it is
        # invalid. Quiesce this multi-process stress test before teardown.
        dist.barrier()
        fence.close()
        view.close()
        view.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _peer_access_available(world_size: int) -> bool:
    if not current_platform.is_cuda() or torch.cuda.device_count() < world_size:
        return False
    return all(
        src == dst or torch.cuda.can_device_access_peer(src, dst)
        for src in range(world_size)
        for dst in range(world_size)
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_rank_major_peer_reads_writes_and_repeated_close(world_size: int) -> None:
    if not _peer_access_available(world_size):
        pytest.skip(f"CUDA VMM peer view requires {world_size} peer-accessible GPUs")
    mp.spawn(
        _rank_major_peer_worker,
        args=(world_size, get_open_port()),
        nprocs=world_size,
    )
