# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/sgl-project/sglang/pull/31435
"""CUDA VMM-backed rank-major peer tensor views.

The setup path exchanges allocation handles. Once constructed, the returned
views use ordinary device loads and stores; they do not issue runtime pulls,
gets, collectives, or copies.
"""

from __future__ import annotations

import array
import ctypes
import math
import os
import socket
import struct
import tempfile
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)

_driver: Any | None = None
_FD_EXCHANGE_TIMEOUT_S = 120.0
_FD_HEADER = struct.Struct("<Q")

# A PyTorch tensor may be kept alive after its owning peer view is closed. The
# VMM address is invalid after close, but the DLPack deleter callback must still
# be callable when such an alias is eventually destroyed.
_RETIRED_DLPACK_REFS: list[list[Any]] = []


def _get_cuda_driver() -> Any:
    global _driver
    if _driver is None:
        try:
            from cuda.bindings import driver
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CUDA VMM peer views require the cuda-bindings package."
            ) from exc
        _driver = driver
    return _driver


def _check_driver(result: Any, operation: str) -> Any:
    """Return a CUDA driver result value or raise an actionable error."""
    if not isinstance(result, tuple):
        result = (result,)
    error = result[0]
    driver = _get_cuda_driver()
    if error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed with {error}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def _error_text(error: BaseException | None) -> str | None:
    if error is None:
        return None
    return f"{type(error).__name__}: {error}"


def _gather_errors(
    group: ProcessGroup, local_error: BaseException | None
) -> list[str | None]:
    errors: list[str | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, _error_text(local_error), group=group)
    return errors


def _raise_if_any_rank_failed(
    group: ProcessGroup,
    rank: int,
    stage: str,
    local_error: BaseException | None,
) -> None:
    errors = _gather_errors(group, local_error)
    for failed_rank, error in enumerate(errors):
        if error is None:
            continue
        message = f"CUDA VMM {stage} failed on group rank {failed_rank}: {error}"
        if failed_rank == rank:
            raise RuntimeError(message) from local_error
        raise RuntimeError(message)


def _validate_request(
    group: ProcessGroup,
    local_shape: tuple[int, ...],
    dtype: torch.dtype,
    first_dim_multiple: int,
    map_rank_local: bool,
    require_native_atomics: bool,
) -> tuple[int, int]:
    if not isinstance(group, ProcessGroup):
        raise TypeError("group must be a torch.distributed.ProcessGroup")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if dist.get_backend(group) == dist.Backend.NCCL:
        raise ValueError("CUDA VMM setup requires a CPU-capable ProcessGroup, not NCCL")

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    local_error: BaseException | None = None
    try:
        if not local_shape:
            raise ValueError("local_shape must have at least one dimension")
        if any(not isinstance(dim, int) or dim <= 0 for dim in local_shape):
            raise ValueError(f"local_shape dimensions must be positive: {local_shape}")
        if first_dim_multiple <= 0:
            raise ValueError("first_dim_multiple must be positive")
        _dtype_to_dlpack(dtype)
    except BaseException as exc:
        local_error = exc
    _raise_if_any_rank_failed(group, rank, "request validation", local_error)

    requests: list[Any] = [None] * world_size
    request = (
        local_shape,
        dtype,
        first_dim_multiple,
        map_rank_local,
        require_native_atomics,
    )
    dist.all_gather_object(requests, request, group=group)
    if any(peer_request != request for peer_request in requests):
        details = ", ".join(
            f"rank {peer_rank}={peer_request!r}"
            for peer_rank, peer_request in enumerate(requests)
        )
        raise ValueError(
            "CUDA VMM peer view arguments must match on every rank; " + details
        )

    hosts: list[str | None] = [None] * world_size
    dist.all_gather_object(hosts, socket.gethostname(), group=group)
    if len(set(hosts)) != 1:
        details = ", ".join(
            f"rank {peer_rank}={host!r}" for peer_rank, host in enumerate(hosts)
        )
        raise ValueError(
            "CUDA VMM peer views require every group rank on one host; " + details
        )
    return rank, world_size


def _validate_peer_capabilities(
    driver: Any,
    group: ProcessGroup,
    rank: int,
    device_id: int,
    require_native_atomics: bool,
) -> None:
    """Collectively validate directed access from each rank to every peer."""
    world_size = dist.get_world_size(group)
    device_ids: list[int | None] = [None] * world_size
    dist.all_gather_object(device_ids, device_id, group=group)

    local_error: BaseException | None = None
    try:
        if any(peer_device is None for peer_device in device_ids):
            raise RuntimeError("A peer rank did not report its CUDA device.")
        if len(set(device_ids)) != world_size:
            raise ValueError(
                "CUDA VMM peer views require one distinct CUDA device per rank; "
                f"got {device_ids}."
            )

        source_device = _check_driver(
            driver.cuDeviceGet(device_id), f"cuDeviceGet({device_id})"
        )
        atomic_attribute = (
            driver.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED
        )
        for peer_rank, peer_device_id in enumerate(device_ids):
            if peer_rank == rank:
                continue
            assert peer_device_id is not None
            peer_device = _check_driver(
                driver.cuDeviceGet(peer_device_id),
                f"cuDeviceGet({peer_device_id})",
            )
            can_access = int(
                _check_driver(
                    driver.cuDeviceCanAccessPeer(source_device, peer_device),
                    f"cuDeviceCanAccessPeer({device_id}, {peer_device_id})",
                )
            )
            if not can_access:
                raise NotImplementedError(
                    "CUDA device "
                    f"{device_id} cannot access PCP rank {peer_rank} device "
                    f"{peer_device_id}."
                )
            if require_native_atomics:
                native_atomics = int(
                    _check_driver(
                        driver.cuDeviceGetP2PAttribute(
                            atomic_attribute, source_device, peer_device
                        ),
                        "cuDeviceGetP2PAttribute(NATIVE_ATOMIC_SUPPORTED, "
                        f"{device_id}, {peer_device_id})",
                    )
                )
                if not native_atomics:
                    raise NotImplementedError(
                        "Owner-history PCP requires native peer atomics, but CUDA "
                        f"device {device_id} cannot issue them to PCP rank "
                        f"{peer_rank} device {peer_device_id}."
                    )
    except BaseException as exc:
        local_error = exc
    _raise_if_any_rank_failed(group, rank, "peer capability validation", local_error)


def _make_allocation_property(driver: Any, device_id: int, handle_types: Any) -> Any:
    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location = driver.CUmemLocation()
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    prop.requestedHandleTypes = handle_types
    return prop


def _make_rw_access(driver: Any, device_id: int) -> Any:
    access = driver.CUmemAccessDesc()
    access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = device_id
    access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return access


def _element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype, device="cpu").element_size()


def _aligned_local_shape(
    local_shape: tuple[int, ...],
    dtype: torch.dtype,
    granularity: int,
    first_dim_multiple: int,
) -> tuple[tuple[int, ...], int]:
    row_bytes = math.prod(local_shape[1:]) * _element_size(dtype)
    rows_per_granularity = granularity // math.gcd(granularity, row_bytes)
    row_alignment = math.lcm(rows_per_granularity, first_dim_multiple)
    rows = math.ceil(local_shape[0] / row_alignment) * row_alignment
    return (rows, *local_shape[1:]), rows * row_bytes


def _release_handle(driver: Any, handle: Any, label: str) -> None:
    _check_driver(driver.cuMemRelease(handle), label)


def _allocate_local_segment(
    driver: Any,
    group: ProcessGroup,
    rank: int,
    device_id: int,
    local_shape: tuple[int, ...],
    dtype: torch.dtype,
    first_dim_multiple: int,
) -> tuple[Any, tuple[int, ...], int, int]:
    posix = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    granularity_flag = (
        driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    )

    prop: Any | None = None
    granularity = 0
    aligned_shape: tuple[int, ...] = ()
    segment_bytes = 0
    local_error: BaseException | None = None
    try:
        # The Python CUDA bindings do not consistently accept combined
        # CUmemAllocationHandleType bitmasks across releases. POSIX FDs are the
        # portable same-host transport required by this primitive; transport
        # choice does not affect runtime peer loads or stores after mapping.
        prop = _make_allocation_property(driver, device_id, posix)
        granularity = int(
            _check_driver(
                driver.cuMemGetAllocationGranularity(prop, granularity_flag),
                "cuMemGetAllocationGranularity",
            )
        )
        aligned_shape, segment_bytes = _aligned_local_shape(
            local_shape, dtype, granularity, first_dim_multiple
        )
    except BaseException as exc:
        local_error = exc
    _raise_if_any_rank_failed(group, rank, "allocation planning", local_error)
    assert prop is not None

    plan = (aligned_shape, segment_bytes, granularity)
    plans: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(plans, plan, group=group)
    if any(peer_plan != plan for peer_plan in plans):
        details = ", ".join(
            f"rank {peer_rank}={peer_plan!r}"
            for peer_rank, peer_plan in enumerate(plans)
        )
        raise RuntimeError(
            "CUDA VMM allocation granularity must match on every rank; " + details
        )

    local_handle: Any | None = None
    allocation_error: BaseException | None = None
    try:
        local_handle = _check_driver(
            driver.cuMemCreate(segment_bytes, prop, 0),
            "cuMemCreate(POSIX_FD)",
        )
    except BaseException as exc:
        allocation_error = exc
    try:
        _raise_if_any_rank_failed(group, rank, "local allocation", allocation_error)
    except BaseException:
        if local_handle is not None:
            _release_handle(driver, local_handle, "cuMemRelease(failed allocation)")
        raise
    return local_handle, aligned_shape, segment_bytes, granularity


def _export_local_handle(
    driver: Any,
    group: ProcessGroup,
    rank: int,
    local_handle: Any,
) -> tuple[Literal["fabric", "posix_fd"], bytes | int]:
    posix = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

    local_fd: int | None = None
    posix_error: BaseException | None = None
    try:
        local_fd = int(
            _check_driver(
                driver.cuMemExportToShareableHandle(local_handle, posix, 0),
                "cuMemExportToShareableHandle(POSIX_FD)",
            )
        )
    except BaseException as exc:
        posix_error = exc
    try:
        _raise_if_any_rank_failed(group, rank, "handle export", posix_error)
    except BaseException:
        if local_fd is not None:
            os.close(local_fd)
        raise
    assert local_fd is not None
    return "posix_fd", local_fd


def _send_fd(sock: socket.socket, rank: int, fd: int) -> None:
    payload = _FD_HEADER.pack(rank)
    fds = array.array("i", [fd])
    sent = sock.sendmsg(
        [payload],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())],
    )
    if sent <= 0:
        raise RuntimeError("sendmsg did not send the FD header")
    if sent < len(payload):
        # SOCK_SEQPACKET sends the header atomically. The SOCK_STREAM fallback
        # may split it; SCM_RIGHTS is already attached to the first byte, so
        # only the remaining framing bytes are sent here.
        sock.sendall(payload[sent:])


def _recv_fd(sock: socket.socket) -> tuple[int, int] | None:
    fd_size = array.array("i").itemsize
    payload = bytearray()
    received = array.array("i")
    try:
        while len(payload) < _FD_HEADER.size:
            chunk, ancillary, _, _ = sock.recvmsg(
                _FD_HEADER.size - len(payload), socket.CMSG_SPACE(fd_size)
            )
            if not chunk:
                if not payload and not received:
                    return None
                raise RuntimeError(
                    "peer closed with an incomplete POSIX FD header: "
                    f"{len(payload)}/{_FD_HEADER.size} bytes"
                )
            payload.extend(chunk)
            for level, message_type, data in ancillary:
                if level == socket.SOL_SOCKET and message_type == socket.SCM_RIGHTS:
                    received.frombytes(data[: len(data) - len(data) % fd_size])
        if len(received) != 1:
            raise RuntimeError(
                f"expected one file descriptor, received {len(received)}"
            )
        return _FD_HEADER.unpack(payload)[0], int(received[0])
    except BaseException:
        for received_fd in received:
            os.close(received_fd)
        raise


def _exchange_posix_fds(
    group: ProcessGroup, rank: int, world_size: int, local_fd: int
) -> dict[int, int]:
    """Exchange one process-local CUDA allocation FD per group rank."""
    socket_kind = getattr(socket, "SOCK_SEQPACKET", socket.SOCK_STREAM)
    socket_dir: str | None = None
    socket_path: str | None = None
    server: socket.socket | None = None
    received: dict[int, int] = {}
    receive_errors: list[BaseException] = []
    active_connections: set[socket.socket] = set()
    connection_lock = threading.Lock()

    def receive_loop() -> None:
        assert server is not None
        try:
            for _ in range(world_size - 1):
                connection, _ = server.accept()
                with connection_lock:
                    active_connections.add(connection)
                try:
                    connection.settimeout(_FD_EXCHANGE_TIMEOUT_S)
                    message = _recv_fd(connection)
                    if message is None:
                        raise RuntimeError(
                            "peer closed its socket before sending an FD"
                        )
                    peer_rank, peer_fd = message
                    if peer_rank in received:
                        os.close(peer_fd)
                        raise RuntimeError(f"duplicate FD from group rank {peer_rank}")
                    received[peer_rank] = peer_fd
                finally:
                    with connection_lock:
                        active_connections.discard(connection)
                    connection.close()
        except BaseException as exc:
            receive_errors.append(exc)

    try:
        setup_error: BaseException | None = None
        try:
            socket_dir = tempfile.mkdtemp(prefix="vllm_vmm_fd_")
            socket_path = os.path.join(socket_dir, f"rank_{rank}.sock")
            server = socket.socket(socket.AF_UNIX, socket_kind)
            server.settimeout(_FD_EXCHANGE_TIMEOUT_S)
            server.bind(socket_path)
            server.listen(world_size)
        except BaseException as exc:
            setup_error = exc
        _raise_if_any_rank_failed(group, rank, "POSIX FD socket setup", setup_error)

        socket_paths: list[str | None] = [None] * world_size
        dist.all_gather_object(socket_paths, socket_path, group=group)
        thread = threading.Thread(
            target=receive_loop,
            name=f"vllm-vmm-fd-rank-{rank}",
            daemon=True,
        )
        thread.start()

        send_error: BaseException | None = None
        try:
            for peer_rank, peer_path in enumerate(socket_paths):
                if peer_rank == rank:
                    continue
                assert peer_path is not None
                with socket.socket(socket.AF_UNIX, socket_kind) as client:
                    client.settimeout(_FD_EXCHANGE_TIMEOUT_S)
                    client.connect(peer_path)
                    _send_fd(client, rank, local_fd)
        except BaseException as exc:
            send_error = exc
        thread.join(_FD_EXCHANGE_TIMEOUT_S)
        transfer_error = send_error
        if thread.is_alive():
            if transfer_error is None:
                transfer_error = RuntimeError("timed out receiving POSIX FDs")
            # Stop accept/recv before inspecting or closing ``received`` so a
            # late SCM_RIGHTS message cannot race cleanup and leak its FD.
            assert server is not None
            server.close()
            with connection_lock:
                connections = tuple(active_connections)
            for connection in connections:
                with suppress(OSError):
                    connection.shutdown(socket.SHUT_RDWR)
                connection.close()
            thread.join()
        if transfer_error is None and receive_errors:
            transfer_error = receive_errors[0]
        expected = set(range(world_size)) - {rank}
        if transfer_error is None and set(received) != expected:
            transfer_error = RuntimeError(
                "FD sender mismatch: "
                f"missing={sorted(expected - set(received))}, "
                f"extra={sorted(set(received) - expected)}"
            )
        try:
            _raise_if_any_rank_failed(group, rank, "POSIX FD transfer", transfer_error)
        except BaseException:
            for peer_fd in received.values():
                os.close(peer_fd)
            received.clear()
            raise
        return received
    finally:
        if server is not None:
            server.close()
        if socket_path is not None:
            with suppress(FileNotFoundError):
                os.unlink(socket_path)
        if socket_dir is not None:
            with suppress(OSError):
                os.rmdir(socket_dir)


def _import_peer_handle(
    driver: Any,
    transport: Literal["fabric", "posix_fd"],
    exported_handle: bytes | int,
    peer_rank: int,
) -> Any:
    handle_types = driver.CUmemAllocationHandleType
    if transport == "fabric":
        handle_type = handle_types.CU_MEM_HANDLE_TYPE_FABRIC
        return _check_driver(
            driver.cuMemImportFromShareableHandle(exported_handle, handle_type),
            f"cuMemImportFromShareableHandle(FABRIC, rank={peer_rank})",
        )

    handle_type = handle_types.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    duplicate_fd = os.dup(int(exported_handle))
    try:
        return _check_driver(
            driver.cuMemImportFromShareableHandle(duplicate_fd, handle_type),
            f"cuMemImportFromShareableHandle(POSIX_FD, rank={peer_rank})",
        )
    finally:
        os.close(duplicate_fd)


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLPACK_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLPACK_DELETER),
]


def _dtype_to_dlpack(dtype: torch.dtype) -> tuple[int, int]:
    types: dict[torch.dtype, tuple[int, int]] = {
        torch.uint8: (1, 8),
        torch.int8: (0, 8),
        torch.int16: (0, 16),
        torch.int32: (0, 32),
        torch.int64: (0, 64),
        torch.float16: (2, 16),
        torch.bfloat16: (4, 16),
        torch.float32: (2, 32),
        torch.float64: (2, 64),
        torch.bool: (6, 8),
        torch.complex64: (5, 64),
        torch.complex128: (5, 128),
    }
    optional_types = (
        ("float8_e4m3fn", 10),
        ("float8_e4m3fnuz", 11),
        ("float8_e5m2", 12),
        ("float8_e5m2fnuz", 13),
    )
    for name, code in optional_types:
        if hasattr(torch, name):
            types[getattr(torch, name)] = (code, 8)
    try:
        return types[dtype]
    except KeyError as exc:
        raise TypeError(f"CUDA VMM peer views do not support dtype {dtype}") from exc


def _tensor_from_cuda_pointer(
    pointer: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device_id: int,
    refs: list[Any],
) -> torch.Tensor:
    dtype_code, dtype_bits = _dtype_to_dlpack(dtype)
    shape_array = (ctypes.c_int64 * len(shape))(*shape)
    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(pointer)
    managed.dl_tensor.device = _DLDevice(2, device_id)
    managed.dl_tensor.ndim = len(shape)
    managed.dl_tensor.dtype = _DLDataType(dtype_code, dtype_bits, 1)
    managed.dl_tensor.shape = shape_array
    managed.dl_tensor.strides = None
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None

    @_DLPACK_DELETER
    def deleter(_: ctypes.POINTER(_DLManagedTensor)) -> None:
        return None

    managed.deleter = deleter
    refs.extend((managed, shape_array, deleter))
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    ctypes.pythonapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    capsule = ctypes.pythonapi.PyCapsule_New(ctypes.byref(managed), b"dltensor", None)
    return torch.from_dlpack(capsule)


def _rollback_mapping(
    driver: Any,
    base_va: int | None,
    total_bytes: int,
    mapped_addresses: list[int],
    segment_bytes: int,
) -> None:
    failed_addresses: list[int] = []
    while mapped_addresses:
        address = mapped_addresses.pop()
        try:
            _check_driver(
                driver.cuMemUnmap(address, segment_bytes),
                "cuMemUnmap(rollback)",
            )
        except Exception:
            logger.exception("Failed to roll back CUDA VMM mapping at %#x", address)
            failed_addresses.append(address)
    if base_va is not None and not failed_addresses:
        try:
            _check_driver(
                driver.cuMemAddressFree(base_va, total_bytes),
                "cuMemAddressFree(rollback)",
            )
        except Exception:
            logger.exception(
                "Failed to roll back CUDA VMM address range at %#x", base_va
            )


@dataclass
class RankMajorPeerView:
    """Rank-segmented CUDA memory exposed through rank-major tensor aliases.

    ``global_view`` stores group-rank ``r`` at rows
    ``[r * rows_per_rank:(r + 1) * rows_per_rank]``. If requested,
    ``rank_local_view`` contains the same physical segments rotated so the
    current rank is first. ``local_view`` is the current rank's owner segment.

    All tensor aliases become invalid when :meth:`close` is called. Callers
    must arrange any inter-rank write/read synchronization themselves.
    """

    global_view: torch.Tensor | None
    rank_local_view: torch.Tensor | None
    local_view: torch.Tensor | None
    requested_shape: tuple[int, ...]
    aligned_local_shape: tuple[int, ...]
    rows_per_rank: int
    bytes_per_rank: int
    rank: int
    world_size: int
    device: torch.device
    transport: Literal["fabric", "posix_fd"]
    _global_base_va: int | None
    _rank_local_base_va: int | None
    _total_bytes: int
    _global_mappings: list[int] = field(repr=False)
    _rank_local_mappings: list[int] = field(repr=False)
    _dlpack_refs: list[Any] = field(repr=False)
    _refs_retired: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> RankMajorPeerView:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Tensor views can outlive the allocation wrapper. Keep the ctypes
        # DLPack metadata and callback callable until process teardown even
        # when a caller forgot the required explicit close(). CUDA mappings
        # are intentionally not touched from a Python finalizer.
        if not self._refs_retired and self._dlpack_refs:
            _RETIRED_DLPACK_REFS.append(self._dlpack_refs)
            self._refs_retired = True

    def close(self) -> None:
        """Synchronize the local device and deterministically release mappings."""
        if self._closed:
            return
        torch.cuda.synchronize(self.device)

        self.local_view = None
        self.rank_local_view = None
        self.global_view = None
        if not self._refs_retired:
            _RETIRED_DLPACK_REFS.append(self._dlpack_refs)
            self._refs_retired = True

        driver = _get_cuda_driver()
        failures: list[str] = []
        regions = (
            ("global", "_global_base_va", self._global_mappings),
            ("rank-local", "_rank_local_base_va", self._rank_local_mappings),
        )
        for label, base_attribute, mappings in regions:
            while mappings:
                address = mappings[-1]
                try:
                    _check_driver(
                        driver.cuMemUnmap(address, self.bytes_per_rank),
                        f"cuMemUnmap({label})",
                    )
                except BaseException as exc:
                    failures.append(str(exc))
                    break
                mappings.pop()
            base_va = getattr(self, base_attribute)
            if base_va is not None and not mappings:
                try:
                    _check_driver(
                        driver.cuMemAddressFree(base_va, self._total_bytes),
                        f"cuMemAddressFree({label})",
                    )
                except BaseException as exc:
                    failures.append(str(exc))
                else:
                    setattr(self, base_attribute, None)

        self._closed = self._global_base_va is None and self._rank_local_base_va is None
        if failures:
            raise RuntimeError("CUDA VMM close failed: " + "; ".join(failures))


def create_rank_major_peer_view(
    local_shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    group: ProcessGroup,
    first_dim_multiple: int = 1,
    map_rank_local: bool = False,
    require_native_atomics: bool = False,
    device: torch.device | str | int | None = None,
) -> RankMajorPeerView:
    """Collectively create a same-host rank-major CUDA peer tensor.

    Each rank owns one physical VMM allocation. Every rank maps all allocations
    in group-rank order into one contiguous virtual address range. Setup uses
    same-host POSIX file descriptors to transfer allocation handles.

    Args:
        local_shape: Requested owner-local tensor shape. Its first dimension is
            padded to CUDA allocation granularity and ``first_dim_multiple``.
        dtype: Tensor element type.
        group: Same-host, CPU-capable process group used for setup collectives.
        first_dim_multiple: Additional alignment for the padded first dimension.
        map_rank_local: Also create a rotated alias with the local rank first.
        require_native_atomics: Require directed native peer-atomic support
            between every pair of devices in addition to peer load/store access.
        device: CUDA device. It must be the process's current CUDA device.

    Returns:
        A peer view whose ``local_view`` is the owner-local allocation.

    Raises:
        RuntimeError: If CUDA VMM setup fails on any group rank.
        ValueError: If arguments, topology, or current device are invalid.
    """
    local_shape = tuple(local_shape)
    rank, world_size = _validate_request(
        group,
        local_shape,
        dtype,
        first_dim_multiple,
        map_rank_local,
        require_native_atomics,
    )

    local_error: BaseException | None = None
    driver: Any | None = None
    device_id = -1
    resolved_device = torch.device("cuda")
    try:
        driver = _get_cuda_driver()
        current_device = torch.cuda.current_device()
        if device is None:
            resolved_device = torch.device(f"cuda:{current_device}")
        elif isinstance(device, int):
            resolved_device = torch.device(f"cuda:{device}")
        else:
            resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError(f"device must be CUDA, got {resolved_device}")
        device_id = (
            current_device
            if resolved_device.index is None
            else int(resolved_device.index)
        )
        if device_id != current_device:
            raise ValueError(
                f"device {device_id} is not the current CUDA device {current_device}"
            )
        resolved_device = torch.device(f"cuda:{device_id}")
    except BaseException as exc:
        local_error = exc
    _raise_if_any_rank_failed(group, rank, "CUDA initialization", local_error)
    assert driver is not None
    _validate_peer_capabilities(driver, group, rank, device_id, require_native_atomics)

    local_handle, aligned_shape, segment_bytes, granularity = _allocate_local_segment(
        driver,
        group,
        rank,
        device_id,
        local_shape,
        dtype,
        first_dim_multiple,
    )

    local_fd: int | None = None
    peer_fds: dict[int, int] = {}
    imported_handles: list[Any] = []
    dlpack_refs: list[Any] = []
    global_base_va: int | None = None
    rank_local_base_va: int | None = None
    global_mappings: list[int] = []
    rank_local_mappings: list[int] = []
    total_bytes = segment_bytes * world_size
    transport: Literal["fabric", "posix_fd"] = "posix_fd"
    try:
        transport, exported_handle = _export_local_handle(
            driver, group, rank, local_handle
        )
        peer_handles: list[bytes | int | None] = [None] * world_size
        if transport == "fabric":
            dist.all_gather_object(peer_handles, exported_handle, group=group)
        else:
            local_fd = int(exported_handle)
            peer_fds = _exchange_posix_fds(group, rank, world_size, local_fd)

        mapping_error: BaseException | None = None
        try:
            global_base_va = int(
                _check_driver(
                    driver.cuMemAddressReserve(total_bytes, granularity, 0, 0),
                    "cuMemAddressReserve(global)",
                )
            )
            if map_rank_local:
                rank_local_base_va = int(
                    _check_driver(
                        driver.cuMemAddressReserve(total_bytes, granularity, 0, 0),
                        "cuMemAddressReserve(rank-local)",
                    )
                )

            access = _make_rw_access(driver, device_id)
            for peer_rank in range(world_size):
                handle = local_handle
                if peer_rank != rank:
                    exported = (
                        peer_handles[peer_rank]
                        if transport == "fabric"
                        else peer_fds[peer_rank]
                    )
                    assert exported is not None
                    handle = _import_peer_handle(driver, transport, exported, peer_rank)
                    imported_handles.append(handle)

                global_address = global_base_va + peer_rank * segment_bytes
                _check_driver(
                    driver.cuMemMap(global_address, segment_bytes, 0, handle, 0),
                    f"cuMemMap(global, rank={peer_rank})",
                )
                global_mappings.append(global_address)
                _check_driver(
                    driver.cuMemSetAccess(global_address, segment_bytes, [access], 1),
                    f"cuMemSetAccess(global, rank={peer_rank})",
                )

                if rank_local_base_va is not None:
                    rank_local_segment = (peer_rank - rank) % world_size
                    rank_local_address = (
                        rank_local_base_va + rank_local_segment * segment_bytes
                    )
                    _check_driver(
                        driver.cuMemMap(
                            rank_local_address, segment_bytes, 0, handle, 0
                        ),
                        f"cuMemMap(rank-local, rank={peer_rank})",
                    )
                    rank_local_mappings.append(rank_local_address)
                    _check_driver(
                        driver.cuMemSetAccess(
                            rank_local_address, segment_bytes, [access], 1
                        ),
                        f"cuMemSetAccess(rank-local, rank={peer_rank})",
                    )

                if peer_rank != rank:
                    _release_handle(
                        driver, handle, f"cuMemRelease(imported rank={peer_rank})"
                    )
                    imported_handles.pop()
        except BaseException as exc:
            mapping_error = exc
        _raise_if_any_rank_failed(group, rank, "peer mapping", mapping_error)

        release_error: BaseException | None = None
        try:
            _release_handle(driver, local_handle, "cuMemRelease(local)")
            local_handle = None
        except BaseException as exc:
            release_error = exc
        _raise_if_any_rank_failed(
            group, rank, "allocation handle release", release_error
        )

        global_view: torch.Tensor | None = None
        rank_local_view: torch.Tensor | None = None
        local_view: torch.Tensor | None = None
        tensor_error: BaseException | None = None
        try:
            global_shape = (world_size * aligned_shape[0], *aligned_shape[1:])
            assert global_base_va is not None
            global_view = _tensor_from_cuda_pointer(
                global_base_va, global_shape, dtype, device_id, dlpack_refs
            )
            if rank_local_base_va is not None:
                rank_local_view = _tensor_from_cuda_pointer(
                    rank_local_base_va,
                    global_shape,
                    dtype,
                    device_id,
                    dlpack_refs,
                )
            local_segment_va = global_base_va + rank * segment_bytes
            local_view = _tensor_from_cuda_pointer(
                local_segment_va,
                aligned_shape,
                dtype,
                device_id,
                dlpack_refs,
            )
        except BaseException as exc:
            tensor_error = exc
        _raise_if_any_rank_failed(group, rank, "tensor construction", tensor_error)
        assert global_view is not None and local_view is not None

        return RankMajorPeerView(
            global_view=global_view,
            rank_local_view=rank_local_view,
            local_view=local_view,
            requested_shape=local_shape,
            aligned_local_shape=aligned_shape,
            rows_per_rank=aligned_shape[0],
            bytes_per_rank=segment_bytes,
            rank=rank,
            world_size=world_size,
            device=resolved_device,
            transport=transport,
            _global_base_va=global_base_va,
            _rank_local_base_va=rank_local_base_va,
            _total_bytes=total_bytes,
            _global_mappings=global_mappings,
            _rank_local_mappings=rank_local_mappings,
            _dlpack_refs=dlpack_refs,
        )
    except BaseException:
        if dlpack_refs:
            _RETIRED_DLPACK_REFS.append(dlpack_refs)
        _rollback_mapping(
            driver,
            global_base_va,
            total_bytes,
            global_mappings,
            segment_bytes,
        )
        _rollback_mapping(
            driver,
            rank_local_base_va,
            total_bytes,
            rank_local_mappings,
            segment_bytes,
        )
        for handle in imported_handles:
            try:
                _check_driver(
                    driver.cuMemRelease(handle),
                    "cuMemRelease(imported rollback)",
                )
            except Exception:
                logger.exception("Failed to release imported CUDA VMM handle")
        if local_handle is not None:
            try:
                _check_driver(
                    driver.cuMemRelease(local_handle),
                    "cuMemRelease(local rollback)",
                )
            except Exception:
                logger.exception("Failed to release local CUDA VMM handle")
        raise
    finally:
        if local_fd is not None:
            os.close(local_fd)
        for peer_fd in peer_fds.values():
            os.close(peer_fd)
