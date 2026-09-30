# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA Engram: cuMem-shared host tables and sorted or inline host lookups."""

import ctypes
import os
import socket
import uuid
import weakref
from contextlib import ExitStack
from types import SimpleNamespace

import torch
from cuda.bindings import driver as cu

from vllm.distributed import get_engram_dp_group, get_engram_dp_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.deepseek_v41.common.engram import (
    DPSharedEngramStorage,
    EngramLayout,
)
from vllm.models.deepseek_v41.common.engram import (
    Engram as BaseEngram,
)
from vllm.models.deepseek_v41.common.engram import (
    ParallelEngramEmbedding as BaseParallelEngramEmbedding,
)
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

_POSIX_FD = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
_FD_EXCHANGE_TIMEOUT_S = 30
_INLINE_LOOKUP_MIN_TOKENS = 4096


def can_share_engram_tables(layout: EngramLayout, block_size: int = 32) -> bool:
    """Whether co-located DP replicas exist and cuMem can hold the full tables
    on the host NUMA node of every replica's GPU."""
    if get_engram_dp_size() == 1:
        logger.warning_once(
            "Engram DP replicas are not co-located on one node; "
            "storing the offloaded tables per rank instead of sharing them."
        )
        return False
    device = torch.accelerator.current_device_index()
    num_bytes = sum(layout.num_embeddings) * (
        layout.head_dim + layout.head_dim // block_size
    )
    attr = cu.CUdevice_attribute
    vmm = attr.CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
    reason = None
    if cu.cuDeviceGetAttribute(vmm, device) != (cu.CUresult.CUDA_SUCCESS, 1):
        reason = "cuMem cannot allocate host NUMA memory"
    else:
        numa = _host_numa_id(device)
        try:
            with open(f"/sys/devices/system/node/node{numa}/meminfo") as meminfo:
                numa_bytes = int(meminfo.read().split("MemTotal:")[1].split()[0]) * 1024
            if numa_bytes < num_bytes:
                reason = f"needs {num_bytes / 1024**3:.1f} GiB on host NUMA node {numa}"
        except OSError as exc:
            reason = f"cannot read host NUMA node {numa} memory: {exc}"
    # Replicas on other NUMA nodes may disagree, but must share or shard together.
    group = get_engram_dp_group()
    assert group is not None
    reasons: list[str | None] = [None] * group.world_size
    torch.distributed.all_gather_object(reasons, reason, group=group.cpu_group)
    failures = "; ".join(
        f"EDP rank {rank}: {reason}"
        for rank, reason in enumerate(reasons)
        if reason is not None
    )
    if failures:
        logger.warning_once(
            "Not sharing Engram tables across DP replicas (%s); "
            "sharding them across replicas instead.",
            failures,
        )
        return False
    return True


def _unwrap_cu(result):
    """Unwrap a cuda-python ``(error, value)`` result."""
    err, *value = result
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err.name}")
    return value[0] if value else None


def _host_numa_id(device: int) -> int:
    numa = cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID
    return max(0, _unwrap_cu(cu.cuDeviceGetAttribute(numa, device)))


class CuMemEngramStorage(DPSharedEngramStorage):
    """DP-shared host weights in cuMem, which GPUs map with 2M pages
    (/dev/shm only gets 64K ones)."""

    def _allocate(self, num_bytes: int) -> torch.Tensor:
        """Map one host allocation into every rank; the leader sends its fd."""
        group = self.group
        device = torch.accelerator.current_device_index()
        location = cu.CUmemLocationType
        with ExitStack() as stack:
            handle = fd = address = size = error = None
            if group.rank_in_group == 0:
                try:
                    prop = cu.CUmemAllocationProp()
                    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
                    prop.location.type = location.CU_MEM_LOCATION_TYPE_HOST_NUMA
                    prop.location.id = _host_numa_id(device)
                    prop.requestedHandleTypes = _POSIX_FD
                    granularity = _unwrap_cu(cu.cuMemGetAllocationGranularity(prop, 0))
                    size = round_up(num_bytes, granularity)
                    handle = _unwrap_cu(cu.cuMemCreate(size, prop, 0))
                    stack.callback(cu.cuMemRelease, handle)
                    fd = _unwrap_cu(
                        cu.cuMemExportToShareableHandle(handle, _POSIX_FD, 0)
                    )
                    stack.callback(os.close, fd)
                    server = stack.enter_context(socket.socket(socket.AF_UNIX))
                    address = f"\0vllm_engram_{uuid.uuid4().hex}"
                    server.bind(address)
                    server.listen()
                    server.settimeout(_FD_EXCHANGE_TIMEOUT_S)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            address, size, error = group.broadcast_object((address, size, error))
            if error is not None:
                raise RuntimeError(
                    "Engram shared-memory creation failed on EDP rank 0: " + error
                )

            owner = tensor = None
            try:
                if handle is not None:
                    assert fd is not None
                    for _ in range(group.world_size - 1):
                        with server.accept()[0] as conn:
                            socket.send_fds(conn, [b"\0"], [fd])
                else:
                    peer = stack.enter_context(socket.socket(socket.AF_UNIX))
                    # Don't hang if the leader fails before sending our fd.
                    peer.settimeout(_FD_EXCHANGE_TIMEOUT_S)
                    peer.connect(address)
                    fd = socket.recv_fds(peer, 1, 1)[1][0]
                    stack.callback(os.close, fd)
                    handle = _unwrap_cu(
                        cu.cuMemImportFromShareableHandle(fd, _POSIX_FD)
                    )
                    stack.callback(cu.cuMemRelease, handle)
                # The mapping keeps the allocation alive after its handle is released.
                pointer = _unwrap_cu(cu.cuMemAddressReserve(size, 0, 0, 0))
                owner = (ctypes.c_uint8 * size).from_address(int(pointer))
                finalizer = weakref.finalize(owner, self._unmap, pointer, size)
                finalizer.atexit = False  # type: ignore[misc]
                _unwrap_cu(cu.cuMemMap(pointer, size, 0, handle, 0))
                gpu, cpu = cu.CUmemAccessDesc(), cu.CUmemAccessDesc()
                gpu.location.type = location.CU_MEM_LOCATION_TYPE_DEVICE
                gpu.location.id = device
                cpu.location.type = location.CU_MEM_LOCATION_TYPE_HOST
                rw = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                gpu.flags = cpu.flags = rw
                _unwrap_cu(cu.cuMemSetAccess(pointer, size, [gpu, cpu], 2))
                tensor = torch.frombuffer(owner, dtype=torch.uint8)[:num_bytes]
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            # Also fences peer imports before the leader closes its fd.
            errors: list[str | None] = [None] * group.world_size
            torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
            failures = "; ".join(
                f"EDP rank {rank}: {error}"
                for rank, error in enumerate(errors)
                if error is not None
            )
            if failures:
                # Dropping the owner runs the finalizer, which unmaps.
                del tensor, owner
                raise RuntimeError(
                    "Engram shared-memory initialization failed: " + failures
                )
            assert tensor is not None
            return tensor

    @staticmethod
    def _unmap(pointer: cu.CUdeviceptr, size: int) -> None:
        cu.cuMemUnmap(pointer, size)
        cu.cuMemAddressFree(pointer, size)

    def _device_views(self) -> tuple[torch.Tensor, torch.Tensor]:
        # The UVA helper would tag the views with the creator's device.
        weight_bytes = self.weight.nbytes
        array = {
            "shape": (weight_bytes + self.weight_scale_inv.nbytes,),
            "typestr": "|u1",
            "data": (self.weight.data_ptr(), False),
            "version": 3,
        }
        storage = torch.as_tensor(
            SimpleNamespace(__cuda_array_interface__=array, owner=self.weight),
            device=torch.device("cuda", torch.accelerator.current_device_index()),
        )
        return (
            storage[:weight_bytes].view(self.weight.dtype).view(self.weight.shape),
            storage[weight_bytes:].view(self.weight_scale_inv.shape),
        )


class ParallelEngramEmbedding(BaseParallelEngramEmbedding):
    """Engram embedding whose DP-shared host tables live in cuMem."""

    _shared_storage_cls = CuMemEngramStorage


class Engram(BaseEngram):
    """Engram whose big host lookups run on the main stream."""

    _embedding_cls = ParallelEngramEmbedding

    def _lookup_inline(self, num_tokens: int) -> bool:
        # Big lookups stall persistent main-stream kernels anyway: run them inline.
        # All DP ranks must agree, or EP collectives wait on the slowest one.
        if is_forward_context_available() and (dp := get_forward_context().dp_metadata):
            num_tokens = int(dp.num_tokens_across_dp_cpu.max())
        return num_tokens >= _INLINE_LOOKUP_MIN_TOKENS
