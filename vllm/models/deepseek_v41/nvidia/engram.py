# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA Engram: cuMem-shared host tables and inline big host lookups."""

import ctypes
import os
import socket
import uuid
import weakref
from contextlib import ExitStack
from types import SimpleNamespace

import torch
from cuda.bindings import driver as cu

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.deepseek_v41.common.engram import DPSharedEngramStorage
from vllm.models.deepseek_v41.common.engram import Engram as BaseEngram
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

_POSIX_FD = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
_FD_EXCHANGE_TIMEOUT_S = 30
_INLINE_LOOKUP_MIN_TOKENS = 4096


def _unwrap_cu(result):
    """Unwrap a cuda-python ``(error, value)`` result."""
    err, *value = result
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err.name}")
    return value[0] if value else None


def _host_numa_id(device: int) -> int:
    numa = cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID
    return max(0, _unwrap_cu(cu.cuDeviceGetAttribute(numa, device)))


def _host_prop(device: int) -> cu.CUmemAllocationProp:
    """A shareable pinned allocation on the host NUMA node of `device`."""
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    prop.location.id = _host_numa_id(device)
    prop.requestedHandleTypes = _POSIX_FD
    return prop


def _map_host(pointer: cu.CUdeviceptr, size: int, handle, device: int) -> None:
    """Map `handle` at `pointer`, readable and writable by `device` and the CPU."""
    _unwrap_cu(cu.cuMemMap(pointer, size, 0, handle, 0))
    gpu, cpu = cu.CUmemAccessDesc(), cu.CUmemAccessDesc()
    gpu.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    gpu.location.id = device
    cpu.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST
    gpu.flags = cpu.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _unwrap_cu(cu.cuMemSetAccess(pointer, size, [gpu, cpu], 2))


class CuMemEngramStorage(DPSharedEngramStorage):
    """DP-shared host weights in cuMem, which GPUs map with 2M pages
    (/dev/shm only gets 64K ones)."""

    @staticmethod
    def check_available(num_bytes: int) -> None:
        device = torch.accelerator.current_device_index()
        attr = cu.CUdevice_attribute
        vmm = attr.CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
        if cu.cuDeviceGetAttribute(vmm, device) != (cu.CUresult.CUDA_SUCCESS, 1):
            raise RuntimeError("the driver cannot allocate host NUMA memory with cuMem")
        numa = _host_numa_id(device)
        with open(f"/sys/devices/system/node/node{numa}/meminfo") as meminfo:
            numa_bytes = int(meminfo.read().split("MemTotal:")[1].split()[0]) * 1024
        if numa_bytes < num_bytes:
            gib = num_bytes / 1024**3
            raise RuntimeError(f"needs {gib:.1f} GiB on host NUMA node {numa}")
        # Map one granule as _allocate would, so an unsupported step fails here.
        with ExitStack() as stack:
            prop = _host_prop(device)
            size = _unwrap_cu(cu.cuMemGetAllocationGranularity(prop, 0))
            handle = _unwrap_cu(cu.cuMemCreate(size, prop, 0))
            stack.callback(cu.cuMemRelease, handle)
            os.close(_unwrap_cu(cu.cuMemExportToShareableHandle(handle, _POSIX_FD, 0)))
            pointer = _unwrap_cu(cu.cuMemAddressReserve(size, 0, 0, 0))
            stack.callback(CuMemEngramStorage._unmap, pointer, size)
            _map_host(pointer, size, handle, device)

    def _allocate(self, num_bytes: int) -> torch.Tensor:
        """Map one host allocation into every rank; the leader sends its fd."""
        group = self.group
        device = torch.accelerator.current_device_index()
        with ExitStack() as stack:
            handle = fd = address = size = error = None
            if group.rank_in_group == 0:
                try:
                    prop = _host_prop(device)
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
                _map_host(pointer, size, handle, device)
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


class Engram(BaseEngram):
    """Engram whose big host lookups run on the main stream."""

    def _lookup_inline(self, num_tokens: int) -> bool:
        # Big lookups stall persistent main-stream kernels anyway: run them inline.
        # All DP ranks must agree, or EP collectives wait on the slowest one.
        if is_forward_context_available() and (dp := get_forward_context().dp_metadata):
            num_tokens = int(dp.num_tokens_across_dp_cpu.max())
        return num_tokens >= _INLINE_LOOKUP_MIN_TOKENS
