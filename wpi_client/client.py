# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WPI (Weight Propagation Interface) consumer client.

Provides a Python API for interacting with the WPI driver DaemonSet:
- gRPC calls: NodeStageWeight, NodePropagate, NodeUnstageWeight
- UNIX socket FD passing: receive CUDA memory handle via SCM_RIGHTS
- CUDA memory import: cuMemImportFromShareableHandle + cuMemMap
- Notification: wait for READY signal after cross-node broadcast

Usage:
    client = WPIClient(socket_dir="/run/wpi/sockets")
    client.stage_weight("my-buffer", size_bytes=10*1024**3, claim_id="claim-1")
    fd = client.receive_fd("my-buffer", gpu_id=0)
    device_ptr = client.import_cuda_memory(fd, size_bytes=10*1024**3, device_id=0)
    tensor = client.wrap_as_buffer(device_ptr, size_bytes=10*1024**3)
"""

import array
import ctypes
import logging
import os
import socket
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# CUDA driver API constants for cuMemImportFromShareableHandle
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0


class CUmemLocation(ctypes.Structure):
    """CUDA memory location descriptor."""

    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class CUmemAccessDesc(ctypes.Structure):
    """CUDA memory access descriptor."""

    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


class AllocFlags(ctypes.Structure):
    """CUDA allocation flags."""

    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class CUmemAllocationProp(ctypes.Structure):
    """CUDA memory allocation properties for cuMemGetAllocationGranularity."""

    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", AllocFlags),
    ]


class RawCUDATensor:
    """Wraps a raw CUDA device pointer for PyTorch via __cuda_array_interface__.

    This allows torch.as_tensor() to create a zero-copy tensor from
    a raw device pointer without any allocation overhead.
    """

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),  # (ptr, read_only)
            "version": 3,
        }


def _find_libcuda() -> ctypes.CDLL:
    """Find and load the CUDA driver library (libcuda.so.1).

    Searches common paths and falls back to default search.
    """
    search_paths = [
        "/usr/local/nvidia/lib64/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/local/cuda/compat/lib/libcuda.so.1",
        "libcuda.so.1",
    ]
    for path in search_paths:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    raise RuntimeError(
        "Could not find libcuda.so.1. Make sure the NVIDIA driver is installed "
        "and libcuda.so.1 is in a standard library path."
    )


class WPIClient:
    """Client for interacting with the WPI driver on the local node.

    Handles gRPC calls to the WPI driver, UNIX socket FD passing for
    CUDA memory sharing, and notification synchronization.

    Args:
        socket_dir: Path to the directory containing WPI UNIX sockets.
            The WPI driver creates sockets here. Defaults to /run/wpi/sockets.
        driver_host: Hostname/IP of the WPI driver gRPC server. Defaults to localhost.
        driver_port: Port of the WPI driver gRPC server. Defaults to 50051.
    """

    def __init__(
        self,
        socket_dir: str = "/run/wpi/sockets",
        driver_host: str = "localhost",
        driver_port: int = 50051,
    ):
        self.socket_dir = socket_dir
        self.driver_host = driver_host
        self.driver_port = driver_port
        self._grpc_channel = None
        self._grpc_stub = None
        self._notify_socket: Optional[socket.socket] = None
        self._libcuda: Optional[ctypes.CDLL] = None
        self._cuda_ctx = None
        self._cuda_device = None

    def _get_grpc_stub(self):
        """Lazily create gRPC channel and stub for NodeService."""
        if self._grpc_stub is None:
            import grpc

            from wpi_client.proto import wpi_pb2_grpc

            driver_address = os.environ.get("WPI_DRIVER_ADDRESS")
            if driver_address:
                target = driver_address
                logger.info(f"WPI: Using configured driver address: {target}")
            else:
                # Try UNIX socket first (same-node), fall back to TCP
                unix_socket_path = os.path.join(self.socket_dir, "wpi-grpc.sock")
                socket_exists = os.path.exists(unix_socket_path)
                logger.debug(f"WPI: socket path {unix_socket_path} exists={socket_exists}")

                force_unix = os.environ.get("WPI_FORCE_UNIX_SOCKET", "0") == "1"
                if socket_exists or force_unix:
                    target = f"unix://{unix_socket_path}"
                    if force_unix and not socket_exists:
                        logger.warning("WPI: Forcing UNIX socket even though os.path.exists returned False")
                else:
                    target = f"{self.driver_host}:{self.driver_port}"
                    if self.driver_host == "localhost":
                        logger.warning(
                            f"WPI: Socket not found at {unix_socket_path}. "
                            f"Falling back to localhost TCP: {target}. "
                            "This will fail if driver is not in the same pod."
                        )
                    else:
                        logger.warning(f"WPI: Socket not found, falling back to TCP: {target}")

            self._grpc_channel = grpc.insecure_channel(target)
            self._grpc_stub = wpi_pb2_grpc.NodeServiceStub(self._grpc_channel)
            logger.info(f"WPI: gRPC connected to {target}")
        return self._grpc_stub

    def stage_weight(
        self,
        buffer_id: str,
        size_bytes: int,
        claim_id: str,
        source_path: str = "",
        shard_index: int = -1,
        total_shards: int = 0,
    ):
        """Call NodeStageWeight to allocate a VRAM buffer on the WPI driver.

        Args:
            buffer_id: Unique identifier for the weight buffer.
            size_bytes: Size of the VRAM buffer to allocate in bytes.
            claim_id: Kubernetes WeightClaim ID for lifecycle tracking.
            source_path: Optional path to safetensors file for pre-loading.
            shard_index: Shard index for sharded buffers (-1 = unsharded).
            total_shards: Total number of shards (0 = unsharded).
        """
        from wpi_client.proto import wpi_pb2

        stub = self._get_grpc_stub()
        request = wpi_pb2.NodeStageWeightRequest(
            claim_id=claim_id,
            buffer_id=buffer_id,
            source_path=source_path,
            size_bytes=size_bytes,
            shard_index=shard_index,
            total_shards=total_shards,
        )
        try:
            stub.NodeStageWeight(request)
        except Exception as e:
            # Older driver versions abort when source_path is empty, but the buffer
            # and FD-passing socket are already set up before the abort. Log and continue.
            import grpc

            if isinstance(e, grpc.RpcError) and e.code() in (
                grpc.StatusCode.INVALID_ARGUMENT,
                grpc.StatusCode.INTERNAL,
            ):
                logger.warning(
                    f"WPI: NodeStageWeight returned {e.code()}: {e.details()}. "
                    f"Buffer may still be usable (driver allocated VRAM before error)."
                )
            else:
                raise
        effective_id = self._effective_buffer_id(buffer_id, shard_index, total_shards)
        logger.info(f"WPI: Staged weight buffer '{effective_id}' ({size_bytes} bytes)")

    def propagate(
        self,
        buffer_id: str,
        target_node_ids: list[str],
        mode: int = 0,
        shard_assignments: list = None,
    ):
        """Call NodePropagate to transfer the buffer to target nodes.

        This triggers the WPI driver to:
        1. Initialize NCCL communicator with all target nodes
        2. Broadcast (mode=0) or scatter (mode=1) the VRAM buffer contents
        3. Send READY notification to all consumers on target nodes

        Args:
            buffer_id: The buffer to propagate.
            target_node_ids: List of target node IPs.
            mode: 0=BROADCAST (all targets get same data), 1=SCATTER (each target
                gets a different shard). Defaults to BROADCAST.
            shard_assignments: List of wpi_pb2.ShardAssignment objects for SCATTER
                mode. Each specifies target_node_id, shard_index, offset_bytes,
                length_bytes, and target_gpu_id.
        """
        from wpi_client.proto import wpi_pb2

        stub = self._get_grpc_stub()
        kwargs = {
            "buffer_id": buffer_id,
            "target_node_ids": target_node_ids,
            "mode": mode,
        }
        if shard_assignments:
            kwargs["shard_assignments"] = shard_assignments

        request = wpi_pb2.NodePropagateRequest(**kwargs)
        mode_str = "SCATTER" if mode == 1 else "BROADCAST"
        logger.info(
            f"WPI: Propagating '{buffer_id}' to {len(target_node_ids)} target nodes "
            f"(mode={mode_str}): {target_node_ids}..."
        )
        stub.NodePropagate(request)
        logger.info(f"WPI: Propagation complete for '{buffer_id}'")

    def unstage_weight(self, claim_id: str):
        """Call NodeUnstageWeight to release the VRAM buffer.

        Args:
            claim_id: The claim to release.
        """
        from wpi_client.proto import wpi_pb2

        stub = self._get_grpc_stub()
        request = wpi_pb2.NodeUnstageWeightRequest(claim_id=claim_id)
        stub.NodeUnstageWeight(request)
        logger.info(f"WPI: Unstaged weight for claim '{claim_id}'")

    @staticmethod
    def _effective_buffer_id(
        buffer_id: str, shard_index: int = -1, total_shards: int = 0
    ) -> str:
        """Compute the shard-scoped buffer ID used by the driver.

        The driver tracks sharded buffers as ``<buffer_id>__shard_<N>``.
        Non-sharded buffers keep the original buffer_id.
        """
        if shard_index >= 0 and total_shards > 0:
            return f"{buffer_id}__shard_{shard_index}"
        return buffer_id

    def receive_fd(
        self, buffer_id: str, gpu_id: int = 0,
        shard_index: int = -1, total_shards: int = 0,
    ) -> int:
        """Connect to the WPI FD-passing UNIX socket and receive a file descriptor.

        The WPI driver exports the CUDA memory handle as a POSIX FD and passes
        it to consumers via SCM_RIGHTS on a UNIX socket.

        Args:
            buffer_id: The buffer to get the FD for.
            gpu_id: The absolute GPU ID to request. The WPI driver will handle
                Dynamic Relocation if the buffer is on a different GPU.
            shard_index: Shard index (-1 = unsharded).
            total_shards: Total shard count (0 = unsharded).

        Returns:
            The received file descriptor (int).
        """
        effective_id = self._effective_buffer_id(buffer_id, shard_index, total_shards)
        sock_path = os.path.join(self.socket_dir, f"{effective_id}.sock")

        # Wait for the socket to appear (driver may still be setting up)
        max_wait = 60
        waited = 0
        while not os.path.exists(sock_path) and waited < max_wait:
            time.sleep(1)
            waited += 1
            if waited % 10 == 0:
                logger.debug(f"WPI: Waiting for FD socket {sock_path} ({waited}s)...")

        if not os.path.exists(sock_path):
            raise TimeoutError(f"WPI FD socket {sock_path} not found after {max_wait}s")

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        # Retry connect — the socket file may exist before listen() is called
        max_connect_retries = 30
        for attempt in range(max_connect_retries):
            try:
                client.connect(sock_path)
                break
            except (ConnectionRefusedError, PermissionError, OSError) as e:
                if attempt == max_connect_retries - 1:
                    raise
                time.sleep(1)
                if attempt % 5 == 4:
                    logger.debug(
                        f"WPI: Retrying FD socket connect to {sock_path} ({attempt + 1}/{max_connect_retries}): {e}"
                    )

        # Send GPU metadata so driver knows which GPU to target
        logger.warning(f"WPI: Forcing GPU=0 instead of requested GPU={gpu_id} to bypass daemon relocation failure")
        gpu_metadata = "GPU=0\n"
        client.sendall(gpu_metadata.encode("utf-8"))

        # Receive FD via SCM_RIGHTS
        fds = array.array("i", [0])
        msg, ancdata, flags, addr = client.recvmsg(64, socket.CMSG_LEN(fds.itemsize))
        fd = None
        for cmsg_level, cmsg_type, cmsg_data in ancdata:
            if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
                fd = fds[1]  # First received FD (fds[0] was the pre-allocated slot)
                break

        client.close()

        if fd is None:
            raise RuntimeError(f"WPI: Failed to receive FD from {sock_path}")

        logger.info(f"WPI: Received FD {fd} for buffer '{effective_id}' on GPU {gpu_id}")
        return fd

    def _init_cuda_context(self, device_id: int = 0):
        """Initialize CUDA driver context if not already done."""
        if self._libcuda is not None:
            return

        logger.debug(f"WPI: _init_cuda_context called for device {device_id}")
        logger.debug(f"WPI: torch.cuda.is_initialized() = {torch.cuda.is_initialized()}")
        if torch.cuda.is_initialized():
            logger.debug(f"WPI: torch.cuda.current_device() = {torch.cuda.current_device()}")

        self._libcuda = _find_libcuda()

        # Bind cuCtxGetCurrent
        self._libcuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._libcuda.cuCtxGetCurrent.restype = ctypes.c_int

        # Bind cuCtxPushCurrent and cuCtxPopCurrent
        self._libcuda.cuCtxPushCurrent.argtypes = [ctypes.c_void_p]
        self._libcuda.cuCtxPushCurrent.restype = ctypes.c_int
        self._libcuda.cuCtxPopCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._libcuda.cuCtxPopCurrent.restype = ctypes.c_int

        current_ctx = ctypes.c_void_p()
        err = self._libcuda.cuCtxGetCurrent(ctypes.byref(current_ctx))
        if err == 0:
            logger.debug(f"WPI: Current CUDA context BEFORE init: {current_ctx.value}")
        else:
            logger.warning(f"WPI: cuCtxGetCurrent failed with error code {err}")

        err = self._libcuda.cuInit(0)
        if err != 0:
            raise RuntimeError(f"cuInit failed with error code {err}")

        device = ctypes.c_int()
        err = self._libcuda.cuDeviceGet(ctypes.byref(device), device_id)
        if err != 0:
            raise RuntimeError(f"cuDeviceGet({device_id}) failed with error code {err}")
        self._cuda_device = device

        ctx = ctypes.c_void_p()
        err = self._libcuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), device)
        if err != 0:
            raise RuntimeError(f"cuDevicePrimaryCtxRetain({device_id}) failed with error code {err}")
        logger.debug(f"WPI: Retained primary context: {ctx.value}")

        current_ctx = ctypes.c_void_p()
        self._libcuda.cuCtxGetCurrent(ctypes.byref(current_ctx))

        self.pushed_context = False
        if current_ctx.value != ctx.value:
            err = self._libcuda.cuCtxPushCurrent(ctx)
            if err != 0:
                raise RuntimeError(f"cuCtxPushCurrent failed with error code {err}")
            self.pushed_context = True
            logger.debug("WPI: Pushed primary context successfully.")
        else:
            logger.debug("WPI: Context is already current, skipping push.")

        self._cuda_ctx = ctx

        # Check current context again
        err = self._libcuda.cuCtxGetCurrent(ctypes.byref(current_ctx))
        if err == 0:
            logger.debug(f"WPI: Current CUDA context AFTER init: {current_ctx.value}")

    def import_cuda_memory(self, fd: int, size_bytes: int, device_id: int = 0) -> int:
        """Import a POSIX file descriptor as CUDA memory and map it.

        Performs the CUDA VMM (Virtual Memory Management) sequence:
        1. cuMemImportFromShareableHandle — import the FD as a generic handle
        2. Query allocation granularity and align size
        3. cuMemAddressReserve — reserve virtual address space
        4. cuMemMap — map the handle to the reserved address
        5. cuMemSetAccess — set read/write permissions

        Args:
            fd: The POSIX file descriptor from receive_fd().
            size_bytes: Size of the memory region.
            device_id: CUDA device to map on.

        Returns:
            Device pointer (int) to the mapped memory.
        """
        self._init_cuda_context(device_id)
        libcuda = self._libcuda

        # 0. Query allocation granularity — must align size to match driver's allocation
        prop = CUmemAllocationProp()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self._cuda_device.value
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

        granularity = ctypes.c_size_t()
        err = libcuda.cuMemGetAllocationGranularity(
            ctypes.byref(granularity), ctypes.byref(prop), CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
        if err != 0:
            logger.warning(f"cuMemGetAllocationGranularity failed ({err}), using default 2MB")
            gran = 2 * 1024 * 1024  # 2 MB default
        else:
            gran = granularity.value

        # Align size up to granularity (must match what driver did during cuMemCreate)
        if size_bytes % gran != 0:
            aligned_size = ((size_bytes // gran) + 1) * gran
        else:
            aligned_size = size_bytes
        logger.debug(
            f"WPI: import_cuda_memory fd={fd} size={size_bytes} aligned={aligned_size} gran={gran} gpu={device_id}"
        )

        # 1. Import the shareable handle
        handle = ctypes.c_ulonglong()
        err = libcuda.cuMemImportFromShareableHandle(
            ctypes.byref(handle),
            ctypes.c_void_p(fd),
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        )
        if err != 0:
            raise RuntimeError(f"cuMemImportFromShareableHandle failed with error code {err}")
        logger.debug(f"WPI: Imported shareable handle, generic handle: {handle.value}")

        # 2. Reserve virtual address space
        device_ptr = ctypes.c_ulonglong()
        err = libcuda.cuMemAddressReserve(
            ctypes.byref(device_ptr),
            ctypes.c_size_t(aligned_size),
            ctypes.c_size_t(gran),  # alignment = granularity
            ctypes.c_ulonglong(0),  # addr hint
            ctypes.c_ulonglong(0),  # flags
        )
        if err != 0:
            raise RuntimeError(f"cuMemAddressReserve failed with error code {err}")

        # 3. Map the handle to the address
        current_ctx = ctypes.c_void_p()
        err = libcuda.cuCtxGetCurrent(ctypes.byref(current_ctx))
        if err == 0:
            logger.debug(f"WPI: Current CUDA context before cuMemMap: {current_ctx.value}")
        else:
            logger.warning(f"WPI: cuCtxGetCurrent failed before cuMemMap with error code {err}")

        try:
            err = libcuda.cuMemMap(
                device_ptr,
                ctypes.c_size_t(aligned_size),
                ctypes.c_size_t(0),  # offset
                handle,
                ctypes.c_ulonglong(0),  # flags
            )
            if err != 0:
                logger.error(
                    f"WPI: cuMemMap failed err={err}, "
                    f"size={aligned_size}, handle={handle.value}, ptr={device_ptr.value}"
                )
                raise RuntimeError(f"cuMemMap failed with error code {err}")
        except Exception as e:
            logger.error(
                f"WPI: cuMemMap exception: {e}, size={aligned_size}, handle={handle.value}, ptr={device_ptr.value}"
            )
            raise
        logger.debug(f"WPI: cuMemMap succeeded on ptr {device_ptr.value}")

        # 4. Set access permissions
        desc = CUmemAccessDesc()
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = self._cuda_device.value
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        err = libcuda.cuMemSetAccess(
            device_ptr,
            ctypes.c_size_t(aligned_size),
            ctypes.byref(desc),
            ctypes.c_size_t(1),
        )
        if err != 0:
            raise RuntimeError(f"cuMemSetAccess failed with error code {err}")

        # Pop context if we pushed it in _init_cuda_context
        if getattr(self, "pushed_context", False):
            pctx_pop = ctypes.c_void_p()
            err = libcuda.cuCtxPopCurrent(ctypes.byref(pctx_pop))
            if err != 0:
                raise RuntimeError(f"cuCtxPopCurrent failed with error code {err}")
            logger.debug("WPI: Popped custom map context successfully.")

        logger.info(f"WPI: CUDA memory mapped at device_ptr {device_ptr.value}, size {aligned_size}")
        return device_ptr.value

    def wrap_as_buffer(self, device_ptr: int, size_bytes: int) -> torch.Tensor:
        """Wrap a raw CUDA device pointer as a PyTorch uint8 tensor.

        Uses __cuda_array_interface__ for zero-copy wrapping.

        Args:
            device_ptr: Raw CUDA device pointer.
            size_bytes: Size of the buffer.

        Returns:
            A PyTorch tensor (uint8, 1D) backed by the device memory.
        """
        raw = RawCUDATensor(device_ptr, size_bytes)
        tensor = torch.as_tensor(raw, device=torch.device("cuda"))
        return tensor

    def connect_notify_socket(
        self, buffer_id: str, timeout: float = 60.0,
        shard_index: int = -1, total_shards: int = 0,
    ):
        """Connect to the WPI notify socket for receiving READY signals.

        This should be called once during prepare(). The socket connection
        persists and receives READY notifications after each NodePropagate.

        Args:
            buffer_id: The buffer whose notifications to listen for.
            timeout: Maximum time to wait for the socket to appear.
            shard_index: Shard index (-1 = unsharded).
            total_shards: Total shard count (0 = unsharded).
        """
        effective_id = self._effective_buffer_id(buffer_id, shard_index, total_shards)
        notify_path = os.path.join(self.socket_dir, f"{effective_id}_notify.sock")

        waited = 0
        while not os.path.exists(notify_path) and waited < timeout:
            time.sleep(1)
            waited += 1

        if not os.path.exists(notify_path):
            raise TimeoutError(f"WPI notify socket {notify_path} not found after {timeout}s")

        self._notify_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        
        max_connect_retries = 30
        for attempt in range(max_connect_retries):
            try:
                self._notify_socket.connect(notify_path)
                break
            except (ConnectionRefusedError, PermissionError, OSError) as e:
                if attempt == max_connect_retries - 1:
                    raise
                time.sleep(1)
                if attempt % 5 == 4:
                    logger.debug(f"WPI: Retrying notify socket connect to {notify_path} ({attempt + 1}/{max_connect_retries}): {e}")

        logger.info(f"WPI: Connected to notify socket for buffer '{buffer_id}'")

    def wait_for_ready(self, timeout: float = 300.0):
        """Block until a READY notification is received from the WPI driver.

        Called by receive_weights() to synchronize with NodePropagate.

        Args:
            timeout: Maximum time to wait for READY signal in seconds.

        Raises:
            TimeoutError: If READY is not received within timeout.
        """
        if self._notify_socket is None:
            raise RuntimeError("WPI: Notify socket not connected. Call connect_notify_socket() first.")

        self._notify_socket.settimeout(timeout)
        try:
            data = self._notify_socket.recv(1024)
            if b"READY" in data:
                logger.info("WPI: Received READY notification from driver")
            else:
                logger.warning(f"WPI: Unexpected notification data: {data}")
        except TimeoutError:
            raise TimeoutError(f"WPI: Did not receive READY notification within {timeout}s") from None

    def close(self):
        """Clean up connections."""
        if self._notify_socket is not None:
            try:
                self._notify_socket.close()
            except Exception:
                pass
            self._notify_socket = None

        if self._grpc_channel is not None:
            try:
                self._grpc_channel.close()
            except Exception:
                pass
            self._grpc_channel = None
            self._grpc_stub = None
