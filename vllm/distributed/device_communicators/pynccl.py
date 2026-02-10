# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PyNccl Communicator - NCCL bindings for vLLM.

This module provides the PyNcclCommunicator class for GPU collective operations.
By default, it uses nccl4py (official NVIDIA Python bindings). Set environment
variable VLLM_DISABLE_NCCL4PY=1 to use the legacy ctypes-based implementation.
"""

# ===================== import region =====================
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

import vllm.envs as envs
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

_NCCL_SYMM_OPS_REGISTERED = False


def register_nccl_symmetric_ops(pynccl_comm):
    from vllm.distributed.device_communicators.pynccl_allocator import (
        nccl_symm_mem_context,
    )
    from vllm.utils.torch_utils import direct_register_custom_op

    global _NCCL_SYMM_OPS_REGISTERED
    if _NCCL_SYMM_OPS_REGISTERED:
        return
    _NCCL_SYMM_OPS_REGISTERED = True

    def all_reduce_symmetric_with_copy_impl(input_tensor: torch.Tensor) -> torch.Tensor:
        with nccl_symm_mem_context(pynccl_comm):
            symm_input = torch.empty_like(input_tensor)
            symm_output = torch.empty_like(input_tensor)
        symm_input.copy_(input_tensor)
        symm_output = pynccl_comm.all_reduce(symm_input, symm_output)
        return symm_output

    def all_reduce_symmetric_with_copy_fake(input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(input_tensor)

    direct_register_custom_op(
        op_name="all_reduce_symmetric_with_copy",
        op_func=all_reduce_symmetric_with_copy_impl,
        fake_impl=all_reduce_symmetric_with_copy_fake,
    )


# ===================== nccl4py implementation =====================


def _torch_reduce_op_to_nccl4py(op: ReduceOp):
    """Convert torch.distributed.ReduceOp to nccl4py reduction operator."""
    import nccl.core as nccl

    mapping = {
        ReduceOp.SUM: nccl.SUM,
        ReduceOp.PRODUCT: nccl.PROD,
        ReduceOp.MAX: nccl.MAX,
        ReduceOp.MIN: nccl.MIN,
        ReduceOp.AVG: nccl.AVG,
    }
    return mapping[op]


def _get_stream_for_nccl4py(stream):
    """Convert torch stream to format acceptable by nccl4py."""
    if stream is None:
        return None
    # nccl4py accepts the stream directly or as an integer pointer
    return stream.cuda_stream


class PyNcclCommunicator:
    """
    NCCL Communicator for GPU collective operations.

    This implementation uses nccl4py (official NVIDIA Python bindings) by default.
    Set VLLM_DISABLE_NCCL4PY=1 to use the legacy ctypes-based implementation.
    """

    def __init__(
        self,
        group: ProcessGroup | StatelessProcessGroup,
        device: int | str | torch.device,
        library_path: str | None = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyNcclCommunicator to. If None,
                it will be bound to f"cuda:{local_rank}".
            library_path: the path to the NCCL library. If None, it will
                use the default library path. (Ignored when using nccl4py)
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert dist.get_backend(group) != dist.Backend.NCCL, (
                "PyNcclCommunicator should be attached to a non-NCCL group."
            )
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1 or envs.VLLM_DISABLE_PYNCCL:
            self.available = False
            self.disabled = True
            return

        try:
            import nccl.core as nccl

            self._nccl = nccl
        except ImportError:
            logger.warning(
                "nccl4py is not available. NCCL operations will be disabled. "
                "Install with: pip install nccl4py[cu12]"
            )
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        # Get version information
        version_info = nccl.get_version()
        self.nccl_version = int(
            version_info.nccl_version.major * 10000
            + version_info.nccl_version.minor * 100
            + version_info.nccl_version.micro
        )

        # Generate unique ID on rank 0
        if self.rank == 0:
            self.unique_id = nccl.get_unique_id()
            logger.info_once(
                "vLLM is using nccl==%s (via nccl4py)",
                str(version_info.nccl_version),
                scope="local",
            )
        else:
            self.unique_id = nccl.get_unique_id(empty=True)

        # Broadcast unique ID to all ranks
        if not isinstance(group, StatelessProcessGroup):
            ranks = dist.get_process_group_ranks(group)
            # Convert unique ID to bytes and broadcast
            unique_id_bytes = list(self.unique_id.as_bytes)
            tensor = torch.ByteTensor(unique_id_bytes)
            dist.broadcast(tensor, src=ranks[0], group=group)
            if self.rank != 0:
                self.unique_id = nccl.UniqueId.from_bytes(bytes(tensor.tolist()))
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)

        # Parse device
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        # Initialize NCCL communicator on the specified device
        with torch.cuda.device(device):
            self.comm = nccl.Communicator.init(
                nranks=self.world_size,
                rank=self.rank,
                unique_id=self.unique_id,
            )

            # Warmup with a small all_reduce
            stream = current_stream()
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            stream.synchronize()
            del data

        # Track registered windows for cleanup
        self._registered_windows: list = []

    def all_reduce(
        self,
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor = None,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ) -> torch.Tensor:
        if self.disabled:
            return None

        assert in_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}"
        )

        if out_tensor is None:
            out_tensor = torch.empty_like(in_tensor)

        if stream is None:
            stream = current_stream()

        nccl_op = _torch_reduce_op_to_nccl4py(op)
        self.comm.allreduce(
            in_tensor,
            out_tensor,
            nccl_op,
            stream=_get_stream_for_nccl4py(stream),
        )
        return out_tensor

    def all_gather(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, stream=None
    ):
        if self.disabled:
            return

        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )

        if stream is None:
            stream = current_stream()

        self.comm.allgather(
            input_tensor,
            output_tensor,
            stream=_get_stream_for_nccl4py(stream),
        )

    def all_gatherv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        stream=None,
    ):
        if self.disabled:
            return

        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )

        if stream is None:
            stream = current_stream()

        assert output_tensor.shape[0] == sum(sizes)

        # nccl4py doesn't have native all_gatherv, implement using group + broadcast
        split_offset = 0
        self._nccl.group_start()
        for root, split_size in enumerate(sizes):
            dst_slice = output_tensor[split_offset : split_offset + split_size]
            self.comm.broadcast(
                input_tensor,
                dst_slice,
                root,
                stream=_get_stream_for_nccl4py(stream),
            )
            split_offset += split_size
        self._nccl.group_end()

    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        if self.disabled:
            return

        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )

        if stream is None:
            stream = current_stream()

        nccl_op = _torch_reduce_op_to_nccl4py(op)
        self.comm.reduce_scatter(
            input_tensor,
            output_tensor,
            nccl_op,
            stream=_get_stream_for_nccl4py(stream),
        )

    def reduce_scatterv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        if self.disabled:
            return

        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )

        if stream is None:
            stream = current_stream()

        nccl_op = _torch_reduce_op_to_nccl4py(op)

        # nccl4py doesn't have native reduce_scatterv, implement using group + reduce
        split_offset = 0
        self._nccl.group_start()
        for root, split_size in enumerate(sizes):
            chunk = input_tensor[split_offset : split_offset + split_size, ...]
            self.comm.reduce(
                chunk,
                output_tensor,
                nccl_op,
                root=root,
                stream=_get_stream_for_nccl4py(stream),
            )
            split_offset += split_size
        self._nccl.group_end()

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        if self.disabled:
            return

        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )

        if stream is None:
            stream = current_stream()

        self.comm.send(
            tensor,
            dst,
            stream=_get_stream_for_nccl4py(stream),
        )

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return

        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )

        if stream is None:
            stream = current_stream()

        self.comm.recv(
            tensor,
            src,
            stream=_get_stream_for_nccl4py(stream),
        )

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return

        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )

        if stream is None:
            stream = current_stream()

        # nccl4py broadcast: sendbuf only used on root, recvbuf on all ranks
        # For in-place broadcast, pass same tensor for both
        self.comm.broadcast(
            tensor,  # sendbuf (only used on root)
            tensor,  # recvbuf (receives data on all ranks)
            src,
            stream=_get_stream_for_nccl4py(stream),
        )

    def group_start(self):
        self._nccl.group_start()

    def group_end(self):
        self._nccl.group_end()

    def register_comm_window(self, tensor: torch.Tensor):
        """Register a tensor for window-based communication (e.g., symmetric memory)."""
        handle = self.comm.register_window(tensor)
        if handle is not None:
            self._registered_windows.append(handle)
        return handle

    def register_comm_window_raw(self, ptr: int, size: int):
        """Register a raw memory region for window-based communication."""
        # Use nccl4py's low-level bindings for raw pointer registration
        from nccl import bindings as nccl_bindings

        # winFlags=1 for CollSymmetric (matches WindowFlag.CollSymmetric)
        handle = nccl_bindings.comm_window_register(self.comm.ptr, ptr, size, 1)
        if handle != 0:
            # Store as tuple (comm_ptr, window_handle) for deregistration
            self._registered_windows.append((self.comm.ptr, handle))
        return handle

    def deregister_comm_window(self, window):
        """Deregister a previously registered window."""
        if window is None:
            return

        from nccl import bindings as nccl_bindings

        # Check if it's a handle from register_window or register_comm_window_raw
        if hasattr(window, "close"):
            # nccl4py RegisteredWindowHandle
            window.close()
            if window in self._registered_windows:
                self._registered_windows.remove(window)
        elif isinstance(window, tuple):
            # Raw registration (comm_ptr, window_handle)
            comm_ptr, win_handle = window
            nccl_bindings.comm_window_deregister(comm_ptr, win_handle)
            if window in self._registered_windows:
                self._registered_windows.remove(window)
        else:
            # Assume it's a raw window handle from register_comm_window_raw
            nccl_bindings.comm_window_deregister(self.comm.ptr, window)

    def __del__(self):
        """Cleanup registered windows to prevent resource leaks."""
        # Use a copy of the list to safely modify it while iterating
        for window in list(self._registered_windows):
            try:
                self.deregister_comm_window(window)
            except Exception:
                # Log errors during cleanup, but don't raise exceptions
                # from __del__ as it can cause issues during interpreter shutdown
                logger.warning(
                    "Error deregistering NCCL window during cleanup: %s",
                    window,
                    exc_info=True,
                )


# ===================== Legacy implementation fallback =====================

if envs.VLLM_DISABLE_NCCL4PY:
    # Import the legacy implementation and replace PyNcclCommunicator
    logger.info("Using legacy ctypes-based NCCL bindings (VLLM_DISABLE_NCCL4PY=1)")

    from vllm.distributed.device_communicators.pynccl_wrapper_legacy import (
        NCCLLibrary,
        buffer_type,
        cudaStream_t,
        ncclComm_t,
        ncclDataTypeEnum,
        ncclRedOpTypeEnum,
        ncclUniqueId,
    )

    class PyNcclCommunicatorLegacy:
        """Legacy PyNcclCommunicator using ctypes-based NCCL bindings."""

        def __init__(
            self,
            group: ProcessGroup | StatelessProcessGroup,
            device: int | str | torch.device,
            library_path: str | None = None,
        ):
            if not isinstance(group, StatelessProcessGroup):
                assert dist.is_initialized()
                assert dist.get_backend(group) != dist.Backend.NCCL, (
                    "PyNcclCommunicator should be attached to a non-NCCL group."
                )
                self.rank = dist.get_rank(group)
                self.world_size = dist.get_world_size(group)
            else:
                self.rank = group.rank
                self.world_size = group.world_size

            self.group = group

            if self.world_size == 1 or envs.VLLM_DISABLE_PYNCCL:
                self.available = False
                self.disabled = True
                return
            try:
                self.nccl = NCCLLibrary(library_path)
            except Exception:
                self.available = False
                self.disabled = True
                return

            self.available = True
            self.disabled = False

            self.nccl_version = self.nccl.ncclGetRawVersion()
            if self.rank == 0:
                self.unique_id = self.nccl.ncclGetUniqueId()
                logger.info_once(
                    "vLLM is using nccl==%s (legacy)",
                    self.nccl.ncclGetVersion(),
                    scope="local",
                )
            else:
                self.unique_id = ncclUniqueId()

            if not isinstance(group, StatelessProcessGroup):
                tensor = torch.ByteTensor(list(self.unique_id.internal))
                ranks = dist.get_process_group_ranks(group)
                dist.broadcast(tensor, src=ranks[0], group=group)
                byte_list = tensor.tolist()
                for i, byte in enumerate(byte_list):
                    self.unique_id.internal[i] = byte
            else:
                self.unique_id = group.broadcast_obj(self.unique_id, src=0)

            if isinstance(device, int):
                device = torch.device(f"cuda:{device}")
            elif isinstance(device, str):
                device = torch.device(device)
            assert isinstance(device, torch.device)
            self.device = device

            with torch.cuda.device(device):
                self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                    self.world_size, self.unique_id, self.rank
                )

                stream = current_stream()
                data = torch.zeros(1, device=device)
                self.all_reduce(data)
                stream.synchronize()
                del data

        def all_reduce(
            self,
            in_tensor: torch.Tensor,
            out_tensor: torch.Tensor = None,
            op: ReduceOp = ReduceOp.SUM,
            stream=None,
        ) -> torch.Tensor:
            if self.disabled:
                return None
            assert in_tensor.device == self.device

            if out_tensor is None:
                out_tensor = torch.empty_like(in_tensor)

            if stream is None:
                stream = current_stream()
            self.nccl.ncclAllReduce(
                buffer_type(in_tensor.data_ptr()),
                buffer_type(out_tensor.data_ptr()),
                in_tensor.numel(),
                ncclDataTypeEnum.from_torch(in_tensor.dtype),
                ncclRedOpTypeEnum.from_torch(op),
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )
            return out_tensor

        def all_gather(
            self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, stream=None
        ):
            if self.disabled:
                return
            assert input_tensor.device == self.device
            if stream is None:
                stream = current_stream()
            self.nccl.ncclAllGather(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(output_tensor.data_ptr()),
                input_tensor.numel(),
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )

        def all_gatherv(
            self,
            output_tensor: torch.Tensor,
            input_tensor: torch.Tensor,
            sizes: list[int],
            stream=None,
        ):
            if self.disabled:
                return
            assert input_tensor.device == self.device
            if stream is None:
                stream = current_stream()
            assert output_tensor.shape[0] == sum(sizes)
            split_offset = 0
            self.nccl.ncclGroupStart()
            for root, split_size in enumerate(sizes):
                dst_slice = output_tensor[split_offset : split_offset + split_size]
                self.nccl.ncclBroadcast(
                    buffer_type(input_tensor.data_ptr()),
                    buffer_type(dst_slice.data_ptr()),
                    dst_slice.numel(),
                    ncclDataTypeEnum.from_torch(input_tensor.dtype),
                    root,
                    self.comm,
                    cudaStream_t(stream.cuda_stream),
                )
                split_offset += split_size
            self.nccl.ncclGroupEnd()

        def reduce_scatter(
            self,
            output_tensor: torch.Tensor,
            input_tensor: torch.Tensor,
            op: ReduceOp = ReduceOp.SUM,
            stream=None,
        ):
            if self.disabled:
                return
            assert input_tensor.device == self.device
            if stream is None:
                stream = current_stream()
            self.nccl.ncclReduceScatter(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(output_tensor.data_ptr()),
                output_tensor.numel(),
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                ncclRedOpTypeEnum.from_torch(op),
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )

        def reduce_scatterv(
            self,
            output_tensor: torch.Tensor,
            input_tensor: torch.Tensor,
            sizes: list[int],
            op: ReduceOp = ReduceOp.SUM,
            stream=None,
        ):
            if self.disabled:
                return
            assert input_tensor.device == self.device
            if stream is None:
                stream = current_stream()

            split_offset = 0
            self.nccl.ncclGroupStart()
            for root, split_size in enumerate(sizes):
                chunk = input_tensor[split_offset : split_offset + split_size, ...]
                self.nccl.ncclReduce(
                    buffer_type(chunk.data_ptr()),
                    buffer_type(output_tensor.data_ptr()),
                    chunk.numel(),
                    ncclDataTypeEnum.from_torch(input_tensor.dtype),
                    ncclRedOpTypeEnum.from_torch(op),
                    root,
                    self.comm,
                    cudaStream_t(stream.cuda_stream),
                )
                split_offset += split_size
            self.nccl.ncclGroupEnd()

        def send(self, tensor: torch.Tensor, dst: int, stream=None):
            if self.disabled:
                return
            assert tensor.device == self.device
            if stream is None:
                stream = current_stream()
            self.nccl.ncclSend(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                dst,
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )

        def recv(self, tensor: torch.Tensor, src: int, stream=None):
            if self.disabled:
                return
            assert tensor.device == self.device
            if stream is None:
                stream = current_stream()
            self.nccl.ncclRecv(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                src,
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )

        def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
            if self.disabled:
                return
            assert tensor.device == self.device
            if stream is None:
                stream = current_stream()
            if src == self.rank:
                sendbuff = buffer_type(tensor.data_ptr())
                recvbuff = buffer_type(tensor.data_ptr())
            else:
                sendbuff = buffer_type()
                recvbuff = buffer_type(tensor.data_ptr())
            self.nccl.ncclBroadcast(
                sendbuff,
                recvbuff,
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                src,
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )

        def group_start(self):
            self.nccl.ncclGroupStart()

        def group_end(self):
            self.nccl.ncclGroupEnd()

        def register_comm_window(self, tensor: torch.Tensor):
            return self.nccl.ncclCommWindowRegister(
                self.comm,
                buffer_type(tensor.data_ptr()),
                tensor.numel() * tensor.element_size(),
                1,
            )

        def register_comm_window_raw(self, ptr: int, size: int):
            return self.nccl.ncclCommWindowRegister(
                self.comm, buffer_type(ptr), size, 1
            )

        def deregister_comm_window(self, window):
            return self.nccl.ncclCommWindowDeregister(self.comm, window)

    # Replace PyNcclCommunicator with legacy version
    PyNcclCommunicator = PyNcclCommunicatorLegacy  # type: ignore[misc,assignment]
