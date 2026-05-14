# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# ===================== import region =====================
import threading

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

import vllm.envs as envs
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclShrinkFlagEnum,
    ncclUniqueId,
)
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


class PyNcclCommunicator:
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
                use the default library path.
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
            self.nccl = NCCLLibrary(library_path)
        except Exception:
            # disable because of missing NCCL library
            # e.g. in a non-GPU environment
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        self.nccl_version = self.nccl.ncclGetRawVersion()
        if self.rank == 0:
            # get the unique id from NCCL
            self.unique_id = self.nccl.ncclGetUniqueId()
            logger.info_once("vLLM is using nccl==%s", self.nccl.ncclGetVersion())
        else:
            # construct an empty unique id
            self.unique_id = ncclUniqueId()

        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            # arg `src` in `broadcast` is the global rank
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
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        # nccl communicator and stream will use this device
        with torch.accelerator.device_index(device.index):
            self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                self.world_size, self.unique_id, self.rank
            )

            stream = current_stream()
            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            stream.synchronize()
            del data

    def destroy(self):
        if self.available and not self.disabled:
            # ncclCommAbort can block until all CUDA graphs that
            # captured NCCL ops on this comm are destroyed — and
            # those graphs are released later in this same main-
            # thread teardown, so a direct call here self-deadlocks.
            # Run it in a daemon thread with a timeout: the main
            # thread proceeds, the graphs drop, and the abort returns.
            def _abort():
                with torch.accelerator.device_index(self.device.index):
                    self.nccl.ncclCommAbort(self.comm)

            abort_thread = threading.Thread(target=_abort, daemon=True)
            abort_thread.start()
            abort_thread.join(timeout=5.0)
            self.available = False
            self.disabled = True

    def supports_comm_shrink_grow(self) -> bool:
        return (
            not self.disabled
            and self.available
            and self.nccl.has_comm_shrink_grow()
        )

    def _replace_comm(
        self,
        comm: ncclComm_t,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        self.comm = comm
        if rank is not None:
            self.rank = rank
        if world_size is not None:
            self.world_size = world_size
        self.available = True
        self.disabled = False

    def get_grow_unique_id(self) -> ncclUniqueId:
        """Return an NCCL grow id generated from this communicator.

        The caller must distribute this id to joining ranks with an out-of-band
        mechanism such as the existing CPU group or RPC layer.
        """
        if self.disabled:
            raise RuntimeError("Cannot get grow unique id from disabled communicator")
        return self.nccl.ncclCommGetUniqueId(self.comm)

    def shrink(
        self,
        exclude_ranks: list[int],
        shrink_flags: int = ncclShrinkFlagEnum.NCCL_SHRINK_DEFAULT,
        destroy_old: bool = False,
    ) -> bool:
        """Shrink this communicator in place.

        Returns True if this rank is retained in the new communicator. Ranks in
        ``exclude_ranks`` must not call ncclCommShrink per NCCL's contract; this
        method therefore only marks them inactive and leaves parent communicator
        cleanup to the caller's higher-level shutdown path.
        """
        if self.disabled:
            raise RuntimeError("Cannot shrink disabled communicator")
        excluded = sorted(set(int(rank) for rank in exclude_ranks))
        invalid = [
            rank for rank in excluded if rank < 0 or rank >= self.world_size
        ]
        if invalid:
            raise ValueError(
                f"exclude_ranks must be in [0, {self.world_size}), got {invalid}"
            )
        if not excluded:
            return True
        if len(excluded) == self.world_size:
            raise ValueError("Cannot shrink communicator by excluding every rank")

        torch.accelerator.synchronize()
        if self.rank in excluded:
            self.available = False
            self.disabled = True
            return False

        old_comm = self.comm
        new_comm = self.nccl.ncclCommShrink(old_comm, excluded, shrink_flags)
        new_rank = self.rank - sum(rank < self.rank for rank in excluded)
        new_world_size = self.world_size - len(excluded)
        self._replace_comm(new_comm, rank=new_rank, world_size=new_world_size)
        if destroy_old:
            self.nccl.ncclCommDestroy(old_comm)
        return True

    def shrink_abort(
        self,
        exclude_ranks: list[int],
        destroy_old: bool = False,
    ) -> bool:
        return self.shrink(
            exclude_ranks,
            shrink_flags=ncclShrinkFlagEnum.NCCL_SHRINK_ABORT,
            destroy_old=destroy_old,
        )

    def grow(
        self,
        new_world_size: int,
        grow_unique_id: ncclUniqueId | None = None,
        new_rank: int = -1,
        destroy_old: bool = True,
    ) -> None:
        """Grow this communicator in place.

        Existing ranks call this with ``new_rank=-1`` and ``grow_unique_id=None``.
        Joining ranks call it on a disabled placeholder with ``new_rank`` set to
        their assigned rank and ``grow_unique_id`` received out of band.
        """
        if new_world_size <= 0:
            raise ValueError(f"new_world_size must be positive, got {new_world_size}")

        torch.accelerator.synchronize()
        if new_rank == -1:
            if self.disabled:
                raise RuntimeError(
                    "Existing rank grow requires an enabled parent communicator"
                )
            if new_world_size <= self.world_size:
                raise ValueError(
                    "Existing rank grow requires new_world_size greater than "
                    f"current world_size ({self.world_size}), got {new_world_size}"
                )
            old_comm = self.comm
            new_comm = self.nccl.ncclCommGrow(old_comm, new_world_size, None, -1)
            self._replace_comm(new_comm, world_size=new_world_size)
            if destroy_old:
                self.nccl.ncclCommDestroy(old_comm)
            return

        if grow_unique_id is None:
            raise ValueError("Joining rank grow requires grow_unique_id")
        if new_rank < 0 or new_rank >= new_world_size:
            raise ValueError(
                f"new_rank must be in [0, {new_world_size}), got {new_rank}"
            )

        new_comm = self.nccl.ncclCommGrow(
            None, new_world_size, grow_unique_id, new_rank
        )
        self._replace_comm(new_comm, rank=new_rank, world_size=new_world_size)

    def all_reduce(
        self,
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor = None,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ) -> torch.Tensor:
        if self.disabled:
            return None
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert in_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}"
        )

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
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
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
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
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
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
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
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
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
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        if tensor.dtype in [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ]:
            nccl_dtype = ncclDataTypeEnum.from_torch(torch.uint8)
        else:
            nccl_dtype = ncclDataTypeEnum.from_torch(tensor.dtype)
        self.nccl.ncclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            nccl_dtype,
            dst,
            self.comm,
            cudaStream_t(stream.cuda_stream),
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
        if tensor.dtype in [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ]:
            nccl_dtype = ncclDataTypeEnum.from_torch(torch.uint8)
        else:
            nccl_dtype = ncclDataTypeEnum.from_torch(tensor.dtype)
        self.nccl.ncclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            nccl_dtype,
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
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
        if src == self.rank:
            sendbuff = buffer_type(tensor.data_ptr())
            # NCCL requires the sender also to have a receive buffer
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
        return self.nccl.ncclCommWindowRegister(self.comm, buffer_type(ptr), size, 1)

    def deregister_comm_window(self, window):
        return self.nccl.ncclCommWindowDeregister(self.comm, window)

    def batch_isend_irecv(self, p2p_ops: list, stream=None):
        if self.disabled:
            return
        if stream is None:
            stream = current_stream()
        self.group_start()
        for op in p2p_ops:
            if op.op is torch.distributed.isend:
                self.send(op.tensor, op.group_peer, stream)
            elif op.op is torch.distributed.irecv:
                self.recv(op.tensor, op.group_peer, stream)

        self.group_end()
