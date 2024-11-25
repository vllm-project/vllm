from contextlib import contextmanager
from typing import Optional, Union

# ===================== import region =====================
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary, buffer_type, cudaStream_t, ncclComm_t, ncclDataTypeEnum,
    ncclRedOpTypeEnum, ncclUniqueId)
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

logger = init_logger(__name__)


class PyNcclCommunicator:

    def __init__(
        self,
        group: Union[ProcessGroup, StatelessProcessGroup],
        device: Union[int, str, torch.device],
        library_path: Optional[str] = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyNcclCommunicator to. If None,
                it will be bind to f"cuda:{local_rank}".
            library_path: the path to the NCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert dist.get_backend(group) != dist.Backend.NCCL, (
                "PyNcclCommunicator should be attached to a non-NCCL group.")
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            self.stream = None
            return
        try:
            self.nccl = NCCLLibrary(library_path)
        except Exception:
            # disable because of missing NCCL library
            # e.g. in a non-GPU environment
            self.available = False
            self.disabled = True
            self.stream = None
            return

        self.available = True
        self.disabled = False

        logger.info("vLLM is using nccl==%s", self.nccl.ncclGetVersion())

        if self.rank == 0:
            # get the unique id from NCCL
            self.unique_id = self.nccl.ncclGetUniqueId()
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
        # `torch.cuda.device` is a context manager that changes the
        # current cuda device to the specified one
        with torch.cuda.device(device):
            self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                self.world_size, self.unique_id, self.rank)
            self.stream = torch.cuda.Stream()

            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            self.stream.synchronize()
            del data

        # by default it is disabled, e.g. in profiling models and prefill phase.
        # to use it, use under `with obj.change_state(enable=True)`, usually
        # when we are using CUDA graph.
        self.disabled = True

    def all_reduce(self,
                   tensor: torch.Tensor,
                   op: ReduceOp = ReduceOp.SUM,
                   stream=None):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = self.stream
        self.nccl.ncclAllReduce(buffer_type(tensor.data_ptr()),
                                buffer_type(tensor.data_ptr()), tensor.numel(),
                                ncclDataTypeEnum.from_torch(tensor.dtype),
                                ncclRedOpTypeEnum.from_torch(op), self.comm,
                                cudaStream_t(stream.cuda_stream))

    def all_gather(self,
                   output_tensor: torch.Tensor,
                   input_tensor: torch.Tensor,
                   stream=None):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}")
        if stream is None:
            stream = self.stream
        self.nccl.ncclAllGather(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()), input_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype), self.comm,
            cudaStream_t(stream.cuda_stream))

    def reduce_scatter(self,
                       output_tensor: torch.Tensor,
                       input_tensor: torch.Tensor,
                       op: ReduceOp = ReduceOp.SUM,
                       stream=None):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}")
        if stream is None:
            stream = self.stream
        self.nccl.ncclReduceScatter(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()), output_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op), self.comm,
            cudaStream_t(stream.cuda_stream))

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = self.stream
        self.nccl.ncclSend(buffer_type(tensor.data_ptr()), tensor.numel(),
                           ncclDataTypeEnum.from_torch(tensor.dtype), dst,
                           self.comm, cudaStream_t(stream.cuda_stream))

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = self.stream
        self.nccl.ncclRecv(buffer_type(tensor.data_ptr()), tensor.numel(),
                           ncclDataTypeEnum.from_torch(tensor.dtype), src,
                           self.comm, cudaStream_t(stream.cuda_stream))

    @contextmanager
    def change_state(self,
                     enable: Optional[bool] = None,
                     stream: Optional[torch.cuda.Stream] = None):
        """
        A context manager to change the state of the communicator.
        """
        if enable is None:
            # guess a default value when not specified
            enable = self.available

        if stream is None:
            stream = self.stream

        old_disable = self.disabled
        old_stream = self.stream

        self.stream = stream
        self.disabled = not enable
        yield

        self.disabled = old_disable
        self.stream = old_stream
