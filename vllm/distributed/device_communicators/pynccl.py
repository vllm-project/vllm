# This file is a pure Python wrapper for the NCCL library.
# The main purpose is to use NCCL combined with CUDA graph.
# Before writing this script, we tried the following approach:
# 1. We tried to use `cupy`, it calls NCCL correctly, but `cupy` itself
#  often gets stuck when initializing the NCCL communicator.
# 2. We tried to use `torch.distributed`, but `torch.distributed.all_reduce`
#  contains many other potential cuda APIs, that are not allowed during
#  capturing the CUDA graph. For further details, please check
# https://discuss.pytorch.org/t/pytorch-cudagraph-with-nccl-operation-failed/ .
#
# Another rejected idea is to write a C/C++ binding for NCCL. It is usually
# doable, but we often encounter issues related with nccl versions, and need
# to switch between different versions of NCCL. See
# https://github.com/NVIDIA/nccl/issues/1234 for more details.
# A C/C++ binding is not flexible enough to handle this. It requires
# recompilation of the code every time we want to switch between different
# versions. This current implementation, with a **pure** Python wrapper, is
# more flexible. We can easily switch between different versions of NCCL by
# changing the environment variable `VLLM_NCCL_SO_PATH`, or the `so_file`
# variable in the code.

import ctypes
import platform
from typing import Optional, Union

# ===================== import region =====================
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from vllm.distributed.parallel_state import get_cpu_world_group, get_local_rank
from vllm.logger import init_logger
from vllm.utils import find_nccl_library, nccl_integrity_check
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary, ncclDataType_t, ncclRedOp_t, ncclUniqueId, ncclComm_t, cudaStream_t, buffer_type)

logger = init_logger(__name__)

class NCCLCommunicator:
    def __init__(
        self,
        group: Optional[ProcessGroup] = None,
        device: Optional[Union[int, str, torch.device]] = None,
        library_path: Optional[str] = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the NCCLCommunicator to. If None,
                it will be bind to f"cuda:{local_rank}".
            library_path: the path to the NCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        so_file = library_path or find_nccl_library()

        try:
            # load the library in another process.
            # if it core dumps, it will not crash the current process
            nccl_integrity_check(so_file)
            self.nccl = NCCLLibrary(so_file)
        except Exception as e:
            logger.error(
                "Failed to load NCCL library from %s ."
                "It is expected if you are not running on NVIDIA/AMD GPUs."
                "Otherwise, the nccl library might not exist, be corrupted "
                "or it does not support the current platform %s."
                "One solution is to download libnccl2 version 2.18 from "
                "https://developer.download.nvidia.com/compute/cuda/repos/ "
                "and extract the libnccl.so.2 file. If you already have the "
                "library, please set the environment variable VLLM_NCCL_SO_PATH"
                " to point to the correct nccl library path.", so_file,
                platform.platform())
            raise e

        assert dist.is_initialized()
        group = get_cpu_world_group() if group is None else group
        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "NCCLCommunicator should be attached to a non-NCCL group.")
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if self.rank == 0:
            # get the unique id from NCCL
            self.unique_id = self.nccl.ncclGetUniqueId()
        else:
            # construct an empty unique id
            self.unique_id = ncclUniqueId()
        tensor = torch.ByteTensor(list(self.unique_id.internal))
        dist.broadcast(tensor, src=0, group=group)
        byte_list = tensor.tolist()
        for i, byte in enumerate(byte_list):
            self.unique_id.internal[i] = byte
        if device is None:
            local_rank = get_local_rank()
            device = torch.device(f"cuda:{local_rank}")
        elif isinstance(device, int):
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
            self.comm = self.nccl.ncclCommInitRank(self.world_size, self.unique_id,
                                       self.rank)
            self.stream = torch.cuda.Stream()

    def all_reduce(self,
                   tensor: torch.Tensor,
                   op: ReduceOp = ReduceOp.SUM,
                   stream=None):
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = self.stream
        self.nccl.ncclAllReduce(buffer_type(tensor.data_ptr()),
                                buffer_type(tensor.data_ptr()),
                                tensor.numel(),
                                ncclDataTypeEnum.from_torch(tensor.dtype),
                                ncclRedOpTypeEnum.from_torch(op), self.comm,
                                cudaStream_t(stream.cuda_stream))

    def __del__(self):
        # `dist` module might have been already destroyed
        if hasattr(dist, 'destroy_process_group'):
            dist.destroy_process_group()
        self.nccl.ncclCommDestroy(self.comm)
