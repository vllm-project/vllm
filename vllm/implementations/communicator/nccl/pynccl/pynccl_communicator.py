import logging
import os
from contextlib import contextmanager
from typing import Any, Optional

import torch

from vllm.implementations.communicator.nccl.pynccl.wrapper import (
    NCCLLibrary, buffer_type, cudaStream_t, ncclDataType_t, ncclRedOp_t,
    ncclUniqueId)
from vllm.interfaces.communicator import Communicator, ReduceOp
from vllm.interfaces.coordinator import Coordinator

logger = logging.getLogger(__name__)

# script to manage the path of the nccl library

so_file: Optional[str] = None


def set_pynccl_path(path: str) -> None:
    global so_file
    so_file = path


def get_pynccl_path() -> Optional[str]:
    return so_file


@contextmanager
def change_pynccl_path(path: str) -> None:
    global so_file
    old_path = so_file
    so_file = path
    yield
    so_file = old_path


class NCCLCommunicator(Communicator):

    def __init__(
        self,
        coordinator: Coordinator,
        path_of_nccl: str = None,
    ):
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            assert len(visible_gpus) >= coordinator.get_group_size(), \
                (f"Number of visible GPUs {len(visible_gpus)} is less than"
                f" the number of processes in the group {coordinator.group}.")

        super().__init__(coordinator)

        # search priority:
        # 1. path_of_nccl (passed in the constructor of NCCLCommunicator)
        # 2. so_file (set by users calling `set_pynccl_path`)
        # 3. VLLM_NCCL_SO_PATH environment variable
        # 4. default path
        path_of_nccl = path_of_nccl or so_file or os.environ.get(
            "VLLM_NCCL_SO_PATH", "")
        if not path_of_nccl:
            # not set yet, try a decent guess as default
            if torch.version.cuda is not None:
                path_of_nccl = "libnccl.so.2"
            elif torch.version.hip is not None:
                path_of_nccl = "librccl.so.1"
            else:
                raise ValueError("NCCL only supports CUDA and ROCm backends.")
        logger.debug(f"Loading nccl from library {so_file}")

        try:
            self.lib = NCCLLibrary(path_of_nccl)
        except Exception as e:
            logger.error(
                f"Failed to load NCCL library from {path_of_nccl} ."
                "It is expected if you are not running on NVIDIA/AMD GPUs."
                "Otherwise please set environment variable VLLM_NCCL_SO_PATH"
                " to point to the correct nccl library path.")
            raise e

        logger.info(f"vLLM is using nccl=={self.lib.ncclGetVersion()}")
        local_rank = coordinator.get_local_rank()
        torch.cuda.set_device(local_rank)
        self.stream = torch.cuda.Stream(device=f"cuda:{local_rank}")
        if coordinator.is_group_master():
            # get a unique id by calling nccl library
            self.unique_id = self.lib.ncclGetUniqueId()
        else:
            # default initialization of unique_id
            self.unique_id = ncclUniqueId()
        data = bytearray(self.unique_id.internal)
        coordinator.broadcast(data, src=coordinator.get_group_master_rank())
        for i in range(len(data)):
            self.unique_id.internal[i] = data[i]
        nrank = coordinator.get_group_size()
        rank = coordinator.get_group_rank()
        self.comm = self.lib.ncclCommInitRank(nrank, self.unique_id, rank)

    @staticmethod
    def convert_reduce_op(op: ReduceOp) -> ncclRedOp_t:
        return {
            ReduceOp.SUM: ncclRedOp_t.ncclSum,
            ReduceOp.PRODUCT: ncclRedOp_t.ncclProd,
            ReduceOp.MAX: ncclRedOp_t.ncclMax,
            ReduceOp.MIN: ncclRedOp_t.ncclMin,
            ReduceOp.AVG: ncclRedOp_t.ncclAvg,
        }[op]

    @staticmethod
    def convert_data_type(dtype: torch.dtype) -> ncclDataType_t:
        return {
            torch.int8: ncclDataType_t.ncclInt8,
            torch.uint8: ncclDataType_t.ncclUint8,
            torch.int32: ncclDataType_t.ncclInt32,
            torch.int64: ncclDataType_t.ncclInt64,
            torch.float16: ncclDataType_t.ncclFloat16,
            torch.float32: ncclDataType_t.ncclFloat32,
            torch.float64: ncclDataType_t.ncclFloat64,
            torch.bfloat16: ncclDataType_t.ncclBfloat16,
        }[dtype]

    def all_reduce(self,
                   tensor_in: torch.Tensor,
                   tensor_out: Optional[torch.Tensor] = None,
                   op: ReduceOp = ReduceOp.SUM,
                   stream: Optional[Any] = None):
        assert tensor_in.is_cuda and tensor_in.is_contiguous()
        if tensor_out is None:
            tensor_out = tensor_in
        op = self.convert_reduce_op(op)
        dtype = self.convert_data_type(tensor_in.dtype)
        if stream is None:
            stream = self.stream
        self.lib.ncclAllReduce(buffer_type(tensor_in.data_ptr()),
                               buffer_type(tensor_out.data_ptr()),
                               tensor_in.numel(), dtype, op, self.comm,
                               cudaStream_t(stream.cuda_stream))

    def __del__(self):
        self.lib.ncclCommDestroy(self.comm)
