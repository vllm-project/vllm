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

logger = init_logger(__name__)

so_file = find_nccl_library()

try:
    # load the library in another process.
    # if it core dumps, it will not crash the current process
    nccl_integrity_check(so_file)
    nccl = ctypes.CDLL(so_file)
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

# === export types and functions from nccl to Python ===
# for the original nccl definition, please check
# https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in

ncclResult_t = ctypes.c_int

_c_ncclGetErrorString = nccl.ncclGetErrorString
_c_ncclGetErrorString.restype = ctypes.c_char_p
_c_ncclGetErrorString.argtypes = [ncclResult_t]


def NCCL_CHECK(result: ncclResult_t) -> None:
    if result != 0:
        error_str = _c_ncclGetErrorString(result)
        error_str = error_str.decode("utf-8")
        raise RuntimeError(f"NCCL error: {error_str}")


# equivalent to c declaration:
# ncclResult_t  ncclGetVersion(int *version);
_c_ncclGetVersion = nccl.ncclGetVersion
_c_ncclGetVersion.restype = ctypes.c_int
_c_ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]


def ncclGetVersion() -> str:
    version = ctypes.c_int()
    NCCL_CHECK(_c_ncclGetVersion(ctypes.byref(version)))
    # something like 21903 --> "2.19.3"
    version_str = str(version.value)
    major = version_str[0].lstrip("0")
    minor = version_str[1:3].lstrip("0")
    patch = version_str[3:].lstrip("0")
    return f"{major}.{minor}.{patch}"


class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


# equivalent to c declaration:
# ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
_c_ncclGetUniqueId = nccl.ncclGetUniqueId
_c_ncclGetUniqueId.restype = ctypes.c_int
_c_ncclGetUniqueId.argtypes = [ctypes.POINTER(NcclUniqueId)]


def ncclGetUniqueId() -> NcclUniqueId:
    unique_id = NcclUniqueId()
    NCCL_CHECK(_c_ncclGetUniqueId(ctypes.byref(unique_id)))
    return unique_id


# equivalent to c declaration:
# ncclResult_t  ncclCommInitRank(
#   ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
# note that ncclComm_t is a pointer type, so the first argument
# is a pointer to a pointer
_c_ncclCommInitRank = nccl.ncclCommInitRank
_c_ncclCommInitRank.restype = ctypes.c_int
_c_ncclCommInitRank.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, NcclUniqueId, ctypes.c_int
]

ncclDataType_t = ctypes.c_int


class ncclDataTypeEnum:
    ncclInt8 = 0
    ncclChar = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclHalf = 6
    ncclFloat32 = 7
    ncclFloat = 7
    ncclFloat64 = 8
    ncclDouble = 8
    ncclBfloat16 = 9
    ncclNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.ncclInt8
        if dtype == torch.uint8:
            return cls.ncclUint8
        if dtype == torch.int32:
            return cls.ncclInt32
        if dtype == torch.int64:
            return cls.ncclInt64
        if dtype == torch.float16:
            return cls.ncclFloat16
        if dtype == torch.float32:
            return cls.ncclFloat32
        if dtype == torch.float64:
            return cls.ncclFloat64
        if dtype == torch.bfloat16:
            return cls.ncclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


ncclRedOp_t = ctypes.c_int


class ncclRedOpTypeEnum:
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4
    ncclNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.ncclSum
        if op == ReduceOp.PRODUCT:
            return cls.ncclProd
        if op == ReduceOp.MAX:
            return cls.ncclMax
        if op == ReduceOp.MIN:
            return cls.ncclMin
        if op == ReduceOp.AVG:
            return cls.ncclAvg
        raise ValueError(f"Unsupported op: {op}")


# equivalent to c declaration:
# ncclResult_t  ncclAllReduce(
#   const void* sendbuff, void* recvbuff, size_t count,
#   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
#   udaStream_t stream);
# note that cudaStream_t is a pointer type, so the last argument is a pointer
_c_ncclAllReduce = nccl.ncclAllReduce
_c_ncclAllReduce.restype = ctypes.c_int
_c_ncclAllReduce.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ncclRedOp_t,
    ncclDataType_t, ctypes.c_void_p, ctypes.c_void_p
]

# be cautious! this is a collective call, it will block until all
# processes in the communicator have called this function.
# because Python object destruction can happen in random order,
# it is better not to call it at all.
# equivalent to c declaration:
# ncclResult_t  ncclCommDestroy(ncclComm_t comm);
_c_ncclCommDestroy = nccl.ncclCommDestroy
_c_ncclCommDestroy.restype = ctypes.c_int
_c_ncclCommDestroy.argtypes = [ctypes.c_void_p]


class NCCLCommunicator:

    def __init__(
        self,
        group: Optional[ProcessGroup] = None,
        device: Optional[Union[int, str, torch.device]] = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the NCCLCommunicator to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        assert dist.is_initialized()
        group = get_cpu_world_group() if group is None else group
        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "NCCLCommunicator should be attached to a non-NCCL group.")
        self.group = group
        # note: this rank is the rank in the group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if self.rank == 0:
            self.unique_id = ncclGetUniqueId()
        else:
            self.unique_id = NcclUniqueId()
        tensor = torch.ByteTensor(list(self.unique_id.internal))
        ranks = dist.get_process_group_ranks(group)
        # arg `src` in `broadcast` is the global rank
        dist.broadcast(tensor, src=ranks[0], group=group)
        byte_list = tensor.tolist()
        for i, byte in enumerate(byte_list):
            self.unique_id.internal[i] = byte
        self.comm = ctypes.c_void_p()
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
            NCCL_CHECK(
                _c_ncclCommInitRank(ctypes.byref(self.comm), self.world_size,
                                    self.unique_id, self.rank))
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
        NCCL_CHECK(
            _c_ncclAllReduce(ctypes.c_void_p(tensor.data_ptr()),
                             ctypes.c_void_p(tensor.data_ptr()),
                             tensor.numel(),
                             ncclDataTypeEnum.from_torch(tensor.dtype),
                             ncclRedOpTypeEnum.from_torch(op), self.comm,
                             ctypes.c_void_p(stream.cuda_stream)))
