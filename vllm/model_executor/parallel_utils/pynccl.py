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
import datetime
import logging
import os

# ===================== import region =====================
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

logger = logging.getLogger(__name__)

so_file = os.environ.get("VLLM_NCCL_SO_PATH", "")

# manually load the nccl library
if so_file:
    logger.info(
        f"Loading nccl from environment variable VLLM_NCCL_SO_PATH={so_file}")
else:
    if torch.version.cuda is not None:
        so_file = "libnccl.so"
    elif torch.version.hip is not None:
        so_file = "librccl.so"
    else:
        raise ValueError("NCCL only supports CUDA and ROCm backends.")
    logger.debug(f"Loading nccl from library {so_file}")

try:
    nccl = ctypes.CDLL(so_file)
except Exception as e:
    logger.error(
        f"Failed to load NCCL library from {so_file} ."
        "It is expected if you are not running on NVIDIA/AMD GPUs."
        "Otherwise please set the environment variable VLLM_NCCL_SO_PATH"
        " to point to the correct nccl library path.")
    raise e

# === export types and functions from nccl to Python ===
# for the original nccl definition, please check
# https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in

ncclResult_t = ctypes.c_int

# equivalent to c declaration:
# ncclResult_t  ncclGetVersion(int *version);
_c_ncclGetVersion = nccl.ncclGetVersion
_c_ncclGetVersion.restype = ctypes.c_int
_c_ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]


def ncclGetVersion() -> str:
    version = ctypes.c_int()
    result = _c_ncclGetVersion(ctypes.byref(version))
    assert result == 0
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
    result = _c_ncclGetUniqueId(ctypes.byref(unique_id))
    assert result == 0
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


# enums
class ncclDataType_t(ctypes.c_int):
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
    def from_torch(cls, dtype: torch.dtype) -> 'ncclDataType_t':
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


class ncclRedOp_t(ctypes.c_int):
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4
    ncclNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> 'ncclRedOp_t':
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
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ncclDataType_t,
    ncclRedOp_t, ctypes.c_void_p, ctypes.c_void_p
]

# equivalent to c declaration:
# ncclResult_t  ncclCommDestroy(ncclComm_t comm);
_c_ncclCommDestroy = nccl.ncclCommDestroy
_c_ncclCommDestroy.restype = ctypes.c_int
_c_ncclCommDestroy.argtypes = [ctypes.c_void_p]


class NCCLCommunicator:

    def __init__(
        self,
        backend=None,
        init_method=None,
        timeout=datetime.timedelta(seconds=10),
        world_size: int = -1,
        rank: int = -1,
        store=None,
        group_name: str = "",
        pg_options=None,
    ):
        if not dist.is_initialized():
            backend = backend or "nccl"
            assert backend == 'nccl', (
                "only use nccl backend for starting the NCCL communicator")
            dist.init_process_group(backend=backend,
                                    init_method=init_method,
                                    timeout=timeout,
                                    world_size=world_size,
                                    rank=rank,
                                    store=store,
                                    group_name=group_name,
                                    pg_options=pg_options)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        torch.cuda.set_device(self.rank)
        if self.rank == 0:
            self.unique_id = ncclGetUniqueId()
        else:
            self.unique_id = NcclUniqueId()
        tensor = torch.ByteTensor(list(self.unique_id.internal)).cuda(
            self.rank)
        dist.broadcast(tensor, src=0)
        byte_list = tensor.cpu().tolist()
        self.unique_id = NcclUniqueId()
        for i, byte in enumerate(byte_list):
            self.unique_id.internal[i] = byte
        self.comm = ctypes.c_void_p()
        result = _c_ncclCommInitRank(ctypes.byref(self.comm), self.world_size,
                                     self.unique_id, self.rank)
        assert result == 0
        self.stream = torch.cuda.Stream(device=f"cuda:{self.rank}")

    def all_reduce(self,
                   tensor: torch.Tensor,
                   op: ReduceOp = ReduceOp.SUM,
                   stream=None):
        if stream is None:
            stream = self.stream
        result = _c_ncclAllReduce(ctypes.c_void_p(tensor.data_ptr()),
                                  ctypes.c_void_p(tensor.data_ptr()),
                                  tensor.numel(),
                                  ncclDataType_t.from_torch(tensor.dtype),
                                  ncclRedOp_t.from_torch(op), self.comm,
                                  ctypes.c_void_p(stream.cuda_stream))
        assert result == 0

    def __del__(self):
        dist.destroy_process_group()
        _c_ncclCommDestroy(self.comm)
