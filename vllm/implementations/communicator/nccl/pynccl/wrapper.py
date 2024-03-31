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
from dataclasses import dataclass
from typing import Any, List

# === export types and functions from nccl to Python ===
# for the original nccl definition, please check
# https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in

ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p


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


class ncclRedOp_t(ctypes.c_int):
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4
    ncclNumOps = 5


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class NCCLLibrary:
    exported_functions = [
        # ncclResult_t  ncclGetVersion(int *version);
        Function("ncclGetVersion", ncclResult_t,
                 [ctypes.POINTER(ctypes.c_int)]),
        # ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
        Function("ncclGetUniqueId", ncclResult_t,
                 [ctypes.POINTER(ncclUniqueId)]),
        # ncclResult_t  ncclCommInitRank(
        #   ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
        # note that ncclComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function("ncclCommInitRank", ncclResult_t, [
            ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId,
            ctypes.c_int
        ]),
        # ncclResult_t  ncclAllReduce(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        #   cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function("ncclAllReduce", ncclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t,
            ncclRedOp_t, ncclComm_t, cudaStream_t
        ]),
        # ncclResult_t  ncclCommDestroy(ncclComm_t comm);
        Function("ncclCommDestroy", ncclResult_t, [ncclComm_t]),
    ]

    def __init__(self, so_file: str):
        self.lib = ctypes.CDLL(so_file)
        self._funcs = {}
        for func in self.exported_functions:
            f = getattr(self.lib, func.name)
            f.restype = func.restype
            f.argtypes = func.argtypes
            self._funcs[func.name] = f

    def ncclGetVersion(self) -> str:
        version = ctypes.c_int()
        result = self._funcs["ncclGetVersion"](ctypes.byref(version))
        assert result == 0
        version_str = str(version.value)
        # something like 21903 --> "2.19.3"
        major = version_str[0].lstrip("0")
        minor = version_str[1:3].lstrip("0")
        patch = version_str[3:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def ncclGetUniqueId(self) -> ncclUniqueId:
        unique_id = ncclUniqueId()
        result = self._funcs["ncclGetUniqueId"](ctypes.byref(unique_id))
        assert result == 0
        return unique_id

    def ncclCommInitRank(self, world_size: int, unique_id: ncclUniqueId,
                         rank: int) -> ncclComm_t:
        comm = ncclComm_t()
        result = self._funcs["ncclCommInitRank"](ctypes.byref(comm),
                                                 world_size, unique_id, rank)
        assert result == 0
        return comm

    def ncclAllReduce(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: ncclDataType_t, op: ncclRedOp_t,
                      comm: ncclComm_t, stream: cudaStream_t) -> None:
        result = self._funcs["ncclAllReduce"](sendbuff, recvbuff, count,
                                              datatype, op, comm, stream)
        assert result == 0

    def ncclCommDestroy(self, comm: ncclComm_t) -> None:
        result = self._funcs["ncclCommDestroy"](comm)
        assert result == 0


__all__ = [
    "NCCLLibrary", "ncclDataType_t", "ncclRedOp_t", "ncclUniqueId",
    "ncclComm_t", "cudaStream_t", "buffer_type"
]
