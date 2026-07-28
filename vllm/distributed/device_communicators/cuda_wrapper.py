# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This file is a pure Python wrapper for the cudart library.
It avoids the need to compile a separate shared library, and is
convenient for use when we just need to call a few functions.
"""

import ctypes
from dataclasses import dataclass
from typing import Any

# this line makes it possible to directly load `libcudart.so` using `ctypes`
import torch  # noqa

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.system_utils import find_loaded_library

logger = init_logger(__name__)

# === export types and functions from cudart to Python ===
# for the original cudart definition, please check
# https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

cudaError_t = ctypes.c_int
cudaMemcpyKind = ctypes.c_int


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


class CudaRTLibrary:
    exported_functions = [
        # ​cudaError_t cudaSetDevice ( int  device )
        Function("cudaSetDevice", cudaError_t, [ctypes.c_int]),
        # cudaError_t 	cudaDeviceSynchronize ( void )
        Function("cudaDeviceSynchronize", cudaError_t, []),
        # ​cudaError_t cudaDeviceReset ( void )
        Function("cudaDeviceReset", cudaError_t, []),
        # const char* 	cudaGetErrorString ( cudaError_t error )
        Function("cudaGetErrorString", ctypes.c_char_p, [cudaError_t]),
        # ​cudaError_t 	cudaMalloc ( void** devPtr, size_t size )
        Function(
            "cudaMalloc",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        ),
        # ​cudaError_t 	cudaFree ( void* devPtr )
        Function("cudaFree", cudaError_t, [ctypes.c_void_p]),
        # ​cudaError_t cudaMemset ( void* devPtr, int  value, size_t count )
        Function(
            "cudaMemset", cudaError_t, [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        ),
        # ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) # noqa
        Function(
            "cudaMemcpy",
            cudaError_t,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, cudaMemcpyKind],
        ),
        # cudaError_t cudaIpcGetMemHandle ( cudaIpcMemHandle_t* handle, void* devPtr ) # noqa
        Function(
            "cudaIpcGetMemHandle",
            cudaError_t,
            [ctypes.POINTER(cudaIpcMemHandle_t), ctypes.c_void_p],
        ),
        # ​cudaError_t cudaIpcOpenMemHandle ( void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags ) # noqa
        Function(
            "cudaIpcOpenMemHandle",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), cudaIpcMemHandle_t, ctypes.c_uint],
        ),
        # ​cudaError_t cudaGetLastError ( void )
        Function("cudaGetLastError", cudaError_t, []),
        # cudaError_t cudaHostRegister ( void* ptr, size_t size, unsigned int flags )
        Function(
            "cudaHostRegister",
            cudaError_t,
            [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint],
        ),
        # cudaError_t cudaHostUnregister ( void* ptr )
        Function(
            "cudaHostUnregister",
            cudaError_t,
            [ctypes.c_void_p],
        ),
    ]

    # https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Runtime_API_functions_supported_by_HIP.html # noqa
    cuda_to_hip_mapping = {
        "cudaSetDevice": "hipSetDevice",
        "cudaDeviceSynchronize": "hipDeviceSynchronize",
        "cudaDeviceReset": "hipDeviceReset",
        "cudaGetErrorString": "hipGetErrorString",
        "cudaMalloc": "hipMalloc",
        "cudaFree": "hipFree",
        "cudaMemset": "hipMemset",
        "cudaMemcpy": "hipMemcpy",
        "cudaIpcGetMemHandle": "hipIpcGetMemHandle",
        "cudaIpcOpenMemHandle": "hipIpcOpenMemHandle",
        "cudaGetLastError": "hipGetLastError",
        "cudaHostRegister": "hipHostRegister",
        "cudaHostUnregister": "hipHostUnregister",
    }

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: str | None = None):
        if so_file is None:
            so_file = (
                find_loaded_library(
                    "libamdhip64" if current_platform.is_rocm() else "libcudart"
                )
                or envs.VLLM_CUDART_SO_PATH  # fallback to env var
            )
            assert so_file is not None, (
                "libcudart is not loaded in the current process, "
                "try setting VLLM_CUDART_SO_PATH"
            )
        if so_file not in CudaRTLibrary.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            CudaRTLibrary.path_to_library_cache[so_file] = lib
        self.lib = CudaRTLibrary.path_to_library_cache[so_file]

        if so_file not in CudaRTLibrary.path_to_dict_mapping:
            _funcs = {}
            for func in CudaRTLibrary.exported_functions:
                f = getattr(
                    self.lib,
                    CudaRTLibrary.cuda_to_hip_mapping[func.name]
                    if current_platform.is_rocm()
                    else func.name,
                )
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            CudaRTLibrary.path_to_dict_mapping[so_file] = _funcs
        self.funcs = CudaRTLibrary.path_to_dict_mapping[so_file]

    def CUDART_CHECK(self, result: cudaError_t) -> None:
        if result != 0:
            error_str = self.cudaGetErrorString(result)
            # Drain any thread-local pending CUDA error so subsequent API
            # calls are not poisoned by a stale error (e.g. from a previous
            # best-effort cudaHostRegister that deliberately did not raise).
            self.drain_pending_error()
            raise RuntimeError(f"CUDART error: {error_str}")

    def cudaGetErrorString(self, error: cudaError_t) -> str:
        return self.funcs["cudaGetErrorString"](error).decode("utf-8")

    def cudaSetDevice(self, device: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaSetDevice"](device))

    def cudaDeviceSynchronize(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceSynchronize"]())

    def cudaDeviceReset(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceReset"]())

    def cudaMalloc(self, size: int) -> ctypes.c_void_p:
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(self.funcs["cudaMalloc"](ctypes.byref(devPtr), size))
        return devPtr

    def cudaFree(self, devPtr: ctypes.c_void_p) -> None:
        self.CUDART_CHECK(self.funcs["cudaFree"](devPtr))

    def cudaMemset(self, devPtr: ctypes.c_void_p, value: int, count: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaMemset"](devPtr, value, count))

    def cudaMemcpy(
        self, dst: ctypes.c_void_p, src: ctypes.c_void_p, count: int
    ) -> None:
        cudaMemcpyDefault = 4
        kind = cudaMemcpyDefault
        self.CUDART_CHECK(self.funcs["cudaMemcpy"](dst, src, count, kind))

    def cudaIpcGetMemHandle(self, devPtr: ctypes.c_void_p) -> cudaIpcMemHandle_t:
        handle = cudaIpcMemHandle_t()
        self.CUDART_CHECK(
            self.funcs["cudaIpcGetMemHandle"](ctypes.byref(handle), devPtr)
        )
        return handle

    def cudaIpcOpenMemHandle(self, handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        cudaIpcMemLazyEnablePeerAccess = 1
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(
            self.funcs["cudaIpcOpenMemHandle"](
                ctypes.byref(devPtr), handle, cudaIpcMemLazyEnablePeerAccess
            )
        )
        return devPtr

    def cudaGetLastError(self) -> int:
        """Return (and clear) the thread-local pending CUDA runtime error.

        Unlike other methods this deliberately does **not** call
        ``CUDART_CHECK`` because a nonzero value is the expected/consumed
        state (e.g. consuming a stale error left by a failed best-effort
        ``cudaHostRegister``).  Returns the prior error code as a plain
        Python int.
        """
        return int(self.funcs["cudaGetLastError"]())

    def drain_pending_error(self) -> int:
        """Canonical same-handle drain helper.

        Drains and returns the thread-local pending CUDA runtime error
        through this library handle.  Unlike ``CUDART_CHECK``, does not
        raise — a nonzero return is the expected consumed state for
        best-effort paths (e.g. after a failed ``cudaHostRegister``).
        """
        return self.cudaGetLastError()

    def cudaHostRegister(self, ptr: int, size: int, flags: int = 0) -> int:
        """Register host memory for CUDA DMA access (best-effort safe).

        Returns the raw ``cudaError_t`` as a plain Python int — does
        **not** call ``CUDART_CHECK`` so callers on best-effort paths
        can inspect the result without raising.  Use ``drain_pending_error``
        on the same instance to consume any thread-local pending error
        after a nonzero return.
        """
        return int(self.funcs["cudaHostRegister"](ctypes.c_void_p(ptr), size, flags))

    def cudaHostUnregister(self, ptr: int) -> int:
        """Unregister host memory previously registered with cudaHostRegister.

        Returns the raw ``cudaError_t`` as a plain Python int.
        """
        return int(self.funcs["cudaHostUnregister"](ctypes.c_void_p(ptr)))
