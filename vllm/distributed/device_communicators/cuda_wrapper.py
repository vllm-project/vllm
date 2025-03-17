# SPDX-License-Identifier: Apache-2.0
"""This file is a pure Python wrapper for the cudart library.
It avoids the need to compile a separate shared library, and is
convenient for use when we just need to call a few functions.
"""
import ctypes
import os
import platform
from dataclasses import dataclass
from shutil import which
from typing import Any, Dict, List, Optional

# this line makes it possible to directly load `libcudart.so` using `ctypes`
import torch  # noqa

import vllm.envs as envs
from vllm.logger import init_logger

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
    argtypes: List[Any]


def find_loaded_library(lib_name) -> Optional[str]:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """ # noqa
    path = None
    if platform.system() != 'Windows':
        found = False
        with open("/proc/self/maps") as f:
            for line in f:
                if lib_name in line:
                    found = True
                    break
        if not found:
            # the library is not loaded in the current process
            return None
        # if lib_name is libcudart, we need to match a line with:
        # address /path/to/libcudart-hash.so.11.0
        start = line.index("/")
        path = line[start:].strip()
        filename = path.split("/")[-1]
        assert filename.rpartition(".so")[0].startswith(lib_name), \
            f"Unexpected filename: {filename} for library {lib_name}"
    elif "cudart" in lib_name:
        cudart_path = os.getenv("VLLM_CUDART_SO_PATH", None)
        if cudart_path is None:
            cuda_path = None
            if os.environ.get("CUDA_HOME", None):
                cuda_path = os.environ.get("CUDA_HOME")
            elif os.environ.get("CUDA_ROOT", None):
                cuda_path = os.environ.get("CUDA_ROOT")
            elif os.environ.get("CUDA_PATH"):
                cuda_path = os.environ.get("CUDA_PATH", None)

            cuda_major_version = torch.version.cuda.split(
                '.')[0] if torch.version.cuda else "12"
            if cuda_major_version < "12":
                cuda_major_version += "0"
            if cuda_path:
                cudart_path = os.path.abspath(
                    os.path.join(cuda_path, "bin",
                                 f"cudart64_{cuda_major_version}.dll"))
            else:
                nvcc_path = which("nvcc")
                if nvcc_path:
                    cudart_path = os.path.abspath(
                        os.path.join(nvcc_path, "..",
                                     f"cudart64_{cuda_major_version}.dll"))
            if cudart_path:
                os.environ["VLLM_CUDART_SO_PATH"] = cudart_path
                logger.info('VLLM_CUDART_SO_PATH resolved to %s', cudart_path)
            else:
                raise ValueError(
                    'VLLM_CUDART_SO_PATH is not set. '
                    'VLLM_CUDART_SO_PATH need to be set with the absolute path '
                    'to cudart dll on Windows (for example, set '
                    'VLLM_CUDART_SO_PATH=C:\\CUDA\\v12.4\\bin\\cudart64_12.dll)'
                )
    return path


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
        Function("cudaMalloc", cudaError_t,
                 [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
        # ​cudaError_t 	cudaFree ( void* devPtr )
        Function("cudaFree", cudaError_t, [ctypes.c_void_p]),
        # ​cudaError_t cudaMemset ( void* devPtr, int  value, size_t count )
        Function("cudaMemset", cudaError_t,
                 [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]),
        # ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) # noqa
        Function("cudaMemcpy", cudaError_t, [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, cudaMemcpyKind
        ]),

        # cudaError_t cudaIpcGetMemHandle ( cudaIpcMemHandle_t* handle, void* devPtr ) # noqa
        Function("cudaIpcGetMemHandle", cudaError_t,
                 [ctypes.POINTER(cudaIpcMemHandle_t), ctypes.c_void_p]),
        # ​cudaError_t cudaIpcOpenMemHandle ( void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags ) # noqa
        Function("cudaIpcOpenMemHandle", cudaError_t, [
            ctypes.POINTER(ctypes.c_void_p), cudaIpcMemHandle_t, ctypes.c_uint
        ]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            so_file = find_loaded_library("libcudart")
            if so_file is None:
                so_file = envs.VLLM_CUDART_SO_PATH  # fallback to env var
            assert so_file is not None, \
                (
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
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            CudaRTLibrary.path_to_dict_mapping[so_file] = _funcs
        self.funcs = CudaRTLibrary.path_to_dict_mapping[so_file]

    def CUDART_CHECK(self, result: cudaError_t) -> None:
        if result != 0:
            error_str = self.cudaGetErrorString(result)
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

    def cudaMemset(self, devPtr: ctypes.c_void_p, value: int,
                   count: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaMemset"](devPtr, value, count))

    def cudaMemcpy(self, dst: ctypes.c_void_p, src: ctypes.c_void_p,
                   count: int) -> None:
        cudaMemcpyDefault = 4
        kind = cudaMemcpyDefault
        self.CUDART_CHECK(self.funcs["cudaMemcpy"](dst, src, count, kind))

    def cudaIpcGetMemHandle(self,
                            devPtr: ctypes.c_void_p) -> cudaIpcMemHandle_t:
        handle = cudaIpcMemHandle_t()
        self.CUDART_CHECK(self.funcs["cudaIpcGetMemHandle"](
            ctypes.byref(handle), devPtr))
        return handle

    def cudaIpcOpenMemHandle(self,
                             handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        cudaIpcMemLazyEnablePeerAccess = 1
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(self.funcs["cudaIpcOpenMemHandle"](
            ctypes.byref(devPtr), handle, cudaIpcMemLazyEnablePeerAccess))
        return devPtr
