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
    }

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    @classmethod
    def _runtime_lib_name(cls) -> str:
        return "libamdhip64" if current_platform.is_rocm() else "libcudart"

    @classmethod
    def _symbol_name(cls, func: Function) -> str:
        if current_platform.is_rocm():
            return cls.cuda_to_hip_mapping[func.name]
        return func.name

    @classmethod
    def _loaded_library_paths(cls, lib_name: str) -> list[str]:
        paths: list[str] = []
        try:
            with open("/proc/self/maps") as maps:
                for line in maps:
                    if lib_name not in line or "/" not in line:
                        continue
                    path = line[line.index("/") :].strip()
                    filename = path.rsplit("/", 1)[-1]
                    if not filename.rpartition(".so")[0].startswith(lib_name):
                        continue
                    if path not in paths:
                        paths.append(path)
        except OSError:
            return paths
        return paths

    @classmethod
    def _load_library(cls, so_file: str) -> Any:
        if so_file not in cls.path_to_library_cache:
            cls.path_to_library_cache[so_file] = ctypes.CDLL(so_file)
        return cls.path_to_library_cache[so_file]

    @classmethod
    def _missing_exported_functions(cls, lib: Any) -> list[str]:
        missing: list[str] = []
        for func in cls.exported_functions:
            symbol = cls._symbol_name(func)
            try:
                getattr(lib, symbol)
            except AttributeError:
                missing.append(symbol)
        return missing

    @classmethod
    def _select_runtime_library(cls) -> str:
        lib_name = cls._runtime_lib_name()
        candidates = cls._loaded_library_paths(lib_name)
        if envs.VLLM_CUDART_SO_PATH and envs.VLLM_CUDART_SO_PATH not in candidates:
            candidates.append(envs.VLLM_CUDART_SO_PATH)

        rejected: list[str] = []
        for so_file in candidates:
            try:
                lib = cls._load_library(so_file)
            except OSError as e:
                rejected.append(f"{so_file} ({e})")
                continue
            missing = cls._missing_exported_functions(lib)
            if missing:
                rejected.append(f"{so_file} (missing {', '.join(missing)})")
                continue
            return so_file

        message = (
            f"{lib_name} with the required runtime symbols is not loaded in "
            "the current process; try setting VLLM_CUDART_SO_PATH."
        )
        if rejected:
            message += " Rejected candidates: " + "; ".join(rejected)
        raise RuntimeError(message)

    def __init__(self, so_file: str | None = None):
        if so_file is None:
            so_file = self._select_runtime_library()
        self.lib = self._load_library(so_file)

        if so_file not in CudaRTLibrary.path_to_dict_mapping:
            missing = self._missing_exported_functions(self.lib)
            if missing:
                raise RuntimeError(
                    f"{so_file} is missing required runtime symbols: "
                    f"{', '.join(missing)}"
                )
            _funcs = {}
            for func in CudaRTLibrary.exported_functions:
                f = getattr(self.lib, self._symbol_name(func))
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
