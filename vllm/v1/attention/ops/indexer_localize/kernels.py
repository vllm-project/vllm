# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA kernels of the die-local indexer, compiled at runtime with NVRTC.

`build.cu` (the per-step build), `merge.cu` (the merge of both domains'
top-k candidates) and `topk_shape.cu` (the top-k's CTA shape, for occupancy
queries) ship as source. Each configuration is compiled for the
current device at first use, once per process, and loaded with
`cuLibraryLoadData`, whose kernel handle launches in any context: on full-GPU
streams and on the locality domains' green-context streams alike. A launch is
one `cuLaunchKernel` on the given stream, so CUDA graphs capture it like any
other kernel.
"""

import ctypes
import os
import threading
import time
from collections.abc import Sequence

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_DIR = os.path.dirname(os.path.abspath(__file__))


def _check(result):
    """The values of a cuda.bindings call, raising on an error code."""
    err, *values = result if isinstance(result, tuple) else (result,)
    if int(err) != 0:
        raise RuntimeError(f"CUDA call failed: {err!r}")
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def compile_source(
    source: str, arch: str, defines: Sequence[tuple[str, int]] = ()
) -> bytes:
    """Compile a kernel source of this package to a cubin with NVRTC."""
    from cuda.bindings import nvrtc

    with open(os.path.join(_DIR, source), "rb") as f:
        code = f.read()
    prog = _check(nvrtc.nvrtcCreateProgram(code, source.encode(), 0, [], []))
    try:
        options = [
            f"--gpu-architecture={arch}".encode(),
            b"-std=c++17",
            b"-default-device",
            *(f"-D{name}={value}".encode() for name, value in defines),
        ]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(options), options)
        if int(err) != 0:
            size = _check(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(
                f"NVRTC failed to compile {source} for {arch}: "
                f"{log.decode(errors='replace')}"
            )
        size = _check(nvrtc.nvrtcGetCUBINSize(prog))
        cubin = b" " * size
        _check(nvrtc.nvrtcGetCUBIN(prog, cubin))
        return bytes(cubin)
    finally:
        nvrtc.nvrtcDestroyProgram(prog)


class CudaKernel:
    """One compiled kernel; launches take their arguments as ctypes values."""

    def __init__(
        self,
        source: str,
        name: str,
        defines: Sequence[tuple[str, int]],
        device_index: int,
    ) -> None:
        from cuda.bindings import driver as cu

        major, minor = torch.cuda.get_device_capability(device_index)
        arch = f"sm_{major}{minor}{'a' if major >= 9 else ''}"
        start = time.perf_counter()
        self._cubin = compile_source(source, arch, defines)
        self.compile_seconds = time.perf_counter() - start
        self._library = _check(
            cu.cuLibraryLoadData(self._cubin, None, None, 0, None, None, 0)
        )
        self._kernel = _check(cu.cuLibraryGetKernel(self._library, name.encode()))
        logger.debug(
            "Compiled %s %s for %s in %.2f s",
            source,
            dict(defines),
            arch,
            self.compile_seconds,
        )

    def launch(
        self,
        grid: int,
        block: int,
        smem: int,
        args: Sequence[ctypes._SimpleCData | ctypes.Structure],
        stream: torch.cuda.Stream,
    ) -> None:
        """Launch a 1-D grid on `stream`; `args` in the kernel's order."""
        from cuda.bindings import driver as cu

        ptrs = (ctypes.c_void_p * len(args))(*(ctypes.addressof(a) for a in args))
        _check(
            cu.cuLaunchKernel(
                self._kernel,
                grid,
                1,
                1,
                block,
                1,
                1,
                smem,
                stream.cuda_stream,
                ctypes.addressof(ptrs),
                0,
            )
        )

    def max_active_clusters(
        self,
        cluster: int,
        block: int,
        smem: int,
        stream: torch.cuda.Stream,
        device_index: int,
    ) -> int:
        """How many clusters of `cluster` CTAs (`block` threads and `smem`
        bytes of dynamic shared memory each) run at once on the SMs `stream`
        launches to: for a green-context stream, that context's SMs. CUDA's
        occupancy query; clusters cannot span GPCs."""
        from cuda.bindings import driver as cu

        dev = _check(cu.cuDeviceGet(device_index))
        attr = cu.CUfunction_attribute
        _check(
            cu.cuKernelSetAttribute(
                attr.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem,
                self._kernel,
                dev,
            )
        )
        _check(
            cu.cuKernelSetAttribute(
                attr.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                1,
                self._kernel,
                dev,
            )
        )
        config = cu.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = 1, cluster, 1
        config.blockDimX, config.blockDimY, config.blockDimZ = block, 1, 1
        config.sharedMemBytes = smem
        config.hStream = cu.CUstream(stream.cuda_stream)
        dims = cu.CUlaunchAttribute()
        dims.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        dims.value.clusterDim.x = 1
        dims.value.clusterDim.y = cluster
        dims.value.clusterDim.z = 1
        config.attrs = [dims]
        config.numAttrs = 1
        func = _check(cu.cuKernelGetFunction(self._kernel))
        return int(_check(cu.cuOccupancyMaxActiveClusters(func, config)))


_KERNELS: dict[tuple, CudaKernel | None] = {}
_LOCK = threading.Lock()


def get_kernel(
    source: str,
    name: str,
    defines: Sequence[tuple[str, int]],
    device_index: int,
    build: bool = True,
) -> CudaKernel | None:
    """The kernel of one configuration, compiled once per process.

    Returns None, after one warning, when it cannot be compiled or loaded
    (cuda-bindings or NVRTC missing, an NVRTC that does not know the device),
    and without `build` when it has not been compiled yet.
    """
    key = (source, name, tuple(defines), device_index)
    with _LOCK:
        if key not in _KERNELS:
            if not build:
                return None
            try:
                _KERNELS[key] = CudaKernel(source, name, defines, device_index)
            except Exception as e:  # noqa: BLE001 - callers fall back
                logger.warning(
                    "Die-local indexer: cannot build %s %s (%s: %s)",
                    source,
                    dict(defines),
                    type(e).__name__,
                    e,
                )
                _KERNELS[key] = None
        return _KERNELS[key]
