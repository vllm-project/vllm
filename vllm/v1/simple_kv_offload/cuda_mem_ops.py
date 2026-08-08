# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-level CUDA/HIP memory helpers: pinning and batch DMA transfers."""

import ctypes
import os
from typing import Any, NamedTuple

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# CUmemcpySrcAccessOrder values (CUDA driver API). STREAM(1): source read in
# stream order, safe when the source may still be written. ANY(3): source may
# be read early, only safe for a stable source (e.g. pinned host memory).
CU_MEMCPY_SRC_ACCESS_ORDER_STREAM = 1
CU_MEMCPY_SRC_ACCESS_ORDER_ANY = 3


def pin_tensor(tensor: torch.Tensor) -> None:
    """Pin a CPU tensor via cudaHostRegister.

    This bypasses PyTorch's CUDACachingHostAllocator which rounds
    every ``pin_memory=True`` allocation up to the next power of 2
    (e.g. 100 GB becomes 128 GB).
    """
    err = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)
    if err.value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {err}")


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("id", ctypes.c_int)]


class _CUmemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_uint),
        ("srcLocHint", _CUmemLocation),
        ("dstLocHint", _CUmemLocation),
        ("flags", ctypes.c_uint),
    ]


_BATCH_MEMCPY_FUNC_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_uint,  # CUresult / hipError_t
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
)

# Resolved lazily on first use: (entry point, numAttrs to pass).
_batch_memcpy: tuple[Any, int] | None = None

# Max copy descriptors per batch call, resolved lazily on first use.
_max_batch_descriptors: int | None = None

# Safe default descriptor cap on ROCm. hipMemcpyBatchAsync GPU-faults (memory
# access fault in __amd_rocclr_copyBufferBatch, process abort/segfault) for any
# call with count > 8192 on ROCm 7.15 / MI350X (gfx950) — independent of copy
# size, direction, and numAttrs.
# 4096 keeps a 2x margin below the ceiling; the ~10% lower
# per-batch bandwidth vs 8192 is negligible since the copy is overlapped.
_ROCM_DEFAULT_MAX_BATCH_DESCRIPTORS = 4096


def _resolve_max_batch_descriptors() -> int:
    """Max copy descriptors to pass to one batch-memcpy call (0 = unlimited).

    ROCm's ``hipMemcpyBatchAsync`` faults above 8192 descriptors per call, so
    on ROCm we cap and chunk larger transfers. CUDA's ``cuMemcpyBatchAsync``
    handles arbitrary counts and is left uncapped. Set
    ``VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS`` (>0) to override on any platform.
    """
    global _max_batch_descriptors
    if _max_batch_descriptors is None:
        override = os.getenv("VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS")
        val: int | None = None
        if override is not None:
            try:
                parsed = int(override)
            except ValueError:
                logger.warning(
                    "Ignoring invalid VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS=%r",
                    override,
                )
            else:
                if parsed > 0:
                    val = parsed
        if val is None:
            val = (
                _ROCM_DEFAULT_MAX_BATCH_DESCRIPTORS if current_platform.is_rocm() else 0
            )
        _max_batch_descriptors = val
    return _max_batch_descriptors


def _resolve_batch_memcpy() -> tuple[Any, int]:
    """Resolve the batch-memcpy entry point and its ``numAttrs`` (one-time).

    CUDA uses ``cuMemcpyBatchAsync``; ROCm uses ``hipMemcpyBatchAsync``.
    Raises ``RuntimeError`` if the symbol is unavailable (old CUDA driver,
    ROCm < 7.1, unusual install).
    """
    if current_platform.is_rocm():
        try:
            lib = _load_hip_runtime()
            fn = lib.hipMemcpyBatchAsync
        except (OSError, AttributeError) as e:
            raise RuntimeError(
                "hipMemcpyBatchAsync is unavailable in this ROCm install; "
                "SimpleCPUOffloadConnector requires ROCm 7.1+."
            ) from e
        fn.restype = ctypes.c_uint
        fn.argtypes = [
            ctypes.c_void_p,  # dsts
            ctypes.c_void_p,  # srcs
            ctypes.c_void_p,  # sizes
            ctypes.c_size_t,  # count
            ctypes.c_void_p,  # attrs
            ctypes.c_void_p,  # attrIdxs
            ctypes.c_size_t,  # numAttrs
            ctypes.c_void_p,  # failIdx
            ctypes.c_void_p,  # stream
        ]
        return fn, _rocm_num_attrs(lib)

    from cuda.bindings import driver as drv

    err, ptr, _ = drv.cuGetProcAddress(b"cuMemcpyBatchAsync", 12080, 0)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuGetProcAddress(cuMemcpyBatchAsync) failed: {err}")
    return _BATCH_MEMCPY_FUNC_TYPE(ptr), 1


def _load_hip_runtime() -> ctypes.CDLL:
    """Load ``libamdhip64``, tolerating installs without the devel symlink.

    The unversioned ``libamdhip64.so`` only ships with the ROCm devel package;
    runtime-only and wheel-packaged ROCm installs provide just the versioned
    soname. ``dlopen`` returns the already-mapped library when asked for a
    soname the process has loaded — torch loads HIP at import — so the
    versioned names resolve even when they are not on the loader search path.
    """
    errors = []
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        try:
            return ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            errors.append(f"{name}: {e}")
    raise OSError("could not load the HIP runtime: " + "; ".join(errors))


def _rocm_num_attrs(lib: ctypes.CDLL) -> int:
    """``numAttrs`` for ``hipMemcpyBatchAsync`` on the running HIP runtime."""
    ver = ctypes.c_int(0)
    try:
        if lib.hipRuntimeGetVersion(ctypes.byref(ver)) != 0:
            ver.value = 0
    except (OSError, AttributeError):
        ver.value = 0
    return _num_attrs_for_hip_version(ver.value)


def _num_attrs_for_hip_version(version: int) -> int:
    """``numAttrs`` for ``hipMemcpyBatchAsync`` given a HIP runtime version int.

    ROCm 7.2.1-7.2.3 reject ``numAttrs > 0`` (ROCm/clr @ rocm-7.2.1
    hipamd/src/hip_memory.cpp:2819-2822); 7.13+ accept it. ``version`` 0
    (unknown) yields the conservative 0.
    """
    # HIP encodes version as major*10_000_000 + minor*100_000 + patch.
    major, minor = version // 10_000_000, (version // 100_000) % 100
    return 1 if (major, minor) >= (7, 13) else 0


class BatchMemcpyParams(NamedTuple):
    src_bases: np.ndarray  # [num_layers] uint64 — data_ptr per layer
    dst_bases: np.ndarray  # [num_layers] uint64
    bpb: np.ndarray  # [num_layers] uint64 — bytes per block
    num_layers: int
    # One attributes entry carrying srcAccessOrder. Ignored when num_attrs is
    # 0, which is what ROCm runtimes older than 7.13 require (see
    # _num_attrs_for_hip_version).
    attrs: _CUmemcpyAttributes
    attrs_idx: ctypes.c_size_t
    num_attrs: int
    # NOTE: cuMemcpyBatchAsync_v2() removed fail_idx field, but we use
    # cuMemcpyBatchAsync() with fail_idx for backward compatibility
    fail_idx: ctypes.c_size_t
    stream_handle: int  # raw cudaStream_t / CUstream


def build_params(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
    src_access_order: int = CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
) -> BatchMemcpyParams:
    global _batch_memcpy
    if _batch_memcpy is None:
        _batch_memcpy = _resolve_batch_memcpy()
    _, num_attrs = _batch_memcpy

    assert list(src_caches.keys()) == list(dst_caches.keys())
    src_tensors = list(src_caches.values())
    dst_tensors = list(dst_caches.values())

    src_bases, dst_bases, bpb = [], [], []
    for s, d in zip(src_tensors, dst_tensors):
        s_bpb = s.stride(0) * s.element_size()
        assert s_bpb == d.stride(0) * d.element_size()
        src_bases.append(s.data_ptr())
        dst_bases.append(d.data_ptr())
        bpb.append(s_bpb)

    attrs = _CUmemcpyAttributes(srcAccessOrder=src_access_order)

    return BatchMemcpyParams(
        src_bases=np.array(src_bases, dtype=np.uint64),
        dst_bases=np.array(dst_bases, dtype=np.uint64),
        bpb=np.array(bpb, dtype=np.uint64),
        num_layers=len(src_tensors),
        attrs=attrs,
        attrs_idx=ctypes.c_size_t(0),
        num_attrs=num_attrs,
        fail_idx=ctypes.c_size_t(0),
        stream_handle=stream.cuda_stream,
    )


def copy_blocks(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    params: BatchMemcpyParams,
) -> None:
    """Copy blocks via cuMemcpyBatchAsync / hipMemcpyBatchAsync."""
    n = len(src_block_ids)
    if n == 0:
        return

    assert _batch_memcpy is not None, "build_params() must run before copy_blocks()"
    fn, _ = _batch_memcpy

    src_ids = np.array(src_block_ids, dtype=np.uint64)
    dst_ids = np.array(dst_block_ids, dtype=np.uint64)

    src_all = (
        params.src_bases[:, None] + src_ids[None, :] * params.bpb[:, None]
    ).ravel()
    dst_all = (
        params.dst_bases[:, None] + dst_ids[None, :] * params.bpb[:, None]
    ).ravel()
    sz_all = np.repeat(params.bpb, n)
    total = n * params.num_layers

    # Chunk on ROCm (hipMemcpyBatchAsync faults above 8192 descriptors/call);
    # CUDA is uncapped (max_desc == 0), so it issues a single call as before.
    max_desc = _resolve_max_batch_descriptors()
    step = total if max_desc <= 0 else max_desc
    for off in range(0, total, step):
        cnt = min(step, total - off)
        err = fn(
            dst_all[off : off + cnt].ctypes.data,
            src_all[off : off + cnt].ctypes.data,
            sz_all[off : off + cnt].ctypes.data,
            cnt,
            ctypes.addressof(params.attrs),
            ctypes.byref(params.attrs_idx),
            params.num_attrs,
            ctypes.byref(params.fail_idx),
            params.stream_handle,
        )
        if err != 0:
            raise RuntimeError(
                f"batch memcpy failed: err={err} failIdx={params.fail_idx.value}"
            )
