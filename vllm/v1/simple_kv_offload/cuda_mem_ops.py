# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-level CUDA/HIP memory helpers: pinning and batch DMA transfers."""

import ctypes
from typing import Any, NamedTuple

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


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

# Resolved lazily on first use.
_batch_memcpy_fn: Any = None


def _resolve_batch_memcpy():
    """Resolve the platform batch-memcpy entry point (one-time).

    * CUDA: ``cuMemcpyBatchAsync`` via ``cuGetProcAddress`` (uses
      srcAccessOrder=STREAM via one attributes entry).
    * ROCm: ``hipMemcpyBatchAsync`` from libamdhip64 (ROCm 7.1+). ROCm
      7.2.1 or 7.2.2 rejects any call with ``numAttrs > 0``
      (see ROCm/clr @ rocm-7.2.1 hipamd/src/hip_memory.cpp:2819-2822), so
      we call with ``numAttrs=0``.

    Raises ``RuntimeError`` if the symbol is unavailable (older CUDA
    driver, ROCm < 7.1, unusual install). The connector requires the
    batch API.
    """
    if current_platform.is_rocm():
        try:
            lib = ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
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
        return fn

    from cuda.bindings import driver as drv

    err, ptr, _ = drv.cuGetProcAddress(b"cuMemcpyBatchAsync", 12080, 0)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuGetProcAddress(cuMemcpyBatchAsync) failed: {err}")
    return _BATCH_MEMCPY_FUNC_TYPE(ptr)


class BatchMemcpyParams(NamedTuple):
    src_bases: np.ndarray  # [num_layers] uint64 — data_ptr per layer
    dst_bases: np.ndarray  # [num_layers] uint64
    bpb: np.ndarray  # [num_layers] uint64 — bytes per block
    num_layers: int
    # CUDA only: one attributes entry with srcAccessOrder=ANY. Unused on
    # ROCm (7.2.1 or 7.2.2) because the current runtime rejects numAttrs > 0.
    attrs: _CUmemcpyAttributes
    attrs_idx: ctypes.c_size_t
    # NOTE: cuMemcpyBatchAsync_v2() removed fail_idx field, but we use
    # cuMemcpyBatchAsync() with fail_idx for backward compatibility
    fail_idx: ctypes.c_size_t
    stream_handle: int  # raw cudaStream_t / CUstream


def build_params(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
) -> BatchMemcpyParams:
    global _batch_memcpy_fn
    if _batch_memcpy_fn is None:
        _batch_memcpy_fn = _resolve_batch_memcpy()

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

    # ``srcAccessOrder=3`` == CU_MEMCPY_SRC_ACCESS_ORDER_ANY /
    # hipMemcpySrcAccessOrderAny. See
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g6f1ff58e3065df3eb4b573dba77ad31f  # noqa: E501
    attrs = _CUmemcpyAttributes(srcAccessOrder=3)

    return BatchMemcpyParams(
        src_bases=np.array(src_bases, dtype=np.uint64),
        dst_bases=np.array(dst_bases, dtype=np.uint64),
        bpb=np.array(bpb, dtype=np.uint64),
        num_layers=len(src_tensors),
        attrs=attrs,
        attrs_idx=ctypes.c_size_t(0),
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

    # ROCm 7.2.1/7.2.2 rejects any call with numAttrs>0 (hipMemcpyBatchAsync
    # hipamd/src/hip_memory.cpp:2819-2822); CUDA uses one attrs entry so
    # srcAccessOrder is honored. attrs / attrsIdxs are ignored when
    # numAttrs==0, so we pass the same values from both paths.
    num_attrs = 0 if current_platform.is_rocm() else 1
    err = _batch_memcpy_fn(
        dst_all.ctypes.data,
        src_all.ctypes.data,
        sz_all.ctypes.data,
        total,
        ctypes.addressof(params.attrs),
        ctypes.byref(params.attrs_idx),
        num_attrs,
        ctypes.byref(params.fail_idx),
        params.stream_handle,
    )
    if err != 0:
        raise RuntimeError(
            f"batch memcpy failed: err={err} failIdx={params.fail_idx.value}"
        )
