# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)

_CUDA_INPUT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_CUDA_KV_CACHE_DTYPES = {
    "auto",
    "float16",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
}
_NATIVE_CACHE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@cache
def _has_cuda_diffkv_op() -> bool:
    return hasattr(torch.ops, "_C_cache_ops") and hasattr(
        torch.ops._C_cache_ops, "reshape_and_cache_flash_diffkv"
    )


def _is_supported_cuda_cache_dtype(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
) -> bool:
    if key.dtype not in _CUDA_INPUT_DTYPES or value.dtype != key.dtype:
        return False
    if kv_cache_dtype not in _CUDA_KV_CACHE_DTYPES:
        return False

    if is_quantized_kv_cache(kv_cache_dtype):
        return kv_cache.dtype == torch.uint8

    if kv_cache.dtype != key.dtype:
        return False
    expected_dtype = _NATIVE_CACHE_DTYPES.get(kv_cache_dtype)
    return expected_dtype is None or kv_cache.dtype == expected_dtype


def _has_supported_cuda_scales(
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> bool:
    return k_scale.numel() == 1 and v_scale.numel() == 1


def _has_supported_cuda_layout(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> bool:
    if key.ndim != 3 or value.ndim != 3 or kv_cache.ndim != 4:
        return False
    if slot_mapping.ndim != 1 or slot_mapping.dtype != torch.long:
        return False
    if key.shape[0] < slot_mapping.shape[0] or value.shape[0] < slot_mapping.shape[0]:
        return False
    if key.shape[:2] != value.shape[:2]:
        return False
    head_stride = key.shape[2] + value.shape[2]
    if kv_cache.shape[2] != key.shape[1] or kv_cache.shape[3] != head_stride:
        return False
    return kv_cache.stride(2) == head_stride


def _can_use_cuda_diffkv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> bool:
    return (
        current_platform.is_cuda()
        and _has_cuda_diffkv_op()
        and _is_supported_cuda_cache_dtype(key, value, kv_cache, kv_cache_dtype)
        and _has_supported_cuda_scales(k_scale, v_scale)
        and _has_supported_cuda_layout(key, value, kv_cache, slot_mapping)
    )


def reshape_and_cache_flash_diffkv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    backend = envs.VLLM_DIFFKV_CACHE_BACKEND.lower()
    use_cuda = _can_use_cuda_diffkv_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    if backend == "cuda" and not use_cuda:
        raise NotImplementedError(
            "CUDA DiffKV reshape-and-cache does not support this input. "
            "Use VLLM_DIFFKV_CACHE_BACKEND=triton or auto to allow fallback."
        )
    if backend == "cuda" or (backend == "auto" and use_cuda):
        ops.reshape_and_cache_flash_diffkv(
            key,
            value,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        return

    triton_reshape_and_cache_flash_diffkv(
        key,
        value,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
