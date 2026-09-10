# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""RDNA4 backend selection for SplitKV paged attention."""

from __future__ import annotations

import math
from functools import lru_cache
from inspect import signature
from typing import Any

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)


def _can_use_rdna4_splitkv_decode(
    *,
    query_dtype: torch.dtype,
    key_cache_dtype: torch.dtype,
    value_cache_dtype: torch.dtype,
    kv_quant_mode: KVQuantMode,
    is_e4m3_kv_cache: bool,
    head_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi_slopes: bool,
    sliding_window: int,
    has_sinks: bool,
    has_output_scale: bool,
    is_gfx1x: bool,
    is_gfx12x: bool,
) -> bool:
    """Return whether the validated SplitKV decode route can be used."""
    if (
        query_dtype not in (torch.float16, torch.bfloat16)
        or key_cache_dtype != value_cache_dtype
        or head_size not in (128, 256)
        or num_kv_heads <= 0
        or num_query_heads % num_kv_heads != 0
        or not 1 <= num_query_heads // num_kv_heads <= 16
        or use_alibi_slopes
        or sliding_window != 0
        or has_sinks
        or has_output_scale
    ):
        return False

    if kv_quant_mode == KVQuantMode.FP8_PER_TENSOR:
        e4m3_dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        return is_gfx12x and is_e4m3_kv_cache and key_cache_dtype in e4m3_dtypes
    if kv_quant_mode != KVQuantMode.NONE:
        return False

    return is_gfx1x and key_cache_dtype == query_dtype


@lru_cache(maxsize=1)
def _load_flydsl_splitkv() -> tuple[Any, Any]:
    """Load the in-tree kernels without making FlyDSL a vLLM dependency."""
    import flydsl.compiler  # noqa: F401
    import flydsl.expr as fx

    scalar_fp8_params = tuple(signature(fx.rocdl.cvt_f32_fp8).parameters)
    if scalar_fp8_params[:2] != ("src", "byte_sel"):
        raise RuntimeError("FlyDSL lacks the scalar FP8 conversion API")
    required_memory_apis = (
        (fx.llvm, "memory_fence"),
        (fx.llvm, "atomic_add"),
        (fx.llvm, "generic_store"),
        (fx, "AtomicOrdering"),
        (fx.rocdl, "SyncScope"),
    )
    missing = [
        f"{module.__name__}.{name}"
        for module, name in required_memory_apis
        if not hasattr(module, name)
    ]
    if missing:
        raise RuntimeError(f"FlyDSL lacks required ordered memory APIs: {missing}")

    from .flydsl_kernels.rdna4_splitkv import (
        rdna4_splitkv_paged_attention,
        select_kernel_config,
    )

    return select_kernel_config, rdna4_splitkv_paged_attention


@lru_cache(maxsize=1)
def is_rdna4_flydsl_splitkv_available() -> bool:
    """Return whether the FlyDSL compiler and in-tree kernels import."""
    try:
        _load_flydsl_splitkv()
    except Exception as exc:  # noqa: BLE001
        logger.warning_once(
            "RDNA4 FlyDSL SplitKV is unavailable (%s); using another backend.",
            exc,
        )
        return False
    return True


def get_rdna4_flydsl_splitkv_config(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    scale: float,
    actual_max_splits: int,
    max_seq_len: int,
    filter_by_query_len: bool,
) -> Any | None:
    """Return a validated FlyDSL configuration, or ``None`` for fallback."""
    from vllm.platforms.rocm import on_rdna4

    if (
        not on_rdna4()
        or not is_rdna4_flydsl_splitkv_available()
        or query.ndim != 3
        or key_cache.ndim != 5
        or value_cache.ndim != 4
        or block_tables.ndim != 2
        or seq_lens.ndim != 1
        or query_start_loc is None
        or query_start_loc.ndim != 1
        or not query.is_cuda
    ):
        return None
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    head_size = query.shape[2]
    page_size = key_cache.shape[3]
    cache_pack = 16 // key_cache.element_size()
    cache_groups = head_size // cache_pack
    supported_dtypes = (
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    )
    valid = (
        query.dtype in (torch.bfloat16, torch.float16)
        and output.dtype == query.dtype
        and key_cache.dtype == value_cache.dtype
        and key_cache.dtype in supported_dtypes
        and head_size in (128, 256)
        and output.shape == query.shape
        and query.shape[0] >= seq_lens.numel()
        and num_kv_heads > 0
        and num_query_heads % num_kv_heads == 0
        and page_size >= 8
        and page_size % 8 == 0
        and key_cache.shape[2:] == (cache_groups, page_size, cache_pack)
        and value_cache.shape
        == (key_cache.shape[0], num_kv_heads, head_size, page_size)
        and block_tables.shape[0] == seq_lens.numel()
        and block_tables.shape[1] * page_size >= max_seq_len
        and query_start_loc.numel() == seq_lens.numel() + 1
        and query.stride(2) == 1
        and output.stride(2) == 1
        and key_cache.stride(4) == 1
        and value_cache.stride(3) == 1
        and block_tables.stride(1) == 1
        and seq_lens.stride(0) == 1
        and query_start_loc.stride(0) == 1
        and query.device
        == output.device
        == key_cache.device
        == value_cache.device
        == block_tables.device
        == seq_lens.device
        == query_start_loc.device
        and block_tables.dtype == torch.int32
        and seq_lens.dtype == torch.int32
        and query_start_loc.dtype == torch.int32
        and filter_by_query_len
        and isinstance(k_scale, torch.Tensor)
        and isinstance(v_scale, torch.Tensor)
        and k_scale.dtype == torch.float32
        and v_scale.dtype == torch.float32
        and k_scale.device == query.device
        and v_scale.device == query.device
        and k_scale.numel() == 1
        and v_scale.numel() == 1
        and math.isfinite(scale)
        and max_seq_len > 0
        and actual_max_splits in (2, 4, 8, 16)
    )
    if not valid:
        return None
    select_kernel_config, _ = _load_flydsl_splitkv()
    return select_kernel_config(query, key_cache, seq_lens, max_seq_len)


def can_use_rdna4_flydsl_splitkv_paged_attention(**kwargs) -> bool:
    """Return whether a validated RDNA4 FlyDSL route applies."""
    return get_rdna4_flydsl_splitkv_config(**kwargs) is not None


def try_rdna4_splitkv_paged_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    scale: float,
    actual_max_splits: int,
    max_seq_len: int,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    filter_by_query_len: bool,
) -> bool:
    """Run the selected RDNA4 FlyDSL route and report whether one was used."""
    if not envs.VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL:
        return False

    flydsl_config = get_rdna4_flydsl_splitkv_config(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        output=output,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        k_scale=k_scale,
        v_scale=v_scale,
        scale=scale,
        actual_max_splits=actual_max_splits,
        max_seq_len=max_seq_len,
        filter_by_query_len=filter_by_query_len,
    )
    if flydsl_config is None:
        return False

    assert query_start_loc is not None
    assert isinstance(k_scale, torch.Tensor)
    assert isinstance(v_scale, torch.Tensor)
    _, launch = _load_flydsl_splitkv()
    launch(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_loc,
        k_scale,
        v_scale,
        output,
        mid_out,
        mid_lse,
        actual_max_splits,
        scale,
        max_seq_len,
        config=flydsl_config,
    )
    logger.info_once("Using RDNA4 FlyDSL SplitKV route: %s", flydsl_config.route.value)
    return True


__all__ = [
    "can_use_rdna4_flydsl_splitkv_paged_attention",
    "get_rdna4_flydsl_splitkv_config",
    "is_rdna4_flydsl_splitkv_available",
    "try_rdna4_splitkv_paged_attention",
]
