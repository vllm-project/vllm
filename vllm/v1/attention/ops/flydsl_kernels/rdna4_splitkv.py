# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public router for the RDNA4 SplitKV paged-attention kernel family."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any
from weakref import ReferenceType, ref

import torch

from .rdna4_splitkv_common import HEAD_DIM
from .rdna4_splitkv_gqa16_d128 import compile_gqa_stage
from .rdna4_splitkv_native_d128 import compile_native_d128_gqa16_stage
from .rdna4_splitkv_reduce import _launch_reduce
from .rdna4_splitkv_wave8 import compile_wave8_stage
from .runtime import get_compiled_static_tensors as _get_compiled_static_tensors
from .runtime import run_compiled as _run_compiled


@dataclass
class _Wave8FusedLaunch:
    query_ref: ReferenceType[torch.Tensor]
    tensors: tuple[torch.Tensor, ...]
    params: tuple
    compiled: Any
    k_scale: torch.Tensor
    v_scale: torch.Tensor


_WAVE8_FUSED_CACHE_SIZE = 128
_WAVE8_FUSED_CACHE: dict[int, _Wave8FusedLaunch] = {}
_WAVE8_COUNTER_CACHE_SIZE = 32
_WAVE8_COUNTER_CACHE: dict[tuple[int, int], torch.Tensor] = {}
_WAVE8_COUNTER_CACHE_LOCK = Lock()


def _drop_wave8_fused_entry(query_ref: ReferenceType[torch.Tensor], key: int) -> None:
    entry = _WAVE8_FUSED_CACHE.get(key)
    if entry is not None and entry.query_ref is query_ref:
        _WAVE8_FUSED_CACHE.pop(key, None)


def _wave8_split_counters(
    query: torch.Tensor,
    stream: torch.cuda.Stream,
    rows: int,
) -> torch.Tensor:
    device_index = query.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    key = (device_index, int(stream.cuda_stream))
    required = rows * 16
    counters = _WAVE8_COUNTER_CACHE.get(key)
    if counters is not None and counters.numel() >= required:
        return counters
    with _WAVE8_COUNTER_CACHE_LOCK:
        counters = _WAVE8_COUNTER_CACHE.get(key)
        if counters is None or counters.numel() < required:
            if (
                key not in _WAVE8_COUNTER_CACHE
                and len(_WAVE8_COUNTER_CACHE) >= _WAVE8_COUNTER_CACHE_SIZE
            ):
                _WAVE8_COUNTER_CACHE.pop(next(iter(_WAVE8_COUNTER_CACHE)))
            counters = torch.zeros(required, dtype=torch.int32, device=query.device)
            _WAVE8_COUNTER_CACHE[key] = counters
    return counters


class SplitKVRoute(str, Enum):
    """Validated RDNA4 kernel specializations."""

    NATIVE_D128_GQA16_DIRECT = "native_d128_gqa16_direct"
    NATIVE_D128_GQA16_LDS = "native_d128_gqa16_lds"
    D128_GQA16_TILE32 = "d128_gqa16_tile32"
    D256_GQA4_8_TILE32 = "d256_gqa4_8_tile32"
    D256_GQA16_TILE32 = "d256_gqa16_tile32"
    D128_GQA8_TILE32 = "d128_gqa8_tile32"
    GENERIC_TILE32 = "generic_tile32"
    WAVE8 = "wave8"


@dataclass(frozen=True)
class SplitKVConfig:
    """A promoted kernel route selected from the runtime tensor contract."""

    route: SplitKVRoute


def select_kernel_config(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
) -> SplitKVConfig | None:
    """Select a specialization safe for arbitrary values in the API contract.

    Only routes validated against unrestricted-value accuracy are selectable.
    """
    if query.ndim != 3 or key_cache.ndim != 5:
        return None
    num_kv_heads = int(key_cache.shape[1])
    if num_kv_heads <= 0 or int(query.shape[1]) % num_kv_heads:
        return None

    gqa = int(query.shape[1]) // num_kv_heads
    head_size = int(query.shape[2])
    page_size = int(key_cache.shape[3])
    batch_size = int(seq_lens.numel())
    page_aligned = page_size >= 8 and page_size % 8 == 0

    if (
        query.dtype in (torch.bfloat16, torch.float16)
        and key_cache.dtype == query.dtype
        and head_size == 128
        and batch_size >= 8
        and gqa == 16
        and page_size >= 16
        and page_size % 16 == 0
    ):
        route = (
            SplitKVRoute.NATIVE_D128_GQA16_DIRECT
            if batch_size == 8
            else SplitKVRoute.NATIVE_D128_GQA16_LDS
            if batch_size >= 16
            else None
        )
        if route is not None:
            if max_seq_len >= 8192:
                return SplitKVConfig(SplitKVRoute.D128_GQA16_TILE32)
            return SplitKVConfig(route)

    if (
        query.dtype in (torch.bfloat16, torch.float16)
        and (
            key_cache.dtype == query.dtype
            or key_cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        )
        and head_size == 128
        and batch_size == 1
        and gqa == 16
        and page_size >= 16
        and page_size % 16 == 0
    ):
        return SplitKVConfig(SplitKVRoute.D128_GQA16_TILE32)

    if (
        query.dtype in (torch.bfloat16, torch.float16)
        and key_cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        and head_size == 128
        and batch_size < 8
        and gqa == 8
        and page_size >= 16
        and page_size % 16 == 0
    ):
        return SplitKVConfig(SplitKVRoute.D128_GQA8_TILE32)

    supported_d256_cache = key_cache.dtype == query.dtype or key_cache.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    )
    if (
        query.dtype in (torch.bfloat16, torch.float16)
        and supported_d256_cache
        and head_size == 256
        and batch_size < 8
        and gqa == 16
        and page_size >= 16
        and page_size % 16 == 0
    ):
        return SplitKVConfig(SplitKVRoute.D256_GQA16_TILE32)

    native_d256_gqa4_8 = key_cache.dtype == query.dtype and (
        gqa == 4 or (1 < batch_size < 8 and gqa == 8)
    )
    fnuz_d256_gqa8 = (
        key_cache.dtype == torch.float8_e4m3fnuz and batch_size == 1 and gqa == 8
    )
    if (
        query.dtype in (torch.bfloat16, torch.float16)
        and head_size == 256
        and page_size >= 16
        and page_size % 16 == 0
        and (native_d256_gqa4_8 or fnuz_d256_gqa8)
    ):
        return SplitKVConfig(SplitKVRoute.D256_GQA4_8_TILE32)

    generic_d256 = head_size == 256 and 5 <= gqa <= 15
    generic_d128_native_gqa8 = (
        head_size == 128 and gqa == 8 and key_cache.dtype == query.dtype
    )
    if (
        query.dtype in (torch.bfloat16, torch.float16)
        and supported_d256_cache
        and page_size >= 16
        and page_size % 16 == 0
        and (generic_d256 or generic_d128_native_gqa8)
    ):
        return SplitKVConfig(SplitKVRoute.GENERIC_TILE32)

    # Per-query-head Wave8 repeats native KV reads as GQA/batch grow. The
    # grouped Triton fallback wins these long-context shapes on gfx1201.
    if max_seq_len >= 2048 and (
        (key_cache.dtype == query.dtype and (gqa >= 3 or (gqa == 2 and batch_size > 1)))
        or (head_size == 128 and gqa == 4 and batch_size > 1)
    ):
        return None

    if (
        query.dtype in (torch.bfloat16, torch.float16)
        and (
            key_cache.dtype == query.dtype
            or key_cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        )
        and head_size in (128, 256)
        and batch_size < 8
        and 1 <= gqa <= 4
        and page_aligned
    ):
        return SplitKVConfig(SplitKVRoute.WAVE8)

    return None


def _run_d256_tile32(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    splits: int,
    scale: float,
) -> torch.Tensor:
    """Run the accuracy-validated native-query D256 Tile32 stage."""

    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"query must be bfloat16 or float16, got {query.dtype}")
    if key_cache.dtype != value_cache.dtype:
        raise ValueError("key_cache and value_cache must have the same dtype")
    if key_cache.dtype not in (
        query.dtype,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ):
        raise ValueError(f"unsupported cache dtype {key_cache.dtype}")
    if int(query.shape[2]) != HEAD_DIM:
        raise ValueError(f"head dimension must be {HEAD_DIM}, got {query.shape[2]}")
    num_kv_heads = int(key_cache.shape[1])
    query_group_size = int(query.shape[1]) // num_kv_heads
    if not 1 <= query_group_size <= 16:
        raise ValueError(f"query group size must be in [1, 16], got {query_group_size}")
    if k_scale.ndim == 0:
        k_scale = k_scale.reshape(1)
    if v_scale.ndim == 0:
        v_scale = v_scale.reshape(1)

    kv_dtype = (
        "fp8fnuz"
        if key_cache.dtype == torch.float8_e4m3fnuz
        else (
            "fp8"
            if key_cache.element_size() == 1
            else "fp16"
            if key_cache.dtype == torch.float16
            else "bf16"
        )
    )
    stage = compile_gqa_stage(
        dtype="fp16" if query.dtype == torch.float16 else "bf16",
        kv_dtype=kv_dtype,
        head_dim=HEAD_DIM,
        splits=int(splits),
        num_kv_heads=num_kv_heads,
        query_group_size=query_group_size,
        page_size=int(key_cache.shape[3]),
        softmax_scale=float(scale),
    )
    stream = torch.cuda.current_stream(query.device)
    _run_compiled(
        stage,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_loc,
        k_scale,
        v_scale,
        mid_out,
        mid_lse,
        int(seq_lens.numel()),
        int(block_tables.stride(0)),
        int(query.stride(0)),
        int(query.stride(1)),
        *map(int, key_cache.stride()),
        *map(int, value_cache.stride()),
        *map(int, mid_out.stride()[:3]),
        *map(int, mid_lse.stride()),
        stream,
    )
    _launch_reduce(
        query,
        seq_lens,
        query_start_loc,
        output,
        mid_out,
        mid_lse,
        int(splits),
        1,
        stream,
    )
    return output


def _run_native_d128_gqa16(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    splits: int,
    scale: float,
    schedule: str,
) -> torch.Tensor:
    """Run a four-wave native D128/GQA16 schedule."""

    if schedule not in ("direct", "lds", "tile32_native"):
        raise ValueError(f"unsupported D128 schedule: {schedule!r}")

    if k_scale.ndim == 0:
        k_scale = k_scale.reshape(1)
    if v_scale.ndim == 0:
        v_scale = v_scale.reshape(1)
    num_kv_heads = int(key_cache.shape[1])
    query_group_size = int(query.shape[1]) // num_kv_heads
    page_size = int(key_cache.shape[3])
    dtype = "fp16" if query.dtype == torch.float16 else "bf16"
    kv_dtype = (
        "fp8fnuz"
        if key_cache.dtype == torch.float8_e4m3fnuz
        else (
            "fp8"
            if key_cache.element_size() == 1
            else "fp16"
            if key_cache.dtype == torch.float16
            else "bf16"
        )
    )
    compile_stage = (
        compile_gqa_stage
        if schedule == "tile32_native"
        else compile_native_d128_gqa16_stage
    )
    compile_kwargs = dict(
        dtype=dtype,
        kv_dtype=kv_dtype,
        splits=int(splits),
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        softmax_scale=float(scale),
    )
    if schedule == "tile32_native":
        compile_kwargs["head_dim"] = 128
        compile_kwargs["query_group_size"] = query_group_size
    else:
        compile_kwargs["schedule"] = schedule
    stage = compile_stage(**compile_kwargs)
    stream = torch.cuda.current_stream(query.device)
    _run_compiled(
        stage,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_loc,
        k_scale,
        v_scale,
        mid_out,
        mid_lse,
        int(seq_lens.numel()),
        int(block_tables.stride(0)),
        int(query.stride(0)),
        int(query.stride(1)),
        *map(int, key_cache.stride()),
        *map(int, value_cache.stride()),
        *map(int, mid_out.stride()[:3]),
        *map(int, mid_lse.stride()),
        stream,
    )
    _launch_reduce(
        query,
        seq_lens,
        query_start_loc,
        output,
        mid_out,
        mid_lse,
        int(splits),
        page_size if schedule == "direct" else 1,
        stream,
    )
    return output


def _run_wave8(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    splits: int,
    scale: float,
) -> torch.Tensor:
    """Launch the fused eight-wave per-query-head stage."""

    original_k_scale = k_scale
    original_v_scale = v_scale
    stream = torch.cuda.current_stream(query.device)
    split_counters = _wave8_split_counters(
        query,
        stream,
        int(seq_lens.numel()) * int(query.shape[1]),
    )
    cache_params = (int(splits), float(scale))
    cache_tensors = (
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_loc,
        original_k_scale,
        original_v_scale,
        output,
        mid_out,
        mid_lse,
        split_counters,
    )
    cached = _WAVE8_FUSED_CACHE.get(id(query))
    if (
        cached is not None
        and cached.query_ref() is query
        and cached.params == cache_params
        and all(
            actual is expected
            for actual, expected in zip(cache_tensors, cached.tensors)
        )
    ):
        cached.compiled(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            query_start_loc,
            cached.k_scale,
            cached.v_scale,
            mid_out,
            mid_lse,
            output,
            split_counters,
            int(seq_lens.numel()),
            stream,
        )
        return output

    if k_scale.ndim == 0:
        k_scale = k_scale.reshape(1)
    if v_scale.ndim == 0:
        v_scale = v_scale.reshape(1)
    num_query_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[1])
    layout_tensors = (
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_loc,
        k_scale,
        v_scale,
        mid_out,
        mid_lse,
        output,
        split_counters,
    )
    layout_key = tuple(
        (tuple(tensor.shape), tuple(tensor.stride())) for tensor in layout_tensors
    )
    stage = compile_wave8_stage(
        query_dtype="fp16" if query.dtype == torch.float16 else "bf16",
        kv_dtype=(
            "fp8fnuz"
            if key_cache.dtype == torch.float8_e4m3fnuz
            else (
                "fp8"
                if key_cache.element_size() == 1
                else "fp16"
                if key_cache.dtype == torch.float16
                else "bf16"
            )
        ),
        head_dim=int(query.shape[2]),
        splits=int(splits),
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        query_group_size=num_query_heads // num_kv_heads,
        page_size=int(key_cache.shape[3]),
        softmax_scale=float(scale),
        layout_key=layout_key,
    )
    launch_args = (
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_loc,
        k_scale,
        v_scale,
        mid_out,
        mid_lse,
        output,
        split_counters,
        int(seq_lens.numel()),
        stream,
    )
    compiled = _get_compiled_static_tensors(stage, *launch_args)
    if (
        id(query) not in _WAVE8_FUSED_CACHE
        and len(_WAVE8_FUSED_CACHE) >= _WAVE8_FUSED_CACHE_SIZE
    ):
        _WAVE8_FUSED_CACHE.pop(next(iter(_WAVE8_FUSED_CACHE)))
    query_id = id(query)

    def drop_entry(dead_ref: ReferenceType[torch.Tensor]) -> None:
        _drop_wave8_fused_entry(dead_ref, query_id)

    query_ref = ref(query, drop_entry)
    _WAVE8_FUSED_CACHE[query_id] = _Wave8FusedLaunch(
        query_ref=query_ref,
        tensors=cache_tensors,
        params=cache_params,
        compiled=compiled,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    compiled(*launch_args)
    return output


def rdna4_splitkv_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    splits: int,
    scale: float,
    max_seq_len: int,
    config: SplitKVConfig | None = None,
) -> SplitKVConfig:
    """Validate, select, and launch an RDNA4 SplitKV specialization."""
    selected_config = select_kernel_config(query, key_cache, seq_lens, max_seq_len)
    if selected_config is None:
        raise ValueError("RDNA4 FlyDSL SplitKV does not support this configuration")
    if config is not None and config != selected_config:
        raise ValueError("RDNA4 FlyDSL SplitKV configuration does not match the inputs")
    config = selected_config

    args = (
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_loc,
        k_scale,
        v_scale,
        output,
    )
    if config.route == SplitKVRoute.NATIVE_D128_GQA16_DIRECT:
        _run_native_d128_gqa16(
            *args, mid_out, mid_lse, splits, scale, schedule="direct"
        )
    elif config.route == SplitKVRoute.NATIVE_D128_GQA16_LDS:
        _run_native_d128_gqa16(*args, mid_out, mid_lse, splits, scale, schedule="lds")
    elif config.route in (
        SplitKVRoute.D128_GQA16_TILE32,
        SplitKVRoute.D128_GQA8_TILE32,
    ):
        _run_native_d128_gqa16(
            *args, mid_out, mid_lse, splits, scale, schedule="tile32_native"
        )
    elif config.route in (
        SplitKVRoute.D256_GQA4_8_TILE32,
        SplitKVRoute.D256_GQA16_TILE32,
    ):
        _run_d256_tile32(*args, mid_out, mid_lse, splits, scale)
    elif config.route == SplitKVRoute.GENERIC_TILE32:
        if int(query.shape[2]) == 256:
            _run_d256_tile32(*args, mid_out, mid_lse, splits, scale)
        else:
            _run_native_d128_gqa16(
                *args,
                mid_out,
                mid_lse,
                splits,
                scale,
                schedule="tile32_native",
            )
    elif config.route == SplitKVRoute.WAVE8:
        _run_wave8(*args, mid_out, mid_lse, splits, scale)
    else:
        raise AssertionError(f"unhandled RDNA4 SplitKV route: {config.route}")
    return config


__all__ = [
    "SplitKVConfig",
    "SplitKVRoute",
    "rdna4_splitkv_paged_attention",
    "select_kernel_config",
]
