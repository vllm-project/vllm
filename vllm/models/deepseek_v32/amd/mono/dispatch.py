# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter#6173 at b02df0db8 (MIT License),
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# aiter/ops/flydsl/kernels/glm5_mono/dispatch.py

"""Construction-time mode selection for native model MonoKernels."""

from __future__ import annotations

from vllm.models.deepseek_v32.amd.mono.config import GLM5_GRAPH_BATCHES, glm5_tp_config

MODES = ("off", "auto", "mono", "staged")
SAMPLES = (4, 8)


class MonoUnsupported(Exception):
    """The loaded model or runtime state cannot use the native path."""


def tp_uniform_local_validation(
    error: Exception | None,
    *,
    group,
    world_size: int,
    context: str,
) -> None:
    """Make a local adapter validation result uniform across the TP group."""
    local = None if error is None else f"{type(error).__name__}: {error}"
    if world_size == 1:
        reports = [local]
    else:
        import torch.distributed as dist

        reports = [None] * world_size
        dist.all_gather_object(reports, local, group=group)
    failed = [
        (rank, report) for rank, report in enumerate(reports) if report is not None
    ]
    if failed:
        detail = "; ".join(f"rank {rank}: {report}" for rank, report in failed)
        raise MonoUnsupported(f"{context}: {detail}")


def normalize_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode not in MODES:
        raise ValueError(
            f"MonoKernel mode must be one of {', '.join(MODES)}, got {mode!r}"
        )
    return mode


def glm52_native_config(
    *, samples: int, tp_size: int, kv_cache_dtype: str, mtp: bool, query_length: int
):
    """Return the canonical eligible GLM-5 shard geometry, or ``None``."""
    try:
        config = glm5_tp_config(tp_size)
    except ValueError:
        return None
    if (
        query_length <= 0
        or samples % query_length
        or samples // query_length not in GLM5_GRAPH_BATCHES
    ):
        return None
    if query_length in (5, 6):
        return config if tp_size == 4 and kv_cache_dtype == "fp8" and mtp else None
    if query_length == 1:
        return (
            config
            if tp_size == 8
            and kv_cache_dtype == "bf16"
            and not mtp
            and samples in SAMPLES
            else None
        )
    return None


def select_backend(
    model: str,
    mode: str,
    *,
    samples: int,
    tp_size: int,
    kv_cache_dtype: str,
    native: bool = True,
    decode: bool = True,
    mtp: bool = False,
    q: int = 1,
    dpa: bool = False,
    dcp: bool = False,
    plugin: bool = False,
    is_kda: bool = True,
    has_moe: bool = True,
    external_indexer: bool = True,
    cache_layout: str = "atom",
    query_length: int = 1,
) -> str | None:
    """Return a production backend name, or ``None`` for baseline fallback."""
    mode = normalize_mode(mode)
    if mode == "off" or not native or not decode or dpa or dcp or plugin:
        return None
    if model == "glm52":
        if (
            glm52_native_config(
                samples=samples,
                tp_size=tp_size,
                kv_cache_dtype=kv_cache_dtype,
                mtp=mtp,
                query_length=query_length,
            )
            and has_moe
            and external_indexer
            and cache_layout == "atom"
            and mode in ("auto", "mono")
        ):
            return "mono"
        return None
    if model == "kimi_k3":
        if (
            tp_size != 8
            or kv_cache_dtype not in ("bf16", "fp8")
            or samples <= 0
            or q <= 0
            or samples % q
            or mtp != (q > 1)
        ):
            return None
        if is_kda and has_moe:
            if mode in ("auto", "mono"):
                return "mono"
            if mode == "staged":
                return "staged"
    return None
