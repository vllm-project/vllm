# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup compilation of Kimi-K3 vision FA4 kernels."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial

import torch

from vllm.model_executor.warmup.cutedsl_warmup import (
    CuTeDSLCompileUnit,
    register_cutedsl_warmup_provider,
)
from vllm.platforms import current_platform

_FA4_TILE_SIZE = 128
_FA4_MAX_SPLITS = 128


@dataclass(frozen=True)
class KimiK3VisionFA4WarmupConfig:
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    max_batch_size: int
    max_seqlen: int


@dataclass(frozen=True)
class _FA4WarmupProbe:
    batch_size: int
    max_seqlen: int
    dispatch_key: tuple[int, bool, int | None]


def _combine_log_max_splits(head_dim: int, num_splits: int) -> int:
    k_block_size = 64 if head_dim <= 64 else 128
    tile_m = 8 if k_block_size == 128 else 16
    min_log_max_splits = 5 if tile_m == 8 else 4
    return max(math.ceil(math.log2(num_splits)), min_log_max_splits)


def _get_dispatch(
    config: KimiK3VisionFA4WarmupConfig,
    *,
    batch_size: int,
    max_seqlen: int,
    num_sms: int,
) -> tuple[int, bool, int | None]:
    # These are the shape-dependent fields in FA4's forward and combine
    # compile keys for noncausal varlen MHA. Compile units still pass
    # num_splits=0 so the production selector makes the final decision.
    q_stage = 2 if max_seqlen > _FA4_TILE_SIZE else 1
    effective_q_tile = q_stage * _FA4_TILE_SIZE
    num_m_blocks = math.ceil(max_seqlen / effective_q_tile)
    num_n_blocks = math.ceil(max_seqlen / _FA4_TILE_SIZE)

    if num_n_blocks <= 4:
        num_splits = 1
    else:
        total_m_blocks = batch_size * config.num_heads * num_m_blocks
        num_splits = min(
            num_sms // total_m_blocks,
            _FA4_MAX_SPLITS,
            num_n_blocks,
        )

    is_split_kv = num_splits > 1
    combine_key = (
        _combine_log_max_splits(config.head_dim, num_splits) if is_split_kv else None
    )
    return q_stage, is_split_kv, combine_key


def _get_warmup_probes(
    config: KimiK3VisionFA4WarmupConfig,
    *,
    num_sms: int,
) -> tuple[_FA4WarmupProbe, ...]:
    if config.max_batch_size <= 0 or config.max_seqlen <= 0:
        return ()

    probes: dict[tuple[int, bool, int | None], _FA4WarmupProbe] = {}

    def add_probe(batch_size: int, max_seqlen: int) -> None:
        dispatch_key = _get_dispatch(
            config,
            batch_size=batch_size,
            max_seqlen=max_seqlen,
            num_sms=num_sms,
        )
        if dispatch_key not in probes:
            probes[dispatch_key] = _FA4WarmupProbe(
                batch_size=batch_size,
                max_seqlen=max_seqlen,
                dispatch_key=dispatch_key,
            )

    add_probe(batch_size=1, max_seqlen=1)
    if config.max_seqlen > _FA4_TILE_SIZE:
        add_probe(batch_size=1, max_seqlen=_FA4_TILE_SIZE + 1)

    max_n_blocks = math.ceil(config.max_seqlen / _FA4_TILE_SIZE)
    for num_n_blocks in range(5, max_n_blocks + 1):
        max_seqlen = min(
            (num_n_blocks - 1) * _FA4_TILE_SIZE + 1,
            config.max_seqlen,
        )
        num_m_blocks = math.ceil(max_seqlen / (2 * _FA4_TILE_SIZE))
        max_split_batch_size = min(
            config.max_batch_size,
            num_sms // (2 * config.num_heads * num_m_blocks),
        )
        if max_split_batch_size == 0:
            break
        for batch_size in range(1, max_split_batch_size + 1):
            add_probe(batch_size, max_seqlen)

    return tuple(probes.values())


def _iter_compile_units(
    config: KimiK3VisionFA4WarmupConfig,
) -> Iterator[CuTeDSLCompileUnit]:
    device_id = current_platform.current_device()
    num_sms = current_platform.num_compute_units(device_id)
    for probe in _get_warmup_probes(config, num_sms=num_sms):
        yield CuTeDSLCompileUnit(
            name="kimi_k3_vision_fa4",
            key=("kimi_k3_vision_fa4", config, probe.dispatch_key),
            compile=partial(_compile, config, probe),
        )


def _compile(
    config: KimiK3VisionFA4WarmupConfig,
    probe: _FA4WarmupProbe,
) -> None:
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    device = current_platform.current_device()
    total_tokens = probe.batch_size * probe.max_seqlen
    qkv = torch.empty(
        3,
        total_tokens,
        config.num_heads,
        config.head_dim,
        device=device,
        dtype=config.dtype,
    )
    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        probe.max_seqlen,
        device=qkv.device,
        dtype=torch.int32,
    )
    flash_attn_varlen_func(
        qkv[0],
        qkv[1],
        qkv[2],
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=probe.max_seqlen,
        max_seqlen_k=probe.max_seqlen,
        dropout_p=0.0,
        causal=False,
        softmax_scale=config.head_dim**-0.5,
        fa_version=4,
        num_splits=0,
    )


class _WarmupProvider:
    def __init__(self) -> None:
        self.configs: set[KimiK3VisionFA4WarmupConfig] = set()

    def get_cutedsl_warmup_compile_units(self) -> tuple[CuTeDSLCompileUnit, ...]:
        return tuple(
            unit for config in self.configs for unit in _iter_compile_units(config)
        )


_PROVIDER = _WarmupProvider()


def register_kimi_k3_vision_fa4_warmup(
    config: KimiK3VisionFA4WarmupConfig,
) -> None:
    _PROVIDER.configs.add(config)
    register_cutedsl_warmup_provider(_PROVIDER)
