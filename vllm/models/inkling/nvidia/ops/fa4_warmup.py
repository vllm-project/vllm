# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup compilation of the FA4 kernels used by Inkling."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial

import torch

from vllm.model_executor.warmup.cutedsl_warmup import (
    CuTeDSLCompileUnit,
    register_cutedsl_warmup_provider,
)

from .fa4_rel_attention import (
    bucket_max_seqlen_q,
    inkling_fa4_num_splits,
    inkling_fa4_rel_attention,
)


@dataclass(frozen=True)
class InklingFA4WarmupConfig:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rel_extent: int
    window_size: tuple[int, int]
    is_local: bool
    max_kv_len: int
    dtype: torch.dtype
    kv_dtype: torch.dtype
    block_size: int
    max_num_reqs: int
    max_num_batched_tokens: int


def _num_warps_bucket(num_reqs: int) -> int:
    num_warps = min((num_reqs + 30) // 31, 32)
    return 1 << (num_warps - 1).bit_length()


def _compile(config: InklingFA4WarmupConfig, max_seqlen_q: int, num_reqs: int) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    num_splits = inkling_fa4_num_splits(
        is_local=config.is_local,
        batch_size=num_reqs,
        max_query_len=max_seqlen_q,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        max_kv_len=config.max_kv_len,
    )
    with FakeTensorMode():
        device = torch.accelerator.current_accelerator()
        total_q = max_seqlen_q + num_reqs - 1
        q = torch.empty(
            total_q,
            config.num_heads,
            config.head_dim,
            dtype=config.dtype,
            device=device,
        )
        kv = torch.empty(
            1,
            2,
            config.block_size,
            config.num_kv_heads,
            config.head_dim,
            dtype=config.kv_dtype,
            device=device,
        )
        key_cache, value_cache = kv.unbind(1)
        inkling_fa4_rel_attention(
            q,
            key_cache,
            value_cache,
            block_table=torch.empty(num_reqs, 1, dtype=torch.int32, device=device),
            cache_seqlens=torch.empty(num_reqs, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.empty(num_reqs + 1, dtype=torch.int32, device=device),
            max_seqlen_q=max_seqlen_q,
            softmax_scale=1.0 / config.head_dim,
            causal=True,
            window_size=config.window_size,
            rel_extent=config.rel_extent,
            rel_logits=torch.empty(
                total_q,
                config.num_heads,
                config.rel_extent,
                dtype=config.dtype,
                device=device,
            ),
            num_splits=num_splits,
            out=torch.empty_like(q),
        )


def _iter_compile_units(
    config: InklingFA4WarmupConfig,
) -> Iterator[CuTeDSLCompileUnit]:
    max_bucket = bucket_max_seqlen_q(config.max_num_batched_tokens)
    max_seqlen_q = 1
    while max_seqlen_q <= max_bucket:
        min_query_len = max_seqlen_q // 2 + 1
        max_num_reqs = min(
            config.max_num_reqs,
            config.max_num_batched_tokens - min_query_len + 1,
        )
        seen: set[tuple[int, int, int | None, bool]] = set()
        for num_reqs in range(1, max_num_reqs + 1):
            num_splits = inkling_fa4_num_splits(
                is_local=config.is_local,
                batch_size=num_reqs,
                max_query_len=max_seqlen_q,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                max_kv_len=config.max_kv_len,
            )
            key = (
                max_seqlen_q,
                num_splits,
                _num_warps_bucket(num_reqs) if num_splits > 1 else None,
                num_reqs > 1024,
            )
            if key in seen:
                continue
            seen.add(key)
            yield CuTeDSLCompileUnit(
                name="inkling_fa4",
                key=("inkling_fa4", config, key),
                compile=partial(_compile, config, max_seqlen_q, num_reqs),
            )
        max_seqlen_q *= 2


class _WarmupProvider:
    def __init__(self) -> None:
        self.configs: set[InklingFA4WarmupConfig] = set()

    def get_cutedsl_warmup_compile_units(self) -> tuple[CuTeDSLCompileUnit, ...]:
        return tuple(
            unit for config in self.configs for unit in _iter_compile_units(config)
        )


_PROVIDER = _WarmupProvider()


def register_fa4_warmup(config: InklingFA4WarmupConfig) -> None:
    _PROVIDER.configs.add(config)
    register_cutedsl_warmup_provider(_PROVIDER)
