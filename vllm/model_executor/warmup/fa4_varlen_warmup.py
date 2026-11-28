# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm the FA4 kernels of non-causal, non-paged varlen attention.

FA4 picks its kernels from each call's shape: Q-tile stages, static or dynamic
persistent scheduling, Split-KV with its combine and scheduler metadata, and
measured per-GPU overrides. This warmup does not restate those rules. It runs a
grid of call shapes under `FakeTensorMode` with FA4's compile-miss recorder
set, so FA4 itself reports the kernels each shape would compile, then compiles
one shape per group of kernels not yet covered.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.platforms import current_platform

logger = init_logger(__name__)

# SM100 forward tiles span 128 tokens. Probing both ends of every tile reaches
# every Q/K block count and every threshold that falls on a tile boundary.
_SEQLEN_TILE = 128
# Sequence counts up to this are probed one by one, larger ones geometrically.
_DENSE_NUM_SEQS = 64
# Skewed calls probe every tile up to this one, geometrically spaced tiles past it.
_DENSE_SEQLEN_TILES = 16


@dataclass(frozen=True, slots=True)
class FA4VarlenWarmupSpec:
    """Bounds of the varlen FA4 calls an attention layer makes.

    Args:
        max_seqlen: Longest sequence in one call, e.g. one image's patches.
        max_num_tokens: Most tokens across the sequences of one call.
        max_num_seqs: Most sequences in one call.

    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    max_seqlen: int
    max_num_tokens: int
    max_num_seqs: int


@dataclass(frozen=True, slots=True)
class FA4VarlenCallShape:
    num_seqs: int
    max_seqlen: int
    num_tokens: int


def _seqlen_grid(max_seqlen: int, *, dense_tiles: int | None = None) -> list[int]:
    """Both ends of every tile, or of the first `dense_tiles` and a geometric
    set of tiles past them."""
    num_tiles = math.ceil(max_seqlen / _SEQLEN_TILE)
    tiles = set(range(num_tiles if dense_tiles is None else dense_tiles))
    if dense_tiles is not None:
        n = dense_tiles
        while n < num_tiles:
            tiles.update((n, n + n // 3, n + 2 * n // 3))
            n *= 2
    seqlens = {max_seqlen}
    for tile in tiles:
        if tile < num_tiles:
            start = tile * _SEQLEN_TILE
            seqlens.update((start + 1, min(start + _SEQLEN_TILE, max_seqlen)))
    return sorted(seqlens)


def _geometric_num_seqs(max_num_seqs: int) -> list[int]:
    num_seqs = {1, max_num_seqs}
    n = 2
    while n <= max_num_seqs:
        num_seqs.update((n, n + 1, n + n // 2))
        n *= 2
    return sorted(n for n in num_seqs if n <= max_num_seqs)


def get_candidate_shapes(spec: FA4VarlenWarmupSpec) -> list[FA4VarlenCallShape]:
    """Call shapes within `spec` that bracket FA4's shape-dependent choices.

    Equal-length calls probe every tile and every sequence count up to
    `_DENSE_NUM_SEQS`. Calls with one long sequence among single-token ones,
    the other extreme of the token total, reach sequence counts whose
    equal-length calls exceed `max_num_tokens`; they probe a geometric set of
    sequence counts and of tiles past `_DENSE_SEQLEN_TILES`.
    """
    max_num_seqs = min(spec.max_num_seqs, spec.max_num_tokens)
    dense = range(1, min(max_num_seqs, _DENSE_NUM_SEQS) + 1)
    geometric = _geometric_num_seqs(max_num_seqs)
    uniform_num_seqs = sorted({*dense, *geometric})
    max_seqlen_bound = min(spec.max_seqlen, spec.max_num_tokens)

    shapes: list[FA4VarlenCallShape] = []
    for max_seqlen in _seqlen_grid(max_seqlen_bound):
        for num_seqs in uniform_num_seqs:
            if num_seqs * max_seqlen > spec.max_num_tokens:
                break
            shapes.append(
                FA4VarlenCallShape(num_seqs, max_seqlen, num_seqs * max_seqlen)
            )
    for max_seqlen in _seqlen_grid(max_seqlen_bound, dense_tiles=_DENSE_SEQLEN_TILES):
        for num_seqs in geometric:
            if max_seqlen + num_seqs - 1 > spec.max_num_tokens:
                break
            shapes.append(
                FA4VarlenCallShape(num_seqs, max_seqlen, max_seqlen + num_seqs - 1)
            )
    return list(dict.fromkeys(shapes))


def run_fa4_varlen_call(
    spec: FA4VarlenWarmupSpec, shape: FA4VarlenCallShape, device: Any
) -> None:
    """Make the FA4 call of `flash_attn_maxseqlen_wrapper` on fake tensors."""
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    q = torch.empty(
        (shape.num_tokens, spec.num_heads, spec.head_dim),
        dtype=spec.dtype,
        device=device,
    )
    k = torch.empty(
        (shape.num_tokens, spec.num_kv_heads, spec.head_dim),
        dtype=spec.dtype,
        device=device,
    )
    cu_seqlens = torch.empty((shape.num_seqs + 1,), dtype=torch.int32, device=device)
    flash_attn_varlen_func(
        q,
        k,
        torch.empty_like(k),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=shape.max_seqlen,
        max_seqlen_k=shape.max_seqlen,
        dropout_p=0.0,
        causal=False,
        softmax_scale=spec.head_dim**-0.5,
        out=torch.empty_like(q),
        fa_version=4,
    )


def get_missing_fa4_kernels(
    spec: FA4VarlenWarmupSpec, shape: FA4VarlenCallShape, device: Any
) -> set[Any]:
    """Kernels FA4 would compile for `shape`; call under `FakeTensorMode`."""
    from vllm.vllm_flash_attn.cute.interface import _compile_misses

    misses: set[Any] = set()
    token = _compile_misses.set(misses)
    try:
        run_fa4_varlen_call(spec, shape, device)
    finally:
        _compile_misses.reset(token)
    return misses


class FA4VarlenAttentionKernel(VllmJitKernel["FA4VarlenAttentionKernel.CompileKey"]):
    """Compile the FA4 kernels a layer's varlen calls can reach."""

    @dataclass(frozen=True, slots=True)
    class CompileKey:
        spec: FA4VarlenWarmupSpec
        shape: FA4VarlenCallShape

    def get_warmup_keys(self, spec: FA4VarlenWarmupSpec) -> list[CompileKey]:
        if not current_platform.is_device_capability_family(100):
            return []
        from torch._subclasses.fake_tensor import FakeTensorMode

        start = time.perf_counter()
        shapes = get_candidate_shapes(spec)
        device = current_platform.current_device()
        covered: set[Any] = set()
        keys: list[FA4VarlenAttentionKernel.CompileKey] = []
        with FakeTensorMode():
            for shape in shapes:
                misses = get_missing_fa4_kernels(spec, shape, device)
                if not misses <= covered:
                    covered |= misses
                    keys.append(self.CompileKey(spec, shape))
        logger.info_once(
            "Probed %d FA4 varlen call shapes for %s in %.1fs: %d kernels to "
            "compile from %d shapes.",
            len(shapes),
            spec,
            time.perf_counter() - start,
            len(covered),
            len(keys),
        )
        return keys

    def compile(self, compile_key: CompileKey) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        # Fake tensors compile FA4's kernels, combine included, without launching.
        with FakeTensorMode():
            run_fa4_varlen_call(
                compile_key.spec, compile_key.shape, current_platform.current_device()
            )


FA4_VARLEN_ATTENTION_KERNEL = FA4VarlenAttentionKernel()
