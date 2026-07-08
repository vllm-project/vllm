# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA4 MLA prefill CuTeDSL compile warmup config."""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from vllm.v1.attention.backends.fa_utils import (
        FlashAttentionCuTeDSLCompileSpec,
    )

FA4ArchitectureFamily = Literal["sm90", "sm100f", "sm120"]

FA4_STANDARD_DTYPES = (torch.bfloat16, torch.float16)

# Current vLLM MLA prefill expands K/V to num_heads before FA4, so this plan
# covers qhead_per_kvhead=1.
# Batch is not a current FA4 MLA-prefill key field. Use b1 for compile-only
# specs because it is the conservative case for Split-KV shape heuristics.
# TODO(roberto): FA4 also has direct-GQA and qv/top-k absorbed-MLA paths, but vLLM
# does not use them in this backend yet; they need a separate
# num_kv_heads/qv/top-k-aware warmup plan if wired in later.
FA4_MLA_PREFILL_COMPILE_BATCH_SIZE = 1
FA4_MLA_PREFILL_Q_TILE = 128
FA4_MLA_PREFILL_K_TILE = 128
FA4_MLA_PREFILL_LONG_K_BLOCKS = 32
FA4_MLA_PREFILL_VERY_LONG_K_BLOCKS = 64
FA4_MLA_PREFILL_CAUSAL_OPTIONS = (False, True)
FA4_MLA_PREFILL_LSE_OPTIONS = (False, True)


@dataclass(frozen=True)
class FA4MLAPrefillCompileContext:
    dtype: torch.dtype
    num_heads: int
    qk_head_dim: int
    v_head_dim: int
    kv_nope_head_dim: int
    requires_v_padding: bool
    scale: float
    num_splits: int
    fa_version: int

    # Return the V head dim FA4 sees.
    @property
    def effective_v_head_dim(self) -> int:
        if self.requires_v_padding:
            return self.qk_head_dim
        return self.v_head_dim


@dataclass(frozen=True)
class FA4MLAPrefillCompileRequest:
    """One compile-only FA4 MLA prefill request."""

    key: Hashable
    compile_spec: FlashAttentionCuTeDSLCompileSpec

    # Compile this request.
    def compile(self) -> None:
        self.compile_spec.compile()


# Yield deduped compile requests.
def iter_fa4_mla_prefill_compile_requests(
    ctx: FA4MLAPrefillCompileContext,
) -> Iterator[FA4MLAPrefillCompileRequest]:
    """Yield compile requests for this fixed MLA backend.

    FA4 dedupes duplicate atomic kernel selections in its own JIT cache.
    """
    seen: set[Hashable] = set()
    for compile_spec in iter_fa4_mla_prefill_compile_specs(ctx):
        key = compile_spec.request_key()
        if key in seen:
            continue
        seen.add(key)
        yield FA4MLAPrefillCompileRequest(
            key=key,
            compile_spec=compile_spec,
        )


# Build compile specs for this setup.
def iter_fa4_mla_prefill_compile_specs(
    ctx: FA4MLAPrefillCompileContext,
) -> Iterator[FlashAttentionCuTeDSLCompileSpec]:
    """Yield compile-only FA4 MLA prefill requests for this fixed setup."""

    arch_family = _fa4_architecture_family_from_compute_capability(
        *torch.cuda.get_device_capability()
    )
    if not _supports_fa4_mla_prefill(ctx, arch_family):
        return

    from vllm.v1.attention.backends.fa_utils import (
        FlashAttentionCuTeDSLCompileSpec,
    )

    batch_size = FA4_MLA_PREFILL_COMPILE_BATCH_SIZE
    v_stride = None
    if not ctx.requires_v_padding:
        v_stride = (
            ctx.num_heads * ctx.kv_nope_head_dim,
            ctx.kv_nope_head_dim,
            1,
        )

    for _, max_seqlen_q, max_seqlen_k in _shape_probes_for_context(ctx, arch_family):
        total_q_tokens = batch_size * max_seqlen_q
        total_kv_tokens = batch_size * max_seqlen_k
        for causal in FA4_MLA_PREFILL_CAUSAL_OPTIONS:
            for return_lse in FA4_MLA_PREFILL_LSE_OPTIONS:
                yield FlashAttentionCuTeDSLCompileSpec(
                    q_shape=(total_q_tokens, ctx.num_heads, ctx.qk_head_dim),
                    k_shape=(total_kv_tokens, ctx.num_heads, ctx.qk_head_dim),
                    v_shape=(
                        total_kv_tokens,
                        ctx.num_heads,
                        ctx.effective_v_head_dim,
                    ),
                    v_stride=v_stride,
                    q_dtype=ctx.dtype,
                    cu_seqlens_q_shape=(batch_size + 1,),
                    cu_seqlens_k_shape=(batch_size + 1,),
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=ctx.scale,
                    causal=causal,
                    return_softmax_lse=return_lse,
                    num_splits=ctx.num_splits,
                    fa_version=ctx.fa_version,
                )


# Pick one q/k point per current FA4 MLA-prefill shape regime.
def _shape_probes_for_context(
    ctx: FA4MLAPrefillCompileContext,
    arch_family: FA4ArchitectureFamily,
) -> tuple[tuple[str, int, int], ...]:
    q_stage1_q = 1
    q_stage2_q = FA4_MLA_PREFILL_Q_TILE + 1
    # FA4 never auto-splits when ceil(max_seqlen_k / tile_n) <= 4.
    no_split_k = 4 * FA4_MLA_PREFILL_K_TILE
    long_k = FA4_MLA_PREFILL_LONG_K_BLOCKS * FA4_MLA_PREFILL_K_TILE
    # Diff-head-dim Blackwell Split-KV switches tile_n at 64 K blocks.
    very_long_k = FA4_MLA_PREFILL_VERY_LONG_K_BLOCKS * FA4_MLA_PREFILL_K_TILE

    base_probes = (
        ("q_stage1", q_stage1_q, FA4_MLA_PREFILL_K_TILE),
        ("q_stage2", q_stage2_q, no_split_k),
    )
    # SM120 currently rejects Split-KV in FA4; num_splits=1 also has no split
    # shape regimes on any architecture.
    if ctx.num_splits == 1 or arch_family == "sm120":
        return base_probes

    long_k_probes = (
        ("q_stage1_long_k", q_stage1_q, long_k),
        ("q_stage2_long_k", q_stage2_q, long_k),
    )

    # SM90 does not have the SM100 q_stage or diff-head-dim tile_n=64 branch.
    # Same-dim SM100-family MLA also does not need the very-long-K probe.
    if arch_family == "sm90" or ctx.qk_head_dim == ctx.effective_v_head_dim:
        return (*base_probes, *long_k_probes)

    very_long_k_probes = (
        ("q_stage1_very_long_k", q_stage1_q, very_long_k),
        ("q_stage2_very_long_k", q_stage2_q, very_long_k),
    )
    return (*base_probes, *long_k_probes, *very_long_k_probes)


# Check whether this setup can use FA4 MLA prefill.
def _supports_fa4_mla_prefill(
    ctx: FA4MLAPrefillCompileContext,
    arch_family: FA4ArchitectureFamily,
) -> bool:
    return (
        ctx.dtype in FA4_STANDARD_DTYPES
        and ctx.num_heads > 0
        and (arch_family != "sm120" or ctx.num_splits == 1)
    )


# Map CUDA capability to the FA4 arch family used by warmup checks.
def _fa4_architecture_family_from_compute_capability(
    major: int,
    minor: int,
) -> FA4ArchitectureFamily:
    if (major, minor) == (9, 0):
        return "sm90"
    if major == 10:
        return "sm100f"
    if (major, minor) == (12, 0):
        return "sm120"
    raise ValueError(f"FA4 warmup does not know CUDA capability {major}.{minor}")
