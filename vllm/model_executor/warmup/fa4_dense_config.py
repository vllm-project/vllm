# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA4 dense (paged) CuTeDSL compile warmup config.

The dense FLASH_ATTN FA4 path (``FlashAttentionForwardSm100`` via
``flash_attn_varlen_func``) uses a paged KV cache and ``seqused_k``. main's
existing CuTeDSL warmup only covers FA4 MLA prefill, so the dense prefill
(``q_stage=2``) specialization JIT-compiles during serving. This config
compiles the dense specializations up front so serving sees no CuTeDSL JIT.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from dataclasses import dataclass

import torch

from vllm.model_executor.warmup.cutedsl_warmup import CuTeDSLCompileUnit

FA4_DENSE_STANDARD_DTYPES = (torch.bfloat16, torch.float16)

# A prefill-sized query (rows > q tile) selects the q_stage=2 kernel; a
# single-token-per-seq query selects q_stage=1. Concrete magnitudes below the
# tile boundary do not change the compiled kernel, only the q_stage selection.
_PREFILL_Q_ROWS = 256
_DECODE_BATCH = 8
_WARMUP_NUM_BLOCKS = 256
_DECODE_MAX_SEQLEN_K = 8192


@dataclass(frozen=True)
class FA4DenseCompileContext:
    dtype: torch.dtype
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    max_blocks_per_seq: int
    scale: float
    fa_version: int
    has_fp8_kv: bool = False


def _supports_fa4_dense(ctx: FA4DenseCompileContext) -> bool:
    major, _ = torch.cuda.get_device_capability()
    return (
        ctx.fa_version == 4
        and ctx.dtype in FA4_DENSE_STANDARD_DTYPES
        and ctx.num_qo_heads > 0
        and ctx.num_kv_heads > 0
        and ctx.head_dim > 0
        and ctx.page_size > 0
        and ctx.max_blocks_per_seq > 0
        and major == 10
    )


def iter_fa4_dense_compile_units(
    ctx: FA4DenseCompileContext,
) -> Iterator[CuTeDSLCompileUnit]:
    """Yield compile-only units for the dense FA4 serving specializations."""
    if not _supports_fa4_dense(ctx):
        return

    from vllm.v1.attention.backends.fa_utils import (
        compile_flash_attn_varlen_func_from_specs,
    )

    if compile_flash_attn_varlen_func_from_specs is None:
        return

    kv_shape = (_WARMUP_NUM_BLOCKS, ctx.page_size, ctx.num_kv_heads, ctx.head_dim)
    # key_cache/value_cache are views of a (num_blocks, 2, page_size, num_kv_heads,
    # head_dim) cache, so their num_blocks stride carries the paired-kv factor of 2.
    kv_stride = (
        2 * ctx.page_size * ctx.num_kv_heads * ctx.head_dim,
        ctx.num_kv_heads * ctx.head_dim,
        ctx.head_dim,
        1,
    )
    descale_options = (False, True) if ctx.has_fp8_kv else (False,)

    # (regime, q_rows, max_seqlen_q, batch, max_seqlen_k)
    regimes = (
        ("prefill", _PREFILL_Q_ROWS, _PREFILL_Q_ROWS, 1, _PREFILL_Q_ROWS),
        ("decode", _DECODE_BATCH, 1, _DECODE_BATCH, _DECODE_MAX_SEQLEN_K),
    )

    seen: set[Hashable] = set()
    for regime, q_rows, max_seqlen_q, batch, max_seqlen_k in regimes:
        for descale in descale_options:
            key: Hashable = (
                "fa4_dense",
                regime,
                ctx.dtype,
                ctx.num_qo_heads,
                ctx.num_kv_heads,
                ctx.head_dim,
                ctx.page_size,
                descale,
            )
            if key in seen:
                continue
            seen.add(key)

            spec = dict(
                q_shape=(q_rows, ctx.num_qo_heads, ctx.head_dim),
                k_shape=kv_shape,
                v_shape=kv_shape,
                k_stride=kv_stride,
                v_stride=kv_stride,
                q_dtype=ctx.dtype,
                cu_seqlens_q_shape=(batch + 1,),
                cu_seqlens_k_shape=None,
                seqused_k_shape=(batch,),
                page_table_shape=(batch, ctx.max_blocks_per_seq),
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=ctx.scale,
                causal=True,
                num_splits=1,
                fa_version=ctx.fa_version,
                return_softmax_lse=False,
                q_descale=descale,
                k_descale=descale,
                v_descale=descale,
            )

            def _compile(spec: dict = spec) -> None:
                compile_flash_attn_varlen_func_from_specs(**spec)

            yield CuTeDSLCompileUnit(
                name=f"fa4_dense_{regime}", key=key, compile=_compile
            )
