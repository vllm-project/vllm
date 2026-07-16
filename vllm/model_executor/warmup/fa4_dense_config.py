# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA4 dense (paged) CuTeDSL compile warmup config.

The dense FLASH_ATTN FA4 path (``FlashAttentionForwardSm100`` via
``flash_attn_varlen_func``) uses a paged KV cache and ``seqused_k``. main's
existing CuTeDSL warmup only covers FA4 MLA prefill, so the dense
specializations JIT-compile during serving. This config compiles them up front
so serving sees no CuTeDSL JIT.

For a fixed model and device, the FA4 dense ``compile_key`` only moves along a
few atomic factors (the rest are fixed head/dtype fields):

* ``q_stage`` -- ``2`` (multi-token / prefill) only on SM100/SM110 when
  ``max_seqlen_q * qhead_per_kvhead > tile_m``; always ``1`` on SM90 and SM120.
* ``is_split_kv`` (``num_splits > 1``) -- serving passes ``num_splits=0`` (auto),
  so decode with small batch / long context splits. Rejected on SM120.

We enumerate those factors directly rather than via serving-side proxies
(prefill size, decode batch), mirroring the MLA-prefill config's ``q_stage``
probes in ``fa4_cutedsl_config.py``.

fp8-KV dense warmup is intentionally not emitted: the FA4 fp8 kernel takes fp8
q/k/v but writes a bf16 output, and ``compile_flash_attn_varlen_func_from_specs``
hardcodes the compile-only output dtype to ``q_dtype`` (no ``out_dtype``), so an
fp8-in/bf16-out spec cannot be built. fp8-KV serving therefore falls back to
in-serving JIT for the dense path (as under eager); enabling it needs an
``out_dtype`` on that helper.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from dataclasses import dataclass
from typing import Literal

import torch

from vllm.model_executor.warmup.cutedsl_warmup import CuTeDSLCompileUnit
from vllm.platforms import current_platform

FA4DenseArchFamily = Literal["sm90", "sm100f", "sm120"]

FA4_DENSE_STANDARD_DTYPES = (torch.bfloat16, torch.float16)

# SM100/SM110 dense FA4 uses the default FwdConfig(128, 128) (no SM100 tile
# tuning), so q_stage=2 is selected once max_seqlen_q * qhead_per_kvhead > 128.
# max_seqlen_q = tile + 1 forces q_stage=2 for any qhead_per_kvhead >= 1.
_SM100_DENSE_Q_TILE = 128

_WARMUP_NUM_BLOCKS = 256
# max_seqlen_k is not a compile_key field; use a value large enough to be a
# structurally valid split-KV shape.
_WARMUP_MAX_SEQLEN_K = 8192


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


# Map a device capability to the FA4 arch family used by warmup checks. Returns
# None for arches vLLM does not run dense FA4 on (skip warmup, don't raise --
# the provider is registered whenever fa_version==4).
def _fa4_dense_arch_family(
    cap: object | None,
) -> FA4DenseArchFamily | None:
    major = getattr(cap, "major", None)
    if major == 9:
        return "sm90"
    if major in (10, 11):
        return "sm100f"
    if major == 12:
        return "sm120"
    return None


def _supports_fa4_dense(
    ctx: FA4DenseCompileContext,
    family: FA4DenseArchFamily | None,
) -> bool:
    return (
        family is not None
        and ctx.fa_version == 4
        and ctx.dtype in FA4_DENSE_STANDARD_DTYPES
        and ctx.num_qo_heads > 0
        and ctx.num_kv_heads > 0
        and ctx.head_dim > 0
        and ctx.page_size > 0
        and ctx.max_blocks_per_seq > 0
    )


# Yield the (q_stage, is_split_kv) atomic-factor combinations the dense FA4
# serving path can produce on this arch family.
def _iter_compile_factors(
    family: FA4DenseArchFamily,
) -> Iterator[tuple[int, bool]]:
    # q_stage=2 (multi-token) only exists on SM100/SM110.
    q_stages = (1, 2) if family == "sm100f" else (1,)
    # SM120 rejects split-KV; SM90 and SM100/SM110 support it.
    split_states = (False,) if family == "sm120" else (False, True)

    for q_stage in q_stages:
        for is_split_kv in split_states:
            yield q_stage, is_split_kv


def iter_fa4_dense_compile_units(
    ctx: FA4DenseCompileContext,
) -> Iterator[CuTeDSLCompileUnit]:
    """Yield compile-only units for the dense FA4 serving specializations."""
    family = _fa4_dense_arch_family(current_platform.get_device_capability())
    if not _supports_fa4_dense(ctx, family):
        return
    assert family is not None

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

    # A single warmup sequence (batch=1); batch is not a compile_key field.
    batch = 1

    seen: set[Hashable] = set()
    for q_stage, is_split_kv in _iter_compile_factors(family):
        # q_stage is selected from max_seqlen_q relative to the q tile; total
        # q rows only needs to hold that many tokens for the single sequence.
        max_seqlen_q = _SM100_DENSE_Q_TILE + 1 if q_stage == 2 else 1
        # is_split_kv == (num_splits > 1); the concrete value is not a key field.
        num_splits = 2 if is_split_kv else 1

        key: Hashable = (
            "fa4_dense",
            family,
            q_stage,
            is_split_kv,
            ctx.dtype,
            ctx.num_qo_heads,
            ctx.num_kv_heads,
            ctx.head_dim,
            ctx.page_size,
        )
        if key in seen:
            continue
        seen.add(key)

        spec = dict(
            q_shape=(max_seqlen_q, ctx.num_qo_heads, ctx.head_dim),
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
            max_seqlen_k=_WARMUP_MAX_SEQLEN_K,
            softmax_scale=ctx.scale,
            causal=True,
            num_splits=num_splits,
            fa_version=ctx.fa_version,
            return_softmax_lse=False,
        )

        def _compile(spec: dict = spec) -> None:
            compile_flash_attn_varlen_func_from_specs(**spec)

        name = f"fa4_dense_q{q_stage}_split{int(is_split_kv)}"
        yield CuTeDSLCompileUnit(name=name, key=key, compile=_compile)
