# ruff: ignore
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal, overload

import torch

def get_scheduler_metadata(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    headdim: int,
    cache_seqlens: torch.Tensor,
    qkv_dtype: torch.dtype = ...,
    headdim_v: int | None = ...,
    cu_seqlens_q: torch.Tensor | None = ...,
    cu_seqlens_k_new: torch.Tensor | None = ...,
    cache_leftpad: torch.Tensor | None = ...,
    page_size: int = ...,
    max_seqlen_k_new: int = ...,
    causal: bool = ...,
    window_size: tuple[int, int] = ...,
    has_softcap: bool = ...,
    num_splits: int = ...,
    pack_gqa: Any | None = ...,
    sm_margin: int = ...,
): ...
@overload
def flash_attn_varlen_func(
    q: tuple[int, int, int],
    k: tuple[int, int, int],
    v: tuple[int, int, int],
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor | None,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor | None = ...,
    seqused_k: Any | None = ...,
    q_v: Any | None = ...,
    dropout_p: float = ...,
    causal: bool = ...,
    window_size: list[int] | None = ...,
    softmax_scale: float = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    deterministic: bool = ...,
    return_attn_probs: bool = ...,
    block_table: Any | None = ...,
    return_softmax_lse: Literal[False] = ...,
    out: Any = ...,
    # FA3 Only
    scheduler_metadata: Any | None = ...,
    q_descale: Any | None = ...,
    k_descale: Any | None = ...,
    v_descale: Any | None = ...,
    # Version selector
    fa_version: int = ...,
) -> tuple[int, int, int]: ...
@overload
def flash_attn_varlen_func(
    q: tuple[int, int, int],
    k: tuple[int, int, int],
    v: tuple[int, int, int],
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor | None,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor | None = ...,
    seqused_k: Any | None = ...,
    q_v: Any | None = ...,
    dropout_p: float = ...,
    causal: bool = ...,
    window_size: list[int] | None = ...,
    softmax_scale: float = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    deterministic: bool = ...,
    return_attn_probs: bool = ...,
    block_table: Any | None = ...,
    return_softmax_lse: Literal[True] = ...,
    out: Any = ...,
    # FA3 Only
    scheduler_metadata: Any | None = ...,
    q_descale: Any | None = ...,
    k_descale: Any | None = ...,
    v_descale: Any | None = ...,
    # Version selector
    fa_version: int = ...,
) -> tuple[tuple[int, int, int], tuple[int, int]]: ...
@overload
def flash_attn_with_kvcache(
    q: tuple[int, int, int, int],
    k_cache: tuple[int, int, int, int],
    v_cache: tuple[int, int, int, int],
    k: tuple[int, int, int, int] | None = ...,
    v: tuple[int, int, int, int] | None = ...,
    rotary_cos: tuple[int, int] | None = ...,
    rotary_sin: tuple[int, int] | None = ...,
    cache_seqlens: int | torch.Tensor | None = None,
    cache_batch_idx: torch.Tensor | None = None,
    cache_leftpad: torch.Tensor | None = ...,
    block_table: torch.Tensor | None = ...,
    softmax_scale: float = ...,
    causal: bool = ...,
    window_size: tuple[int, int] = ...,  # -1 means infinite context window
    softcap: float = ...,
    rotary_interleaved: bool = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    num_splits: int = ...,
    return_softmax_lse: Literal[False] = ...,
    *,
    out: Any = ...,
    # FA3 Only
    scheduler_metadata: Any | None = ...,
    q_descale: Any | None = ...,
    k_descale: Any | None = ...,
    v_descale: Any | None = ...,
    # Version selector
    fa_version: int = ...,
) -> tuple[int, int, int, int]: ...
@overload
def flash_attn_with_kvcache(
    q: tuple[int, int, int, int],
    k_cache: tuple[int, int, int, int],
    v_cache: tuple[int, int, int, int],
    k: tuple[int, int, int, int] | None = ...,
    v: tuple[int, int, int, int] | None = ...,
    rotary_cos: tuple[int, int] | None = ...,
    rotary_sin: tuple[int, int] | None = ...,
    cache_seqlens: int | torch.Tensor | None = None,
    cache_batch_idx: torch.Tensor | None = None,
    cache_leftpad: torch.Tensor | None = ...,
    block_table: torch.Tensor | None = ...,
    softmax_scale: float = ...,
    causal: bool = ...,
    window_size: tuple[int, int] = ...,  # -1 means infinite context window
    softcap: float = ...,
    rotary_interleaved: bool = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    num_splits: int = ...,
    return_softmax_lse: Literal[True] = ...,
    *,
    out: Any = ...,
    # FA3 Only
    scheduler_metadata: Any | None = ...,
    q_descale: Any | None = ...,
    k_descale: Any | None = ...,
    v_descale: Any | None = ...,
    # Version selector
    fa_version: int = ...,
) -> tuple[tuple[int, int, int], tuple[int, int]]: ...
@overload
def sparse_attn_func(
    q: tuple[int, int, int, int],
    k: tuple[int, int, int, int],
    v: tuple[int, int, int, int],
    block_count: tuple[int, int, float],
    block_offset: tuple[int, int, float, int],
    column_count: tuple[int, int, float],
    column_index: tuple[int, int, float, int],
    dropout_p: float = ...,
    softmax_scale: float = ...,
    causal: bool = ...,
    softcap: float = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    deterministic: bool = ...,
    return_attn_probs: bool = ...,
    *,
    return_softmax_lse: Literal[False] = ...,
    out: Any = ...,
) -> tuple[int, int, int]: ...
@overload
def sparse_attn_func(
    q: tuple[int, int, int, int],
    k: tuple[int, int, int, int],
    v: tuple[int, int, int, int],
    block_count: tuple[int, int, float],
    block_offset: tuple[int, int, float, int],
    column_count: tuple[int, int, float],
    column_index: tuple[int, int, float, int],
    dropout_p: float = ...,
    softmax_scale: float = ...,
    causal: bool = ...,
    softcap: float = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    deterministic: bool = ...,
    return_attn_probs: bool = ...,
    *,
    return_softmax_lse: Literal[True] = ...,
    out: Any = ...,
) -> tuple[tuple[int, int, int], tuple[int, int]]: ...
@overload
def sparse_attn_varlen_func(
    q: tuple[int, int, int],
    k: tuple[int, int, int],
    v: tuple[int, int, int],
    block_count: tuple[int, int, float],
    block_offset: tuple[int, int, float, int],
    column_count: tuple[int, int, float],
    column_index: tuple[int, int, float, int],
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = ...,
    softmax_scale: float = ...,
    causal: bool = ...,
    softcap: float = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    deterministic: bool = ...,
    return_attn_probs: bool = ...,
    *,
    return_softmax_lse: Literal[False] = ...,
    out: Any = ...,
) -> tuple[int, int, int]: ...
@overload
def sparse_attn_varlen_func(
    q: tuple[int, int, int],
    k: tuple[int, int, int],
    v: tuple[int, int, int],
    block_count: tuple[int, int, float],
    block_offset: tuple[int, int, float, int],
    column_count: tuple[int, int, float],
    column_index: tuple[int, int, float, int],
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = ...,
    softmax_scale: float = ...,
    causal: bool = ...,
    softcap: float = ...,
    alibi_slopes: tuple[int] | tuple[int, int] | None = ...,
    deterministic: bool = ...,
    return_attn_probs: bool = ...,
    *,
    return_softmax_lse: Literal[True] = ...,
    out: Any = ...,
) -> tuple[tuple[int, int, int], tuple[int, int]]: ...
def is_fa_version_supported(
    fa_version: int, device: torch.device | None = None
) -> bool: ...
def fa_version_unsupported_reason(
    fa_version: int, device: torch.device | None = None
) -> str | None: ...
