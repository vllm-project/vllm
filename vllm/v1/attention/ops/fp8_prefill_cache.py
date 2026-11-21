# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, exact FP8-to-BF16 page expansion for native prefill.

Materializing referenced page entries once per layer and prefill step trades
scratch memory for fewer repeated conversions across attention query tiles.
Calibration remains in the attention operation. This helper never applies
scales and never mutates persistent KV storage. The caller must gate device,
attention mode, query dtype and graph compatibility before planning.
"""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

MAX_EXPANSION_BYTES = 1 << 30
MIN_QUERY_TOKENS = 1024


def prefill_page_range(
    indptr: torch.Tensor, available: int
) -> tuple[int, int, torch.Tensor]:
    if indptr.device.type != "cpu" or indptr.ndim != 1 or indptr.numel() < 2:
        raise ValueError(
            "Prefill indptr must be a CPU vector with at least two entries"
        )
    if indptr.dtype not in (torch.int32, torch.int64):
        raise ValueError("Page offsets must be integers")
    start, end = int(indptr[0]), int(indptr[-1])
    if start < 0 or end > available or bool(torch.any(indptr[1:] < indptr[:-1])):
        raise ValueError("Invalid prefill page offsets")
    rebased = indptr - start
    if indptr.is_pinned():
        rebased = rebased.pin_memory()
    return start, end, rebased


def expansion_bytes(
    *,
    num_query_tokens: int,
    num_pages: int,
    num_kv_heads: int,
    page_size: int,
    head_dim: int,
    budget: int,
) -> int:
    if not 0 <= budget <= MAX_EXPANSION_BYTES:
        raise ValueError("Expansion budget must be between zero and 1 GiB")
    if min(num_pages, num_kv_heads, page_size, head_dim) < 0:
        raise ValueError("Negative cache geometry")
    size = num_pages * 2 * num_kv_heads * page_size * head_dim * 2
    if num_query_tokens < MIN_QUERY_TOKENS or size > budget:
        return 0
    return size


@dataclass
class PrefillCacheExpansion:
    source_indices: torch.Tensor
    indptr: torch.Tensor
    indices: torch.Tensor
    cache: torch.Tensor
    layout: str = "HND"


def plan_expansion(
    indptr: torch.Tensor,
    page_indices: torch.Tensor,
    *,
    num_query_tokens: int,
    num_kv_heads: int,
    page_size: int,
    head_dim: int,
    budget: int,
    layout: str = "HND",
) -> PrefillCacheExpansion | None:
    """Allocate a compact prefill cache, or retain native FP8 on budget/OOM.

    ``indptr`` is the CPU prefill slice of the native paged offsets and may
    start after decode entries. Each returned plan owns its scratch storage;
    fixed capacity allows reuse by the allocator without aliasing live plans.
    """
    if layout not in ("HND", "NHD"):
        raise ValueError("Only HND and NHD cache layouts are supported")
    start, end, compact_indptr = prefill_page_range(indptr, page_indices.numel())
    count = end - start
    if not expansion_bytes(
        num_query_tokens=num_query_tokens,
        num_pages=count,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        head_dim=head_dim,
        budget=budget,
    ):
        return None
    if page_indices.device.type != "cuda" or page_indices.ndim != 1:
        raise ValueError("Page indices must be a CUDA vector")
    if page_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Page indices must be integers")
    if torch.cuda.is_current_stream_capturing():
        return None
    try:
        # A stable capacity lets the allocator reuse buffers as context grows,
        # instead of retaining a different allocation for each prefill chunk.
        # Each metadata object still owns its buffer; live batches never alias.
        capacity = budget // (2 * num_kv_heads * page_size * head_dim * 2)
        shape = (
            (capacity, 2, num_kv_heads, page_size, head_dim)
            if layout == "HND"
            else (capacity, 2, page_size, num_kv_heads, head_dim)
        )
        cache = torch.empty(shape, dtype=torch.bfloat16, device=page_indices.device)[
            :count
        ]
        compact_indices = torch.arange(
            count, device=page_indices.device, dtype=page_indices.dtype
        )
    except torch.OutOfMemoryError:
        # The caller has not planned attention yet and can use its FP8 path.
        return None
    return PrefillCacheExpansion(
        page_indices[start:end], compact_indptr, compact_indices, cache, layout
    )


@triton.jit
def _expand_pages(
    K,
    V,
    INDICES,
    OUTPUT,
    INDEX_STRIDE: tl.constexpr,
    KP: tl.constexpr,
    KH: tl.constexpr,
    KT: tl.constexpr,
    KD: tl.constexpr,
    VP: tl.constexpr,
    VH: tl.constexpr,
    VT: tl.constexpr,
    VD: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM: tl.constexpr,
    NHD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page = tl.program_id(0).to(tl.int64)
    physical = tl.load(INDICES + page * INDEX_STRIDE).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < HEADS * PAGE_SIZE * DIM
    head = offsets // (PAGE_SIZE * DIM)
    token = (offsets // DIM) % PAGE_SIZE
    dim = offsets % DIM
    key = tl.load(K + physical * KP + head * KH + token * KT + dim * KD, mask, 0.0)
    val = tl.load(V + physical * VP + head * VH + token * VT + dim * VD, mask, 0.0)
    page_elements = HEADS * PAGE_SIZE * DIM
    output_offset = token * HEADS * DIM + head * DIM + dim if NHD else offsets
    destination = page * 2 * page_elements + output_offset
    tl.store(OUTPUT + destination, key.to(tl.bfloat16), mask)
    tl.store(OUTPUT + destination + page_elements, val.to(tl.bfloat16), mask)


def expand_cache(
    keys: torch.Tensor, values: torch.Tensor, plan: PrefillCacheExpansion
) -> torch.Tensor:
    """Gather referenced pages exactly, preserving K/V calibration for attention.

    K/V are strided ``[pages, heads, tokens, dim]`` or NHD views. The result
    is ``[referenced_pages, 2, ...]`` in the same layout. Shared source pages
    remain separate logical entries; persistent cache bytes are not modified.
    """
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("Expected four-dimensional K/V views with identical geometry")
    if keys.dtype != torch.float8_e4m3fn or values.dtype != keys.dtype:
        raise ValueError("Only E4M3 FP8 storage is qualified")
    if keys.device != values.device or keys.device != plan.cache.device:
        raise ValueError("Cache, indices and expansion must share a device")
    if plan.cache.shape[2:] != keys.shape[1:]:
        raise ValueError("Expansion geometry does not match the KV cache")
    if plan.layout == "NHD":
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prefill cache expansion is not qualified for full capture")
    heads, page_size, dim = keys.shape[1:]
    _expand_pages[(plan.cache.shape[0], triton.cdiv(heads * page_size * dim, 2048))](
        keys,
        values,
        plan.source_indices,
        plan.cache,
        plan.source_indices.stride(0),
        *keys.stride(),
        *values.stride(),
        heads,
        page_size,
        dim,
        plan.layout == "NHD",
        2048,
        num_warps=4,
    )
    return plan.cache
