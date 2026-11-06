# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER's FlyDSL sparse prefill for the Qwen4Exp QSA owner on gfx950.

When vLLM uses AITER (``VLLM_ROCM_USE_AITER``) on gfx950 and the installed
AITER has the op for gfx950, the ROCm metadata builder describes each step's
trailing prefilling requests with their host lengths, and
``qsa_sparse_paged_attention`` hands those requests' rows to the op when they
number at least ``FLYDSL_PREFILL_MIN_ROWS`` and the op can read the K/V cache
in place. The leading decode rows, and every step the op does not take, run
the Triton kernel.
"""

from __future__ import annotations

import importlib
from functools import cache

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d

from ...common.qsa import QSAPrefillLayout

logger = init_logger(__name__)

HEAD_DIM = 256
# The op holds at most this many query heads per KV head in its direct wave.
MAX_GROUP = 16
# Prefill rows from which the op beats the Triton kernel on MI355X at every
# measured prefix length (256 rows over a 132K-token prefix are 1.18x slower).
FLYDSL_PREFILL_MIN_ROWS = 768
# Requests (rows, no prefix) of the warmup calls: one below the 1536 rows from
# which AITER plans union tiles on gfx950 (direct kernels only); one request
# long enough for the wider union tile AITER gives a whole prefill at 6 or 3
# query heads per KV head; and two requests, which keep the narrower tile.
WARMUP_LAYOUTS = ((FLYDSL_PREFILL_MIN_ROWS,), (4096,), (2048, 2048))


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


@cache
def _flydsl_op():
    """AITER's ``flydsl_qsa_sparse_prefill``, or None when this AITER lacks it
    or has no gfx950 entry (AITER lists gfx950 from ROCm/aiter#7556, which also
    lets the op read strided K/V)."""
    try:
        module = importlib.import_module("aiter.ops.flydsl.qsa_sparse_prefill")
        common = importlib.import_module(
            "aiter.ops.flydsl.kernels.qsa.sparse_prefill_common"
        )
    except ImportError as exc:
        logger.debug("No AITER FlyDSL QSA sparse prefill (%s).", exc)
        return None
    if not any(gfx == "gfx950" for gfx, _ in common.PLATFORMS):
        logger.debug("AITER's FlyDSL QSA sparse prefill does not support gfx950.")
        return None
    return module.flydsl_qsa_sparse_prefill


def flydsl_prefill_available() -> bool:
    """Whether vLLM uses AITER (``VLLM_ROCM_USE_AITER``), the device is gfx950
    and the installed AITER runs the op."""
    return rocm_aiter_ops.is_enabled() and _on_gfx950() and _flydsl_op() is not None


def _selection_blocks(width: int) -> int | None:
    """T of the op's ``[rows, 4 T + 3]`` indices (T = 256, 512, 768 or 1024)."""
    blocks, rest = divmod(width - 3, 4)
    return blocks if rest == 0 and blocks in (256, 512, 768, 1024) else None


def flydsl_prefill_applies(
    num_rows: int,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    """Whether a prefill tail of ``num_rows`` rows takes the FlyDSL op (the
    caller has validated ``q`` and the cache)."""
    return (
        num_rows >= FLYDSL_PREFILL_MIN_ROWS
        and k_cache.dtype == torch.bfloat16
        and q.shape[2] == HEAD_DIM
        and q.shape[1] // k_cache.shape[2] <= MAX_GROUP
        and _selection_blocks(logical_indices.shape[1]) is not None
        and logical_indices.is_contiguous()
        and out.is_contiguous()
        and out.data_ptr() % 16 == 0
        and not torch.cuda.is_current_stream_capturing()
        and flydsl_prefill_available()
    )


def _pools(
    k_cache: torch.Tensor, v_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """K and V as the op's ``[rows, heads, head_dim]`` token pools, read in
    place, and the rows from one page to the next; None when the op cannot read
    the cache that way.

    K and V are the two halves of one record per token (one split of the QSA
    cache), so they share every stride. A pool row is ``heads`` adjacent head
    vectors, 16-byte aligned. The op reads four-token blocks at four
    consecutive rows from a multiple of four: a block starts in a page at a
    multiple of four (pages hold a multiple of 16 tokens), so its row is a
    multiple of four when pages are a multiple of four rows apart.
    """
    pages, page_size, heads, dim = k_cache.shape
    page_stride, token_stride, head_stride, _ = k_cache.stride()
    per_page, rest = divmod(page_stride, token_stride)
    if (
        rest
        or per_page % 4
        or (heads > 1 and head_stride != dim)
        or k_cache.data_ptr() % 16
    ):
        return None
    shape = ((pages - 1) * per_page + page_size, heads, dim)
    stride = (token_stride, dim, 1)
    return (
        k_cache.as_strided(shape, stride),
        v_cache.as_strided(shape, stride),
        per_page,
    )


def token_slots(
    layout: QSAPrefillLayout,
    block_table: torch.Tensor,
    page_size: int,
    per_page: int,
) -> torch.Tensor:
    """int32 pool row of every key of the layout's requests, in packed order
    (request r's tokens 0 .. prefix_r + query_r - 1, requests in batch order):
    token t of request r is row ``block_table[r, t // page_size] * per_page +
    t % page_size``. Built once per step and block table, for every layer. The
    step never waits for it: the lengths are host values copied without a sync,
    and the rest is device work."""
    key = (block_table.data_ptr(), tuple(block_table.shape), page_size, per_page)
    slots = layout.cache.get(key)
    if slots is None:
        device = block_table.device
        lengths = [q + p for q, p in zip(layout.query_lens, layout.prefix_lens)]
        counts = async_tensor_h2d(lengths, device, torch.int64)
        total = sum(lengths)
        requests = torch.arange(
            layout.first_req, layout.first_req + len(lengths), device=device
        )
        request = torch.repeat_interleave(requests, counts, output_size=total)
        first_token = torch.cumsum(counts, 0) - counts
        token = torch.arange(total, device=device) - torch.repeat_interleave(
            first_token, counts, output_size=total
        )
        page = block_table[request, token // page_size].long()
        slots = (page * per_page + token % page_size).to(torch.int32)
        layout.cache[key] = slots
    return slots


def try_flydsl_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    out: torch.Tensor,
    layout: QSAPrefillLayout,
) -> int | None:
    """Run the step's trailing prefill rows through the FlyDSL op.

    Returns how many leading rows are left to the Triton kernel, or None when
    the op does not take this step (nothing was written).
    """
    first = layout.first_row
    if first + layout.num_rows != q.shape[0] or not flydsl_prefill_applies(
        layout.num_rows, q, k_cache, logical_indices[first:], out[first:]
    ):
        return None
    pools = _pools(k_cache, v_cache)
    if pools is None:
        return None
    k, v, per_page = pools
    logger.info_once(
        "Qwen4Exp QSA runs prefill chunks of %d or more rows on AITER's FlyDSL "
        "sparse prefill.",
        FLYDSL_PREFILL_MIN_ROWS,
    )
    _flydsl_op()(
        q[first:].contiguous(),
        k,
        v,
        logical_indices[first:],
        query_lens=layout.query_lens,
        prefix_lens=layout.prefix_lens,
        softmax_scale=HEAD_DIM**-0.5,
        out=out[first:],
        kv_slots=token_slots(layout, block_table, k_cache.shape[1], per_page),
    )
    return first


def _prefill_selection(
    query_lens: tuple[int, ...], blocks: int, width: int, device: torch.device
) -> torch.Tensor:
    """Indices the op takes for requests without a prefix: the row at request
    position p (v = p + 1 tokens) selects its last min(v // 4, T) complete
    four-token blocks, then its v % 4 tail tokens; the rest of the row is -1."""
    visible = torch.cat([torch.arange(1, n + 1, device=device) for n in query_lens])
    first = 4 * (visible // 4 - blocks).clamp(min=0)
    indices = first[:, None] + torch.arange(width, device=device)
    return torch.where(indices < visible[:, None], indices, -1).to(torch.int32)


def warmup_flydsl_qsa_prefill(
    num_query_heads: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selection_width: int,
) -> tuple[tuple[int, ...], ...]:
    """Compile the op's kernels for a layer's K/V cache views, so that no served
    step compiles them: each of ``WARMUP_LAYOUTS`` goes through
    ``try_flydsl_prefill`` on the cache itself, whose row pitch and pool span
    select the kernels. Every key is read from block 0, the null block, which
    is mapped even while an extensible KV cache has mapped nothing else.
    Returns the layouts the op took."""
    blocks = _selection_blocks(selection_width)
    if blocks is None:
        return ()
    device = k_cache.device
    warmed = []
    for query_lens in WARMUP_LAYOUTS:
        q = torch.zeros(
            sum(query_lens),
            num_query_heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        pages = cdiv(max(query_lens), k_cache.shape[1])
        took = try_flydsl_prefill(
            q,
            k_cache,
            v_cache,
            _prefill_selection(query_lens, blocks, selection_width, device),
            torch.zeros(len(query_lens), pages, dtype=torch.int32, device=device),
            torch.empty_like(q),
            QSAPrefillLayout(0, 0, query_lens, (0,) * len(query_lens)),
        )
        if took is not None:
            warmed.append(query_lens)
    torch.accelerator.synchronize(device)
    return tuple(warmed)
