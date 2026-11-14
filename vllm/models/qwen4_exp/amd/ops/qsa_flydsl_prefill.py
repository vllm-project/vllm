# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER's FlyDSL sparse prefill for the Qwen4Exp QSA owner on gfx950.

When vLLM uses AITER (`VLLM_ROCM_USE_AITER`) on gfx950 and the installed
AITER has the QSA sparse prefill API, the ROCm metadata builder describes each
step's trailing prefilling requests with their host lengths, and
`qsa_sparse_paged_attention` hands those requests' rows to the op when they
number at least `FLYDSL_PREFILL_MIN_ROWS` and AITER accepts the call. The
leading decode rows, and every step AITER declines, run the Triton kernel.

What the op can read is AITER's to say: its operand rules
(`qsa_sparse_prefill_problem`) and the paged cache's geometry
(`qsa_paged_kv_pool`). What stays here is vLLM's: the row threshold, the
trailing-rows split, the cache views and the per-step slot table.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

from ...common.qsa import QSAPrefillLayout

logger = init_logger(__name__)

HEAD_DIM = 256
# Prefill rows from which the op beats the Triton kernel on MI355X at every
# measured prefix length (256 rows over a 132K-token prefix are 1.18x slower).
FLYDSL_PREFILL_MIN_ROWS = 768


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def flydsl_prefill_available() -> bool:
    """Whether vLLM uses AITER (`VLLM_ROCM_USE_AITER`), the device is gfx950
    and the installed AITER has the op's API and its FlyDSL backend (it runs on
    gfx950 from ROCm/aiter#7556, which also lets it read strided K/V)."""
    return bool(rocm_aiter_ops.is_qsa_sparse_prefill_available()) and _on_gfx950()


def _kv_slots(
    layout: QSAPrefillLayout, block_table: torch.Tensor, pool: Any
) -> torch.Tensor:
    """The op's `kv_slots` for the layout's requests, built once per step and
    block table for every layer."""
    key = (
        block_table.data_ptr(),
        tuple(block_table.shape),
        pool.page_size,
        pool.rows_per_page,
    )
    slots = layout.cache.get(key)
    if slots is None:
        first = layout.first_req
        slots = rocm_aiter_ops.qsa_paged_kv_slots(
            layout.query_lens,
            layout.prefix_lens,
            block_table[first : first + len(layout.query_lens)],
            pool.page_size,
            pool.rows_per_page,
        )
        layout.cache[key] = slots
    return slots


def _run(
    q: torch.Tensor,
    pool: Any,
    indices: torch.Tensor,
    query_lens: tuple[int, ...],
    prefix_lens: tuple[int, ...],
    out: torch.Tensor,
    slots: torch.Tensor,
) -> bool:
    """The op over the pools; False, with nothing written, when AITER's
    operand rules decline the call."""
    kwargs = {
        "query_lens": query_lens,
        "prefix_lens": prefix_lens,
        "softmax_scale": HEAD_DIM**-0.5,
        "out": out,
        "kv_slots": slots,
    }
    problem = rocm_aiter_ops.qsa_sparse_prefill_problem(
        q, pool.k, pool.v, indices, **kwargs
    )
    if problem is not None:
        logger.warning_once(
            "AITER's QSA sparse prefill declines a call vLLM sent it "
            "(the step runs on the Triton kernel): %s",
            problem,
        )
        return False
    rocm_aiter_ops.qsa_sparse_prefill(q, pool.k, pool.v, indices, **kwargs)
    return True


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
    if (
        layout.num_rows < FLYDSL_PREFILL_MIN_ROWS
        or first + layout.num_rows != q.shape[0]
        or torch.cuda.is_current_stream_capturing()
        or not flydsl_prefill_available()
    ):
        return None
    pool = rocm_aiter_ops.qsa_paged_kv_pool(k_cache, v_cache)
    if pool is None:
        return None
    ran = _run(
        q[first:].contiguous(),
        pool,
        logical_indices[first:],
        layout.query_lens,
        layout.prefix_lens,
        out[first:],
        _kv_slots(layout, block_table, pool),
    )
    if not ran:
        return None
    logger.info_once(
        "Qwen4Exp QSA runs prefill chunks of %d or more rows on AITER's FlyDSL "
        "sparse prefill.",
        FLYDSL_PREFILL_MIN_ROWS,
    )
    return first


def warmup_flydsl_qsa_prefill(
    num_query_heads: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selection_width: int,
) -> tuple[tuple[int, ...], ...]:
    """Compile the op's kernels for a layer's K/V cache views, so that no served
    step compiles them: AITER runs one call per kernel-variant class through
    the call below, on the cache itself, whose row pitch and pool span select
    the kernels. Every key is read from block 0, the null block, which is mapped
    even while an extensible KV cache has mapped nothing else. Returns the
    layouts the op took."""
    pool = rocm_aiter_ops.qsa_paged_kv_pool(k_cache, v_cache)
    if pool is None:
        return ()
    device = k_cache.device

    def run(q, indices, query_lens, prefix_lens) -> bool:
        pages = cdiv(
            max(n + p for n, p in zip(query_lens, prefix_lens)), pool.page_size
        )
        table = torch.zeros(len(query_lens), pages, dtype=torch.int32, device=device)
        slots = rocm_aiter_ops.qsa_paged_kv_slots(
            query_lens, prefix_lens, table, pool.page_size, pool.rows_per_page
        )
        return _run(
            q, pool, indices, query_lens, prefix_lens, torch.empty_like(q), slots
        )

    return rocm_aiter_ops.qsa_sparse_prefill_warmup(
        run,
        num_q_heads=num_query_heads,
        num_kv_heads=k_cache.shape[2],
        selection_width=selection_width,
        device=device,
    )
