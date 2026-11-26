# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER sparse MLA backend for HY V4 top-k rows that are not front-packed (ROCm).

The AITER sparse path reads each row's first `paged_kv_indptr` delta entries
and maps an invalid index to slot 0 rather than skipping it. The DFly draft
composes each top-k row from the target's history, padding, and the draft's
causal block, so its live entries are not at the front of the row and their
count depends on the target's history. This backend packs every row in place
and rebuilds `paged_kv_indptr` from the live counts before each layer reads it,
without a host sync.
"""

from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
    ROCMAiterMLASparseImpl,
)

logger = init_logger(__name__)


def warn_if_aiter_triton_sparse_mla_disabled() -> None:
    """Warn when HY V4 sparse MLA would use AITER's default decode kernel.

    That kernel reads the wrong KV rows once a layer's KV cache passes 4 GiB
    (fixed by ROCm/aiter#6277); the Triton sparse MLA kernel does not.
    """
    if rocm_aiter_ops.is_triton_sparse_mla_enabled():
        return
    logger.warning_once(
        "HYV4 on ROCm should run with VLLM_ROCM_USE_AITER=1 and "
        "VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1 (gfx950): AITER's default "
        "sparse MLA decode kernel reads the wrong KV rows once a layer's KV "
        "cache exceeds 4 GiB (ROCm/aiter#6277)."
    )


@triton.jit
def _pack_live_indices_kernel(
    indices_ptr,
    counts_ptr,
    row_stride,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    in_row = columns < WIDTH
    base = indices_ptr + row.to(tl.int64) * row_stride
    values = tl.load(base + columns, mask=in_row, other=-1)
    live = values >= 0
    destination = tl.cumsum(live.to(tl.int32), axis=0) - 1
    count = tl.sum(live.to(tl.int32), axis=0)
    # Every lane must read its entry before any lane overwrites the row.
    tl.debug_barrier()
    tl.store(base + destination, values, mask=in_row & live)
    tl.store(base + columns, -1, mask=in_row & (columns >= count))
    tl.store(counts_ptr + row, count)


def pack_live_indices_(indices: torch.Tensor) -> torch.Tensor:
    """Stably move each row's live indices to its front, in place.

    Returns the number of live indices per row.
    """
    rows, width = indices.shape
    counts = torch.empty(rows, dtype=torch.int32, device=indices.device)
    if rows:
        _pack_live_indices_kernel[(rows,)](
            indices,
            counts,
            indices.stride(0),
            WIDTH=width,
            BLOCK=triton.next_power_of_2(width),
        )
    return counts


class HYV4AiterMLASparseImpl(ROCMAiterMLASparseImpl):
    """Sparse MLA that packs and counts each top-k row before reading it."""

    supports_shardq: ClassVar[bool] = False

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        assert self.topk_indices_buffer is not None
        num_tokens = (q[0] if isinstance(q, tuple) else q).shape[0]
        counts = pack_live_indices_(self.topk_indices_buffer[:num_tokens])
        indptr = attn_metadata.paged_kv_indptr
        torch.cumsum(counts, dim=0, out=indptr[1 : num_tokens + 1])
        tail = indptr[num_tokens + 1 :]
        tail.copy_(indptr[num_tokens].expand_as(tail))
        return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)


class HYV4AiterMLASparseBackend(ROCMAiterMLASparseBackend):
    @staticmethod
    def get_impl_cls():
        return HYV4AiterMLASparseImpl

    @classmethod
    def supports_pcp(cls):
        return False

    @classmethod
    def supports_dcp(cls):
        return False
