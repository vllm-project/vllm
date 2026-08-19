# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plain FP16 paged-K gather kernels for the Turing backend.

``gather_fp16_k_rows`` is the FP16 analogue of
``dequantize_and_gather_k_cache`` (which dequantizes the UE8M0 FP8 cache to
BF16); it copies FP16 rows from a plain-row paged cache into a dense gather
workspace for sparse prefill. ``gather_fp16_slots`` scatters global slot ids
into a flat FP16 workspace for decode.
"""

import torch

from vllm.models.deepseek_v4.turing.constants import HEAD_DIM
from vllm.triton_utils import tl, triton


@triton.jit
def _gather_fp16_k_rows_kernel(
    out_ptr,  # [num_reqs, max_tokens, HEAD_DIM] fp16
    out_stride0,
    out_stride1,
    k_cache_ptr,  # [num_blocks, block_size, HEAD_DIM] fp16
    seq_lens_ptr,
    block_table_ptr,
    gather_lens_ptr,
    offset,
    max_blocks_per_seq: tl.constexpr,
    cache_block_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Gather the trailing ``gather_len`` FP16 KV rows of each sequence."""
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    offs = tl.arange(0, HEAD_DIM)
    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size
        physical_block_idx = tl.load(
            block_table_ptr + batch_idx * max_blocks_per_seq + block_in_seq
        )
        cache_row = (
            k_cache_ptr
            + physical_block_idx.to(tl.int64) * cache_block_size * HEAD_DIM
            + pos_in_block * HEAD_DIM
        )
        out_row = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1
        tl.store(out_row + offs, tl.load(cache_row + offs))


def gather_fp16_k_rows(
    out: torch.Tensor,  # [num_reqs, max_tokens, HEAD_DIM] fp16
    k_cache: torch.Tensor,  # [num_blocks, block_size, HEAD_DIM] fp16
    seq_lens: torch.Tensor,  # [num_reqs]
    gather_lens: torch.Tensor | None,  # [num_reqs]
    block_table: torch.Tensor,  # [num_reqs, max_blocks_per_seq]
    block_size: int,
    offset: int,
) -> None:
    """Copy FP16 KV rows from a paged cache into a dense gather workspace."""
    num_reqs = seq_lens.shape[0]
    if num_reqs == 0:
        return
    _gather_fp16_k_rows_kernel[(num_reqs, 128)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        gather_lens,
        offset,
        max_blocks_per_seq=block_table.shape[-1],
        cache_block_size=block_size,
        HEAD_DIM=HEAD_DIM,
    )


@triton.jit
def _gather_fp16_slots_kernel(
    out_ptr,  # [total_slots, HEAD_DIM] fp16
    k_cache_ptr,  # [num_blocks, block_size, HEAD_DIM] fp16
    indices_ptr,  # [total_slots] int32, global slot ids
    cache_block_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Copy scattered FP16 KV slots; negative slots write zeros."""
    pid = tl.program_id(0)

    slot = tl.load(indices_ptr + pid).to(tl.int64)
    out_row = out_ptr + pid * HEAD_DIM
    offs = tl.arange(0, HEAD_DIM)
    if slot < 0:
        tl.store(out_row + offs, tl.zeros((HEAD_DIM,), dtype=tl.float16))
        return
    block_idx = slot // cache_block_size
    pos_in_block = slot % cache_block_size
    cache_row = (
        k_cache_ptr + block_idx * cache_block_size * HEAD_DIM + pos_in_block * HEAD_DIM
    )
    tl.store(out_row + offs, tl.load(cache_row + offs))


def gather_fp16_slots(
    out: torch.Tensor,  # [total_slots, HEAD_DIM] fp16
    k_cache: torch.Tensor,  # [num_blocks, block_size, HEAD_DIM] fp16
    indices: torch.Tensor,  # [total_slots] int32 global slot ids
    block_size: int,
) -> None:
    """Scatter global slot ids into a flat FP16 workspace."""
    total_slots = indices.shape[0]
    if total_slots == 0:
        return
    _gather_fp16_slots_kernel[(total_slots,)](
        out,
        k_cache,
        indices,
        cache_block_size=block_size,
        HEAD_DIM=HEAD_DIM,
    )
