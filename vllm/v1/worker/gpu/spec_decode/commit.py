# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Commit of a speculative round along the accepted path: :class:`KvCompactor`
copies the KV of path row ``path[j]`` to chain slot ``j``, so the next step sees
the cache a chain round would have left. Recurrent state resumes from the last
accepted row (``accepted_row``)."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.spec_decode.resources import PlanKind
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.kernels import (
    ABOVE_NONE,
    CommitStep,
    KernelContext,
    SpecKernel,
    SpecLayerSpec,
)
from vllm.v1.worker.gpu.spec_decode.round import AcceptResult


def accept_result_of(
    sampler_output: SamplerOutput, chain_path: torch.Tensor
) -> AcceptResult:
    """The round's :class:`AcceptResult`; a chain's path is a view of the
    constant ``chain_path``, so it costs no launch."""
    sampled = sampler_output.sampled_token_ids
    path = sampler_output.accept_path
    if path is None:
        path = chain_path[: sampled.shape[0], : sampled.shape[1]]
    return AcceptResult(
        path=path,
        path_len=sampler_output.num_sampled,
        sampled=sampled,
    )


def accepted_rows(result: AcceptResult, num_reqs: int) -> torch.Tensor:
    """``[num_reqs]`` int32: the verify row whose recurrent state each request resumes
    from, ``path[path_len - 1]`` (row 0 when nothing was committed)."""
    last = (result.path_len[:num_reqs].to(torch.int64) - 1).clamp_(min=0)
    return result.path[:num_reqs].gather(1, last.unsqueeze(1)).squeeze(1).int()


@dataclass(frozen=True)
class KvCacheLayout:
    """Where token ``t`` of head ``h`` of one layer lives: ``(t // block_size) *
    stride_block + (t % block_size) * stride_slot + h * stride_head``,
    ``head_dim`` contiguous elements."""

    stride_block: int
    stride_slot: int
    stride_head: int
    block_size: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype


def kv_cache_layout(kv_cache: torch.Tensor, num_blocks: int) -> KvCacheLayout:
    """The layout of a layer's ``[blocks, heads, block_size, content]`` (MLA:
    ``[blocks, block_size, content]``) KV cache view, by its own strides."""
    shape, stride = kv_cache.shape, kv_cache.stride()
    if kv_cache.dim() not in (3, 4) or shape[0] != num_blocks:
        raise NotImplementedError(
            f"KV compaction does not know the KV cache layout {tuple(shape)}"
        )
    if stride[-1] != 1:
        raise NotImplementedError("KV compaction needs contiguous head dims")
    _check_dtype(kv_cache.dtype)
    if kv_cache.dim() == 3:
        return KvCacheLayout(
            stride[0], stride[1], 0, shape[1], 1, shape[2], kv_cache.dtype
        )
    return KvCacheLayout(
        stride_block=stride[0],
        stride_slot=stride[2],
        stride_head=stride[1],
        block_size=shape[2],
        num_heads=shape[1],
        head_dim=shape[3],
        dtype=kv_cache.dtype,
    )


@triton.jit
def _compact_kv_kernel(
    layer_ptrs,  # [L] uint64: base address of each layer's KV cache
    table_ptrs,  # [L] uint64: the layer's group block table, [num_reqs, max_blocks]
    table_strides,  # [L] int64: row stride of that table
    table_block_sizes,  # [L] int32: tokens per block of that table
    rows_ptr,  # [R] int32: verify rows per request, by batch row
    seq_lens_ptr,  # [num_reqs] int32
    path_ptr,  # [R, C] int32, by batch row
    stride_path,
    path_len_ptr,  # [R] int32, by batch row
    stride_block,
    stride_slot,
    stride_head,
    block_size,  # tokens per block of the cache as the backend views it
    DIM: tl.constexpr,
    DIM_P: tl.constexpr,
    DTYPE: tl.constexpr,
):
    b, layer, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # The verify rows are the query's last rows, so row r sits at context + r.
    context = tl.load(seq_lens_ptr + b) - tl.load(rows_ptr + b)
    base = tl.load(layer_ptrs + layer).to(tl.pointer_type(DTYPE))
    table = tl.load(table_ptrs + layer).to(tl.pointer_type(tl.int32))
    table_row = table + b * tl.load(table_strides + layer)
    table_block_size = tl.load(table_block_sizes + layer).to(tl.int64)
    offset = head * stride_head
    dims = tl.arange(0, DIM_P)
    # Path order: path[j] >= j, so a destination is only ever a source read earlier.
    for j in range(1, tl.load(path_len_ptr + b)):
        src = context + tl.load(path_ptr + b * stride_path + j)
        dst = context + j
        if src != dst:
            src_block = tl.load(table_row + src // table_block_size).to(tl.int64)
            dst_block = tl.load(table_row + dst // table_block_size).to(tl.int64)
            src_token = src_block * table_block_size + src % table_block_size
            dst_token = dst_block * table_block_size + dst % table_block_size
            src_at = (
                (src_token // block_size) * stride_block
                + (src_token % block_size) * stride_slot
                + offset
            )
            dst_at = (
                (dst_token // block_size) * stride_block
                + (dst_token % block_size) * stride_slot
                + offset
            )
            values = tl.load(base + src_at + dims, mask=dims < DIM)
            tl.store(base + dst_at + dims, values, mask=dims < DIM)
            tl.debug_barrier()


class KvCompactor(SpecKernel):
    """The ``layout:paged_token`` commit: moves the KV of accepted path rows to
    the chain slots, one launch per KV layout. A layer that shares another
    layer's cache must not be bound: it would be compacted twice."""

    def __init__(self) -> None:
        self.launches: list[
            tuple[KvCacheLayout, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

    def supports(self, layer: SpecLayerSpec) -> frozenset[PlanKind]:
        cache = getattr(layer.module, "kv_cache", None)
        if isinstance(cache, torch.Tensor) and not copies_dtype(cache.dtype):
            return frozenset()
        return ABOVE_NONE

    def bind(self, ctx: KernelContext, layers: Sequence[SpecLayerSpec]) -> None:
        # layout -> (caches, block tables, table strides, tokens per block)
        by_layout: dict[KvCacheLayout, tuple[list[int], ...]] = {}
        for layer in layers:
            cache, table = layer.module.kv_cache, layer.block_table
            assert table is not None and table.dtype == torch.int32
            layout = kv_cache_layout(cache, layer.num_blocks)
            ptrs = by_layout.setdefault(layout, ([], [], [], []))
            ptrs[0].append(cache.data_ptr())
            ptrs[1].append(table.data_ptr())
            ptrs[2].append(table.stride(0))
            ptrs[3].append(layer.block_size)
        dtypes = (torch.uint64, torch.uint64, torch.int64, torch.int32)
        self.launches = [
            (
                layout,
                *(
                    torch.tensor(values, dtype=dtype, device=ctx.device)
                    for values, dtype in zip(ptrs, dtypes)
                ),
            )
            for layout, ptrs in by_layout.items()
        ]

    def commit(self, step: CommitStep) -> None:
        """Copy the KV of ``result.path[b, j]`` to chain slot ``j`` of the first
        ``num_reqs`` batch rows, whose ``rows[b]`` verify rows end at
        ``seq_lens[b]``."""
        result = step.result
        for layout, layer_ptrs, table_ptrs, table_strides, table_sizes in self.launches:
            grid = (step.num_reqs, layer_ptrs.shape[0], layout.num_heads)
            _compact_kv_kernel[grid](
                layer_ptrs,
                table_ptrs,
                table_strides,
                table_sizes,
                step.rows,
                step.seq_lens,
                result.path,
                result.path.stride(0),
                result.path_len,
                layout.stride_block,
                layout.stride_slot,
                layout.stride_head,
                layout.block_size,
                DIM=layout.head_dim,
                DIM_P=triton.next_power_of_2(layout.head_dim),
                DTYPE=getattr(tl, _BITS[layout.dtype.itemsize]),
                num_warps=2,
            )


# Any cache dtype copies as an integer; names, as ``tl`` may be a placeholder.
_BITS = {1: "int8", 2: "int16", 4: "int32"}


def copies_dtype(dtype: torch.dtype) -> bool:
    """Whether the compactor copies caches of ``dtype``."""
    return dtype.itemsize in _BITS


def _check_dtype(dtype: torch.dtype) -> None:
    if not copies_dtype(dtype):
        raise NotImplementedError(f"KV compaction does not copy {dtype} caches")
