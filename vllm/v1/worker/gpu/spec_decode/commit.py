# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Commit of a speculative round along the accepted path.

Every verify row wrote its KV at the chain slot of its row index
(``num_computed_tokens + row``). After acceptance the accepted rows of a plan that is
not a chain are not a prefix of the rows, so :class:`KvCompactor` copies the KV of
path row ``path[j]`` to chain slot ``j`` in every attention layer; the next step then
sees the cache a chain round would have left. A chain plan's path is ``arange``: the
commit moves nothing and launches nothing.

The recurrent state of hybrid models commits by row index instead: the model state
resumes from the state the last accepted row wrote (``accepted_row``), and
:class:`ConvWindowAligner` gathers the path's conv inputs into the window the next
step reads.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.capabilities import PlanKind
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
    """The round's :class:`AcceptResult`: the path the speculator's ``accept``
    chose, else the chain's.

    ``chain_path`` is the constant ``[R, C]`` table ``path[r, j] == j``; nothing is
    computed, so a chain step pays no launch for the result.
    """
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
    """Where token ``t`` of head ``h`` of plane ``p`` (K or V) of one layer lives:
    ``(t // block_size) * stride_block + (t % block_size) * stride_slot
    + h * stride_head + p * stride_plane``, ``head_dim`` contiguous elements."""

    stride_block: int
    stride_slot: int
    stride_head: int
    stride_plane: int
    block_size: int
    num_heads: int
    num_planes: int
    head_dim: int
    dtype: torch.dtype


def kv_cache_layout(kv_cache: torch.Tensor, num_blocks: int) -> KvCacheLayout:
    """The layout of a layer's KV cache tensor as the attention backend views it.

    Takes the logical shapes vLLM's backends use: ``[blocks, heads, block_size,
    content]`` (K and V packed in the content, FlashAttention, FlashInfer,
    FlexAttention), ``[2, blocks, block_size, heads, dim]``, ``[blocks, 2,
    block_size, heads, dim]`` and ``[blocks, block_size, dim]`` (MLA); the strides
    are the tensor's own, so a permuted physical order is read correctly. A packed
    content row moves whole, so it is one plane of ``content`` elements.
    """
    shape, stride = kv_cache.shape, kv_cache.stride()
    if kv_cache.dim() == 4 and shape[0] == num_blocks:
        if stride[3] != 1:
            raise NotImplementedError("KV compaction needs contiguous head dims")
        _check_dtype(kv_cache.dtype)
        return KvCacheLayout(
            stride_block=stride[0],
            stride_slot=stride[2],
            stride_head=stride[1],
            stride_plane=0,
            block_size=shape[2],
            num_heads=shape[1],
            num_planes=1,
            head_dim=shape[3],
            dtype=kv_cache.dtype,
        )
    if kv_cache.dim() == 5 and shape[0] == 2 and shape[1] == num_blocks:
        plane, block = 0, 1
    elif kv_cache.dim() == 5 and shape[0] == num_blocks and shape[1] == 2:
        block, plane = 0, 1
    elif kv_cache.dim() == 3 and shape[0] == num_blocks:
        if stride[2] != 1:
            raise NotImplementedError("KV compaction needs contiguous head dims")
        _check_dtype(kv_cache.dtype)
        return KvCacheLayout(
            stride[0], stride[1], 0, 0, shape[1], 1, 1, shape[2], kv_cache.dtype
        )
    else:
        raise NotImplementedError(
            f"KV compaction does not know the KV cache layout {tuple(shape)}"
        )
    if stride[4] != 1:
        raise NotImplementedError("KV compaction needs contiguous head dims")
    _check_dtype(kv_cache.dtype)
    return KvCacheLayout(
        stride_block=stride[block],
        stride_slot=stride[2],
        stride_head=stride[3],
        stride_plane=stride[plane],
        block_size=shape[2],
        num_heads=shape[3],
        num_planes=2,
        head_dim=shape[4],
        dtype=kv_cache.dtype,
    )


@triton.jit
def _compact_kv_kernel(
    layer_ptrs,  # [L] uint64: base address of each layer's KV cache
    table_ptrs,  # [L] uint64: the layer's group block table, [num_reqs, max_blocks]
    table_strides,  # [L] int64: row stride of that table
    table_block_sizes,  # [L] int32: tokens per block of that table
    count_ptr,  # [R] int32: verify rows per request, by batch row
    seq_lens_ptr,  # [num_reqs] int32
    path_ptr,  # [R, C] int32, by batch row
    stride_path,
    path_len_ptr,  # [R] int32, by batch row
    stride_block,
    stride_slot,
    stride_head,
    stride_plane,
    block_size,  # tokens per block of the cache as the backend views it
    NUM_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    DIM_P: tl.constexpr,
    DTYPE: tl.constexpr,
):
    b, layer, plane_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    plane = plane_head // NUM_HEADS
    head = plane_head % NUM_HEADS
    # The verify rows are the query's last rows, so row r sits at context + r.
    context = tl.load(seq_lens_ptr + b) - tl.load(count_ptr + b)
    base = tl.load(layer_ptrs + layer).to(tl.pointer_type(DTYPE))
    table = tl.load(table_ptrs + layer).to(tl.pointer_type(tl.int32))
    table_row = table + b * tl.load(table_strides + layer)
    table_block_size = tl.load(table_block_sizes + layer).to(tl.int64)
    offset = head * stride_head + plane * stride_plane
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
    the chain slots in the layers it is bound to.

    Each layer is read through its KV cache group's block table as the forward
    got it (the batch's rows) and that group's tokens per block. Layers that
    share one layout go in one launch. A layer that reads another layer's cache
    (KV sharing) must not be bound: it would be compacted twice.
    """

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
            assert table is not None
            if table.dtype != torch.int32:
                raise ValueError("block tables must be int32")
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
        ``num_reqs`` batch rows, whose ``count[b]`` verify rows end at
        ``seq_lens[b]``."""
        result = step.result
        for layout, layer_ptrs, table_ptrs, table_strides, table_sizes in self.launches:
            grid = (
                step.num_reqs,
                layer_ptrs.shape[0],
                layout.num_planes * layout.num_heads,
            )
            _compact_kv_kernel[grid](
                layer_ptrs,
                table_ptrs,
                table_strides,
                table_sizes,
                step.count,
                step.seq_lens,
                result.path,
                result.path.stride(0),
                result.path_len,
                layout.stride_block,
                layout.stride_slot,
                layout.stride_head,
                layout.stride_plane,
                layout.block_size,
                NUM_HEADS=layout.num_heads,
                DIM=layout.head_dim,
                DIM_P=triton.next_power_of_2(layout.head_dim),
                DTYPE=getattr(tl, _BITS[layout.dtype.itemsize]),
                num_warps=2,
            )


# The copy moves bits, so any cache dtype (fp8 included) copies as an integer.
# Names, resolved at launch: where vLLM disables Triton, ``tl`` is a placeholder
# without int8 / int16, and the runner imports this module.
_BITS = {1: "int8", 2: "int16", 4: "int32"}


def copies_dtype(dtype: torch.dtype) -> bool:
    """Whether the compactor and the aligner copy caches of ``dtype``."""
    return dtype.itemsize in _BITS


def _check_dtype(dtype: torch.dtype) -> None:
    if not copies_dtype(dtype):
        raise NotImplementedError(f"KV compaction does not copy {dtype} caches")


@triton.jit
def _align_conv_window_kernel(
    window_ptrs,  # [L] uint64: base of each layer's conv state, viewed [slot, dim, tok]
    table_ptrs,  # [L] uint64: the layer group's block table, batch rows
    table_strides,  # [L] int64
    path_ptr,  # [R, C] int32
    stride_path,
    path_len_ptr,  # [R] int32
    stride_slot,
    stride_dim,
    stride_tok,
    dim,
    WIDTH: tl.constexpr,  # conv kernel width k
    DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    layer = tl.program_id(1)
    length = tl.load(path_len_ptr + b)
    if length <= 0:
        return
    last = tl.load(path_ptr + b * stride_path + length - 1)
    table = tl.load(table_ptrs + layer).to(tl.pointer_type(tl.int32))
    slot = tl.load(table + b * tl.load(table_strides + layer)).to(tl.int64)
    if slot <= 0:  # the null block: a request without state in this group
        return
    feats = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask = feats < dim
    base = (
        tl.load(window_ptrs + layer).to(tl.pointer_type(DTYPE))
        + slot * stride_slot
        + feats * stride_dim
    )
    # Column (k-2)+j holds the conv input of row j (j < 0: the history). The next
    # step reads the k-1 columns ending at (k-2)+last, which must hold the path's
    # inputs: path[length-1-i], or the history once the path is exhausted. The path
    # is increasing, so a source never lies above its destination nor at a column
    # already written; descending destinations make the copy safe in place.
    for i in tl.static_range(WIDTH - 1):
        j = length - 1 - i
        row = tl.where(
            j >= 0, tl.load(path_ptr + b * stride_path + tl.maximum(j, 0)), j
        )
        value = tl.load(base + (WIDTH - 2 + row) * stride_tok, mask=mask)
        tl.store(base + (WIDTH - 2 + last - i) * stride_tok, value, mask=mask)


class ConvWindowAligner(SpecKernel):
    """The ``layout:mamba_state@slots`` commit: lines up the conv windows of
    recurrent layers with the accepted path.

    After a verify step the conv state of a request holds the conv inputs of its
    rows in row order, and the next step reads the ``k - 1`` columns that end at
    the last accepted row (``num_accepted_tokens - 1``). For a chain those are
    the accepted rows; for a tree they must be gathered from the path. The SSM
    state needs no move: it resumes from the state row the last accepted row
    wrote. Column 0 of a layer's block table is the request's state slot.
    """

    def __init__(self) -> None:
        self.window_ptrs: torch.Tensor | None = None

    def supports(self, layer: SpecLayerSpec) -> frozenset[PlanKind]:
        return ABOVE_NONE

    def bind(self, ctx: KernelContext, layers: Sequence[SpecLayerSpec]) -> None:
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        windows: list[torch.Tensor] = []
        tables: list[torch.Tensor] = []
        widths: set[int] = set()
        for layer in layers:
            if not hasattr(layer.module, "conv1d"):
                continue
            assert layer.block_table is not None
            conv = layer.module.kv_cache[0]
            # Viewed [slot, dim, tok].
            windows.append(conv if is_conv_state_dim_first() else conv.mT)
            tables.append(layer.block_table)
            widths.add(layer.module.conv1d.weight.shape[-1])
        if not windows:
            return
        if len(widths) != 1:
            raise NotImplementedError(
                "verify plans that are not chains need one conv width per model"
            )
        width = widths.pop()
        first = windows[0]
        for window in windows:
            if window.stride() != first.stride() or window.dtype != first.dtype:
                raise ValueError("every conv state must share one layout and dtype")
        if any(t.dtype != torch.int32 for t in tables):
            raise ValueError("block tables must be int32")
        if first.shape[2] < width - 1:
            raise ValueError("the conv state is narrower than its kernel")
        self.width = width
        self.dim = first.shape[1]
        self.strides = first.stride()
        self.dtype = getattr(tl, _BITS[first.dtype.itemsize])
        self.window_ptrs = torch.tensor(
            [w.data_ptr() for w in windows], dtype=torch.uint64, device=ctx.device
        )
        self.table_ptrs = torch.tensor(
            [t.data_ptr() for t in tables], dtype=torch.uint64, device=ctx.device
        )
        self.table_strides = torch.tensor(
            [t.stride(0) for t in tables], dtype=torch.int64, device=ctx.device
        )

    def commit(self, step: CommitStep) -> None:
        if self.window_ptrs is None or step.num_reqs == 0:
            return
        result = step.result
        block = 256
        grid = (step.num_reqs, self.window_ptrs.shape[0], triton.cdiv(self.dim, block))
        _align_conv_window_kernel[grid](
            self.window_ptrs,
            self.table_ptrs,
            self.table_strides,
            result.path,
            result.path.stride(0),
            result.path_len,
            *self.strides,
            self.dim,
            WIDTH=self.width,
            DTYPE=self.dtype,
            BLOCK=block,
            num_warps=2,
        )
