# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ: Ulysses-style sequence sharding of sparse-MLA prefill under TP.

TP shards Q by heads, so every rank runs the indexer for all tokens and the
sparse-MLA kernel pads its few local heads. ShardQ gives each TP rank a slice of
the query tokens instead: the indexer scores only the rank's tokens, an
all-to-all moves Q from [T, H / tp, D] to [T_rank, H, D], attention runs with
all heads over the replicated KV cache, and a second all-to-all restores
[T, H / tp, D_v]. Each request's tokens fold into ``2 * tp`` pieces; rank ``r``
owns pieces ``r`` and ``2 * tp - 1 - r``, which balances the causal indexer.

Both exchanges push rows straight into peers' symmetric-memory buffers with
one Triton kernel, or fall back to NCCL when the TP group has no symmetric
memory.
"""

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_tp_group
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.cp_common import direct_cp_enabled, symm_mem

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backend import CommonAttentionMetadata


@dataclass
class ShardQSplit:
    """This step's token split over the TP group."""

    world: int
    rank: int
    # Tokens per rank.
    counts: list[int]
    # Every rank's tokens, concatenated in rank order: [num_tokens].
    rank_order: torch.Tensor
    # This rank's tokens (a view of ``rank_order``), ascending.
    local_idx: torch.Tensor
    # This rank's tokens on the host, for per-chunk bounds.
    local_idx_cpu: np.ndarray
    # Destination rank and position there of each ``rank_order`` row.
    row_peer: torch.Tensor
    row_pos: torch.Tensor

    @property
    def num_local(self) -> int:
        return self.counts[self.rank]


def fold_split(query_start_loc: np.ndarray, world: int) -> list[np.ndarray]:
    """Each rank's token ids under the per-request two-piece fold."""
    pieces = 2 * world
    starts, lens = query_start_loc[:-1], np.diff(query_start_loc)
    bounds = starts[:, None] + (np.arange(pieces + 1) * lens[:, None]) // pieces
    out = []
    for rank in range(world):
        mirror = pieces - 1 - rank
        ranges = [np.empty(0, np.int64)]
        for b in bounds:
            ranges.append(np.arange(b[rank], b[rank + 1]))
            ranges.append(np.arange(b[mirror], b[mirror + 1]))
        # Ascending: a request's mirror piece follows its front piece.
        out.append(np.concatenate(ranges).astype(np.int64))
    return out


def make_shardq_split(
    query_start_loc: np.ndarray, world: int, rank: int, device: torch.device
) -> ShardQSplit:
    per_rank = fold_split(query_start_loc, world)
    counts = [len(x) for x in per_rank]
    host = torch.from_numpy(
        np.stack(
            (
                np.concatenate(per_rank),
                np.repeat(np.arange(world), counts),
                np.concatenate([np.arange(n) for n in counts]),
            )
        )
    )
    if device.type == "cuda":
        host = host.pin_memory()
    rank_order, row_peer, row_pos = host.to(device, non_blocking=True)
    start = sum(counts[:rank])
    return ShardQSplit(
        world=world,
        rank=rank,
        counts=counts,
        rank_order=rank_order,
        local_idx=rank_order[start : start + counts[rank]],
        local_idx_cpu=per_rank[rank],
        row_peer=row_peer,
        row_pos=row_pos,
    )


def build_shardq_split(
    common_attn_metadata: "CommonAttentionMetadata",
    vllm_config: "VllmConfig",
    device: torch.device,
) -> tuple[int, ShardQSplit] | None:
    """This step's leading decode tokens and the split of the tokens after
    them, or None when too few tokens follow the decodes.

    Decodes are the indexer's: requests of at most 1 + num_speculative_tokens
    tokens. They stay head-sharded. The main MLA layer's builder and its
    indexer's builder each call this and, like every TP rank, derive the same
    answer from the same batch.
    """
    spec_config = vllm_config.speculative_config
    num_spec = spec_config.num_speculative_tokens if spec_config else 0
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        split_decodes_and_prefills(common_attn_metadata, decode_threshold=1 + num_spec)
    )
    min_tokens = vllm_config.attention_config.sparse_mla_shard_q_min_tokens
    if num_prefills == 0 or num_prefill_tokens < min_tokens:
        return None
    num_reqs = common_attn_metadata.num_reqs
    qsl = (
        common_attn_metadata.query_start_loc_cpu[num_decodes : num_reqs + 1]
        .numpy()
        .astype(np.int64)
        - num_decode_tokens
    )
    if qsl[-1] != num_prefill_tokens:
        return None
    tp = get_tp_group()
    split = make_shardq_split(qsl, tp.world_size, tp.rank_in_group, device)
    return num_decode_tokens, split


def indexer_rows(
    split: ShardQSplit, num_decode_tokens: int
) -> tuple[np.ndarray, torch.Tensor]:
    """Rows the indexer scores for a step whose prefill tokens follow
    ``num_decode_tokens`` head-sharded decode tokens, ascending on the host and
    on ``split``'s device.

    Every rank scores every decode row and its own prefill rows, and writes
    the top-k of the p-th row to buffer row p: decode rows stay in place, and
    this rank's prefill rows follow them compactly.
    """
    host = np.concatenate(
        (np.arange(num_decode_tokens), split.local_idx_cpu + num_decode_tokens)
    )
    device = split.local_idx.device
    rows = torch.cat(
        (
            torch.arange(num_decode_tokens, device=device),
            split.local_idx + num_decode_tokens,
        )
    )
    return host, rows


@dataclass
class ShardQIndexerRows:
    """The rows of an indexer prefill chunk this TP rank scores (see
    indexer_rows)."""

    q_rows: torch.Tensor
    # Their top-k rows: compact, in scoring order.
    topk_rows: slice
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor


def chunk_indexer_rows(
    rows_cpu: np.ndarray,
    rows: torch.Tensor,
    token_start: int,
    token_end: int,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> ShardQIndexerRows:
    """Narrow the chunk of rows [token_start, token_end) and its per-row KV
    bounds to the ``indexer_rows`` it holds."""
    lo, hi = (int(i) for i in np.searchsorted(rows_cpu, (token_start, token_end)))
    q_rows = rows[lo:hi]
    chunk_rows = q_rows - token_start
    return ShardQIndexerRows(
        q_rows=q_rows,
        topk_rows=slice(lo, hi),
        cu_seqlen_ks=cu_seqlen_ks[chunk_rows],
        cu_seqlen_ke=cu_seqlen_ke[chunk_rows],
    )


def reserve_shardq_buffers(
    max_tokens: int,
    max_reqs: int,
    num_local_heads: int,
    qk_dim: int,
    v_dim: int,
    q_dtype: torch.dtype,
    out_dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Allocate the exchange buffers of a worst-case step in the memory profile.

    Profile runs never build a ``ShardQSplit``. Symmetric-memory buffers live
    outside the torch caching allocator for good, so they are allocated here at
    their largest size and counted as non-torch memory; NCCL buffers are
    per-call transients, reserved inside the profiled peak instead.

    ``qk_dim`` and ``v_dim`` are the MQA query and output widths.
    """
    world = get_tp_group().world_size
    # A rank owns at most two pieces of ceil(n / (2 * world)) per request.
    q_rows = -(-max_tokens // world) + 2 * max_reqs
    q_numel = q_rows * world * num_local_heads * qk_dim
    out_numel = max_tokens * num_local_heads * v_dim
    if _use_sm(None):
        group = get_tp_group().device_group
        _symm_buffer(group, "q", q_numel, q_dtype, device)
        _symm_buffer(group, "out", out_numel, out_dtype, device)
    else:
        # Send and receive buffers of both exchanges.
        nbytes = q_numel * q_dtype.itemsize + out_numel * out_dtype.itemsize
        _ = torch.empty(2 * nbytes, dtype=torch.uint8, device=device)


@triton.jit
def _shard_q_kernel(
    q,
    q_stride,
    rank_order,
    row_peer,
    row_pos,
    peer_ptrs,
    dst_stride,
    dst_col,
    width,
    BLOCK: tl.constexpr,
):
    """Push q row ``rank_order[i]`` to rank ``row_peer[i]``, at row
    ``row_pos[i]`` there and column ``dst_col``."""
    i = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < width
    src_row = tl.load(rank_order + i)
    dst_row = tl.load(row_pos + i)
    dst = tl.load(peer_ptrs + tl.load(row_peer + i)).to(
        tl.pointer_type(q.dtype.element_ty)
    )
    vals = tl.load(q + src_row * q_stride + col, mask=mask)
    tl.store(dst + dst_row * dst_stride + dst_col + col, vals, mask=mask)


@triton.jit(do_not_specialize=["num_leading_rows"])
def _unshard_out_kernel(
    out,
    out_stride,
    local_idx,
    peer_ptrs,
    num_leading_rows,
    world,
    width,
    BLOCK: tl.constexpr,
):
    """Push head block ``j`` of local token ``p`` to rank ``j``, at the
    token's row after ``num_leading_rows``."""
    pid = tl.program_id(0)
    p = (pid // world).to(tl.int64)
    j = pid % world
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < width
    dst_row = tl.load(local_idx + p) + num_leading_rows
    dst = tl.load(peer_ptrs + j).to(tl.pointer_type(out.dtype.element_ty))
    vals = tl.load(out + p * out_stride + j * width + col, mask=mask)
    tl.store(dst + dst_row * width + col, vals, mask=mask)


_PUSH_BLOCK = 1024


@dataclass
class _SymmBuffer:
    storage: torch.Tensor
    handle: Any
    # Every rank's buffer address, [world].
    peer_ptrs: torch.Tensor


# Grown geometrically; outgrown buffers stay alive for queued kernels.
_buffers: dict[tuple, _SymmBuffer] = {}
_retired: list[_SymmBuffer] = []


def _symm_buffer(
    group: dist.ProcessGroup, role: str, numel: int, dtype: torch.dtype, device
) -> _SymmBuffer:
    # Sizes depend only on the split, so every rank grows (and rendezvouses)
    # in lockstep.
    key = (group.group_name, role, dtype)
    buf = _buffers.get(key)
    if buf is not None and buf.storage.numel() >= numel:
        return buf
    if buf is not None:
        _retired.append(buf)
    capacity = max(numel, 2 * buf.storage.numel() if buf is not None else 0)
    storage = symm_mem.empty(capacity, dtype=dtype, device=device)
    handle = symm_mem.rendezvous(storage, group.group_name)
    peer_ptrs = torch.tensor(
        [
            handle.get_buffer(peer, [capacity], dtype, 0).data_ptr()
            for peer in range(group.size())
        ],
        dtype=torch.int64,
        device=device,
    )
    buf = _buffers[key] = _SymmBuffer(storage, handle, peer_ptrs)
    return buf


@functools.cache
def _tp_has_symm_mem() -> bool:
    # A collective probe; every TP rank reaches the first exchange together.
    return direct_cp_enabled(get_tp_group(), torch.bfloat16, None)


def _use_sm(group: dist.ProcessGroup | None) -> bool:
    if symm_mem is None:
        return False
    return group is not None or _tp_has_symm_mem()


def shard_q(
    q: torch.Tensor,
    split: ShardQSplit,
    group: dist.ProcessGroup | None = None,
    use_sm: bool | None = None,
) -> torch.Tensor:
    """[T, H_local, D] head shards -> [T_rank, world * H_local, D] token shard.

    The symmetric-memory result views a buffer that the next call reuses.
    """
    _, heads, dim = q.shape
    assert q.stride(2) == 1 and q.stride(1) == dim
    if use_sm is None:
        use_sm = _use_sm(group)
    group = group or get_tp_group().device_group
    all_heads = split.world * heads
    if not use_sm:
        # Exchange bytes: NCCL may lack the dtype (e.g. fp8).
        send = q.index_select(0, split.rank_order).view(torch.uint8)
        recv = send.new_empty((split.world * split.num_local, *send.shape[1:]))
        dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=[split.num_local] * split.world,
            input_split_sizes=split.counts,
            group=group,
        )
        return (
            recv.view(q.dtype)
            .view(split.world, split.num_local, heads, dim)
            .transpose(0, 1)
            .reshape(split.num_local, all_heads, dim)
        )
    buf = _symm_buffer(
        group, "q", max(split.counts) * all_heads * dim, q.dtype, q.device
    )
    width = heads * dim
    # Peers may still read the previous exchange from their buffers.
    buf.handle.barrier()
    _shard_q_kernel[(split.rank_order.numel(), triton.cdiv(width, _PUSH_BLOCK))](
        q,
        q.stride(0),
        split.rank_order,
        split.row_peer,
        split.row_pos,
        buf.peer_ptrs,
        all_heads * dim,
        split.rank * width,
        width,
        BLOCK=_PUSH_BLOCK,
    )
    buf.handle.barrier()
    return buf.storage[: split.num_local * all_heads * dim].view(
        split.num_local, all_heads, dim
    )


def unshard_out(
    out: torch.Tensor,
    split: ShardQSplit,
    group: dist.ProcessGroup | None = None,
    use_sm: bool | None = None,
    num_leading_rows: int = 0,
) -> torch.Tensor:
    """[T_rank, world * H_local, D] token shard -> [T, H_local, D] head shards.

    The result has ``num_leading_rows`` uninitialized rows before the T
    restored ones, for a mixed step's decode output to fill without copying
    the prefill rows again. The symmetric-memory result views a buffer that
    the next call reuses.
    """
    _, all_heads, dim = out.shape
    heads = all_heads // split.world
    assert out.stride(2) == 1 and out.stride(1) == dim
    if use_sm is None:
        use_sm = _use_sm(group)
    group = group or get_tp_group().device_group
    num_tokens = sum(split.counts)
    if not use_sm:
        send = (
            out.view(split.num_local, split.world, heads, dim)
            .transpose(0, 1)
            .reshape(split.world * split.num_local, heads, dim)
        )
        recv = out.new_empty((num_tokens, heads, dim))
        dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=split.counts,
            input_split_sizes=[split.num_local] * split.world,
            group=group,
        )
        restored = recv.new_empty((num_leading_rows + num_tokens, heads, dim))
        restored[num_leading_rows:].index_copy_(0, split.rank_order, recv)
        return restored
    num_rows = num_leading_rows + num_tokens
    buf = _symm_buffer(group, "out", num_rows * heads * dim, out.dtype, out.device)
    width = heads * dim
    buf.handle.barrier()
    _unshard_out_kernel[
        (split.num_local * split.world, triton.cdiv(width, _PUSH_BLOCK))
    ](
        out,
        out.stride(0),
        split.local_idx,
        buf.peer_ptrs,
        num_leading_rows,
        split.world,
        width,
        BLOCK=_PUSH_BLOCK,
    )
    buf.handle.barrier()
    return buf.storage[: num_rows * heads * dim].view(num_rows, heads, dim)
