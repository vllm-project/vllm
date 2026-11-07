# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ: Ulysses-style sequence sharding of sparse-MLA prefill under TP.

TP shards Q by heads, so every rank runs the indexer for all tokens and the
sparse-MLA kernel pads its few local heads. ShardQ gives each TP rank a slice of
the query tokens instead: the indexer scores only the rank's tokens, an
all-to-all moves Q from [T, H / tp, D] to [T_rank, H, D], attention runs with
all heads over the replicated KV cache, and a second all-to-all restores
[T, H / tp, D_v]. Tokens split zigzag: each request cuts into ``2 * tp``
chunks and rank ``r`` owns chunks ``r`` and ``2 * tp - 1 - r``, which balances
the causal work.

Both exchanges push tokens straight into peers' symmetric-memory buffers with
one Triton kernel, or fall back to NCCL when the TP group has no symmetric
memory.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.cp_common import DirectCPWorkspace, direct_cp_enabled
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.sparse_mla_attention import (
        SparseMLACommonMetadata,
    )
    from vllm.v1.attention.backend import CommonAttentionMetadata


@dataclass
class ShardQMetadata:
    """This step's ShardQ: which TP rank attends which of the tokens after
    ``num_decode_tokens`` head-sharded decode tokens (ids relative to the first
    of them), and the forward_mqa metadata of this rank's tokens.
    """

    num_decode_tokens: int
    num_tokens_per_rank: list[int]
    # Every rank's tokens, grouped by rank, and where each group starts.
    token_ids: torch.Tensor
    rank_offsets: torch.Tensor
    # forward_mqa metadata of this rank's tokens, which run with all heads.
    attn_metadata: "SparseMLACommonMetadata" = field(init=False)

    @property
    def local_token_ids(self) -> torch.Tensor:
        """This rank's tokens, ascending."""
        rank = get_tp_group().rank_in_group
        start = sum(self.num_tokens_per_rank[:rank])
        return self.token_ids[start : start + self.num_tokens_per_rank[rank]]


def zigzag_split(query_lens: np.ndarray, world: int) -> tuple[np.ndarray, np.ndarray]:
    """Token ids grouped by owning rank (ascending within a rank), and each
    rank's token count.

    world=4 splits each request into eight chunks, chunk ``i`` spanning
    ``[floor(i * len / 8), floor((i + 1) * len / 8))``, and each rank gets an
    early and a late chunk:

        full:  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
        rank 0:  0                           7
        rank 1:      1                   6
        rank 2:          2           5
        rank 3:              3   4
    """
    num_chunks = 2 * world
    starts = np.cumsum(query_lens) - query_lens
    pos = np.arange(query_lens.sum()) - np.repeat(starts, query_lens)
    # The last chunk i with floor(i * len / num_chunks) <= pos.
    chunk = ((pos + 1) * num_chunks - 1) // np.repeat(query_lens, query_lens)
    owner = np.minimum(chunk, num_chunks - 1 - chunk).astype(np.uint8)
    # A stable sort of uint8 is a linear-time radix sort.
    return np.argsort(owner, kind="stable"), np.bincount(owner, minlength=world)


class ShardQSplit(NamedTuple):
    """Which TP rank attends which of the tokens after ``num_decode_tokens``
    head-sharded decode tokens (ids relative to the first of them)."""

    num_decode_tokens: int
    # Token ids grouped by rank.
    token_ids: np.ndarray
    num_tokens_per_rank: np.ndarray

    @property
    def local_token_ids(self) -> np.ndarray:
        """This TP rank's tokens, ascending."""
        rank = get_tp_group().rank_in_group
        start = self.num_tokens_per_rank[:rank].sum()
        return self.token_ids[start : start + self.num_tokens_per_rank[rank]]


def build_shardq_split(
    common_attn_metadata: "CommonAttentionMetadata", vllm_config: "VllmConfig"
) -> ShardQSplit | None:
    """This step's ``zigzag_split`` over the TP group of the tokens after the
    decodes (requests of at most 1 + num_speculative_tokens tokens). None when
    too few tokens follow the decodes, none of their requests is longer than
    the top-k, or a rank would own no tokens.
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
    seq_lens = common_attn_metadata.seq_lens_cpu_upper_bound
    assert seq_lens is not None
    topk_tokens = vllm_config.model_config.hf_text_config.index_topk
    if int(seq_lens[num_decodes:num_reqs].max()) <= topk_tokens:
        return None
    query_lens = common_attn_metadata.query_start_loc_cpu.diff()
    prefill_lens = query_lens[num_decodes:num_reqs].numpy()
    if prefill_lens.sum() != num_prefill_tokens:
        return None
    token_ids, num_tokens_per_rank = zigzag_split(
        prefill_lens, get_tp_group().world_size
    )
    # A rank without tokens would run FlashMLA's kernel on an empty batch,
    # which it rejects. That takes every request here being shorter than
    # 2 * world tokens; steps large enough for sparse_mla_shard_q_min_tokens
    # virtually never look like that.
    if num_tokens_per_rank.min() == 0:
        return None
    return ShardQSplit(num_decode_tokens, token_ids, num_tokens_per_rank)


@triton.jit
def _shard_q_kernel(
    q,
    q_stride,
    token_ids,
    rank_offsets,
    peer_ptrs,
    dst_stride,
    dst_col,
    width,
    BLOCK: tl.constexpr,
):
    """Push the q of token ``token_ids[rank_offsets[peer] + i]`` to token
    ``i`` of rank ``peer``, at column ``dst_col``."""
    i = tl.program_id(0).to(tl.int64)
    peer = tl.program_id(1)
    start = tl.load(rank_offsets + peer)
    valid = start + i < tl.load(rank_offsets + peer + 1)
    col = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask = valid & (col < width)
    src_token = tl.load(token_ids + start + i, mask=valid, other=0)
    dst = tl.load(peer_ptrs + peer).to(tl.pointer_type(q.dtype.element_ty))
    vals = tl.load(q + src_token * q_stride + col, mask=mask)
    tl.store(dst + i * dst_stride + dst_col + col, vals, mask=mask)


@triton.jit(do_not_specialize=["num_decode_tokens"])
def _unshard_out_kernel(
    out,
    out_stride,
    local_token_ids,
    peer_ptrs,
    num_decode_tokens,
    world,
    width,
    BLOCK: tl.constexpr,
):
    """Push head block ``j`` of local token ``p`` to rank ``j``, at the
    token's index after the ``num_decode_tokens`` decode tokens."""
    pid = tl.program_id(0)
    p = (pid // world).to(tl.int64)
    j = pid % world
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < width
    dst_token = tl.load(local_token_ids + p) + num_decode_tokens
    dst = tl.load(peer_ptrs + j).to(tl.pointer_type(out.dtype.element_ty))
    vals = tl.load(out + p * out_stride + j * width + col, mask=mask)
    tl.store(dst + dst_token * width + col, vals, mask=mask)


_PUSH_BLOCK = 1024


class ShardQWorkspace(DirectCPWorkspace):
    """The exchange buffers in symmetric memory, with one slot per ubatch."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        num_ubatches: int,
        q: tuple[tuple[int, ...], torch.dtype],
        out: tuple[tuple[int, ...], torch.dtype],
    ) -> None:
        super().__init__(group, device, num_ubatches)
        # Each is its storage and every rank's address of each ubatch slot.
        self.q = self._allocate((num_ubatches, *q[0]), q[1])
        self.out = self._allocate((num_ubatches, *out[0]), out[1])
        # Any allocation's handle synchronizes the whole group.
        self.handle = self._allocations[0][1]


# Shared by every ShardQ layer; without it the exchanges use NCCL.
_workspace: ShardQWorkspace | None = None


def allocate_shardq_buffers(
    vllm_config: "VllmConfig",
    num_heads: int,
    query_head_dim: int,
    output_head_dim: int,
    query_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> None:
    """Allocate the worst-case exchange buffers, if the TP group has symmetric
    memory. ``num_heads`` are a TP rank's."""
    logger.info_once(
        "Sharding sparse MLA prefill steps of >= %d tokens across TP ranks by "
        "token (sparse_mla_shard_q).",
        vllm_config.attention_config.sparse_mla_shard_q_min_tokens,
    )
    global _workspace
    group = get_tp_group()
    if _workspace is not None or not direct_cp_enabled(group, torch.bfloat16, None):
        return
    max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    # A rank owns at most two chunks of ceil(n / (2 * world)) per request.
    max_local_tokens = (
        -(-max_num_tokens // group.world_size)
        + 2 * vllm_config.scheduler_config.max_num_seqs
    )
    num_ubatches = max(vllm_config.parallel_config.num_ubatches, 1)
    device = torch.device(
        current_platform.device_type, torch.accelerator.current_device_index()
    )
    _workspace = ShardQWorkspace(
        group.device_group,
        device,
        num_ubatches,
        q=(
            (max_local_tokens, group.world_size * num_heads, query_head_dim),
            query_dtype,
        ),
        out=((max_num_tokens, num_heads, output_head_dim), output_dtype),
    )


def _all_to_all(
    send: torch.Tensor,
    recv: torch.Tensor,
    send_counts: list[int] | None,
    recv_counts: list[int] | None,
) -> None:
    # Exchange bytes: NCCL may lack the dtype (e.g. fp8).
    dist.all_to_all_single(
        recv.view(torch.uint8),
        send.view(torch.uint8),
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
        group=get_tp_group().device_group,
    )


def shard_q(
    q: torch.Tensor | tuple[torch.Tensor, torch.Tensor], metadata: ShardQMetadata
) -> tuple[torch.Tensor, torch.Tensor]:
    """[T, H, D] head shards -> the decode tokens, still head-sharded, and
    this rank's token shard with every rank's heads, [T_rank, world * H, D]."""
    if isinstance(q, tuple):
        q = torch.cat(q, dim=-1)
    decode_q, q = q[: metadata.num_decode_tokens], q[metadata.num_decode_tokens :]
    _, heads, dim = q.shape
    assert q[0].is_contiguous()
    group = get_tp_group()
    world = group.world_size
    num_local = metadata.num_tokens_per_rank[group.rank_in_group]
    if _workspace is None:
        # Every rank sends this rank its num_local tokens, in rank order.
        recv = q.new_empty((world, num_local, heads, dim))
        _all_to_all(
            q.index_select(0, metadata.token_ids),
            recv,
            metadata.num_tokens_per_rank,
            None,
        )
        return decode_q, torch.cat(list(recv), dim=1)
    storage, peer_ptrs = _workspace.q
    assert storage.shape[2:] == (world * heads, dim)
    ubatch = dbo_current_ubatch_id()
    width = heads * dim
    # Peers may still read the previous exchange from their buffers.
    _workspace.handle.barrier(channel=ubatch)
    grid = (max(metadata.num_tokens_per_rank), world, triton.cdiv(width, _PUSH_BLOCK))
    _shard_q_kernel[grid](
        q,
        q.stride(0),
        metadata.token_ids,
        metadata.rank_offsets,
        peer_ptrs[ubatch],
        world * width,
        group.rank_in_group * width,
        width,
        BLOCK=_PUSH_BLOCK,
    )
    _workspace.handle.barrier(channel=ubatch)
    # Views the buffer, which the next exchange reuses.
    return decode_q, storage[ubatch, :num_local]


def unshard_out(
    out: torch.Tensor, metadata: ShardQMetadata, decode_out: torch.Tensor | None
) -> torch.Tensor:
    """This rank's token shard of the attention output [T_rank, world * H, D]
    -> [T, H, D] head shards, after the head-sharded ``decode_out``."""
    assert out[0].is_contiguous()
    world = get_tp_group().world_size
    num_local, heads, dim = out.shape[0], out.shape[1] // world, out.shape[2]
    num_split_tokens = sum(metadata.num_tokens_per_rank)
    num_tokens = metadata.num_decode_tokens + num_split_tokens
    if _workspace is None:
        # One block of heads per destination rank.
        send = torch.stack(out.chunk(world, dim=1))
        recv = out.new_empty((num_split_tokens, heads, dim))
        _all_to_all(send, recv, None, metadata.num_tokens_per_rank)
        restored = out.new_empty((num_tokens, heads, dim))
        restored[metadata.num_decode_tokens :].index_copy_(0, metadata.token_ids, recv)
    else:
        storage, peer_ptrs = _workspace.out
        assert storage.shape[2:] == (heads, dim)
        ubatch = dbo_current_ubatch_id()
        width = heads * dim
        _workspace.handle.barrier(channel=ubatch)
        _unshard_out_kernel[(num_local * world, triton.cdiv(width, _PUSH_BLOCK))](
            out,
            out.stride(0),
            metadata.local_token_ids,
            peer_ptrs[ubatch],
            metadata.num_decode_tokens,
            world,
            width,
            BLOCK=_PUSH_BLOCK,
        )
        _workspace.handle.barrier(channel=ubatch)
        # Views the buffer, which the next exchange reuses.
        restored = storage[ubatch, :num_tokens]
    if decode_out is not None:
        restored[: metadata.num_decode_tokens] = decode_out
    return restored
