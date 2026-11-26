# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ prefill steps under DCP: attention over a gathered context.

A DCP prefill step runs every query token with all heads against each rank's
1/W of the keys, then merges the indexer top-k and combines the attention
partials across ranks. When the DCP group is the TP group, it can instead run
as a TP ShardQ step (vllm.v1.attention.ops.shardq): per layer, the indexer
pushes every rank's KV shard of the step's prefill requests (MLA and indexer
pages) into every rank's symmetric-memory workspace, then each rank scores and
attends its slice of the query tokens with all heads over that context. The
MLA layer writes the new tokens' MLA KV, which every rank computes, into its
own workspace.

The step's leading decodes keep the DCP path. A step whose prefill requests do
not fit the workspace (`max_model_len` tokens plus a page per request) takes
the DCP path whole.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import vllm.envs as envs
from vllm.distributed import get_dcp_group
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV32IndexerPrefillChunkMetadata,
    build_prefill_chunk_metadata,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.ops.cp_common import symm_mem
from vllm.v1.attention.ops.dcp_shardq import PAGE_TOKENS, TABLE_PAGES_ALIGN
from vllm.v1.attention.ops.shardq import ShardQSplit
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

logger = init_logger(__name__)

# Indexer key formats (fp8 or MXFP4): 16-byte aligned values, one 4-byte scale.
INDEX_SCALE_BYTES = 4

DcpShardQPlanner = Callable[
    [CommonAttentionMetadata, ShardQSplit], "DcpShardQContext | None"
]


def _token_bytes(spec: KVCacheSpec, mla_layer: "MLAAttention") -> tuple[int, int]:
    """(MLA, indexer) cache bytes per token of a layer DCP ShardQ can gather,
    or (0, 0): one-byte MLA entries in blocks of whole pages, and an indexer."""
    index_bytes = mla_layer.shardq_index_entry_bytes
    if (
        not isinstance(spec, MLAAttentionSpec)
        or get_dtype_size(spec.dtype) != 1
        or spec.block_size % PAGE_TOKENS
        or index_bytes % 16 != INDEX_SCALE_BYTES
    ):
        return 0, 0
    return spec.state_content_size_bytes, index_bytes


def workspace_pages(vllm_config: "VllmConfig") -> int:
    """Workspace pages: `max_model_len` tokens, each request rounded up."""
    return (
        cdiv(vllm_config.model_config.max_model_len, PAGE_TOKENS)
        + vllm_config.scheduler_config.max_num_seqs
    )


@dataclass
class _Workspace:
    storage: torch.Tensor
    handle: Any
    # Every rank's workspace address, [world].
    peer_ptrs: torch.Tensor


# Shared by every layer of a TP group; sizes come from the configuration.
_workspace_buffer: _Workspace | None = None
_workspace_group_name: str | None = None


def _workspace(nbytes: int, device: torch.device) -> _Workspace:
    """The symmetric-memory workspace; every rank allocates it together."""
    global _workspace_buffer, _workspace_group_name
    group = get_tp_group().device_group
    group_name = group.group_name
    if (
        _workspace_buffer is None
        or _workspace_group_name != group_name
        or _workspace_buffer.storage.numel() < nbytes
    ):
        storage = symm_mem.empty(nbytes, dtype=torch.uint8, device=device)
        handle = symm_mem.rendezvous(storage, group_name)
        peer_ptrs = torch.tensor(
            [
                handle.get_buffer(peer, [nbytes], torch.uint8, 0).data_ptr()
                for peer in range(group.size())
            ],
            dtype=torch.int64,
            device=storage.device,
        )
        _workspace_buffer = _Workspace(storage, handle, peer_ptrs)
        _workspace_group_name = group_name
    return _workspace_buffer


_warmed_up_groups: set[str] = set()


def reserve_dcp_shardq(mla_layer: "MLAAttention", device: torch.device) -> None:
    """In the memory profile run: allocate the workspace (symmetric memory is
    non-torch memory), and run one full step of the DCP path's query gather
    and combine. DCP ShardQ takes most prefill steps off that path, so its
    first step may come late, after the KV cache took the memory its NCCL
    collectives allocate on first use."""
    vllm_config = mla_layer._vllm_config
    mla_bytes, index_bytes = _token_bytes(
        mla_layer.get_kv_cache_config(vllm_config).spec, mla_layer
    )
    if not mla_bytes:
        return
    _workspace(
        workspace_pages(vllm_config) * PAGE_TOKENS * (mla_bytes + index_bytes), device
    )
    dcp = mla_layer.dcp_manager
    assert dcp is not None
    group_name = get_tp_group().device_group.group_name
    if group_name in _warmed_up_groups:
        return
    _warmed_up_groups.add(group_name)
    num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    # The gathered query stays alive through the combine, as in a real step.
    query = None
    if dcp.query_gather is not None:
        query = dcp.query_gather(
            torch.zeros(
                (num_tokens, dcp.num_heads, dcp.query_head_dim),
                dtype=dcp.query_dtype,
                device=dcp.device,
            )
        )
    world = dcp.group.world_size
    output = torch.zeros(
        (num_tokens, world * dcp.num_heads, dcp.output_head_dim),
        dtype=dcp.output_dtype,
        device=dcp.device,
    )
    lse = torch.zeros(output.shape[:2], dtype=torch.float32, device=dcp.device)
    query_start_loc = torch.tensor(
        [0, num_tokens], dtype=torch.int32, device=dcp.device
    )
    seq_lens = torch.ones(1, dtype=torch.int32, device=dcp.device)
    dcp.combine(output, lse, seq_lens=seq_lens, query_start_loc=query_start_loc)
    del query


@dataclass
class DcpShardQContext:
    """One step's gathered-context plan. The workspace holds `pages` MLA pages,
    then as many indexer pages."""

    mla_token_bytes: int
    index_token_bytes: int
    rank: int
    pages: int
    # The step's leading decodes, which keep the DCP path.
    num_decodes: int
    num_decode_tokens: int
    # The prefill requests' int32 sequence lengths and query start locations.
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    # Most local 64-token pages of any prefill request on any rank.
    max_local_pages: int
    # int32 [num_reqs, width]: workspace page of each request's 64-token
    # block (width a multiple of TABLE_PAGES_ALIGN; decode rows unused).
    block_table: torch.Tensor
    # int64 [num_prefill_tokens]: workspace slot of each new prefill token.
    slot_mapping: torch.Tensor
    # Indexer prefill chunks over the workspace, in the non-DCP form, over
    # the tokens of gather_indices.
    indexer_chunks: list[DeepseekV32IndexerPrefillChunkMetadata]
    # The tokens the indexer scores, in top-k order: every decode token in
    # place, then this rank's ShardQ tokens.
    gather_indices: torch.Tensor

    block_size = PAGE_TOKENS

    @property
    def _index_offset(self) -> int:
        return self.pages * PAGE_TOKENS * self.mla_token_bytes

    def _view(self, dtype: torch.dtype, offset: int, token_bytes: int, device):
        nbytes = self._index_offset + self.pages * PAGE_TOKENS * self.index_token_bytes
        return torch.as_strided(
            _workspace(nbytes, device).storage.view(dtype),
            (self.pages, PAGE_TOKENS, token_bytes),
            (PAGE_TOKENS * token_bytes, token_bytes, 1),
            offset,
        )

    def mla_view(self, mla_cache: torch.Tensor) -> torch.Tensor:
        return self._view(mla_cache.dtype, 0, self.mla_token_bytes, mla_cache.device)

    def index_view(self, index_cache: torch.Tensor) -> torch.Tensor:
        return self._view(
            index_cache.dtype,
            self._index_offset,
            self.index_token_bytes,
            index_cache.device,
        )

    def push(
        self,
        mla_cache: torch.Tensor,
        mla_block_table: torch.Tensor,
        index_cache: torch.Tensor,
        index_block_table: torch.Tensor,
        dst_addrs: torch.Tensor,
    ) -> None:
        """Copy this rank's KV shard of the prefill requests from one layer's
        DCP-local caches into the workspaces at `dst_addrs`."""
        if self.max_local_pages:
            # gather imports cutlass, so import it only on the SM100 path.
            from vllm.v1.attention.ops.dcp_shardq.gather import (
                DCP_CONTEXT_PUSH_KERNEL,
            )

            first = self.num_decodes
            DCP_CONTEXT_PUSH_KERNEL(
                mla_cache=mla_cache,
                mla_block_table=mla_block_table[first:],
                index_cache=index_cache,
                index_block_table=index_block_table[first:],
                seq_lens=self.seq_lens,
                query_start_loc=self.query_start_loc,
                dst_block_table=self.block_table[first:],
                max_local_pages=self.max_local_pages,
                dst_addrs=dst_addrs,
                dst_index_offset=self._index_offset,
                rank=self.rank,
            )

    def gather(
        self,
        mla_cache: torch.Tensor,
        mla_block_table: torch.Tensor,
        index_cache: torch.Tensor,
        index_block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Push this rank's KV shard of one layer into every rank's workspace
        and return the workspace's indexer view once every shard landed."""
        index_view = self.index_view(index_cache)
        buffer = _workspace_buffer
        assert buffer is not None
        # Peers may still read the previous layer from their workspaces.
        buffer.handle.barrier()
        self.push(
            mla_cache, mla_block_table, index_cache, index_block_table, buffer.peer_ptrs
        )
        buffer.handle.barrier()
        return index_view


def build_dcp_shardq_context(
    common_attn_metadata: CommonAttentionMetadata,
    split: ShardQSplit,
    *,
    mla_token_bytes: int,
    index_token_bytes: int,
    max_pages: int,
    max_indexer_keys: int,
    max_logits_bytes: int,
    world: int,
    rank: int,
    device: torch.device,
) -> DcpShardQContext | None:
    """The context plan of `split`'s step on DCP rank `rank` of `world`, or
    None when its prefill requests do not fit `max_pages` workspace pages.
    Every rank derives the same plan."""
    num_reqs = common_attn_metadata.num_reqs
    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
    all_qsl = query_start_loc_cpu.numpy().astype(np.int64)
    num_decode_tokens = split.num_decode_tokens
    num_decodes = int(np.searchsorted(all_qsl, num_decode_tokens))
    if all_qsl[num_decodes] != num_decode_tokens:
        return None
    seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
    assert seq_lens_cpu is not None
    # Exact for prefill requests.
    seq_lens_cpu = seq_lens_cpu[:num_reqs]
    all_seq_lens = seq_lens_cpu.numpy().astype(np.int64)
    qsl = all_qsl[num_decodes:]
    query_lens = np.diff(qsl)
    seq_lens = all_seq_lens[num_decodes:]
    pages = -(-seq_lens // PAGE_TOKENS)
    if pages.sum() > max_pages:
        return None
    page_start = np.cumsum(pages) - pages

    # Rank 0 holds the most tokens: ceil(seq_len / world).
    max_local_pages = -(-int(seq_lens.max()) // (world * PAGE_TOKENS))
    # Columns past a request's pages are never addressed.
    width = -(-int(pages.max()) // TABLE_PAGES_ALIGN) * TABLE_PAGES_ALIGN
    block_table = np.zeros((num_reqs, width), dtype=np.int64)
    block_table[num_decodes:] = page_start[:, None] + np.arange(width)
    slot_mapping = np.repeat(
        page_start * PAGE_TOKENS + seq_lens - query_lens - (qsl[:-1] - qsl[0]),
        query_lens,
    ) + np.arange(int(qsl[-1] - qsl[0]))

    # The indexer scores every decode token in place, then this rank's tokens
    # compactly, each at its position; the query ranges are renumbered over
    # those tokens.
    local_start = int(split.num_tokens_per_rank[:rank].sum())
    local_token_ids = split.token_ids[
        local_start : local_start + int(split.num_tokens_per_rank[rank])
    ]
    gather_indices = np.concatenate(
        (np.arange(num_decode_tokens), local_token_ids + num_decode_tokens)
    )
    token_to_req = np.searchsorted(all_qsl[1:], gather_indices, side="right")
    positions = gather_indices + all_seq_lens[token_to_req] - all_qsl[token_to_req + 1]
    compact_qsl = np.searchsorted(gather_indices, all_qsl)

    def to_device(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        host = torch.from_numpy(np.ascontiguousarray(array)).to(dtype)
        if device.type == "cuda":
            host = host.pin_memory()
        return host.to(device, non_blocking=True)

    block_table_gpu = to_device(block_table, torch.int32)
    compact_qsl_cpu = torch.from_numpy(compact_qsl).to(torch.int32)
    compact_qsl_gpu = to_device(compact_qsl, torch.int32)
    positions_gpu = to_device(positions, torch.int32)
    seq_lens_gpu = common_attn_metadata.seq_lens[:num_reqs]
    chunk_specs = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
        seq_lens_cpu[num_decodes:],
        torch.diff(compact_qsl_cpu[num_decodes:]),
        max_indexer_keys,
        max_logits_bytes,
        request_offset=num_decodes,
    )
    indexer_chunks = [
        build_prefill_chunk_metadata(
            req_slice.start,
            req_slice.stop,
            compact_qsl_gpu,
            compact_qsl_cpu,
            seq_lens_gpu,
            seq_lens_gpu,
            seq_lens_cpu,
            block_table_gpu,
            1,
            query_slice=query_slice,
            skip_kv_gather=query_slice.start > 0,
            positions=positions_gpu,
        )
        for req_slice, query_slice in chunk_specs
    ]
    return DcpShardQContext(
        mla_token_bytes=mla_token_bytes,
        index_token_bytes=index_token_bytes,
        rank=rank,
        pages=max_pages,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        seq_lens=seq_lens_gpu[num_decodes:],
        query_start_loc=common_attn_metadata.query_start_loc[
            num_decodes : num_reqs + 1
        ],
        max_local_pages=max_local_pages,
        block_table=block_table_gpu,
        slot_mapping=to_device(slot_mapping, torch.int64),
        indexer_chunks=[chunk for chunk in indexer_chunks if chunk is not None],
        gather_indices=to_device(gather_indices, torch.int64),
    )


def make_dcp_shardq_planner(
    spec: KVCacheSpec, mla_layer: "MLAAttention", device: torch.device
) -> DcpShardQPlanner | None:
    """The step planner of a sparse MLA metadata builder under DCP, or None
    when DCP ShardQ cannot gather the layer's caches."""
    mla_bytes, index_bytes = _token_bytes(spec, mla_layer)
    if not mla_bytes:
        logger.warning_once(
            "DCP ShardQ needs a one-byte MLA cache in blocks of whole 64-token "
            "pages and an fp8 or MXFP4 indexer cache; DCP prefill steps keep "
            "the DCP path."
        )
        return None
    # gather imports cutlass, so import it only on the SM100 path.
    from vllm.v1.attention.ops.dcp_shardq.gather import DCP_CONTEXT_PUSH_KERNEL

    vllm_config = mla_layer._vllm_config
    world = get_dcp_group().world_size
    DCP_CONTEXT_PUSH_KERNEL.register_warmup(
        world=world, mla_token_bytes=mla_bytes, index_token_bytes=index_bytes
    )
    return partial(
        build_dcp_shardq_context,
        mla_token_bytes=mla_bytes,
        index_token_bytes=index_bytes,
        max_pages=workspace_pages(vllm_config),
        max_indexer_keys=get_max_prefill_buffer_size(vllm_config),
        max_logits_bytes=envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024,
        world=world,
        rank=get_dcp_group().rank_in_group,
        device=device,
    )
