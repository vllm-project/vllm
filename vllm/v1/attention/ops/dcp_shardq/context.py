# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ prefill steps under DCP: attention over a gathered context.

DCP shards the KV cache by token, so a DCP prefill step runs every query token
with all heads against each rank's 1/W of the keys, merges the indexer's
per-rank top-k across ranks, and combines the attention partials by LSE: per
layer it moves gigabytes and repeats the attention work W times over.

When the TP group is the DCP group, a pure prefill step can instead run as a
TP ShardQ step (vllm.v1.attention.ops.shardq): each rank gathers its requests'
whole context into a symmetric-memory workspace, in token order and in the
paged formats of the MLA and indexer caches, then scores and attends its slice
of the query tokens with all heads over the workspace, exactly as TP ShardQ
does over a replicated cache. No top-k merge and no LSE combine remain.

Per layer, the indexer pushes every rank's owned rows of the step's requests
into all workspaces (prior-context MLA and indexer rows, plus the indexer rows
of the new tokens); the MLA layer then writes the new tokens' MLA rows, which
every rank computes for all tokens, into its own workspace.

A step runs this way when its prefill rows fit the workspace. Its leading
decode rows (rows of at most ``1 + num_speculative_tokens`` tokens, which the
indexer scores on its decode path) keep the DCP path within the step; its
other rows are sharded, including the short prefill pieces that the sparse
MLA backend classifies as decodes. The workspace holds the prefill rows'
requests only. ``DcpShardQPlanner`` decides, and counts the prefill steps and
tokens of each path.

Scope: any sparse MLA layer with a DSA-style fp8 indexer cache, under DCP
with the DCP group equal to the TP group; nothing here is model-specific.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV32IndexerPrefillChunkMetadata,
    build_prefill_chunk_metadata,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    ConvertReqIndexToGlobalIndexKernel,
)
from vllm.v1.attention.ops.dcp import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.dcp_shardq import PAGE_TOKENS
from vllm.v1.attention.ops.shardq import (
    ShardQSplit,
    _symm_buffer,
    _SymmBuffer,
    make_shardq_split,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.ops.dcp import MLADCPManager

logger = init_logger(__name__)

# Workspace block tables are padded to whole multiples of this many pages, so
# their row stride has one Triton specialization.
TABLE_PAGES_ALIGN = 16


@dataclass(frozen=True)
class ContextLayout:
    """Per-token bytes of a DSA layer's MLA and indexer caches."""

    mla_row_bytes: int
    index_row_bytes: int
    index_scale_bytes: int

    @classmethod
    def from_layer(cls, spec: KVCacheSpec, indexer: object) -> "ContextLayout | None":
        """The layout of a DSA layer that DCP ShardQ steps can gather, or None.

        The MLA cache must keep one-byte elements (per-tensor ``fp8``,
        ``nvfp4_ds_mla``) in kernel blocks of whole 64-token pages (a packed
        cache may use e.g. 1024-token blocks), and the indexer fp8 keys
        with one 4-byte scale per ``quant_block_size`` values (or MXFP4 keys:
        64 B of values and 4 B of scales per token).
        """
        k_cache = getattr(indexer, "k_cache", None)
        quant_block = getattr(indexer, "quant_block_size", 0)
        if (
            not isinstance(spec, MLAAttentionSpec)
            or get_dtype_size(spec.dtype) != 1
            or spec.block_size % PAGE_TOKENS
            or k_cache is None
            or not quant_block
        ):
            return None
        if getattr(indexer, "use_fp4_cache", False):
            # MXFP4 keys: head_dim / 2 B of E2M1 values and one UE8M0 scale per
            # 32 values (17 / 32 B per dim), pages of the values then the scales
            # (as the fp8 layout).
            index_row = k_cache.head_dim * 16 // 17
        else:
            # fp8 keys with one 4-byte scale per quant_block values.
            index_row = k_cache.head_dim * quant_block // (quant_block + 4)
        return cls(
            spec.state_content_size_bytes, index_row, k_cache.head_dim - index_row
        )

    @property
    def mla_page_bytes(self) -> int:
        return PAGE_TOKENS * self.mla_row_bytes

    @property
    def index_page_bytes(self) -> int:
        return PAGE_TOKENS * (self.index_row_bytes + self.index_scale_bytes)


def workspace_max_tokens(vllm_config: "VllmConfig") -> int:
    """Tokens of context the workspace holds: ``max_model_len``, so any one
    request fits. It costs about ``mla_row_bytes + index_row_bytes`` (plus
    the index scales) per token on each rank, ~0.7 GB per 1M tokens with a
    per-tensor ``fp8`` cache and ~0.5 GB with ``nvfp4_ds_mla``. Call it
    through this module so tests can patch it."""
    return vllm_config.model_config.max_model_len


def capacity_pages(max_tokens: int, max_reqs: int) -> int:
    """Workspace pages for ``max_tokens`` tokens, each request rounded up."""
    return -(-max_tokens // PAGE_TOKENS) + max_reqs


def scheduler_context_pages(vllm_config: "VllmConfig") -> int | None:
    """The workspace pages the scheduler keeps each step's prefill rows
    within (the sum over them of ``ceil(seq_len / 64)``), or None when DCP
    ShardQ steps cannot run under this configuration.

    With the cap, no step finds the workspace full, so the DCP path only runs
    a step's leading decode rows and steps of fewer than
    ``sparse_mla_shard_q_min_tokens`` prefill tokens, and the memory profile
    reserves those (``dcp_fallback_max_tokens``) instead of a whole
    ``max_num_batched_tokens`` step. These are the configuration-level and
    platform conditions of SparseMLAShardQ's DCP ShardQ; when its layer-level
    ones fail the cap merely keeps a step's prefill context within
    ``max_model_len``.
    """
    parallel_config = vllm_config.parallel_config
    kv_transfer_config = vllm_config.kv_transfer_config
    dcp = parallel_config.decode_context_parallel_size
    if (
        not vllm_config.attention_config.sparse_mla_shard_q
        or dcp <= 1
        or dcp != parallel_config.tensor_parallel_size
        or parallel_config.cp_kv_cache_interleave_size != 1
        or parallel_config.use_ubatching
        or not current_platform.is_device_capability_family(100)
        or parallel_config.prefill_context_parallel_size != 1
        or (
            kv_transfer_config is not None
            and kv_transfer_config.kv_role == "kv_consumer"
        )
    ):
        return None
    return capacity_pages(
        workspace_max_tokens(vllm_config), vllm_config.scheduler_config.max_num_seqs
    )


def dcp_fallback_max_tokens(vllm_config: "VllmConfig", decode_query_len: int) -> int:
    """Most tokens of a step that keeps the DCP path under DCP ShardQ, given
    the scheduler's context cap: its leading decode rows plus fewer than
    ``sparse_mla_shard_q_min_tokens`` prefill tokens, or a padded step of at
    most the DCP workspace's tokens."""
    from vllm.v1.attention.ops.dcp import get_dcp_workspace_max_num_tokens

    scheduler_config = vllm_config.scheduler_config
    few_tokens = (
        scheduler_config.max_num_seqs * decode_query_len
        + vllm_config.attention_config.sparse_mla_shard_q_min_tokens
        - 1
    )
    return min(
        scheduler_config.max_num_batched_tokens,
        max(few_tokens, get_dcp_workspace_max_num_tokens(vllm_config)),
    )


def _workspace(layout: ContextLayout, pages: int, device) -> _SymmBuffer:
    # Sizes come from the configuration, so every rank rendezvouses together.
    nbytes = pages * (layout.mla_page_bytes + layout.index_page_bytes)
    return _symm_buffer(
        get_tp_group().device_group, "dcp_shardq_context", nbytes, torch.uint8, device
    )


def reserve_workspace(
    layout: ContextLayout, max_tokens: int, max_reqs: int, device: torch.device
) -> None:
    """Allocate the workspace in the memory profile; symmetric memory lives
    outside the torch caching allocator, so it counts as non-torch memory."""
    _workspace(layout, capacity_pages(max_tokens, max_reqs), device)


# The (group, step shape) keys warm_up_dcp_fallback ran.
_WARMED_UP_DCP_FALLBACK: set[tuple[Any, ...]] = set()


def warm_up_dcp_fallback(dcp_manager: "MLADCPManager", num_tokens: int) -> None:
    """Run ``dcp_manager``'s query gather and output combine of a step of
    ``num_tokens`` tokens, and the combine's Triton kernels for every batch
    size, on dummy inputs (once per process and shape).

    A DCP ShardQ instance takes most prefill steps off the DCP path, so the
    steps that keep it (decode rows, few tokens, a full workspace) may come
    only late, after the KV cache and the caching allocator took the memory.
    Beyond the direct workspaces the DCP path falls back to NCCL collectives,
    which connect their peers and allocate their buffers on first use, and its
    Triton kernels compile on first use; calling this in the memory profile
    run makes both happen there and accounts for them. Every rank of the
    group must call it.
    """
    world = dcp_manager.group.world_size
    key = (
        dcp_manager.group.unique_name,
        num_tokens,
        dcp_manager.num_heads,
        dcp_manager.query_head_dim,
        dcp_manager.output_head_dim,
        dcp_manager.query_dtype,
        dcp_manager.output_dtype,
        dcp_manager.is_lse_base_on_e,
    )
    if world == 1 or key in _WARMED_UP_DCP_FALLBACK:
        return
    _WARMED_UP_DCP_FALLBACK.add(key)
    # The gathered query stays alive while the attention writes its partial
    # output and the combine runs, as in a real step.
    gathered_query = None
    if dcp_manager.query_gather is not None:
        gathered_query = dcp_manager.query_gather(
            torch.zeros(
                (num_tokens, dcp_manager.num_heads, dcp_manager.query_head_dim),
                dtype=dcp_manager.query_dtype,
                device=dcp_manager.device,
            )
        )
    # Every head's partial output, as the attention backend returns it.
    partial_output = torch.zeros(
        (num_tokens, world * dcp_manager.num_heads, dcp_manager.output_head_dim),
        dtype=dcp_manager.output_dtype,
        device=dcp_manager.device,
    )
    partial_lse = torch.zeros(
        partial_output.shape[:2], dtype=torch.float32, device=dcp_manager.device
    )

    def batch(num_tokens: int, num_seqs: int) -> dict[str, torch.Tensor]:
        # num_seqs requests of one token each, the last one taking the rest.
        query_start_loc = torch.arange(
            num_seqs + 1, dtype=torch.int32, device=dcp_manager.device
        )
        query_start_loc[-1] = num_tokens
        seq_lens = torch.ones(num_seqs, dtype=torch.int32, device=dcp_manager.device)
        return dict(seq_lens=seq_lens, query_start_loc=query_start_loc)

    dcp_manager.combine(partial_output, partial_lse, **batch(num_tokens, 1))
    if dcp_manager.use_a2a and (
        dcp_manager.direct_a2a is None
        or num_tokens > dcp_manager.direct_a2a.max_num_tokens
    ):
        # The NCCL A2A's pack kernel compiles once per power-of-two
        # bucket of the batch's requests, and both its kernels once per
        # alignment of the per-rank stride of their buffers, a multiple of
        # the token count: a multiple of 16 tokens takes one, an odd count
        # the other one (if any).
        num_seqs = 1
        while num_seqs < 2 * dcp_manager.max_num_seqs and num_seqs <= num_tokens:
            for rows in (round_up(num_seqs, 16), num_seqs | 1):
                if rows > num_tokens:
                    continue
                dcp_a2a_lse_reduce(
                    partial_output[:rows],
                    partial_lse[:rows],
                    cp_group=dcp_manager.group,
                    is_lse_base_on_e=dcp_manager.is_lse_base_on_e,
                    **batch(rows, num_seqs),
                )
            num_seqs *= 2
    del gathered_query
    logger.info_once("Warmed up the MLA DCP collectives of %d-token steps.", num_tokens)


@dataclass
class DcpShardQContext:
    """One step's gathered-context plan (see the module docstring).

    The workspace holds ``pages`` MLA pages, then as many indexer pages.
    """

    layout: ContextLayout
    world: int
    rank: int
    pages: int
    # The step's leading decode rows, which keep the DCP path: its requests
    # before num_decodes and its tokens before num_decode_tokens.
    num_decodes: int
    num_decode_tokens: int
    # The prefill rows' requests: int32 sequence lengths and query start
    # locations, on the device.
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    # Most local 64-token pages of any prefill request on any rank.
    max_local_pages: int
    # int32 [num_reqs, width]: workspace page of each prefill request's
    # 64-token block, addressed like a paged block table by the step's request
    # index (width a multiple of TABLE_PAGES_ALIGN; decode rows' rows unused).
    block_table: torch.Tensor
    # int64 [num_prefill_tokens]: workspace slot of each new prefill token.
    slot_mapping: torch.Tensor
    # Indexer prefill chunks over the workspace, in the non-DCP form.
    indexer_chunks: list[DeepseekV32IndexerPrefillChunkMetadata]

    # The workspace's pages of 64 tokens address its views and block table.
    block_size = PAGE_TOKENS

    def _view(self, dtype: torch.dtype, offset: int, row_bytes: int, device):
        """The workspace pages at ``offset`` as a [pages, 64, row] cache."""
        storage = _workspace(self.layout, self.pages, device).storage
        return torch.as_strided(
            storage.view(dtype),
            (self.pages, PAGE_TOKENS, row_bytes),
            (PAGE_TOKENS * row_bytes, row_bytes, 1),
            offset,
        )

    def mla_view(self, mla_cache: torch.Tensor) -> torch.Tensor:
        """The workspace's MLA pages, typed as ``mla_cache``."""
        layout = self.layout
        return self._view(mla_cache.dtype, 0, layout.mla_row_bytes, mla_cache.device)

    def index_view(self, index_cache: torch.Tensor) -> torch.Tensor:
        """The workspace's indexer pages, typed as ``index_cache``."""
        layout = self.layout
        return self._view(
            index_cache.dtype,
            self.pages * layout.mla_page_bytes,
            layout.index_row_bytes + layout.index_scale_bytes,
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
        """Copy this rank's rows of the prefill requests of one layer's
        DCP-local paged caches (with the step's kernel block tables) into the
        workspaces at ``dst_addrs``."""
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
                dst_index_offset=self.pages * self.layout.mla_page_bytes,
                rank=self.rank,
                index_row_bytes=self.layout.index_row_bytes,
            )

    def gather(
        self,
        mla_cache: torch.Tensor,
        mla_block_table: torch.Tensor,
        index_cache: torch.Tensor,
        index_block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Push this rank's rows of one layer into every rank's workspace (see
        ``push``) and return the workspace as the indexer's cache view, once
        every rank's rows have landed in it."""
        index_view = self.index_view(index_cache)
        buffer = _workspace(self.layout, self.pages, index_cache.device)
        # Peers may still read the previous layer from their workspaces.
        buffer.handle.barrier()
        self.push(
            mla_cache, mla_block_table, index_cache, index_block_table, buffer.peer_ptrs
        )
        buffer.handle.barrier()
        return index_view


def build_dcp_shardq_context(
    common_attn_metadata: CommonAttentionMetadata,
    layout: ContextLayout,
    max_pages: int,
    max_indexer_keys: int,
    max_logits_bytes: int,
    world: int,
    rank: int,
    device: torch.device,
    num_decodes: int = 0,
) -> DcpShardQContext | None:
    """The context plan of a step's prefill rows (its requests from
    ``num_decodes`` on) on DCP rank ``rank`` of ``world``, or None when they
    do not fit the ``max_pages`` workspace pages.

    Every rank derives the same plan; the push kernel finds each rank's rows
    from the step's device sequence lengths.
    """
    num_reqs = common_attn_metadata.num_reqs
    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
    seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
    assert seq_lens_cpu is not None
    # Exact for prefill rows.
    seq_lens_cpu = seq_lens_cpu[:num_reqs]
    qsl = query_start_loc_cpu.numpy().astype(np.int64)[num_decodes:]
    num_decode_tokens = int(qsl[0])
    query_lens = np.diff(qsl)
    seq_lens = seq_lens_cpu.numpy().astype(np.int64)[num_decodes:]
    pages = -(-seq_lens // PAGE_TOKENS)
    if pages.sum() > max_pages:
        return None
    page_start = np.cumsum(pages) - pages

    context_lens = seq_lens - query_lens
    # Rank 0 holds the most rows: ceil(seq_len / world).
    max_local_pages = -(-int(seq_lens.max()) // (world * PAGE_TOKENS))
    # Columns past a request's pages are never addressed.
    width = -(-int(pages.max()) // TABLE_PAGES_ALIGN) * TABLE_PAGES_ALIGN
    block_table = np.zeros((num_reqs, width), dtype=np.int64)
    block_table[num_decodes:] = page_start[:, None] + np.arange(width)
    slot_mapping = np.repeat(
        page_start * PAGE_TOKENS + context_lens - (qsl[:-1] - qsl[0]), query_lens
    ) + np.arange(int(qsl[-1] - qsl[0]))

    def to_device(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        host = torch.from_numpy(np.ascontiguousarray(array)).to(dtype)
        if device.type == "cuda":
            host = host.pin_memory()
        return host.to(device, non_blocking=True)

    block_table_gpu = to_device(block_table, torch.int32)
    seq_lens_gpu = common_attn_metadata.seq_lens[:num_reqs]
    chunk_specs = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
        seq_lens_cpu[num_decodes:],
        torch.from_numpy(query_lens),
        max_indexer_keys,
        max_logits_bytes,
    )
    indexer_chunks = [
        build_prefill_chunk_metadata(
            num_decodes + req_slice.start,
            num_decodes + req_slice.stop,
            common_attn_metadata.query_start_loc,
            query_start_loc_cpu,
            seq_lens_gpu,
            seq_lens_gpu,
            seq_lens_cpu,
            block_table_gpu,
            1,
            query_slice=query_slice,
            skip_kv_gather=query_slice.start > 0,
        )
        for req_slice, query_slice in chunk_specs
    ]
    return DcpShardQContext(
        layout=layout,
        world=world,
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
    )


class _GatheredContextIndexRemap(ConvertReqIndexToGlobalIndexKernel):
    """The sparse-MLA top-k remap as a DCP ShardQ step runs it
    (SparseMLACommonImpl.forward_mqa_shardq): onto the workspace's 64-token
    pages, without DCP, with valid counts, through a block table padded to
    TABLE_PAGES_ALIGN pages. Its Triton kernel is the one every
    ConvertReqIndexToGlobalIndexKernel launches, so warming these keys here
    warms the runtime launches."""

    def get_warmup_keys(  # type: ignore[override]
        self, *, num_topk_tokens: int
    ) -> list[ConvertReqIndexToGlobalIndexKernel.CompileKey]:
        return [
            self.dispatch(
                BLOCK_SIZE=PAGE_TOKENS,
                BLOCK_STRIDE_ROWS=PAGE_TOKENS,
                BLOCK_N=128,
                NUM_TOPK_TOKENS=num_topk_tokens,
                HAS_PREFILL_WORKSPACE=False,
                HAS_IDX_MAPPING=False,
                COUNT_VALID=True,
                COMPACT_TO_FRONT=True,
                DCP_SIZE=1,
                DCP_RANK=0,
                DCP_INTERLEAVE=1,
                max_num_blocks_per_req=TABLE_PAGES_ALIGN,
            )
        ]


_GATHERED_CONTEXT_INDEX_REMAP = _GatheredContextIndexRemap()


def register_warmup(layout: ContextLayout, world: int, num_topk_tokens: int) -> None:
    """Register the JIT kernels that only DCP ShardQ steps launch with the
    active warmup registry: the context push and the top-k remap onto the
    workspace. (The prefill chunk metadata kernel warms its non-DCP form when
    ``sparse_mla_shard_q`` is set under DCP; ShardQ's exchange kernel warms
    with every ShardQ layer.)"""
    # gather imports cutlass, so import it only on the SM100 path.
    from vllm.v1.attention.ops.dcp_shardq.gather import DCP_CONTEXT_PUSH_KERNEL

    DCP_CONTEXT_PUSH_KERNEL.register_warmup(
        world=world,
        mla_row_bytes=layout.mla_row_bytes,
        index_row_bytes=layout.index_row_bytes,
        index_scale_bytes=layout.index_scale_bytes,
    )
    _GATHERED_CONTEXT_INDEX_REMAP.register_warmup(num_topk_tokens=num_topk_tokens)


SHARDQ = "ShardQ"
# Why a step keeps the DCP path.
DENSE_MHA = "dense MHA"
FEW_TOKENS = "few tokens"
PADDED = "padded"
WORKSPACE = "workspace full"


class DcpShardQStats:
    """Steps with prefill rows, and their prefill tokens, by path: ShardQ, or
    the DCP path and why. Rank 0 logs them at most every
    VLLM_LOG_STATS_INTERVAL seconds, for the steps since its previous report
    and in total, labeled with the layers of the metadata builder that plans
    them. Every step the builder plans counts, the runner's profile and warmup
    steps included."""

    def __init__(self, log: bool, label: str = "") -> None:
        self.log = log
        self.label = label
        self.interval = envs.VLLM_LOG_STATS_INTERVAL
        self.last_report = time.monotonic()
        # Path -> [steps, tokens].
        self.recent: dict[str, list[int]] = {}
        self.total: dict[str, list[int]] = {}

    def record(self, path: str, num_tokens: int) -> None:
        for counts in (self.recent, self.total):
            entry = counts.setdefault(path, [0, 0])
            entry[0] += 1
            entry[1] += num_tokens
        if self.log:
            now = time.monotonic()
            if now - self.last_report >= self.interval:
                self.last_report = now
                logger.info("%s", self.report())
                self.recent.clear()

    @staticmethod
    def _share(counts: dict[str, list[int]]) -> tuple[int, int, int, int]:
        steps, tokens = counts.get(SHARDQ, (0, 0))
        all_steps = sum(c[0] for c in counts.values())
        all_tokens = sum(c[1] for c in counts.values())
        return steps, all_steps, tokens, all_tokens

    def report(self) -> str:
        steps, all_steps, tokens, all_tokens = self._share(self.recent)
        _, _, total_tokens, total_all = self._share(self.total)
        fallback = ", ".join(
            f"{path} {c[0]} steps / {c[1]} tokens"
            for path, c in sorted(self.recent.items())
            if path != SHARDQ
        )
        return (
            f"DCP ShardQ prefill{f' [{self.label}]' if self.label else ''}: "
            f"{steps}/{all_steps} steps, "
            f"{tokens}/{all_tokens} tokens "
            f"({100 * tokens / max(all_tokens, 1):.1f}%; "
            f"{100 * total_tokens / max(total_all, 1):.1f}% since start) on "
            f"ShardQ; DCP path: {fallback or 'none'}"
        )


class DcpShardQPlanner:
    """Plans the DCP ShardQ steps of one metadata builder (see the module
    docstring); every rank reaches the same decisions from the same batch."""

    def __init__(
        self,
        layout: ContextLayout,
        max_pages: int,
        max_indexer_keys: int,
        max_logits_bytes: int,
        min_tokens: int,
        max_decode_query_len: int,
        world: int,
        rank: int,
        device: torch.device,
        label: str = "",
        max_fallback_tokens: int | None = None,
    ) -> None:
        self.layout = layout
        # Most tokens of a DCP-path step the memory profile reserved.
        self.max_fallback_tokens = (
            max_fallback_tokens if max_fallback_tokens is not None else 1 << 62
        )
        self.max_pages = max_pages
        self.max_indexer_keys = max_indexer_keys
        self.max_logits_bytes = max_logits_bytes
        self.min_tokens = min_tokens
        self.max_decode_query_len = max_decode_query_len
        self.world = world
        self.rank = rank
        self.device = device
        self.stats = DcpShardQStats(log=rank == 0, label=label)

    def plan(
        self, common_attn_metadata: CommonAttentionMetadata, use_dense_mha: bool
    ) -> tuple[ShardQSplit, DcpShardQContext] | None:
        """The prefill rows' token split and context plan of a step that runs
        as a DCP ShardQ step, or None for the DCP path.

        The step's leading decode rows (the requests of at most
        ``max_decode_query_len`` tokens before its first longer one, which the
        indexer scores on its decode path) keep the DCP path; the split and
        the plan cover the rest. A step without prefill rows is not counted.

        ``use_dense_mha``: the step's prefill rows take dense MHA.
        """
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        if num_reqs == 0 or num_tokens == 0:
            return None
        qsl = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1].numpy()
        qsl = qsl.astype(np.int64)
        query_lens = np.diff(qsl)
        is_prefill = query_lens > self.max_decode_query_len
        if not is_prefill.any():
            return None
        num_decodes = int(is_prefill.argmax())
        num_prefill_tokens = num_tokens - int(qsl[num_decodes])
        context = None
        if qsl[-1] != num_tokens or query_lens.min() == 0:
            path = PADDED
        elif use_dense_mha:
            path = DENSE_MHA
        elif num_prefill_tokens < self.min_tokens:
            path = FEW_TOKENS
        else:
            context = build_dcp_shardq_context(
                common_attn_metadata,
                self.layout,
                self.max_pages,
                self.max_indexer_keys,
                self.max_logits_bytes,
                self.world,
                self.rank,
                self.device,
                num_decodes,
            )
            path = SHARDQ if context is not None else WORKSPACE
        self.stats.record(path, num_prefill_tokens)
        if context is None:
            # Dense MHA rows take no MQA combine; the others are bounded.
            if path != DENSE_MHA and num_tokens > self.max_fallback_tokens:
                logger.warning_once(
                    "A %d-token step keeps the DCP path (%s), more than the %d "
                    "tokens the memory profile reserved for it.",
                    num_tokens,
                    path,
                    self.max_fallback_tokens,
                )
            return None
        split = make_shardq_split(
            qsl[num_decodes:] - qsl[num_decodes], self.world, self.rank, self.device
        )
        return split, context
