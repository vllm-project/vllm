# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 index-cache indexer impl for MiniMax M3 (SM100/SM103/SM107).

The index-K side cache holds packed NVFP4 pages ``[num_blocks, 128, 72]`` (E2M1
codes, then E4M3 block scales, no global scale) written by the fused
qk-norm/rope insert, and index-Q is E4M3. Both sides use the MSA paged indexers
of ``fmha_sm100`` with fused top-k, which write 15 score-ranked history pages
plus the query's local page last (``init_blocks=0, local_blocks=1``) as
logical block ids:

* Decode scores the NVFP4 pages with the Q8KV4 decode indexer, which reads the
  1..8 queries of each request in place (they take the last slots of its
  8-query tile) and writes their top-k straight into the shared buffer.
  Uniform decode batches are captured in full CUDA graphs, so each padded
  batch size and query length gets its own CUDA-graph wrapper whose buffers
  never move; the builder re-plans it outside capture every step.
* Prefill has no NVFP4 kernel. Each layer dequantizes the history pages of the
  prefill requests into compact FP8 workspace pages (what the decode kernel
  scores) and runs the Q8KV8 prefill indexer on them, in request chunks whose
  pages fit the gather budget.

All index heads of the rank share the index keys and select their own pages.
Prefill and decode score every head of the rank in one launch. Prefill plans
are rebuilt every step and only run eagerly (mixed batches are never captured
whole).
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerMetadataBuilder,
)
from vllm.models.minimax_m3.nvidia.indexer_nvfp4_decode_cp import (
    Nvfp4DecodeCPLayer,
    Nvfp4DecodeCPPlan,
    get_nvfp4_indexer_decode_cp,
)
from vllm.models.minimax_m3.nvidia.ops.gather_nvfp4_index_k import (
    minimax_m3_gather_nvfp4_index_k_fp8,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
    max_decode_query_len,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

PAGE_SIZE = 128
HEAD_DIM = 128
TOPK_BLOCKS = 16
# Queries per request the Q8KV4 decode indexer scores.
DECODE_QUERY_LEN = 8
# Longest block table the MSA indexers accept.
MAX_PAGES = 8192
# Index heads the Q8KV4 decode indexer scores in one launch.
DECODE_NUM_HEADS = (1, 2, 4)


def _msa_indexers():
    """Import the MSA indexer wrappers.

    Through ``fmha_sm100.sparse``, which binds the package's csrc JIT loader.
    """
    from vllm.third_party.fmha_sm100.sparse import (
        BatchDecodeIndexerQ8KV4Wrapper,
        BatchPrefillIndexerQ8KV8Wrapper,
    )

    return BatchDecodeIndexerQ8KV4Wrapper, BatchPrefillIndexerQ8KV8Wrapper


def max_gather_pages(vllm_config: VllmConfig) -> int:
    """FP8 pages one prefill chunk may gather; always fits one full request."""
    tokens = max(
        vllm_config.model_config.max_model_len,
        4 * vllm_config.scheduler_config.max_num_batched_tokens,
    )
    return cdiv(tokens, PAGE_SIZE)


class MiniMaxM3IndexerNvfp4Backend(MiniMaxM3IndexerBackend):
    """Indexer side-cache backend selecting the NVFP4 builder."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3IndexerNvfp4MetadataBuilder"]:
        return MiniMaxM3IndexerNvfp4MetadataBuilder


@dataclass
class MiniMaxM3IndexerNvfp4DecodeMetadata:
    wrapper: Any  # planned CUDA-graph wrapper, None when decode CP is on
    num_reqs: int  # includes zero-length full-CUDA-graph padding requests
    query_len: int  # uniform, <= DECODE_QUERY_LEN
    # Set instead of the wrapper when decode CP is on.
    cp_plan: Nvfp4DecodeCPPlan | None = None


@dataclass
class MiniMaxM3IndexerNvfp4PrefillChunk:
    wrapper: Any  # planned BatchPrefillIndexerQ8KV8Wrapper
    # Token range within the prefill region (offsets from num_decode_tokens).
    token_start: int
    token_end: int
    block_table: torch.Tensor  # [n, max_blocks] physical pages
    num_pages: torch.Tensor  # [n] int32 history pages to gather
    page_offsets: torch.Tensor  # [n] int32 first compact page of each request
    total_pages: int
    max_pages: int


@dataclass
class MiniMaxM3IndexerNvfp4Metadata(MiniMaxM3IndexerMetadata):
    """The base ``prefill``/``decode`` fields are unused on this path."""

    decode_q8kv4: MiniMaxM3IndexerNvfp4DecodeMetadata | None = None
    prefill_chunks: list[MiniMaxM3IndexerNvfp4PrefillChunk] = field(
        default_factory=list
    )


class MiniMaxM3IndexerNvfp4MetadataBuilder(MiniMaxM3IndexerMetadataBuilder):
    """Plans the MSA decode and prefill indexers once per step."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if self.max_decode_query_len > DECODE_QUERY_LEN:
            raise NotImplementedError(
                "The MiniMax M3 NVFP4 indexer scores at most "
                f"{DECODE_QUERY_LEN} decode queries per request, got "
                f"{self.max_decode_query_len} (use num_speculative_tokens <= "
                f"{DECODE_QUERY_LEN - 1})."
            )
        if self.num_index_heads not in DECODE_NUM_HEADS:
            raise NotImplementedError(
                "The MiniMax M3 NVFP4 indexer serves "
                f"{DECODE_NUM_HEADS} index heads per rank, got "
                f"{self.num_index_heads}."
            )
        self.max_gather_pages = max_gather_pages(vllm_config)
        # Decode wrapper per padded decode batch size and query length.
        self.decode_wrappers: dict[tuple[int, int], Any] = {}
        # Decode CP (opt-in); static, so all ranks and graphs agree.
        self.decode_cp = get_nvfp4_indexer_decode_cp(
            num_index_heads=self.num_index_heads,
            max_decode_query_len=self.max_decode_query_len,
        )
        # Pages a request can span; decode scores at most this many.
        self.max_decode_pages = min(
            cdiv(vllm_config.model_config.max_model_len, PAGE_SIZE), MAX_PAGES
        )
        # One page-score scratch for every decode wrapper: they never run at
        # once, and a buffer per batch size would hold 256 KiB per request at a
        # 1M context for every batch size the engine sees. Decode batches,
        # padded to a CUDA graph size, never exceed max_decode_reqs.
        self.decode_score_buffer: torch.Tensor | None = None
        if self.decode_cp is None:
            max_decode_reqs = max(
                vllm_config.scheduler_config.max_num_seqs,
                vllm_config.compilation_config.max_cudagraph_capture_size or 0,
            )
            decode_wrapper_cls, _ = _msa_indexers()
            self.decode_score_buffer = torch.empty(
                decode_wrapper_cls.score_buffer_size(
                    max_decode_reqs,
                    self.max_decode_pages,
                    num_heads=self.num_index_heads,
                ),
                dtype=torch.float32,
                device=device,
            )

    def _decode_wrapper(
        self, block_table: torch.Tensor, seq_lens: torch.Tensor, query_len: int
    ) -> Any:
        num_reqs = seq_lens.shape[0]
        wrapper = self.decode_wrappers.get((num_reqs, query_len))
        if wrapper is None:
            # The runner's block table and seq_lens are persistent buffers, so
            # the wrapper reads them in place and plan() copies nothing. A
            # strided (width-capped) block table gets its own buffer.
            if not block_table.is_contiguous():
                block_table = torch.empty(
                    block_table.shape, dtype=torch.int32, device=self.device
                )
            decode_wrapper_cls, _ = _msa_indexers()
            wrapper = self.decode_wrappers[num_reqs, query_len] = decode_wrapper_cls(
                num_heads=self.num_index_heads,
                use_cuda_graph=True,
                block_table_buffer=block_table,
                seq_lens_buffer=seq_lens,
                score_buffer=self.decode_score_buffer,
            )
        return wrapper

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3IndexerNvfp4Metadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        qsl_cpu = common_attn_metadata.query_start_loc_cpu

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        decode: MiniMaxM3IndexerNvfp4DecodeMetadata | None = None
        if num_decode_tokens > 0:
            # num_decodes counts the zero-length requests that pad a full CUDA
            # graph batch, so it is the batch size the graph was captured with.
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            query_len = int(query_lens_cpu[0])
            assert torch.all((query_lens_cpu == query_len) | (query_lens_cpu == 0))
            num_pages = min(block_table.shape[1], self.max_decode_pages)
            if self.decode_cp is not None:
                decode = MiniMaxM3IndexerNvfp4DecodeMetadata(
                    wrapper=None,
                    num_reqs=num_decodes,
                    query_len=query_len,
                    cp_plan=self.decode_cp.plan(
                        block_table[:num_decodes, :num_pages],
                        seq_lens[:num_decodes],
                    ),
                )
            else:
                decode_block_table = block_table[:num_decodes, :num_pages]
                decode_seq_lens = seq_lens[:num_decodes]
                wrapper = self._decode_wrapper(
                    decode_block_table, decode_seq_lens, query_len
                )
                wrapper.plan(decode_block_table, decode_seq_lens, query_len=query_len)
                decode = MiniMaxM3IndexerNvfp4DecodeMetadata(
                    wrapper=wrapper,
                    num_reqs=num_decodes,
                    query_len=query_len,
                )

        prefill_chunks: list[MiniMaxM3IndexerNvfp4PrefillChunk] = []
        if num_prefills > 0:
            prefill_chunks = self._build_prefill_chunks(
                common_attn_metadata, num_decodes
            )

        return MiniMaxM3IndexerNvfp4Metadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            decode_q8kv4=decode,
            prefill_chunks=prefill_chunks,
        )

    def _build_prefill_chunks(
        self, common_attn_metadata: CommonAttentionMetadata, num_decodes: int
    ) -> list[MiniMaxM3IndexerNvfp4PrefillChunk]:
        _, prefill_wrapper_cls = _msa_indexers()
        num_reqs = common_attn_metadata.num_reqs
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        qsl_cpu = common_attn_metadata.query_start_loc_cpu[num_decodes : num_reqs + 1]
        # Host-side lengths keep planning free of device syncs. An upper bound
        # (async spec decode) only over-sizes the plan and the gather: the
        # kernels mask by the device seq_lens and never score the extra pages.
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        assert seq_lens_cpu is not None
        seq_lens_cpu = seq_lens_cpu[num_decodes:num_reqs].to(torch.int32)
        query_lens_cpu = (qsl_cpu[1:] - qsl_cpu[:-1]).to(torch.int32)
        # Only pages before the last query's local page are ever scored.
        history_pages = (seq_lens_cpu - 1) // PAGE_SIZE
        device = seq_lens.device

        chunks = []
        start, num_prefills = 0, seq_lens_cpu.shape[0]
        while start < num_prefills:
            end, total_pages = start, 0
            while end < num_prefills and (
                end == start
                or total_pages + int(history_pages[end]) <= self.max_gather_pages
            ):
                total_pages += int(history_pages[end])
                end += 1
            assert total_pages <= self.max_gather_pages
            num_chunk_reqs = end - start
            max_seq_len = int(seq_lens_cpu[start:end].max())
            width = cdiv(max_seq_len, PAGE_SIZE)
            # [cu_seqlens_q | num_pages | page_offsets | compact block table]
            # in one pinned buffer, so a single H2D copy is truly asynchronous
            # (a pageable one may synchronize with the stream).
            staging = torch.empty(
                3 * num_chunk_reqs + 1 + num_chunk_reqs * width,
                dtype=torch.int32,
                pin_memory=True,
            )
            cu_seqlens_q, pages, offsets, table = staging.split(
                [
                    num_chunk_reqs + 1,
                    num_chunk_reqs,
                    num_chunk_reqs,
                    num_chunk_reqs * width,
                ]
            )
            torch.sub(qsl_cpu[start : end + 1], qsl_cpu[start], out=cu_seqlens_q)
            pages.copy_(history_pages[start:end])
            torch.cumsum(pages, 0, dtype=torch.int32, out=offsets)
            offsets.sub_(pages)
            cols = torch.arange(width, dtype=torch.int32)
            table.view(num_chunk_reqs, width).copy_(
                torch.where(cols < pages[:, None], offsets[:, None] + cols, 0)
            )
            staging_gpu = staging.to(device, non_blocking=True)
            cu_seqlens_q_gpu, pages_gpu, offsets_gpu, table_gpu = staging_gpu.split(
                [
                    num_chunk_reqs + 1,
                    num_chunk_reqs,
                    num_chunk_reqs,
                    num_chunk_reqs * width,
                ]
            )
            lo, hi = num_decodes + start, num_decodes + end
            token_start = int(qsl_cpu[start] - qsl_cpu[0])
            token_end = int(qsl_cpu[end] - qsl_cpu[0])
            wrapper = prefill_wrapper_cls(num_heads=self.num_index_heads)
            wrapper.plan(
                cu_seqlens_q_gpu,
                seq_lens[lo:hi],
                table_gpu.view(num_chunk_reqs, width),
                total_q=token_end - token_start,
                max_seqlen_q=int(query_lens_cpu[start:end].max()),
                max_seqlen_k=max_seq_len,
            )
            chunks.append(
                MiniMaxM3IndexerNvfp4PrefillChunk(
                    wrapper=wrapper,
                    token_start=token_start,
                    token_end=token_end,
                    block_table=block_table[lo:hi],
                    num_pages=pages_gpu,
                    page_offsets=offsets_gpu,
                    total_pages=total_pages,
                    max_pages=int(pages.max()),
                )
            )
            start = end
        return chunks


class MiniMaxM3IndexerNvfp4Impl(MiniMaxM3IndexerImpl):
    """Decode: Q8KV4 on NVFP4 pages. Prefill: FP8 gather + Q8KV8."""

    indexer_backend_cls: ClassVar[type[AttentionBackend]] = MiniMaxM3IndexerNvfp4Backend

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        config = (
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            self.block_size,
            self.index_head_dim,
            self.score_type,
        )
        if config != (TOPK_BLOCKS, 0, 1, PAGE_SIZE, HEAD_DIM, "max"):
            raise NotImplementedError(
                "The MiniMax M3 NVFP4 indexer supports topk_blocks=16, "
                "init_blocks=0, local_blocks=1, 128-token blocks, index head "
                "dim 128 and max scoring; got (topk_blocks, init_blocks, "
                "local_blocks, block_size, index_head_dim, score_type)="
                f"{config}."
            )
        vllm_config = get_current_vllm_config()
        self.max_gather_pages = max_gather_pages(vllm_config)
        # Per-layer decode-CP buffers (collective: same layer order on all ranks).
        self.decode_cp = get_nvfp4_indexer_decode_cp(
            num_index_heads=self.num_index_heads,
            max_decode_query_len=max_decode_query_len(vllm_config),
        )
        self.decode_cp_layer: Nvfp4DecodeCPLayer | None = (
            None if self.decode_cp is None else self.decode_cp.new_layer()
        )

    def reserve_workspace(self) -> None:
        current_workspace_manager().get_simultaneous(
            ((self.max_gather_pages, PAGE_SIZE, HEAD_DIM), torch.float8_e4m3fn)
        )

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        md = attn_metadata[self.index_cache.prefix]
        assert isinstance(md, MiniMaxM3IndexerNvfp4Metadata)

        nd = md.num_decode_tokens
        # Not sliced to the actual tokens: a padded full-CUDA-graph decode
        # batch also scores its padding rows, and every captured tensor must
        # keep its address and shape.
        index_q = index_query.view(-1, self.num_index_heads, self.index_head_dim)
        kv = self.index_cache.kv_cache
        # The kernels write logical block ids straight into the shared
        # [T, H, 16] buffer the attend reads.
        buf = self.topk_indices_buffer
        assert buf is not None

        if md.decode_q8kv4 is not None:
            d = md.decode_q8kv4
            rows = d.num_reqs * d.query_len
            if d.cp_plan is not None:
                assert self.decode_cp is not None and self.decode_cp_layer is not None
                self.decode_cp.forward(
                    self.decode_cp_layer,
                    d.cp_plan,
                    index_q[:rows],
                    kv,
                    buf[:rows],
                    d.query_len,
                )
            else:
                d.wrapper.run(index_q[:rows], kv, out=buf[:rows])
        for chunk in md.prefill_chunks:
            self._prefill(
                index_q[nd + chunk.token_start : nd + chunk.token_end],
                kv,
                chunk,
                buf[nd + chunk.token_start : nd + chunk.token_end],
            )
        return None, None

    def _prefill(
        self,
        index_q: torch.Tensor,
        kv: torch.Tensor,
        chunk: MiniMaxM3IndexerNvfp4PrefillChunk,
        out: torch.Tensor,
    ) -> None:
        (k_pages,) = current_workspace_manager().get_simultaneous(
            ((max(chunk.total_pages, 1), PAGE_SIZE, HEAD_DIM), torch.float8_e4m3fn)
        )
        minimax_m3_gather_nvfp4_index_k_fp8(
            kv,
            chunk.block_table,
            chunk.num_pages,
            chunk.page_offsets,
            chunk.max_pages,
            k_pages,
        )
        chunk.wrapper.run(index_q, k_pages, out=out)
