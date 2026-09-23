# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 indexer backend for the ROCm paged MXFP4 path.

Extends the dense indexer metadata with what aiter's paged MXFP4 MQA-logits
kernel needs to read the cache in place: each prefill chunk split into its
requests, one block-table row per decode query row, and, with
``AttentionConfig.indexer_sparse_logits``, the per-step state the candidate
consumers share. Selected for every V4.1 indexer cache when
``indexer_kv_dtype="mxfp4"`` on ROCm.
"""

import bisect
from dataclasses import dataclass, field, fields

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV41IndexerBackend,
)
from vllm.v1.attention.ops.rocm_mxfp4_indexer import (
    check_rocm_mxfp4_cache_geometry,
    rocm_mxfp4_n_per_tile,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec

logger = init_logger(__name__)

# The consumers walk the pool instead of the context once the context is this
# many pools long. Not measured end to end yet: at speculative decode the
# gather loses to the dense walk until the pool is about a 3x cut (32 heads),
# and at prefill it issues 1.15-1.19x the dense walk's loads per key.
_GATHER_MIN_CUT_DECODE = 3.0
_GATHER_MIN_CUT_PREFILL = 1.5


@dataclass
class RocmMxfp4PrefillPlan:
    """One prefill chunk as the requests the kernel launches on. A request's
    rows are one sequence's query chunk, which is the kernel's next_n."""

    requests: list[tuple[int, int, int]]
    """(first row, end row, request) per request with rows in the chunk; rows
    count from ``chunk.token_start``, requests index ``chunk.block_table``."""
    launches: list[tuple[int, int, int, int]]
    """(first row, end row, first request, requests) per dense launch. The
    kernel's sequences all have next_n rows, so a launch takes a run of
    consecutive requests with equal query rows."""
    width: int
    """Logits columns: an upper bound on the chunk's compressed contexts."""
    row_ends: torch.Tensor
    """[rows] int32 exclusive compressed key bound of each row."""
    context_lens: torch.Tensor
    """[chunk.num_reqs] int32 compressed context of each request."""
    block_ends: torch.Tensor | None = None
    """[rows] int32 candidate blocks each row sees, for the source's pool."""
    use_gather: bool = False
    gathers: list = field(default_factory=list)
    """Per request, the pool the first consumer resolved for the others."""


@dataclass
class DeepseekV41RocmMxfp4IndexerMetadata(DeepseekV32IndexerMetadata):
    prefill_plans: list[RocmMxfp4PrefillPlan] = field(default_factory=list)
    """Parallel to ``prefill.chunks``."""
    decode_row_lens: torch.Tensor | None = None
    """[rows] int32 compressed context of each decode query row."""
    decode_block_ends: torch.Tensor | None = None
    decode_use_gather: bool = False
    decode_gather: tuple[dict, torch.Tensor] | None = None
    """The pool the first consumer resolved for the decode rows."""


def plan_prefill_chunks(
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata],
    query_start_loc: list[int],
    seq_lens_cpu: torch.Tensor,
    context_lens: torch.Tensor,
    compress_ratio: int,
    candidate_block_size: int,
    min_gather_width: float | None,
) -> list[RocmMxfp4PrefillPlan]:
    """Split each chunk into its requests.

    ``query_start_loc`` and ``seq_lens_cpu`` are the step's host copies (the
    latter an upper bound), ``context_lens`` the device compressed lengths.
    A chunk covers whole requests, or a query slice of one request.
    """
    block = candidate_block_size
    plans = []
    for chunk in chunks:
        t0, t1 = chunk.token_start, chunk.token_end
        first = bisect.bisect_right(query_start_loc, t0) - 1
        requests = []
        launches: list[tuple[int, int, int, int]] = []
        for req in range(chunk.num_reqs):
            lo = max(query_start_loc[first + req], t0) - t0
            hi = min(query_start_loc[first + req + 1], t1) - t0
            if hi <= lo:
                continue
            requests.append((lo, hi, req))
            if launches:
                run_lo, run_hi, run_req, seqs = launches[-1]
                if run_req + seqs == req and run_hi - run_lo == seqs * (hi - lo):
                    launches[-1] = (run_lo, hi, run_req, seqs + 1)
                    continue
            launches.append((lo, hi, req, 1))
        reqs = slice(first, first + chunk.num_reqs)
        width = int(seq_lens_cpu[reqs].max()) // compress_ratio
        row_ends = chunk.cu_seqlen_ke - chunk.cu_seqlen_ks
        plans.append(
            RocmMxfp4PrefillPlan(
                requests=requests,
                launches=launches,
                width=width,
                row_ends=row_ends,
                context_lens=context_lens[reqs],
                block_ends=(row_ends + block - 1) // block if block else None,
                use_gather=min_gather_width is not None and width >= min_gather_width,
                gathers=[None] * len(requests),
            )
        )
    return plans


class DeepseekV41RocmMxfp4IndexerBackend(DeepseekV41IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V41_ROCM_MXFP4_INDEXER"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV41RocmMxfp4IndexerMetadataBuilder"]:
        return DeepseekV41RocmMxfp4IndexerMetadataBuilder


class DeepseekV41RocmMxfp4IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """Dense indexer metadata plus the in-place paged launches' layout.

    Requirements are checked here, at engine start.
    """

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        # Decode is always flattened to a row per query token, see __init__.
        return AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        **kwargs,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, **kwargs)
        name = "the ROCm MXFP4 indexer"
        if self.dcp_world_size > 1 or self.use_pcp:
            raise NotImplementedError(
                f"{name} does not support decode/prefill context parallel."
            )
        hf_config = vllm_config.model_config.hf_text_config
        num_heads, head_dim = hf_config.index_n_heads, hf_config.index_head_dim
        self.page_entries = kv_cache_spec.block_size // self.compress_ratio
        check_rocm_mxfp4_cache_geometry(num_heads, head_dim, self.page_entries)
        # The kernel takes one block-table row per query row. Flattening gives
        # every decode token its own; native spec rows would share one.
        self.use_flattening = True
        # Nothing gathers K here, so only the logits budget splits a chunk.
        self.max_prefill_buffer_size = 1 << 62

        self.candidate_block_size = getattr(hf_config, "candidate_block_size", 0)
        self.num_candidate_cols = 0
        if vllm_config.attention_config.indexer_sparse_logits:
            prefix = "attention_config.indexer_sparse_logits"
            block = self.candidate_block_size
            topk_blocks = getattr(hf_config, "candidate_topk_blocks", 0)
            if block <= 0 or topk_blocks <= 0:
                raise ValueError(
                    f"{prefix} requires a two-level (candidate block) indexer "
                    "model such as DeepSeek-V4.1."
                )
            if topk_blocks & (topk_blocks - 1):
                raise ValueError(
                    f"{prefix} on ROCm resolves the candidate pool with a sort "
                    f"that needs a power-of-two block count, not {topk_blocks}."
                )
            n_per_tile = rocm_mxfp4_n_per_tile(num_heads, head_dim)
            if self.page_entries % block or block > n_per_tile:
                raise ValueError(
                    f"{prefix} on ROCm needs {block}-entry candidate blocks to "
                    f"tile a {self.page_entries}-entry indexer page and fit in "
                    f"one {n_per_tile}-entry shuffle group."
                )
            self.num_candidate_cols = topk_blocks * block
            logger.info_once(
                "DeepSeek V4.1 indexer: candidate consumers score the %d "
                "candidate positions with aiter's paged MXFP4 gather once the "
                "context is %.1fx (decode) / %.1fx (prefill) as long.",
                self.num_candidate_cols,
                _GATHER_MIN_CUT_DECODE,
                _GATHER_MIN_CUT_PREFILL,
            )

        scheduler_config = vllm_config.scheduler_config
        int32 = dict(dtype=torch.int32, device=device)
        self.decode_block_ends_buffer = torch.zeros(
            self.arange_buffer.shape[0], **int32
        )
        self.context_lens_buffer = torch.zeros(scheduler_config.max_num_seqs, **int32)

    def _prefill_split_seq_lens(self, seq_lens_cpu: torch.Tensor) -> torch.Tensor:
        # The consumers' logits are [rows, pool] fp32 whatever the context, so
        # the chunker budgets every row as at least that wide.
        return seq_lens_cpu.clamp(min=self.num_candidate_cols)

    def _gather_pays(self, context: int, cut: float) -> bool:
        return self.num_candidate_cols > 0 and context >= cut * self.num_candidate_cols

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV41RocmMxfp4IndexerMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata = DeepseekV41RocmMxfp4IndexerMetadata(
            **{f.name: getattr(base, f.name) for f in fields(base)}
        )
        if base.decode is not None:
            lengths = base.decode.seq_lens.reshape(-1)
            metadata.decode_row_lens = lengths
            block = self.candidate_block_size
            if block:
                ends = self.decode_block_ends_buffer[: lengths.shape[0]]
                torch.add(lengths, block - 1, out=ends)
                ends.floor_divide_(block)
                metadata.decode_block_ends = ends
            metadata.decode_use_gather = self._gather_pays(
                base.max_seq_len // self.compress_ratio, _GATHER_MIN_CUT_DECODE
            )
        if base.prefill is not None:
            cm = common_attn_metadata
            assert cm.seq_lens_cpu_upper_bound is not None
            context_lens = self.context_lens_buffer[: cm.num_reqs]
            torch.floor_divide(
                cm.seq_lens[: cm.num_reqs], self.compress_ratio, out=context_lens
            )
            metadata.prefill_plans = plan_prefill_chunks(
                base.prefill.chunks,
                cm.query_start_loc_cpu.tolist(),
                cm.seq_lens_cpu_upper_bound,
                context_lens,
                self.compress_ratio,
                self.candidate_block_size,
                self.num_candidate_cols * _GATHER_MIN_CUT_PREFILL
                if self.num_candidate_cols
                else None,
            )
        return metadata
