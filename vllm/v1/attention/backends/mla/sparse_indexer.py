# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 indexer backend for the DeepGEMM sparse MQA-logits path.

Candidate-consuming indexer layers score only the candidate blocks published
by the candidate-source layer (``fp8_fp4_(paged_)sparse_mqa_logits``) instead
of dense logits over the whole context. This backend extends the dense
indexer metadata with the per-row scratch that path needs, so the layer op
allocates nothing per step, and lets all consumer layers of a step share one
candidate expansion and DeepGEMM schedule (they all read the same candidate
blocks). Opt in with ``AttentionConfig.indexer_sparse_logits``.
"""

from dataclasses import dataclass, fields

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
    SPARSE_TOPK_KERNEL_SUPPORTED,
    pick_sparse_block_kv,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import has_deep_gemm_sparse_mqa
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV41IndexerBackend,
    dsa_indexer_uses_fp4,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


@dataclass
class SparseMQARowsMetadata:
    """Per-row state for one sparse-logits call: the decode rows, or one
    prefill chunk. Buffers are builder-owned views sized to the rows.

    ``kernel_metadata`` (the DeepGEMM schedule) is filled by the first
    consumer indexer layer of the step together with ``sparse_indices`` /
    ``end``; later consumer layers reuse all three.
    """

    row_ks: torch.Tensor
    """[rows] int32 K-range start per row; zeros for paged decode."""
    row_ke: torch.Tensor
    """[rows] int32 K-range end (prefill) or compressed context length."""
    sparse_indices: torch.Tensor
    """[rows, num_sparse_blocks] int32 DeepGEMM sparse block ids."""
    end: torch.Tensor
    """[rows] int32 valid sparse-column count per row."""
    col_indices: torch.Tensor
    """[rows, topk] int32 scratch for the sparse-column top-k."""
    block_table: torch.Tensor | None = None
    """Decode only: [rows, pages] per-row page table."""
    row_indices: torch.Tensor | None = None
    """Decode only: [rows] row -> request index."""
    kernel_metadata: torch.Tensor | None = None


@dataclass
class DeepseekV41SparseIndexerMetadata(DeepseekV32IndexerMetadata):
    sparse_block_kv: int = 0
    zero_starts: torch.Tensor | None = None
    """[max_rows] int32 zeros: ``row_ks`` for decode and the prefill top-k
    kernel's row starts."""
    sparse_decode: SparseMQARowsMetadata | None = None
    sparse_prefill: list[SparseMQARowsMetadata] | None = None
    """Parallel to ``prefill.chunks``."""


class DeepseekV41SparseIndexerBackend(DeepseekV41IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V41_SPARSE_INDEXER"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV41SparseIndexerMetadataBuilder"]:
        return DeepseekV41SparseIndexerMetadataBuilder


class DeepseekV41SparseIndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """Dense indexer metadata plus the sparse-logits row scratch.

    Every requirement of the sparse kernels is checked here, at engine start,
    rather than falling back to the dense path per step.
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        **kwargs,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, **kwargs)
        hf_config = vllm_config.model_config.hf_text_config
        candidate_block_size = getattr(hf_config, "candidate_block_size", 0)
        candidate_topk_blocks = getattr(hf_config, "candidate_topk_blocks", 0)
        topk_tokens = getattr(hf_config, "index_topk", 0)
        prefix = "attention_config.indexer_sparse_logits"
        if candidate_block_size <= 0 or candidate_topk_blocks <= 0:
            raise ValueError(
                f"{prefix} requires a two-level (candidate block) indexer model "
                "such as DeepSeek-V4.1."
            )
        if not dsa_indexer_uses_fp4(vllm_config):
            raise ValueError(
                f"{prefix} requires the MXFP4 indexer cache "
                "(attention_config.indexer_kv_dtype='mxfp4'): the sparse kernels "
                "take UE8M0-packed granularity-32 scales."
            )
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            raise ValueError(f"{prefix} requires an SM100-class GPU.")
        if not has_deep_gemm_sparse_mqa():
            raise ValueError(
                f"{prefix} requires DeepGEMM >= 2.8 "
                "(fp8_fp4_sparse_mqa_logits not found)."
            )
        if self.dcp_world_size > 1 or self.use_pcp:
            raise NotImplementedError(
                f"{prefix} is not supported with decode/prefill context parallel."
            )
        if not self.supports_varlen:
            raise ValueError(
                f"{prefix} requires the varlen paged MQA logits decode layout."
            )
        if topk_tokens not in SPARSE_TOPK_KERNEL_SUPPORTED:
            raise ValueError(
                f"{prefix} requires index_topk in {SPARSE_TOPK_KERNEL_SUPPORTED}, "
                f"got {topk_tokens}."
            )
        self.sparse_block_kv = pick_sparse_block_kv(candidate_block_size)
        if kv_cache_spec.block_size % self.sparse_block_kv != 0:
            raise ValueError(
                f"{prefix}: indexer cache pages ({kv_cache_spec.block_size} "
                f"tokens) must be a multiple of {self.sparse_block_kv} tokens."
            )
        self.num_sparse_blocks = candidate_topk_blocks * (
            candidate_block_size // self.sparse_block_kv
        )
        # Sparse logits are bf16 plus an fp32 copy for the top-k; make the
        # prefill chunker size chunks as if every row were this wide.
        self.min_split_seq_len = (
            self.num_sparse_blocks * self.sparse_block_kv * (2 + 4) + 3
        ) // 4

        # Rows are indexed by batch token (decode rows first, then prefill
        # chunks by token range), like ``arange_buffer``.
        max_rows = self.arange_buffer.shape[0]
        int32 = dict(dtype=torch.int32, device=device)
        self.sparse_indices_buffer = torch.zeros(
            (max_rows, self.num_sparse_blocks), **int32
        )
        self.end_buffer = torch.zeros(max_rows, **int32)
        self.col_indices_buffer = torch.zeros((max_rows, topk_tokens), **int32)
        self.zero_starts = torch.zeros(max_rows, **int32)
        logger.info_once(
            "DeepSeek V4.1 indexer: candidate consumers score only the candidate "
            "blocks with DeepGEMM sparse MQA logits (%d sparse blocks of %d "
            "tokens per row).",
            self.num_sparse_blocks,
            self.sparse_block_kv,
        )

    def _prefill_split_seq_lens(self, seq_lens_cpu: torch.Tensor) -> torch.Tensor:
        return seq_lens_cpu.clamp(min=self.min_split_seq_len)

    def _rows(self, start: int, end: int, row_ks, row_ke, **kwargs):
        return SparseMQARowsMetadata(
            row_ks=row_ks,
            row_ke=row_ke,
            sparse_indices=self.sparse_indices_buffer[start:end],
            end=self.end_buffer[start:end],
            col_indices=self.col_indices_buffer[start:end],
            **kwargs,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV41SparseIndexerMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata = DeepseekV41SparseIndexerMetadata(
            **{f.name: getattr(base, f.name) for f in fields(base)},
            sparse_block_kv=self.sparse_block_kv,
            zero_starts=self.zero_starts,
        )
        if base.decode is not None:
            decode = base.decode
            rows = decode.seq_lens.shape[0]
            # The varlen layout flattens every query to one row, so the base
            # decode tensors are already per row.
            assert decode.seq_lens.shape[1] == 1 and not decode.requires_padding
            assert decode.block_table.shape[0] == rows
            metadata.sparse_decode = self._rows(
                0,
                rows,
                row_ks=self.zero_starts[:rows],
                row_ke=decode.seq_lens.view(-1),
                block_table=decode.block_table,
                row_indices=(
                    decode.indices
                    if decode.indices is not None
                    else self.arange_buffer[:rows]
                ),
            )
        if base.prefill is not None:
            metadata.sparse_prefill = [
                self._rows(
                    chunk.token_start,
                    chunk.token_end,
                    row_ks=chunk.cu_seqlen_ks,
                    row_ke=chunk.cu_seqlen_ke,
                )
                for chunk in base.prefill.chunks
            ]
        return metadata
