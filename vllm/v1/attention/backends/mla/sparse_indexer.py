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
    check_deep_select_layout,
    has_deep_select,
    pick_sparse_block_kv,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    has_deep_gemm_sparse_mqa,
    is_sm90_paged_sparse_mqa_logits_supported,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
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
        self.sm90_decode_only = (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(90)
        )
        if not self.sm90_decode_only and not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            raise ValueError(f"{prefix} requires an SM90- or SM100-class GPU.")
        if not has_deep_gemm_sparse_mqa():
            raise ValueError(
                f"{prefix} requires DeepGEMM >= 2.8 "
                "(fp8_fp4_sparse_mqa_logits not found)."
            )
        if self.dcp_world_size > 1 or self.use_pcp:
            raise NotImplementedError(
                f"{prefix} is not supported with decode/prefill context parallel."
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
        num_sparse_cols = self.num_sparse_blocks * self.sparse_block_kv
        self._validate_kernel_config(
            vllm_config, kv_cache_spec, num_sparse_cols, topk_tokens
        )
        if not self.sm90_decode_only:
            # Budget the SM100 sparse prefill's BF16 logits in FP32 units.
            self.min_split_seq_len = (num_sparse_cols * 2 + 3) // 4

        # Rows are indexed by batch token; SM90 only needs decode capacity.
        max_rows = (
            self.expanded_block_table_buffer.shape[0]
            if self.sm90_decode_only
            else self.arange_buffer.shape[0]
        )
        int32 = dict(dtype=torch.int32, device=device)
        self.sparse_indices_buffer = torch.zeros(
            (max_rows, self.num_sparse_blocks), **int32
        )
        self.end_buffer = torch.zeros(max_rows, **int32)
        self.col_indices_buffer = torch.zeros((max_rows, topk_tokens), **int32)
        self.zero_row_ks = torch.zeros(max_rows, **int32)
        logger.info_once(
            "DeepSeek V4.1 indexer: candidate consumers score only the candidate "
            "blocks during %s with DeepGEMM sparse MQA logits (%d sparse blocks of %d "
            "tokens per row).",
            "decode" if self.sm90_decode_only else "prefill/decode",
            self.num_sparse_blocks,
            self.sparse_block_kv,
        )

    def _validate_kernel_config(
        self,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
        num_sparse_cols: int,
        topk_tokens: int,
    ) -> None:
        prefix = "attention_config.indexer_sparse_logits"
        topk_backend = vllm_config.kernel_config.sparse_indexer_topk_backend
        if self.sm90_decode_only:
            hf_config = vllm_config.model_config.hf_text_config
            supports_fp8_sparse = not dsa_indexer_uses_fp4(
                vllm_config
            ) and is_sm90_paged_sparse_mqa_logits_supported(
                hf_config.index_n_heads,
                hf_config.index_head_dim,
                kv_cache_spec.num_states,
                self.sparse_block_kv,
            )
            if not supports_fp8_sparse:
                raise ValueError(
                    f"{prefix} on SM90 requires FP8 H=32, D=128, 32/64-token "
                    "cache pages and DeepGEMM with SM90 paged sparse support."
                )
            if topk_backend not in ("auto", "persistent", "cooperative"):
                raise ValueError(
                    f"{prefix} on SM90 supports auto, persistent or cooperative "
                    f"top-k, got {topk_backend!r}."
                )
            if (
                not 0 < self.num_sparse_blocks <= 4096
                or num_sparse_cols < topk_tokens
                or topk_tokens not in (512, 1024, 2048)
            ):
                raise ValueError(f"{prefix}: unsupported SM90 candidate/top-k size.")
            return

        if not self.supports_varlen:
            raise ValueError(
                f"{prefix} requires the varlen paged MQA logits decode layout."
            )
        if not has_deep_select():
            raise ValueError(
                f"{prefix} requires the DeepSelect top-k extension "
                "(vllm._deepselect_C) for this GPU."
            )
        if topk_backend not in ("auto", "deep_select"):
            raise ValueError(
                f"{prefix} uses DeepSelect for the top-k; "
                f"kernel_config.sparse_indexer_topk_backend='{topk_backend}' "
                "is not supported with it."
            )
        check_deep_select_layout(num_sparse_cols, topk_tokens)

    def _prefill_split_seq_lens(self, seq_lens_cpu: torch.Tensor) -> torch.Tensor:
        if self.sm90_decode_only:
            return seq_lens_cpu
        return seq_lens_cpu.clamp(min=self.min_split_seq_len)

    def _rows(
        self,
        start: int,
        end: int,
        *,
        row_ks: torch.Tensor,
        row_ke: torch.Tensor,
        block_table: torch.Tensor | None = None,
        row_indices: torch.Tensor | None = None,
    ) -> SparseMQARowsMetadata:
        return SparseMQARowsMetadata(
            row_ks=row_ks,
            row_ke=row_ke,
            sparse_indices=self.sparse_indices_buffer[start:end],
            end=self.end_buffer[start:end],
            col_indices=self.col_indices_buffer[start:end],
            block_table=block_table,
            row_indices=row_indices,
        )

    def _decode_rows(
        self, decode: DeepSeekV32IndexerDecodeMetadata
    ) -> SparseMQARowsMetadata:
        num_requests, next_n = decode.seq_lens.shape
        num_rows = num_requests * next_n
        block_table = decode.block_table
        if self.sm90_decode_only and next_n > 1:
            # Dense source layers keep their native [B, next_n] layout.
            # Expand the page table once per step for all sparse consumers.
            block_table = self.expanded_block_table_buffer[:num_rows]
            block_table.view(num_requests, next_n, -1).copy_(
                decode.block_table[:, None, :]
            )
        else:
            assert next_n == 1 and not decode.requires_padding
        assert block_table.shape[0] == num_rows
        row_indices = decode.indices
        if row_indices is None:
            row_indices = self.arange_buffer[:num_rows]
        return self._rows(
            0,
            num_rows,
            row_ks=self.zero_row_ks[:num_rows],
            row_ke=decode.seq_lens.view(-1),
            block_table=block_table,
            row_indices=row_indices,
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
        )
        if base.decode is not None:
            metadata.sparse_decode = self._decode_rows(base.decode)
        if base.prefill is not None and not self.sm90_decode_only:
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
