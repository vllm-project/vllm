# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU DeepSeek-V4 sparse-MLA backend descriptor.

DeepSeek-V4 attention runs entirely through
``DeepseekV4Attention``/``DeepseekV4CPUAttention``, never through a generic
``AttentionImpl``. ``DeepseekV4CPUSparseBackend`` exists so the CPU attention
layer has its own backend name and its own metadata-builder class: CPU is not
allowed to run triton-cpu inside the model's forward path, so
``DeepseekV4CPUFlashMLAMetadataBuilder`` overrides the shared base's
``_build_c128a_metadata`` (triton-backed in
``DeepseekV4SparseMLAMetadataBuilder``) with a plain-eager-PyTorch
reimplementation of the same per-token block-table-resolution math.
"""

import torch

from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    _LAYER_TYPE_C4A,
    _LAYER_TYPE_C128A,
    _LAYER_TYPE_SWAONLY,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.attention.ops.flashmla import FlashMLASchedMeta


class DeepseekV4CPUFlashMLAMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    """CPU sparse-MLA metadata builder: same fields as the shared base, but
    skips the C128A dense-topk metadata entirely -- unlike CUDA/XPU/ROCm,
    ``DeepseekV4CPUAttention.forward_mqa`` never reads
    ``c128a_global_decode_topk_indices``/``c128a_decode_topk_lens``/
    ``c128a_prefill_topk_indices``; it recomputes the same local top-k
    directly from ``positions`` instead, so building it here (whether
    eagerly or via the shared base's triton kernel) would be wasted work.
    """

    def _build_c128a_metadata(
        self,
        cm: CommonAttentionMetadata,
        req_id_per_token: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        return {}


class DeepseekV4CPUSparseBackend(DeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "CPU_V4_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV4CPUFlashMLAMetadataBuilder]:
        return DeepseekV4CPUFlashMLAMetadataBuilder


class DeepseekV4CPUIndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """CPU indexer metadata builder: same fields as the shared base, but
    prefill requests are never split into multiple chunks.

    ``DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks`` (the
    shared base's default) bounds two things the CUDA/XPU indexer kernels
    need: the flat-gather workspace size and the dense M*N logits tensor
    those kernels allocate for a chunk. The CPU indexer
    (``sparse_attn_indexer_cpu``) allocates neither -- it reads the paged
    K-cache directly via ``fp8_paged_mqa_logits_cpu``/
    ``topk_transform_512_cpu``, with no chunk-splitting of its own.
    With chunked prefill now enabled for this model on CPU (see
    ``CpuPlatform.check_and_update_config``'s ``amx_mla_or_dsv4_enabled``),
    the total query-token count for one step is already bounded by
    ``max_num_batched_tokens`` -- there is nothing left here to bound, so
    this always returns a single chunk spanning the whole step's prefill
    batch.
    """

    @staticmethod
    def _split_indexer_prefill_chunks(
        compressed_seq_lens_cpu: torch.Tensor,
        prefill_query_lens_cpu: torch.Tensor,
        workspace_size: int,
        max_logits_bytes: int,
        request_offset: int = 0,
    ) -> list[tuple[slice, slice]]:
        # workspace_size/max_logits_bytes are unused: no flat-gather
        # workspace or dense M*N buffer is allocated.
        num_reqs = compressed_seq_lens_cpu.shape[0]
        total_query_len = int(prefill_query_lens_cpu.sum().item())
        return [
            (
                slice(request_offset, request_offset + num_reqs),
                slice(0, total_query_len),
            )
        ]


class DeepseekV4CPUIndexerBackend(DeepseekV4IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "CPU_V4_INDEXER"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV4CPUIndexerMetadataBuilder]:
        return DeepseekV4CPUIndexerMetadataBuilder


class DeepseekV4CPUSparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    """CPU SWA metadata builder: same fields as the shared base."""

    def build_tile_scheduler(
        self, num_decode_tokens: int
    ) -> dict[str, FlashMLASchedMeta | None]:
        """CPU never runs the FlashMLA tile-scheduler planner (that's a CUDA
        C++ decode-path concern) -- always return the all-``None`` sentinel
        the shared base's own CPU branch returns."""
        return {
            _LAYER_TYPE_SWAONLY: None,
            _LAYER_TYPE_C4A: None,
            _LAYER_TYPE_C128A: None,
        }
