# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA indexer metadata, builder, and backend.

A full copy of the common classes, which ROCm still uses, plus the DeepGEMM
prefill plan, so every field of the NVIDIA metadata is visible here.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    MLAAttentionSpec,
)

from ..common.qsa_indexer import (
    QSACompressedKeyCache,
    QSARawKeyCache,
    build_qsa_metadata,
)
from .ops.qsa_prefill_deepgemm import (
    QSADeepGEMMPrefillPlan,
    deepgemm_prefill_supported,
    prepare_deepgemm_prefill,
)


@dataclass
class QSAIndexerMetadata(AttentionMetadata):
    """Per-forward metadata for one raw or compressed QSA indexer cache."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    token_to_req: torch.Tensor
    # Positions in the original uncompressed sequence, padded with -1.
    logical_positions: torch.Tensor  # [num_tokens]
    # Complete compression groups visible to each query.
    visible_blocks: torch.Tensor  # [num_tokens]
    # K compression work items: (request index, local task index), -1 for padding.
    # Task 0 per request also saves the raw-K suffix, even without a complete group.
    # Empty for raw or uncompressed caches.
    k_work_metadata: torch.Tensor  # [num_k_work, 2]
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    max_query_len: int
    decode_query_len: int
    max_seq_len: int
    # Key rows per physical cache block.
    storage_block_size: int
    compress_ratio: int
    # DeepGEMM prefill plan, prepared once per batch and shared by the QSA
    # layers; None unless the compressed FP8 cache takes that path.
    deepgemm_prefill: QSADeepGEMMPrefillPlan | None = None


class QSAIndexerMetadataBuilder(AttentionMetadataBuilder[QSAIndexerMetadata]):
    """Build QSA metadata from vLLM's cache-group-specific common metadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        assert self.reorder_batch_threshold is not None
        # The raw-key ring is the only QSA cache with an uncompressed spec.
        self.is_circular_buffer = not isinstance(kv_cache_spec, MLAAttentionSpec)
        if isinstance(kv_cache_spec, MLAAttentionSpec):
            compress_ratio = kv_cache_spec.tokens_per_state
            assert isinstance(compress_ratio, int), (
                "QSA compression requires an integer tokens_per_state"
            )
            self.compress_ratio = compress_ratio
        else:
            self.compress_ratio = 1
        self.storage_block_size = kv_cache_spec.num_states
        self.use_deepgemm_prefill = (
            current_platform.is_cuda()
            and not self.is_circular_buffer
            and deepgemm_prefill_supported(
                kv_cache_spec.dtype,
                vllm_config.model_config.hf_text_config.indexer_n_heads,
                kv_cache_spec.head_size,
            )
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_to_req_buffer = torch.empty(
            max_tokens, dtype=torch.int32, device=device
        )
        self.slot_mapping_buffer = torch.empty(
            max_tokens, dtype=torch.int64, device=device
        )
        self.logical_positions_buffer = torch.empty(
            max_tokens, dtype=torch.int64, device=device
        )
        self.visible_blocks_buffer = torch.empty(
            max_tokens, dtype=torch.int32, device=device
        )
        max_requests = vllm_config.scheduler_config.max_num_seqs
        self.request_capacity = max_requests
        if not self.is_circular_buffer and self.compress_ratio != 1:
            max_k_work = (
                max_tokens + (self.compress_ratio - 1) * max_requests
            ) // self.compress_ratio
            self.k_work_metadata_buffer = torch.empty(
                max_k_work, 2, dtype=torch.int32, device=device
            )
        else:
            self.k_work_metadata_buffer = torch.empty(
                0, 2, dtype=torch.int32, device=device
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> QSAIndexerMetadata:
        del common_prefix_len, fast_build
        num_tokens = common_attn_metadata.num_actual_tokens
        decode_threshold = self.reorder_batch_threshold
        assert decode_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=decode_threshold,
                require_uniform=True,
            )
        )
        assert num_decodes + num_prefills == common_attn_metadata.num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens
        decode_query_len = 0
        if num_decodes > 0:
            query_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
            )
            nonzero_query_lens = query_lens_cpu[query_lens_cpu > 0]
            if nonzero_query_lens.numel() > 0:
                decode_query_len = int(nonzero_query_lens[0].item())
                assert torch.all(nonzero_query_lens == decode_query_len)
        build_k_work = not self.is_circular_buffer and self.compress_ratio != 1
        k_work_metadata = self.k_work_metadata_buffer
        request_capacity = None
        if build_k_work:
            num_requests = common_attn_metadata.query_start_loc.shape[0] - 1
            request_capacity = self.request_capacity
            max_num_work = (
                num_tokens + (self.compress_ratio - 1) * num_requests
            ) // self.compress_ratio
            k_work_metadata = self.k_work_metadata_buffer[:max_num_work]
        token_to_req, logical_positions, visible_blocks, slot_mapping = (
            build_qsa_metadata(
                common_attn_metadata,
                self.token_to_req_buffer,
                self.logical_positions_buffer,
                self.visible_blocks_buffer,
                self.slot_mapping_buffer,
                storage_block_size=self.storage_block_size,
                compress_ratio=self.compress_ratio,
                circular_buffer_size=(
                    self.kv_cache_spec.block_size if self.is_circular_buffer else 0
                ),
                k_work_metadata_buffer=k_work_metadata if build_k_work else None,
                request_capacity=request_capacity,
            )
        )
        metadata = QSAIndexerMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=slot_mapping,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc=common_attn_metadata.query_start_loc,
            token_to_req=token_to_req,
            logical_positions=logical_positions,
            visible_blocks=visible_blocks,
            k_work_metadata=k_work_metadata,
            num_actual_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            decode_query_len=decode_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
        )
        if self.use_deepgemm_prefill and num_prefill_tokens:
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            metadata.deepgemm_prefill = prepare_deepgemm_prefill(
                metadata,
                common_attn_metadata.query_start_loc_cpu,
                common_attn_metadata.seq_lens_cpu_upper_bound,
            )
        return metadata


class QSAIndexerBackend(AttentionBackend):
    """Key-only dummy backend for out-of-band QSA indexer cache operations."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # fp8 entries allow the optional e4m3 compressed indexer cache.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_INDEXER"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "QSA indexer caches run out-of-band and have no attention impl"
        )

    @staticmethod
    def get_builder_cls() -> type[QSAIndexerMetadataBuilder]:
        return QSAIndexerMetadataBuilder

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # QSA pages are packed beside the main KV pages within each block.
        return (KVCacheLayout.BLNHC, KVCacheLayout.BLHNC)


# The common cache layers are shared with ROCm; these only select the NVIDIA
# backend above.
class NvidiaQSARawKeyCache(QSARawKeyCache):
    def get_attn_backend(self) -> type[AttentionBackend]:
        return QSAIndexerBackend


class NvidiaQSACompressedKeyCache(QSACompressedKeyCache):
    def get_attn_backend(self) -> type[AttentionBackend]:
        return QSAIndexerBackend
