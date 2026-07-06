# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared forward_mha implementation and metadata builder for sparse MLA backends."""

from shutil import which
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBaseImpl,
    MLACommonPrefillMetadata,
    get_mla_dims,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.math_utils import cdiv, round_down
from vllm.utils.torch_utils import np_to_pinned_tensor
from vllm.v1.attention.backend import (
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

T = TypeVar("T", bound=AttentionMetadata)


def dense_mha_fa4_available(qk_head_dim: int) -> bool:
    """Whether an FA4 (>=v4) varlen kernel exists for this qk_head_dim."""
    fa_version = get_flash_attn_version(head_size=qk_head_dim)
    return fa_version is not None and fa_version >= 4


class SparseMLACommonMetadataBuilder(AttentionMetadataBuilder[T]):
    metadata_cls: type[T]

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.device = device
        self.model_config = vllm_config.model_config
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens: int = vllm_config.model_config.hf_config.index_topk
        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )
        self.chunked_prefill_workspace_size = (
            self.determine_chunked_prefill_workspace_size(vllm_config)
        )
        workspace_head_size = (
            self.mla_dims.kv_lora_rank + self.mla_dims.qk_rope_head_dim
        )
        self.chunked_prefill_workspace = torch.empty(
            (self.chunked_prefill_workspace_size, workspace_head_size),
            dtype=self.model_config.dtype,
            device=device,
        )
        # Dense-MHA prefill runs through the shared MLA prefill backend (FA4 /
        # TRT-LLM ragged / FlashInfer), selected per layer. Each builder gets its
        # own clone since the backend caches per-forward metadata.
        self._prefill_backend = vllm_config.compilation_config.static_forward_context[
            layer_names[0]
        ].prefill_backend.clone()

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: "VllmConfig") -> int:
        scheduler_config = vllm_config.scheduler_config
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        topk_tokens = model_config.hf_config.index_topk

        workspace_size = min(
            max(
                8 * model_config.max_model_len,
                4 * scheduler_config.max_num_seqs * cache_config.block_size,
            ),
            64 * 1024,
            scheduler_config.max_num_seqs * topk_tokens,
        )
        return max(
            workspace_size,
            scheduler_config.max_num_seqs * cache_config.block_size,
        )

    def _build_req_id_per_token(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> torch.Tensor:
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            np_to_pinned_tensor(req_id_per_token), non_blocking=True
        )
        return self.req_id_per_token_buffer[:num_tokens]

    def _build_chunked_context_fields(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
        num_decodes: int,
        num_prefills: int,
        prefill_query_lens_cpu: torch.Tensor | None,
    ) -> "MLACommonPrefillMetadata.ChunkedContextMetadata | None":
        if num_prefills == 0 or prefill_query_lens_cpu is None:
            return None

        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        assert seq_lens_cpu is not None
        context_lens_cpu = (
            seq_lens_cpu[num_decodes : num_decodes + num_prefills]
            - prefill_query_lens_cpu
        )
        max_context_len = context_lens_cpu.max().item()
        if max_context_len <= 0:
            return None

        num_prefills_with_context = (context_lens_cpu > 0).sum().item()
        assert num_prefills_with_context > 0

        max_context_chunk = (
            self.chunked_prefill_workspace_size // num_prefills_with_context
        )
        if current_platform.is_cuda():
            max_context_chunk = round_down(
                max_context_chunk, self.kv_cache_spec.block_size
            )
        assert max_context_chunk > 0

        num_chunks = cdiv(max_context_len, max_context_chunk)
        chunk_starts = torch.empty(
            num_chunks, num_prefills, dtype=torch.int32, pin_memory=True
        ).copy_(
            torch.arange(num_chunks, dtype=torch.int32)
            .multiply_(max_context_chunk)
            .unsqueeze(1)
        )
        chunk_ends = torch.min(
            context_lens_cpu.unsqueeze(0), chunk_starts + max_context_chunk
        )
        chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)

        cu_seq_lens_cpu = torch.zeros(
            num_chunks, num_prefills + 1, dtype=torch.int32, pin_memory=True
        )
        torch.cumsum(
            chunk_seq_lens, dim=1, out=cu_seq_lens_cpu[:, 1:], dtype=torch.int32
        )
        chunk_total_token = cu_seq_lens_cpu[:, -1]

        max_tokens_over_chunk = chunk_total_token.max().item()
        token_to_seq_cpu = torch.zeros(
            (num_chunks, max_tokens_over_chunk), dtype=torch.int32, pin_memory=True
        )
        req_indices = torch.arange(num_prefills, dtype=torch.int32)
        for i in range(num_chunks):
            token_to_seq = torch.repeat_interleave(req_indices, chunk_seq_lens[i])
            token_to_seq_cpu[i, : token_to_seq.shape[0]] = token_to_seq

        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        prefill_qsl_cpu = qsl_cpu[num_decodes:] - qsl_cpu[num_decodes]
        prefill_tokens_with_context = prefill_qsl_cpu[num_prefills_with_context].item()

        return MLACommonPrefillMetadata.ChunkedContextMetadata(
            cu_seq_lens=cu_seq_lens_cpu.to(self.device, non_blocking=True),
            starts=chunk_starts.to(self.device, non_blocking=True),
            seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
            max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
            seq_lens=chunk_seq_lens,
            workspace=self.chunked_prefill_workspace,
            token_to_seq=token_to_seq_cpu.to(self.device, non_blocking=True),
            chunk_total_token=chunk_total_token,
            prefill_tokens_with_context=prefill_tokens_with_context,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
    ) -> T:
        req_id_per_token = self._build_req_id_per_token(common_attn_metadata)

        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold or 1,
        )
        (
            prefill_query_start_loc,
            prefill_max_query_len,
            prefill_query_lens_cpu,
        ) = self._build_prefill_fields(common_attn_metadata, num_decodes, num_prefills)

        prefill: MLACommonPrefillMetadata | None = None
        if num_prefills > 0:
            prefill = MLACommonPrefillMetadata(
                block_table=common_attn_metadata.block_table_tensor[num_decodes:, ...],
                query_start_loc=prefill_query_start_loc,
                max_query_len=prefill_max_query_len,
                chunked_context=self._build_chunked_context_fields(
                    common_attn_metadata,
                    num_decodes,
                    num_prefills,
                    prefill_query_lens_cpu,
                ),
                q_data_type=self.model_config.dtype,
                output_dtype=self.model_config.dtype,
                prefill_backend=self._prefill_backend,
            )
            self._prefill_backend.prepare_metadata(prefill)

        return self.metadata_cls(  # type: ignore[call-arg]
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=req_id_per_token,
            seq_lens=common_attn_metadata.seq_lens,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            prefill=prefill,
        )

    @staticmethod
    def _build_prefill_fields(
        common_attn_metadata: "CommonAttentionMetadata",
        num_decodes: int,
        num_prefills: int,
    ) -> tuple[
        torch.Tensor | None,  # prefill_query_start_loc
        int,  # prefill_max_query_len
        torch.Tensor | None,  # prefill_query_lens_cpu
    ]:
        if num_prefills == 0:
            return None, 0, None

        offset = common_attn_metadata.query_start_loc[num_decodes]
        prefill_query_start_loc = (
            common_attn_metadata.query_start_loc[num_decodes:] - offset
        )

        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        prefill_qsl_cpu = qsl_cpu[num_decodes:] - qsl_cpu[num_decodes]
        prefill_query_lens = prefill_qsl_cpu[1:] - prefill_qsl_cpu[:-1]
        prefill_max_query_len = int(prefill_query_lens.max().item())

        return (
            prefill_query_start_loc,
            prefill_max_query_len,
            prefill_query_lens,
        )


class SparseMLACommonImpl(MLACommonBaseImpl[T], Generic[T]):
    """Sparse MLA base: shared dense-MHA prefill (from MLACommonBaseImpl) plus the
    sparse top-k MQA decode path. Subclasses implement forward_mqa."""

    is_sparse = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA-specific
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: "ColumnParallelLinear",
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_b_proj,
        )

        # The indexer carries the shared buffer for normal layers and tests;
        # the explicitly-passed buffer covers backbone skip layers, whose
        # indexer is not constructed (see deepseek_v2.py).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer  # type: ignore[attr-defined]
            if indexer is not None
            else topk_indices_buffer
        )

        self._fa4_available = dense_mha_fa4_available(qk_head_dim)

        self._use_flashinfer_concat_mla_k = (
            has_flashinfer()
            and which("ninja") is not None
            and (self.num_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        )
