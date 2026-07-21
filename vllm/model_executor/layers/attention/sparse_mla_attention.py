# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared forward_mha implementation and metadata builder for sparse MLA backends."""

from shutil import which
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

import numpy as np
import torch

from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBaseImpl,
    MLACommonPrefillMetadata,
    build_mla_chunked_context_metadata,
    get_mla_dims,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.torch_utils import np_to_pinned_tensor
from vllm.v1.attention.backend import (
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

T = TypeVar("T", bound=AttentionMetadata)


class SparseMLACommonMetadataBuilder(AttentionMetadataBuilder[T]):
    metadata_cls: type[T]
    require_uniform_decodes: ClassVar[bool] = False

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
        parallel_config = vllm_config.parallel_config
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        try:
            self.dcp_world_size = get_dcp_group().world_size
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.dcp_local_block_size = self.cp_kv_cache_interleave_size
        self.dcp_virtual_block_size = self.dcp_local_block_size * self.dcp_world_size

        self.chunked_prefill_workspace_size = (
            self.determine_chunked_prefill_workspace_size(vllm_config)
        )
        workspace_head_size = (
            self.mla_dims.kv_lora_rank + self.mla_dims.qk_rope_head_dim
        )
        workspace_rows = self.chunked_prefill_workspace_size
        if self.dcp_world_size > 1:
            # DCP gathers each rank's local KV shard into the workspace, so it
            # needs an extra 1/DCP rows beyond the TP allocation.
            assert self.chunked_prefill_workspace_size % self.dcp_world_size == 0
            workspace_rows += self.chunked_prefill_workspace_size // self.dcp_world_size
        self.chunked_prefill_workspace = torch.empty(
            (workspace_rows, workspace_head_size),
            dtype=self.model_config.dtype,
            device=device,
        )
        layer_prefill_backend = vllm_config.compilation_config.static_forward_context[
            layer_names[0]
        ].prefill_backend
        self._prefill_backend = (
            layer_prefill_backend.clone() if layer_prefill_backend is not None else None
        )

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
        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        prefill_query_start_loc_cpu = qsl_cpu[num_decodes:] - qsl_cpu[num_decodes]

        return build_mla_chunked_context_metadata(
            context_lens_cpu=context_lens_cpu,
            prefill_query_start_loc_cpu=prefill_query_start_loc_cpu,
            num_prefills=num_prefills,
            chunked_prefill_workspace=self.chunked_prefill_workspace,
            chunked_prefill_workspace_size=self.chunked_prefill_workspace_size,
            block_size=self.kv_cache_spec.block_size,
            align_chunk_to_block=current_platform.is_cuda(),
            device=self.device,
            dcp_world_size=self.dcp_world_size,
            dcp_local_block_size=self.dcp_local_block_size,
            dcp_virtual_block_size=self.dcp_virtual_block_size,
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
            treat_short_extends_as_decodes=not self.use_pcp,
            require_uniform=self.require_uniform_decodes,
        )
        (
            prefill_query_start_loc,
            prefill_max_query_len,
            prefill_query_lens_cpu,
        ) = self._build_prefill_fields(common_attn_metadata, num_decodes, num_prefills)

        prefill_max_seq_len = 0
        prefill: MLACommonPrefillMetadata | None = None
        if num_prefills > 0 and self._prefill_backend is not None:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            prefill_max_seq_len = int(
                seq_lens_cpu[num_decodes : num_decodes + num_prefills].max().item()
            )
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
            prefill_max_seq_len=prefill_max_seq_len,
            prefill=prefill,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
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

        self._use_flashinfer_concat_mla_k = (
            has_flashinfer()
            and which("ninja") is not None
            and (self.num_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        )
