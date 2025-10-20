# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionLayer,
    AttentionMetadata,
)
from vllm.attention.backends.utils import get_mla_dims
from vllm.attention.ops.flashmla import (
    flash_mla_sparse_prefill,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import cdiv
from vllm.v1.attention.backends.mla.common import MLACommonBaseImpl
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import PerKVCacheTokenWorkspace, current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)
"""
NOTE: FlashMLA Sparse uses an fp8 cache with the following format

In the "FP8 with scale" format, each token's KV cache is 656 Bytes, 
structured as:
-   **First 512 bytes:** The "quantized NoPE" part, containing 512 
    `float8_e4m3` values.
-   **Next 16 bytes:** Scale factors, containing 4 `float32` values. 
    The first `float32` is the scale for the first 128 `float8_e4m3` values, 
    the second for the next 128, and so on.
-   **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This 
    part is not quantized for accuracy.
"""


class FlashMLASparseBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE"

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return FlashMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashMLASparseMetadataBuilder"]:
        return FlashMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLASparseImpl"]:
        return FlashMLASparseImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # custom storage fromat is 656 bytes
            #  see FlashMLA readme.md for details
            return (num_blocks, block_size, 656)
        else:
            return (num_blocks, block_size, head_size)

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]


@dataclass
class FlashMLASparseMetadata:
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048

    num_prefill_reqs: int = 0
    num_decode_reqs: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    @dataclass
    class FP8KernelMetadata:
        scheduler_metadata: torch.Tensor | None
        num_splits: torch.Tensor
        dummy_block_table: torch.Tensor
        cache_lens: torch.Tensor

    fp8_extra_metadata: FP8KernelMetadata | None = None


class FlashMLASparseMetadataBuilder(AttentionMetadataBuilder[FlashMLASparseMetadata]):
    cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        cache_config = vllm_config.cache_config
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.device = device

        # Treat requests with query length <= 1 as decodes to match the
        # DeepGEMM indexer constraint (fp8_paged_mqa_logits only supports next_n <= 2)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        props = torch.cuda.get_device_properties(device)
        sm_count = props.multi_processor_count

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)

        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.use_fp8_kv_cache = cache_config.cache_dtype == "fp8_ds_mla"
        self.topk_tokens_tensor = torch.tensor(
            [self.topk_tokens], device=device, dtype=torch.int32
        )
        self.max_model_len_tensor = torch.tensor(
            [self.model_config.max_model_len], device=device, dtype=torch.int32
        )
        # this is ignored by `flash_mla_with_kvcache` if indices not None
        self.dummy_block_table = torch.empty(
            (1, 1), dtype=torch.int32, device=self.device
        )

        # Equation taken from FlashMLA/csrc/pybind.cpp
        h_q, h_k = self.num_heads, 1
        s_q = 1  # inversely proportional to s_q, so s_q = 1 is the largest
        max_num_sm_parts = int(
            max((sm_count // 2) / h_k // (cdiv(h_q // h_k, 2 * 64) * s_q), 1)
        )
        if current_platform.is_device_capability(100):
            max_num_sm_parts *= 2
        self.tile_scheduler_metadata_buffer = torch.empty(
            # TileSchedulerMetaDataSize = 8
            # see: FlashMLA/csrc/params.h
            (max_num_sm_parts, 8),
            dtype=torch.int32,
            device=device,
        )
        self.num_splits_buffer = torch.empty(
            # We pack all the tokens into one batch for sparse attention.
            # Otherwise, we can exceed the sm of `get_mla_metadata`.
            (2,),
            dtype=torch.int32,
            device=device,
        )
        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashMLASparseMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        # Zero-fill for cudagraphs
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )
        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]

        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold or 1
            )
        )

        num_prefill_reqs = num_prefills
        num_decode_reqs = num_decodes
        prefill_token_count = num_prefill_tokens
        decode_token_count = num_decode_tokens

        assert num_prefill_reqs + num_decode_reqs == common_attn_metadata.num_reqs
        assert prefill_token_count + decode_token_count == num_tokens

        fp8_extra_metadata = None
        if self.use_fp8_kv_cache:
            tile_scheduler_metadata, num_splits = get_mla_metadata(
                cache_seqlens=self.topk_tokens_tensor,
                num_q_tokens_per_head_k=num_tokens * self.num_heads,
                topk=self.topk_tokens,
                num_heads_q=self.num_heads,
                num_heads_k=1,
                is_fp8_kvcache=True,
            )

            num_sm_parts = tile_scheduler_metadata.size(0)
            # Copy to persistent buffer for full-CG support
            tile_scheduler_metadata_buffer = self.tile_scheduler_metadata_buffer[
                :num_sm_parts
            ]
            tile_scheduler_metadata_buffer.copy_(tile_scheduler_metadata)
            self.num_splits_buffer.copy_(num_splits)

            fp8_extra_metadata = FlashMLASparseMetadata.FP8KernelMetadata(
                scheduler_metadata=tile_scheduler_metadata_buffer,
                num_splits=self.num_splits_buffer,
                # cache_lens and block_table are basically unused in sparse case
                # but the decode kernel will treat -1 and indices >= cache_lens
                # as invalid so we make sure cache_lens is large enough to not
                # accidentally mark indices invalid, we will use -1 exclusively
                # to mark invalid indices
                cache_lens=self.max_model_len_tensor,
                dummy_block_table=self.dummy_block_table,
            )

        metadata = FlashMLASparseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
            num_prefill_tokens=prefill_token_count,
            num_decode_tokens=decode_token_count,
            fp8_extra_metadata=fp8_extra_metadata,
        )
        return metadata


class FlashMLASparseImpl(MLACommonBaseImpl[FlashMLASparseMetadata]):
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
        # MLA Specific Arguments
        topk_indice_buffer: torch.Tensor | None = None,
        indexer: Optional["Indexer"] = None,
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )
        self.softmax_scale = scale
        assert indexer is not None
        self.topk_indices_buffer = indexer.topk_indices_buffer
        self.padding = 128 if current_platform.is_device_capability(100) else 64
        self.workspace_reservation_done = False

        # Workspace specs for prefill buffers
        # Note: these are also registered in FlashMLASparseMetadataBuilder.__init__()
        # before memory profiling to ensure proper memory accounting
        self.prefill_bf16_workspace_spec = PerKVCacheTokenWorkspace(
            shape=(head_size,),
            dtype=torch.bfloat16,
            name="FlashMLASparseImpl.prefill_bf16_workspace",
        )
        self.prefill_seen_workspace_spec = PerKVCacheTokenWorkspace(
            shape=(), dtype=torch.int32, name="FlashMLASparseImpl.prefill_seen_buffer"
        )

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # NOTE(Chen): kernel requires num_local_head to be a multiple of
        # 64 on hopper and 128 on blackwell
        if self.num_heads % self.padding != 0:
            assert self.padding % self.num_heads == 0
            logger.warning_once(
                f"padding num_heads to {self.padding} \
                    due to sparse attn kernel requirement"
            )
            q_padded = q.new_empty((q.shape[0], self.padding, q.shape[2]))
            q_padded[:, : self.num_heads, :] = q
            q = q_padded

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = flash_mla_sparse_prefill(
            q, kv_c_and_k_pe_cache, topk_indices, self.softmax_scale
        )[0]
        output = output[:, : self.num_heads, :]
        return output

    def _forward_fp8_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> torch.Tensor:
        assert attn_metadata.fp8_extra_metadata is not None
        extra_metadata = attn_metadata.fp8_extra_metadata

        _attn_out, _ = flash_mla_with_kvcache(
            q=q.unsqueeze(0),  # unsqueeze to add batch_dim
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=extra_metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=extra_metadata.cache_lens,
            tile_scheduler_metadata=extra_metadata.scheduler_metadata,
            num_splits=extra_metadata.num_splits,
            is_fp8_kvcache=True,
            indices=topk_indices.unsqueeze(0),  # unsqueeze to add batch_dim
            softmax_scale=self.softmax_scale,
        )

        return _attn_out

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode

        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for MLACommonImpl"
            )

        if attn_metadata is None:
            # Profiling run, make sure to reserve the workspace specs if
            # they haven't been reserved yet
            # These need to be registered here (not in FlashMLASparseImpl.__init__)
            # so they're accounted for during memory profiling
            use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"
            if use_fp8_cache and not self.workspace_reservation_done:
                self.workspace_reservation_done = True
                current_workspace_manager().reserve_simultaneous(
                    self.prefill_seen_workspace_spec,
                    self.prefill_bf16_workspace_spec,
                )

            # Dummy run - no need to allocate buffers
            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs

        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        ql_nope = ql_nope.transpose(0, 1)

        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"

        # These buffers are used to track which tokens are attended to by
        # the prefill tokens so we can upconvert (from fp8 to bf16) only
        # these tokens inline within the fused kernel
        prefill_token_mask: torch.Tensor | None = None
        prefill_seen: torch.Tensor | None = None
        prefill_bf16_workspace: torch.Tensor | None = None

        if use_fp8_cache and attn_metadata.num_prefill_tokens > 0:
            if kv_cache.numel() == 0:
                raise RuntimeError(
                    "Expected non-empty kv_cache for fp8_ds_mla prefill handling"
                )

            # Scheduler places decode tokens first, so the remaining suffix maps
            # to prefills for the masked unique-index gather.
            prefill_token_mask = torch.zeros(
                num_actual_toks, dtype=torch.int32, device=topk_indices.device
            )
            prefill_token_mask[attn_metadata.num_decode_tokens :] = True

            (
                prefill_seen,
                prefill_bf16_workspace,
            ) = current_workspace_manager().get_simultaneous(
                self.prefill_seen_workspace_spec,
                self.prefill_bf16_workspace_spec,
            )

            prefill_seen.zero_()

        q = torch.cat([ql_nope, q_pe], dim=-1)

        # CRITICAL: Write to KV cache FIRST before upconverting
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=layer._k_scale,
            )

        # Fused kernel: converts indices and upconverts unique prefill tokens inline
        # MUST happen AFTER concat_and_cache_mla so the FP8 data exists in the cache
        topk_indices_global = ops.convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            block_size=attn_metadata.block_size,
            prefill_token_mask=prefill_token_mask,
            prefill_seen=prefill_seen,
            prefill_bf16_workspace=prefill_bf16_workspace,
            kv_cache=kv_cache if use_fp8_cache else None,
        )

        if not use_fp8_cache:
            attn_out = self._forward_bf16_kv(
                q, kv_cache, topk_indices_global, attn_metadata
            )
        else:
            num_decode_tokens = attn_metadata.num_decode_tokens
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            attn_out = q.new_empty(
                (num_actual_toks, self.num_heads, self.kv_lora_rank),
                dtype=q.dtype,
                device=q.device,
            )

            if num_decode_tokens > 0:
                attn_out[:num_decode_tokens] = self._forward_fp8_kv(
                    q[:num_decode_tokens],
                    kv_cache,
                    topk_indices_global[:num_decode_tokens],
                    attn_metadata,
                )

            if num_prefill_tokens > 0:
                prefill_start = num_decode_tokens
                topk_prefill = topk_indices_global[prefill_start:]
                # Upconversion was done inline by the fused kernel
                # Use bf16 workspace for prefill tokens
                assert prefill_bf16_workspace is not None
                attn_out[prefill_start:] = self._forward_bf16_kv(
                    q[prefill_start:],
                    prefill_bf16_workspace,
                    topk_prefill,
                    attn_metadata,
                )

        self._v_up_proj(attn_out, out=output[:num_actual_toks])
        return output
