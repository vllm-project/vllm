# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.attention.backends.abstract import AttentionLayer, AttentionType, MultipleOf
from vllm.attention.ops.flashmla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
    is_flashmla_dense_supported,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    reshape_attn_output_for_spec_decode,
    reshape_query_for_spec_decode,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class FlashMLABackend(MLACommonBackend):
    @staticmethod
    def get_name() -> str:
        return "FLASHMLA"

    @staticmethod
    def get_metadata_cls() -> type["FlashMLAMetadata"]:
        return FlashMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashMLAMetadataBuilder"]:
        return FlashMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLAImpl"]:
        return FlashMLAImpl

    @staticmethod
    def get_supported_kernel_block_size() -> list[int | MultipleOf]:
        return [64]


@dataclass
class FlashMLADecodeMetadata(MLACommonDecodeMetadata):
    tile_scheduler_metadata: torch.Tensor
    num_splits: torch.Tensor


@dataclass
class FlashMLAMetadata(MLACommonMetadata[FlashMLADecodeMetadata]):
    pass


class FlashMLAMetadataBuilder(MLACommonMetadataBuilder[FlashMLAMetadata]):
    cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    reorder_batch_threshold: int = 128  # process small prefills with decode pathway
    # ^ TODO(matt): tune this

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, FlashMLAMetadata
        )

        self.num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )

        self.cg_buf_tile_scheduler_metadata = None
        self.cg_buf_num_splits = None
        self.is_fp8_kvcache = vllm_config.cache_config.cache_dtype.startswith("fp8")

        device_properties = torch.cuda.get_device_properties(self.device)
        num_sms = device_properties.multi_processor_count

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.cg_buf_tile_scheduler_metadata = torch.zeros(
                # Upper bound on size (<= #SMs, TileSchedulerMetaDataSize)
                # TileSchedulerMetaDataSize = 8
                (num_sms, 8),
                device=self.device,
                dtype=torch.int32,
            )
            self.cg_buf_num_splits = torch.empty(
                (vllm_config.scheduler_config.max_num_seqs + 1),
                device=self.device,
                dtype=torch.int32,
            )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        seq_lens_device: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> FlashMLADecodeMetadata:
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # we use the max but all should be the same due to uniform length requirement
        max_query_len = query_lens_cpu.max().item()
        num_q_tokens_per_head_k = max_query_len * self.num_q_heads // 1
        tile_scheduler_metadata, num_splits = get_mla_metadata(
            seq_lens_device,
            num_q_tokens_per_head_k,
            1,  # MQA for the decode path
            is_fp8_kvcache=self.is_fp8_kvcache,
        )

        # TODO: we can disambiguate between decode and mixed-prefill decode here
        # so we can only use the persistent buffer if a cudagraph is actually
        # being used.
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            assert self.cg_buf_tile_scheduler_metadata is not None
            assert self.cg_buf_num_splits is not None

            sm_parts = tile_scheduler_metadata.size(0)
            # Metadata per-SM, upper bound on size (<= #SMs, TileMetadataSize)
            assert sm_parts <= self.cg_buf_tile_scheduler_metadata.size(0)
            tile_scheduler_metadata_view = self.cg_buf_tile_scheduler_metadata[
                :sm_parts
            ]
            tile_scheduler_metadata_view.copy_(tile_scheduler_metadata)
            tile_scheduler_metadata = tile_scheduler_metadata_view

            # Num splits is per-batch, varying size (batch_size,)
            n = num_splits.size(0)
            # make sure static buffer is large enough
            assert n <= self.cg_buf_num_splits.size(0)
            num_splits_view = self.cg_buf_num_splits[:n]
            num_splits_view.copy_(num_splits)
            # Num splits needs to monotonically increasing
            # (with: https://github.com/vllm-project/FlashMLA/pull/3, otherwise
            #  it needs to monotonically increasing by 1)
            self.cg_buf_num_splits[n:].fill_(num_splits[-1])
            num_splits = num_splits_view

        return FlashMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )


class FlashMLAImpl(MLACommonImpl[FlashMLAMetadata]):
    can_return_lse_for_decode: bool = True

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

        is_supported, reason = is_flashmla_dense_supported()
        assert is_supported, reason

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashMLAImpl"
            )

    def _forward_decode(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # TODO: (zyongye) decode function for mla here
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        # mypy assertion: q is now always a tensor
        assert isinstance(q, torch.Tensor)

        num_decodes = attn_metadata.num_decodes
        q = reshape_query_for_spec_decode(q, num_decodes)

        tile_scheduler_metadata = attn_metadata.decode.tile_scheduler_metadata
        num_splits = attn_metadata.decode.num_splits
        if vllm_is_batch_invariant():
            device = q.device
            dtype = torch.int32

            B = q.shape[0]
            # block_table shape: [batch_size, max_num_blocks_per_seq]
            # The number of blocks per sequence is in the second dimension
            topk = attn_metadata.decode.block_table.shape[-1]
            B_TOPK = 64
            assert topk % B_TOPK == 0, f"topk ({topk}) must be divisible by {B_TOPK}"
            end_block_idx = topk // B_TOPK

            # Single partition => num_sm_parts = 1
            # TileSchedulerMetaDataSize = 8, layout:
            # [begin_idx, begin_block_idx, end_idx, end_block_idx,
            #  begin_n_split_idx, _, _, _]
            tile_scheduler_metadata = torch.zeros((1, 8), dtype=dtype, device=device)
            tile_scheduler_metadata[0, 0] = 0  # begin_idx
            tile_scheduler_metadata[0, 1] = 0  # sched_begin_block_idx
            tile_scheduler_metadata[0, 2] = B - 1  # end_idx
            tile_scheduler_metadata[0, 3] = end_block_idx
            tile_scheduler_metadata[0, 4] = 0  # begin_n_split_idx
            # fields [5..7] stay 0

            # Non-split path ignores num_splits, but the API requires it:
            # zeros of length B+1
            num_splits = torch.zeros((B + 1,), dtype=dtype, device=device)

        o, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # Add head dim of 1
            block_table=attn_metadata.decode.block_table,
            cache_seqlens=attn_metadata.decode.seq_lens,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=self.scale,
            causal=True,
            descale_q=layer._q_scale.reshape(1),
            descale_k=layer._k_scale.reshape(1),
        )

        o = reshape_attn_output_for_spec_decode(o)

        return o, lse
