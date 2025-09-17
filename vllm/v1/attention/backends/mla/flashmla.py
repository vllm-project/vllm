# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar, Optional, Union

import torch

from vllm.attention.backends.abstract import AttentionLayer, AttentionType
from vllm.attention.ops.flashmla import (flash_mla_with_kvcache,
                                         get_mla_metadata,
                                         is_flashmla_supported)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonDecodeMetadata,
                                                   MLACommonImpl,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder)
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class FlashMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_VLLM_V1"

    @staticmethod
    def get_metadata_cls() -> type["FlashMLAMetadata"]:
        return FlashMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashMLAMetadataBuilder"]:
        return FlashMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLAImpl"]:
        return FlashMLAImpl


@dataclass
class FlashMLADecodeMetadata(MLACommonDecodeMetadata):
    tile_scheduler_metadata: torch.Tensor
    num_splits: torch.Tensor


@dataclass
class FlashMLAMetadata(MLACommonMetadata[FlashMLADecodeMetadata]):
    pass


class FlashMLAMetadataBuilder(MLACommonMetadataBuilder[FlashMLAMetadata]):
    cudagraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device,
                         FlashMLAMetadata)

        self.num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config)

        self.cg_buf_tile_scheduler_metadata = None
        self.cg_buf_num_splits = None

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
                dtype=torch.int32)

    def _build_decode(self, block_table_tensor: torch.Tensor,
                      seq_lens_cpu: torch.Tensor,
                      seq_lens_device: torch.Tensor,
                      query_start_loc_cpu: torch.Tensor,
                      query_start_loc_device: torch.Tensor,
                      num_decode_tokens: int) -> FlashMLADecodeMetadata:
        tile_scheduler_metadata, num_splits = \
            get_mla_metadata(
            seq_lens_device,
            self.num_q_heads,
            1, # MQA for the decode path
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
            tile_scheduler_metadata_view = \
                self.cg_buf_tile_scheduler_metadata[:sm_parts]
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
        )


class FlashMLAImpl(MLACommonImpl[FlashMLAMetadata]):

    can_return_lse_for_decode: bool = True

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            logits_soft_cap: Optional[float],
            attn_type: str,
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)

        is_supported, reason = is_flashmla_supported()
        assert is_supported, reason

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashMLAImpl")

    def _forward_decode(
        self,
        q: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        o, lse = flash_mla_with_kvcache(
            q=q.unsqueeze(1),  # Add seqlen dim of 1 (decode)
            k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # Add head dim of 1
            block_table=attn_metadata.decode.block_table,
            cache_seqlens=attn_metadata.decode.seq_lens,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=attn_metadata.decode.
            tile_scheduler_metadata,
            num_splits=attn_metadata.decode.num_splits,
            softmax_scale=self.scale,
            causal=True,
            descale_q=layer._q_scale.reshape(1),
            descale_k=layer._k_scale.reshape(1),
        )

        return o, lse
