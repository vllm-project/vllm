# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.backends.mla.common import (MLACommonBackend,
                                                MLACommonImpl,
                                                MLACommonMetadata,
                                                MLACommonMetadataBuilder,
                                                MLACommonState)
from vllm.attention.ops.flashmla import (flash_mla_with_kvcache,
                                         get_mla_metadata,
                                         is_flashmla_supported)

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata


class FlashMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA"

    @staticmethod
    def get_impl_cls() -> Type["FlashMLAImpl"]:
        return FlashMLAImpl

    @staticmethod
    def get_metadata_cls() -> Type["FlashMLAMetadata"]:
        return FlashMLAMetadata

    @staticmethod
    def get_builder_cls() -> Type["FlashMLAMetadataBuilder"]:
        return FlashMLAMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["FlashMLAState"]:
        return FlashMLAState


@dataclass
class FlashMLAMetadata(MLACommonMetadata):
    decode_tile_scheduler_metadata: Optional[Tuple[torch.Tensor,
                                                   torch.Tensor]] = None
    decode_num_splits: Optional[torch.Tensor] = None

    @property
    def decode_metadata(self):
        decode_metadata = super().decode_metadata
        # TODO: cache assignment?
        if decode_metadata is not None:
            decode_metadata.decode_tile_scheduler_metadata=\
                self.decode_tile_scheduler_metadata
            decode_metadata.decode_num_splits=\
                self.decode_num_splits
        return decode_metadata

    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        raise NotImplementedError(
            "advance_step is not implemented for FlashMLA")


class FlashMLAMetadataBuilder(MLACommonMetadataBuilder[FlashMLAMetadata]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_q_heads = self.runner.model_config.get_num_attention_heads(
            self.runner.parallel_config)

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        m = super().build(seq_lens, query_lens, cuda_graph_pad_size,
                          batch_size)

        if m.num_decode_tokens > 0:
            m.decode_tile_scheduler_metadata, m.decode_num_splits = \
                get_mla_metadata(
                m.seq_lens_tensor[m.num_prefills:],
                self.num_q_heads,
                1, # MQA for the decode path
            )

        return m


class FlashMLAState(MLACommonState[FlashMLAMetadata]):

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

        self.num_q_heads = self.runner.model_config.get_num_attention_heads(
            self.runner.parallel_config)

    @contextmanager
    def graph_capture(self, max_batch_size: int):
        # Run a dummy `get_mla_metadata` so we can get the right shapes
        self._graph_decoder_tile_scheduler_metadata, \
            self._graph_decode_num_splits = get_mla_metadata(
            torch.ones(
                max_batch_size, dtype=torch.int32, device=self.runner.device),
            self.num_q_heads,
            1, # MQA for the decode path
        )

        with super().graph_capture(max_batch_size):
            yield

        del self._graph_decoder_tile_scheduler_metadata
        del self._graph_decode_num_splits

    def graph_capture_get_metadata_for_batch(
            self, batch_size: int, is_encoder_decoder_model: bool = False):
        metadata = super().graph_capture_get_metadata_for_batch(
            batch_size, is_encoder_decoder_model)
        assert metadata.num_decode_tokens > 0

        decoder_tile_scheduler_metadata, decode_num_splits = get_mla_metadata(
            self._graph_seq_lens[:batch_size],
            self.num_q_heads,
            1,  # MQA for the decode path
        )

        self._graph_decoder_tile_scheduler_metadata.copy_(
            decoder_tile_scheduler_metadata)
        self._graph_decode_num_splits[:batch_size + 1].copy_(decode_num_splits)

        metadata.decode_tile_scheduler_metadata=\
            self._graph_decoder_tile_scheduler_metadata
        metadata.decode_num_splits=\
            self._graph_decode_num_splits[:batch_size + 1]

        return metadata

    def get_graph_input_buffers(self,
                                attn_metadata,
                                is_encoder_decoder_model: bool = False):
        input_buffers = super().get_graph_input_buffers(
            attn_metadata, is_encoder_decoder_model)
        input_buffers["decode_tile_scheduler_metadata"] = \
                attn_metadata.decode_metadata.decode_tile_scheduler_metadata
        input_buffers["decode_num_splits"] = \
                attn_metadata.decode_metadata.decode_num_splits

        return input_buffers

    def prepare_graph_input_buffers(self,
                                    input_buffers,
                                    attn_metadata,
                                    is_encoder_decoder_model: bool = False):
        super().prepare_graph_input_buffers(input_buffers, attn_metadata,
                                            is_encoder_decoder_model)

        input_buffers["decode_tile_scheduler_metadata"].copy_(
            attn_metadata.decode_metadata.decode_tile_scheduler_metadata)
        input_buffers["decode_num_splits"].copy_(
            attn_metadata.decode_metadata.decode_num_splits)


class FlashMLAImpl(MLACommonImpl[FlashMLAMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[List[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[Dict[str, Any]],
            logits_soft_cap: Optional[float],
            attn_type: str,
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         **mla_args)

        assert is_flashmla_supported(), \
            "FlashMLA is not supported on this device"

        unsupported_features = [
            alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashMLAImpl")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlashMLA with FP8 KV cache not yet supported")

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLAMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0

        decode_meta = attn_metadata.decode_metadata
        assert decode_meta is not None

        q = torch.cat([q_nope, q_pe], dim=-1)\
            .unsqueeze(1) # Add seqlen dim of 1 (decode)

        o, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # Add head dim of 1
            block_table=decode_meta.block_tables,
            cache_seqlens=decode_meta.seq_lens_tensor,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=decode_meta.decode_tile_scheduler_metadata,
            num_splits=decode_meta.decode_num_splits,
            softmax_scale=self.scale,
            causal=True,
        )

        return self._v_up_proj(o)
