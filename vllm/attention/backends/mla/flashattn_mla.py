# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.mla.common import (MLACommonBackend,
                                                MLACommonImpl,
                                                MLACommonMetadata,
                                                MLACommonMetadataBuilder,
                                                MLACommonState)
from vllm.attention.backends.utils import get_mla_dims
from vllm.logger import init_logger
from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata
from vllm.vllm_flash_attn.fa_utils import (flash_attn_supports_mla,
                                           get_flash_attn_version)

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class FlashAttnMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHATTN_MLA"

    @staticmethod
    def get_metadata_cls() -> type["FlashAttnMLAMetadata"]:
        return FlashAttnMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashAttnMLAMetadataBuilder"]:
        return FlashAttnMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashAttnMLAImpl"]:
        return FlashAttnMLAImpl

    @staticmethod
    def get_state_cls() -> type["FlashAttnMLAState"]:
        return FlashAttnMLAState


@dataclass
class FlashAttnMLAMetadata(MLACommonMetadata):
    decode_scheduler_metadata: Optional[torch.Tensor] = None

    @property
    def decode_metadata(self):
        decode_metadata = super().decode_metadata
        # TODO: cache assignment?
        if decode_metadata is not None:
            decode_metadata.decode_scheduler_metadata=\
                self.decode_scheduler_metadata
        return decode_metadata

    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        raise NotImplementedError(
            "advance_step is not implemented for FlashAttnMLA")


class FlashAttnMLAMetadataBuilder(
        MLACommonMetadataBuilder[FlashAttnMLAMetadata]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_heads_q = self.runner.model_config.get_num_attention_heads(
            self.runner.parallel_config)
        self.fa_aot_schedule = (get_flash_attn_version() == 3)
        self.mla_dims = get_mla_dims(self.runner.model_config)
        self.page_size = self.runner.block_size

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        m = super().build(seq_lens, query_lens, cuda_graph_pad_size,
                          batch_size)

        decode_cu_seqlens_q = m.query_start_loc[
            m.num_prefills:] - m.query_start_loc[m.num_prefills]

        if m.num_decode_tokens > 0 and self.fa_aot_schedule:
            m.decode_scheduler_metadata = get_scheduler_metadata(
                batch_size=batch_size,
                max_seqlen_q=m.max_decode_query_len,
                max_seqlen_k=m.max_decode_seq_len,
                cache_seqlens=m.seq_start_loc[m.num_prefills:],
                num_heads_q=self.num_heads_q,
                num_heads_kv=1,
                headdim=self.mla_dims.qk_rope_head_dim,
                headdim_v=self.mla_dims.kv_lora_rank,
                page_size=self.page_size,
                cu_seqlens_q=decode_cu_seqlens_q,
                causal=True)
        return m


class FlashAttnMLAState(MLACommonState[FlashAttnMLAMetadata]):

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

        self.fa_aot_schedule = (get_flash_attn_version() == 3)
        self.num_heads_q = self.runner.model_config.get_num_attention_heads(
            self.runner.parallel_config)
        self.mla_dims = get_mla_dims(self.runner.model_config)
        self.page_size = self.runner.block_size

    def _dummy_scheduler_metadata(self, max_batch_size: int):
        if self.fa_aot_schedule:
            return get_scheduler_metadata(
                batch_size=max_batch_size,
                max_seqlen_q=1,
                max_seqlen_k=1,
                cache_seqlens=torch.ones(max_batch_size,
                                         dtype=torch.int32,
                                         device=self.runner.device),
                num_heads_q=self.num_heads_q,
                num_heads_kv=1,
                headdim=self.mla_dims.qk_rope_head_dim,
                headdim_v=self.mla_dims.kv_lora_rank,
                page_size=self.page_size,
                cu_seqlens_q=torch.arange(max_batch_size + 1,
                                          dtype=torch.int32,
                                          device=self.runner.device),
                causal=True)
        return None

    @contextmanager
    def graph_capture(self, max_batch_size: int):
        # Run a dummy `get_scheduler_metadata` so we can get the right shapes
        self._graph_scheduler_metadata = self._dummy_scheduler_metadata(
            max_batch_size)
        self._graph_query_start_loc = torch.arange(max_batch_size + 1,
                                                   dtype=torch.int32,
                                                   device=self.runner.device)

        with super().graph_capture(max_batch_size):
            yield

        del self._graph_scheduler_metadata

    def graph_capture_get_metadata_for_batch(
            self, batch_size: int, is_encoder_decoder_model: bool = False):
        metadata = super().graph_capture_get_metadata_for_batch(
            batch_size, is_encoder_decoder_model)
        assert metadata.num_decode_tokens > 0

        decoder_scheduler_metadata = self._dummy_scheduler_metadata(batch_size)

        metadata_size = decoder_scheduler_metadata.numel()
        self._graph_scheduler_metadata[:metadata_size].copy_(
            decoder_scheduler_metadata)

        metadata.decode_scheduler_metadata=\
            self._graph_scheduler_metadata[:metadata_size]
        metadata.query_start_loc=\
            self._graph_query_start_loc[:batch_size + 1]

        return metadata

    def get_graph_input_buffers(self,
                                attn_metadata,
                                is_encoder_decoder_model: bool = False):
        input_buffers = super().get_graph_input_buffers(
            attn_metadata, is_encoder_decoder_model)
        input_buffers["decode_scheduler_metadata"] = \
                attn_metadata.decode_metadata.decode_scheduler_metadata
        input_buffers["query_start_loc"] = \
                attn_metadata.decode_metadata.query_start_loc

        return input_buffers

    def prepare_graph_input_buffers(self,
                                    input_buffers,
                                    attn_metadata,
                                    is_encoder_decoder_model: bool = False):
        super().prepare_graph_input_buffers(input_buffers, attn_metadata,
                                            is_encoder_decoder_model)

        input_buffers["decode_scheduler_metadata"].copy_(
            attn_metadata.decode_metadata.decode_scheduler_metadata)
        input_buffers["query_start_loc"].copy_(
            attn_metadata.decode_metadata.query_start_loc)


class FlashAttnMLAImpl(MLACommonImpl[MLACommonMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[dict[str, Any]],
            logits_soft_cap: Optional[float],
            attn_type: str,
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         **mla_args)

        assert flash_attn_supports_mla(), \
            "FlashAttnMLA is not supported on this device"

        unsupported_features = [
            alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashAttnMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashMLAImpl")

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 FlashMLA not yet supported")

        decode_meta = attn_metadata.decode_metadata
        assert decode_meta is not None

        kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]
        kv_pe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank:]

        o = flash_attn_varlen_func(
            q=q_pe,
            k=kv_pe_cache.unsqueeze(-2),  # Add head dim of 1
            v=kv_c_cache.unsqueeze(-2),  # Add head dim of 1
            q_v=q_nope,
            max_seqlen_q=decode_meta.max_decode_query_len,
            cu_seqlens_q=decode_meta.query_start_loc,
            max_seqlen_k=decode_meta.max_decode_seq_len,
            seqused_k=decode_meta.seq_lens_tensor,
            block_table=decode_meta.block_tables,
            softmax_scale=self.scale,
            causal=True,
            fa_version=3,  # only version 3 is supported
            scheduler_metadata=decode_meta.decode_scheduler_metadata,
        )

        return self._v_up_proj_and_o_proj(o)
