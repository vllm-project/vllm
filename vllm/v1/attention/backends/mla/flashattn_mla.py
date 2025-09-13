# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar, Optional, Union

import torch

from vllm.attention.backends.abstract import (AttentionLayer, AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.utils.fa_utils import (flash_attn_supports_mla,
                                           get_flash_attn_version)
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonDecodeMetadata,
                                                   MLACommonImpl,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder)
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata

logger = init_logger(__name__)

# NOTE(matt): This is an arbitrary number, copied from
# woosuk's implementation in standard FlashAttention backend
_DEFAULT_MAX_NUM_SPLITS_FOR_CUDA_GRAPH = 16


class FlashAttnMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA"

    @staticmethod
    def get_metadata_cls() -> type["FlashAttnMLAMetadata"]:
        return FlashAttnMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashAttnMLAMetadataBuilder"]:
        return FlashAttnMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashAttnMLAImpl"]:
        return FlashAttnMLAImpl


@dataclass
class FlashAttnMLADecodeMetadata(MLACommonDecodeMetadata):
    query_start_loc: torch.Tensor
    max_query_len: int
    max_seq_len: int
    scheduler_metadata: Optional[torch.Tensor] = None
    max_num_splits: int = 0


@dataclass
class FlashAttnMLAMetadata(MLACommonMetadata[FlashAttnMLADecodeMetadata]):
    pass


class FlashAttnMLAMetadataBuilder(
        MLACommonMetadataBuilder[FlashAttnMLAMetadata]):
    cudagraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: ClassVar[int] = 512

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device,
                         FlashAttnMLAMetadata)
        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.fa_aot_schedule = (get_flash_attn_version() == 3)

        self.use_full_cuda_graph = \
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()

        if self.use_full_cuda_graph and self.fa_aot_schedule:
            self.max_cudagraph_size = self.compilation_config.max_capture_size

            if self.max_cudagraph_size > 992:
                # This condition derives from FA3's internal heuristic.
                # TODO(woosuk): Support larger cudagraph sizes.
                raise ValueError(
                    "Capture size larger than 992 is not supported for "
                    "full cuda graph.")

            self.scheduler_metadata = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = _DEFAULT_MAX_NUM_SPLITS_FOR_CUDA_GRAPH

        # TODO(lucas): Until we add support for the DCP custom masking we need
        #   to restrict decodes to q_len == 1 when DCP is enabled.
        self.__class__.reorder_batch_threshold = 1 \
            if get_dcp_group().world_size > 1 else self.reorder_batch_threshold

    def _schedule_decode(self, num_reqs, cu_query_lens, max_query_len, seqlens,
                         max_seq_len, causal):
        if self.fa_aot_schedule:
            return get_scheduler_metadata(
                batch_size=num_reqs,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_seq_len,
                num_heads_q=self.num_heads,
                num_heads_kv=1,
                headdim=self.mla_dims.qk_rope_head_dim,
                cache_seqlens=seqlens,
                qkv_dtype=self.kv_cache_spec.dtype,
                headdim_v=self.mla_dims.kv_lora_rank,
                page_size=self.page_size,
                cu_seqlens_q=cu_query_lens,
                causal=causal,
                num_splits=self.max_num_splits,
            )
        return None

    def _build_decode(self, block_table_tensor: torch.Tensor,
                      seq_lens_cpu: torch.Tensor,
                      seq_lens_device: torch.Tensor,
                      query_start_loc_cpu: torch.Tensor,
                      query_start_loc_device: torch.Tensor,
                      num_decode_tokens: int) -> FlashAttnMLADecodeMetadata:
        query_lens_cpu = (query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])
        max_query_len = query_lens_cpu.max().item()
        max_seq_len = seq_lens_cpu.max().item()

        scheduler_metadata = self._schedule_decode(
            num_reqs=seq_lens_cpu.numel(),
            cu_query_lens=query_start_loc_device,
            max_query_len=max_query_len,
            seqlens=seq_lens_device,
            max_seq_len=max_seq_len,
            causal=True,
        )

        # For FA3 + full cudagraph
        max_num_splits = 0
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            # Ensure the persistent buffer is large enough
            assert n <= self.scheduler_metadata.shape[0], \
                f"Scheduler metadata size {n} exceeds buffer size " + \
                f"{self.scheduler_metadata.shape[0]}"
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

            if num_decode_tokens <= self.max_cudagraph_size:
                # NOTE(woosuk): Setting num_splits > 1 may increase the memory
                # usage, because the intermediate buffers of size [num_splits,
                # num_heads, num_tokens, head_size] are allocated. Therefore,
                # we only set num_splits when using cuda graphs.
                max_num_splits = self.max_num_splits

        return FlashAttnMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            query_start_loc=query_start_loc_device,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            scheduler_metadata=scheduler_metadata,
            max_num_splits=max_num_splits,
        )


class FlashAttnMLAImpl(MLACommonImpl[FlashAttnMLAMetadata]):
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

        assert flash_attn_supports_mla(), \
            "FlashAttnMLA is not supported on this device"

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashAttnMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashAttnMLAImpl")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlashAttnMLA V1 with FP8 KV cache not yet supported")

    def _forward_decode(
        self,
        q: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q_nope, q_pe = q
        else:
            q_nope, q_pe = torch.split(
                q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "FP8 FlashAttention MLA not yet supported")

        kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]
        k_pe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank:]

        # NOTE(matt): During CUDA graph capture, max_query_len can be 0, but the
        # kernel uses this to calculate grid dimensions. Ensure it's at least 1
        # to prevent invalid grid configuration during graph capture.
        max_seqlen_q = max(attn_metadata.decode.max_query_len, 1)

        attn_out = flash_attn_varlen_func(
            q=q_pe,
            k=k_pe_cache.unsqueeze(-2),  # Add head dim of 1
            v=kv_c_cache.unsqueeze(-2),  # Add head dim of 1
            q_v=q_nope,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=attn_metadata.decode.query_start_loc,
            max_seqlen_k=attn_metadata.decode.max_seq_len,
            seqused_k=attn_metadata.decode.seq_lens,
            block_table=attn_metadata.decode.block_table,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=self.need_to_return_lse_for_decode,
            fa_version=3,  # only version 3 is supported
            scheduler_metadata=attn_metadata.decode.scheduler_metadata,
            num_splits=attn_metadata.decode.max_num_splits,
        )

        if self.need_to_return_lse_for_decode:
            o, lse = attn_out
            # FA returns LSE in shape [ H, B ] but DCP wants [ B, H ]
            return o, lse.transpose(0, 1)  # [ H, B ] -> [ B, H ]
        else:
            o = attn_out
            return o, None
