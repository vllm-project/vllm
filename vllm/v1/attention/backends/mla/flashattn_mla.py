# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_mla,
    get_flash_attn_version,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
    flash_attn_varlen_func,
    get_scheduler_metadata,
)

logger = init_logger(__name__)


class FlashAttnMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA"

    @staticmethod
    def get_builder_cls() -> type["FlashAttnMLAMetadataBuilder"]:
        return FlashAttnMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashAttnMLAImpl"]:
        return FlashAttnMLAImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 9

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if not flash_attn_supports_mla():
            return "FlashAttention MLA not supported on this device"
        return None


@dataclass
class FlashAttnMLADecodeMetadata(MLACommonDecodeMetadata):
    query_start_loc: torch.Tensor
    max_query_len: int
    max_seq_len: int
    scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0


@dataclass
class FlashAttnMLAMetadata(MLACommonMetadata[FlashAttnMLADecodeMetadata]):
    pass


class FlashAttnMLAMetadataBuilder(MLACommonMetadataBuilder[FlashAttnMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN
    reorder_batch_threshold: int = 512  # process small prefills with decode pathway

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            FlashAttnMLAMetadata,
            supports_dcp_with_varlen=(interleave_size == 1),
        )
        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.fa_aot_schedule = get_flash_attn_version() == 3

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.fa_aot_schedule:
            self.scheduler_metadata = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = (
                vllm_config.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        if vllm_is_batch_invariant():
            self.max_num_splits = 1

    def _schedule_decode(
        self,
        num_reqs,
        cu_query_lens,
        max_query_len,
        seqlens,
        max_seq_len,
        causal,
        max_num_splits,
    ):
        if self.fa_aot_schedule:
            return get_scheduler_metadata(
                batch_size=num_reqs,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_seq_len,
                num_heads_q=self.num_heads * self.dcp_world_size,
                num_heads_kv=1,
                headdim=self.mla_dims.qk_rope_head_dim,
                cache_seqlens=seqlens,
                qkv_dtype=self.kv_cache_spec.dtype,
                headdim_v=self.mla_dims.kv_lora_rank,
                page_size=self.page_size,
                cu_seqlens_q=cu_query_lens,
                causal=causal,
                num_splits=max_num_splits,
            )
        return None

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> FlashAttnMLADecodeMetadata:
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_query_len = query_lens_cpu.max().item()

        # For Flash Attention MLA + full cudagraph
        max_num_splits = 0
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_decode_tokens <= self.max_cudagraph_size
        ):
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        if vllm_is_batch_invariant():
            max_num_splits = 1

        scheduler_metadata = self._schedule_decode(
            num_reqs=seq_lens_device.shape[0],
            cu_query_lens=query_start_loc_device,
            max_query_len=max_query_len,
            seqlens=seq_lens_device,
            max_seq_len=max_seq_len,
            causal=True,
            max_num_splits=max_num_splits,
        )

        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            # Ensure the persistent buffer is large enough
            assert n <= self.scheduler_metadata.shape[0], (
                f"Scheduler metadata size {n} exceeds buffer size "
                + f"{self.scheduler_metadata.shape[0]}"
            )
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        metadata = FlashAttnMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            query_start_loc=query_start_loc_device,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            scheduler_metadata=scheduler_metadata,
            max_num_splits=max_num_splits,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )
        return metadata


class FlashAttnMLAImpl(MLACommonImpl[FlashAttnMLAMetadata]):
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

        assert flash_attn_supports_mla(), "FlashAttnMLA is not supported on this device"

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashAttnMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashAttnMLAImpl"
            )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlashAttnMLA V1 with FP8 KV cache not yet supported"
            )

    def _forward_decode(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q_nope, q_pe = q
        else:
            q_nope, q_pe = torch.split(
                q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 FlashAttention MLA not yet supported")

        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        k_pe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank :]

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
            cp_world_size=self.dcp_world_size,
            cp_rank=self.dcp_rank,
            cp_tot_seqused_k=attn_metadata.decode.dcp_tot_seq_lens,
        )

        if self.need_to_return_lse_for_decode:
            o, lse = attn_out
            # FA returns LSE in shape [ H, B ] but DCP wants [ B, H ]
            return o, lse.transpose(0, 1)  # [ H, B ] -> [ B, H ]
        else:
            o = attn_out
            return o, None
