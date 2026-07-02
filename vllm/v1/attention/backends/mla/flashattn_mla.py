# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from dataclasses import dataclass
from typing import ClassVar

import torch

import vllm.envs as envs
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
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_mla,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
    flash_attn_varlen_func,
    get_scheduler_metadata,
)

logger = init_logger(__name__)


class FlashAttnMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3)
        return (0, 1, 2)

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

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
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
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
            # FA3 scheduler_metadata size: 1 + round_up(batch_size, 4) * 4
            # The +1 is for the tile_count_semaphore (synchronization).
            # The 4 slots per batch element (num_prepare_batch_vectors) are:
            #   prepare_varlen + dynamic_split + sort_batches + head_swizzle
            # See: https://github.com/vllm-project/flash-attention/blob/5824e6e/hopper/flash_api.cpp#L664-L671  # noqa: E501
            max_batch_size = max(
                vllm_config.scheduler_config.max_num_seqs,
                self.max_cudagraph_size or 0,
            )
            self.scheduler_metadata = torch.zeros(
                1 + round_up(max_batch_size, 4) * 4,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = (
                vllm_config.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        if envs.VLLM_BATCH_INVARIANT:
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

        if envs.VLLM_BATCH_INVARIANT:
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
                f"{self.scheduler_metadata.shape[0]}"
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

    def forward_mqa(
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

        if is_quantized_kv_cache(self.kv_cache_dtype):
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


class FlashAttnStaticSinkMLAImpl(MLACommonImpl[FlashAttnMLAMetadata]):
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

        unsupported_features = [alibi_slopes, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashAttnMLAImpl does not support one of the following: "
                "alibi_slopes, logits_soft_cap"
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

        if sliding_window is None:
            self.window_size = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.window_size = (sliding_window - 1, sliding_window - 1)
        else:
            self.window_size = (sliding_window - 1, 0)

        self.sink_k_pe = None
        self.sink_compressed_kv = None
        self.sink_len = 0
        self.sink_k = None
        self.sink_v = None
        self._init_flash_attn_varlen_helper()

    def _init_flash_attn_varlen_helper(self) -> None:
        assert is_flash_attn_varlen_func_available()
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self._vllm_flash_attn_version = get_flash_attn_version(head_size=qk_head_dim)
        self.flash_attn_varlen_func = flash_attn_varlen_func
        if self._vllm_flash_attn_version is not None:
            self.flash_attn_varlen_func = functools.partial(
                flash_attn_varlen_func,
                fa_version=self._vllm_flash_attn_version,
            )
        device_capability = current_platform.get_device_capability()
        self._requires_v_padding = self._vllm_flash_attn_version is None or not (
            (
                self._vllm_flash_attn_version == 3
                and device_capability is not None
                and device_capability[0] == 9
            )
            or self._vllm_flash_attn_version == 4
        )
        self._is_vllm_fa = current_platform.is_cuda() or current_platform.is_xpu()

    def _flash_attn_varlen_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool = False,
        softmax_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        maybe_padded_v = v
        if self._requires_v_padding:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        if self._is_vllm_fa:
            kwargs["return_softmax_lse"] = return_softmax_lse
        else:
            kwargs["return_attn_probs"] = return_softmax_lse
        if envs.VLLM_BATCH_INVARIANT:
            kwargs["num_splits"] = 1

        attn_out = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=maybe_padded_v,
            softmax_scale=softmax_scale,
            **kwargs,
        )

        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        if self._requires_v_padding:
            attn_out = attn_out[..., : v.shape[-1]]

        if return_softmax_lse:
            return attn_out, lse
        return attn_out

    def update_sink_kv(
        self, sink_k_pe: torch.Tensor, sink_compressed_kv: torch.Tensor
    ) -> None:
        self.sink_k_pe = sink_k_pe.unsqueeze(1).clone()
        self.sink_compressed_kv = sink_compressed_kv.clone()
        self.sink_len = sink_k_pe.shape[0]
        self.sink_k, self.sink_v = self._get_prefill_kv(
            self.sink_compressed_kv, self.sink_k_pe
        )

    def _get_prefill_kv(self, kv_c_normed, k_pe):
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = self._concat_k_nope_k_pe(k_nope, k_pe)
        return k, v

    def _ensure_sink_mha_kv(self) -> None:
        if self.sink_k is None or self.sink_v is None:
            assert self.sink_compressed_kv is not None and self.sink_k_pe is not None
            self.sink_k, self.sink_v = self._get_prefill_kv(
                self.sink_compressed_kv, self.sink_k_pe
            )

    def _compute_sink_prefill_attn(
        self,
        q: torch.Tensor,
        attn_metadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention over static sink tokens only (non-causal prefix)."""
        self._ensure_sink_mha_kv()
        prefill = attn_metadata.prefill
        num_prefills = attn_metadata.num_prefills
        assert self.sink_k is not None and self.sink_v is not None
        sink_k = self.sink_k.repeat(num_prefills, 1, 1)
        sink_v = self.sink_v.repeat(num_prefills, 1, 1)
        cu_seqlens_k = torch.arange(
            0,
            num_prefills * self.sink_len + 1,
            self.sink_len,
            device=q.device,
            dtype=torch.int32,
        )
        sink_o, sink_lse = self._flash_attn_varlen_diff_headdims(
            q=q,
            k=sink_k,
            v=sink_v,
            cu_seqlens_q=prefill.query_start_loc,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=prefill.max_query_len,
            max_seqlen_k=self.sink_len,
            softmax_scale=self.scale,
            causal=False,
            return_softmax_lse=True,
            window_size=None,
        )
        return sink_o, sink_lse

    def _expand_prefill_kv(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        prefill_metadata,
        use_fp8_prefill: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same K/V expansion as MLACommonImpl.forward_mha."""
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = self._concat_k_nope_k_pe(k_nope, k_pe)
        if use_fp8_prefill:
            q_dtype = prefill_metadata.q_data_type
            k = k.to(q_dtype)
            v = v.to(q_dtype)
        return k, v

    def _forward_mha_merge_sink_with_new_tokens(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        sink_o: torch.Tensor,
        sink_lse: torch.Tensor,
        use_fp8_prefill: bool,
    ) -> None:
        """Parent new-token path + sink prefix merge (no chunked context)."""
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.prefill_backend is not None
        k, v = self._expand_prefill_kv(
            kv_c_normed, k_pe, prefill_metadata, use_fp8_prefill
        )
        new_tokens_o, new_tokens_lse = (
            prefill_metadata.prefill_backend.run_prefill_new_tokens(
                q=q,
                k=k,
                v=v,
                return_softmax_lse=True,
            )
        )
        output_view = output.view(-1, self.num_heads, self.v_head_dim)
        merge_attn_states(
            output=output_view,
            output_lse=None,
            prefix_output=sink_o,
            prefix_lse=sink_lse,
            suffix_output=new_tokens_o,
            suffix_lse=new_tokens_lse,
        )

    def _forward_mha_merge_sink_with_sliding_window(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        sink_o: torch.Tensor,
        sink_lse: torch.Tensor,
        use_fp8_prefill: bool,
    ) -> None:
        """Sink prefix + causal sliding-window attention over new tokens."""
        prefill = attn_metadata.prefill
        k, v = self._expand_prefill_kv(kv_c_normed, k_pe, prefill, use_fp8_prefill)
        new_tokens_o, new_tokens_lse = self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=prefill.query_start_loc,
            cu_seqlens_k=prefill.query_start_loc,
            max_seqlen_q=prefill.max_query_len,
            max_seqlen_k=prefill.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=True,
            window_size=self.window_size,
        )
        output_view = output.view(-1, self.num_heads, self.v_head_dim)
        merge_attn_states(
            output=output_view,
            output_lse=None,
            prefix_output=sink_o,
            prefix_lse=sink_lse,
            suffix_output=new_tokens_o,
            suffix_lse=new_tokens_lse,
        )

    def _sink_prefill_attn_latent(
        self,
        q_pe: torch.Tensor,
        ql_nope: torch.Tensor,
        attn_metadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention over static sink tokens in latent space (for paged SWA)."""
        prefill = attn_metadata.prefill
        num_prefills = attn_metadata.num_prefills
        assert self.sink_k_pe is not None and self.sink_compressed_kv is not None
        sink_k_pe = self.sink_k_pe.repeat(num_prefills, 1, 1)
        sink_kv_c = self.sink_compressed_kv
        if sink_kv_c.dim() == 2:
            sink_kv_c = sink_kv_c.unsqueeze(1)
        sink_kv_c = sink_kv_c.repeat(num_prefills, 1, 1)
        cu_seqlens_k = torch.arange(
            0,
            num_prefills * self.sink_len + 1,
            self.sink_len,
            device=q_pe.device,
            dtype=torch.int32,
        )
        return flash_attn_varlen_func(
            q=q_pe,
            k=sink_k_pe,
            v=sink_kv_c,
            q_v=ql_nope,
            max_seqlen_q=prefill.max_query_len,
            cu_seqlens_q=prefill.query_start_loc,
            max_seqlen_k=self.sink_len,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=self.scale,
            causal=False,
            return_softmax_lse=True,
            fa_version=3,
            num_splits=0,
            cp_world_size=self.dcp_world_size,
            cp_rank=self.dcp_rank,
            cp_tot_seqused_k=None,
            window_size=None,
        )

    def _ensure_mqa_bmm_weights(self) -> None:
        if getattr(self, "_w_uk_t", None) is not None:
            return
        kv_b_proj_weight = self.kv_b_proj.weight.T.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        w_uk, w_uv = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        self._w_uk_t = w_uk.permute(1, 2, 0)
        self._w_uv = w_uv.transpose(0, 1)

    def _absorb_uk_into_q(self, q_nope: torch.Tensor) -> torch.Tensor:
        self._ensure_mqa_bmm_weights()
        # (T, N, P) -> (N, T, P) @ (N, P, L) -> (T, N, L)
        q_nope_t = q_nope.transpose(0, 1)
        ql_nope = torch.bmm(q_nope_t, self._w_uk_t)
        return ql_nope.transpose(0, 1)

    def _v_up_proj_latent(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_mqa_bmm_weights()
        # (T, N, L) -> (N, T, L) @ (N, L, V) -> (T, N, V)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        out = torch.bmm(x, self._w_uv)
        return out.transpose(0, 1)

    def _prefill_seq_lens(self, prefill_metadata) -> torch.Tensor:
        assert prefill_metadata.chunked_context is not None
        query_lens = (
            prefill_metadata.query_start_loc[1:] - prefill_metadata.query_start_loc[:-1]
        )
        # chunked_context.seq_lens is built on CPU in MLACommonMetadataBuilder.
        context_lens = prefill_metadata.chunked_context.seq_lens.sum(dim=0).to(
            query_lens.device, non_blocking=True
        )
        return (query_lens + context_lens).to(dtype=torch.int32)

    def _forward_mha_merge_sink_with_swa_paged_attn(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
    ) -> None:
        """SWA extend via paged latent KV cache (same kernel path as MQA decode)."""
        assert self.dcp_world_size == 1, (
            "SWA prefill paged attention is not supported with DCP yet."
        )
        prefill = attn_metadata.prefill
        assert prefill is not None
        num_prefills = attn_metadata.num_prefills

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        ql_nope = self._absorb_uk_into_q(q_nope)
        sink_o, sink_lse = self._sink_prefill_attn_latent(q_pe, ql_nope, attn_metadata)

        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank].unsqueeze(-2)
        k_pe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank :].unsqueeze(-2)
        block_size = kv_c_and_k_pe_cache.size(1)
        seq_lens = self._prefill_seq_lens(prefill)
        max_seqlen_q = max(prefill.max_query_len, 1)
        max_seqlen_k = max(int(seq_lens.max().item()), 1)
        sliding_window_size = (
            list(self.window_size) if self.window_size is not None else None
        )

        scheduler_metadata = get_scheduler_metadata(
            batch_size=num_prefills,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads_q=self.num_heads * self.dcp_world_size,
            num_heads_kv=1,
            headdim=self.qk_rope_head_dim,
            cache_seqlens=seq_lens,
            qkv_dtype=kv_c_and_k_pe_cache.dtype,
            headdim_v=self.kv_lora_rank,
            page_size=block_size,
            cu_seqlens_q=prefill.query_start_loc,
            causal=True,
            num_splits=0,
            window_size=sliding_window_size,
        )

        latent_o, latent_lse = flash_attn_varlen_func(
            q=q_pe,
            k=k_pe_cache,
            v=kv_c_cache,
            q_v=ql_nope,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=prefill.query_start_loc,
            max_seqlen_k=max_seqlen_k,
            seqused_k=seq_lens,
            block_table=prefill.block_table,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=True,
            fa_version=3,
            scheduler_metadata=scheduler_metadata,
            num_splits=0,
            cp_world_size=self.dcp_world_size,
            cp_rank=self.dcp_rank,
            cp_tot_seqused_k=None,
            window_size=sliding_window_size,
        )
        merged_latent = torch.empty_like(latent_o)
        merge_attn_states(
            output=merged_latent,
            output_lse=None,
            prefix_output=sink_o,
            prefix_lse=sink_lse,
            suffix_output=latent_o,
            suffix_lse=latent_lse,
        )
        output_view = output.view(-1, self.num_heads, self.v_head_dim)
        output_view.copy_(self._v_up_proj_latent(merged_latent))

    def _forward_mha_merge_sink_with_chunked_context(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        sink_o: torch.Tensor,
        sink_lse: torch.Tensor,
        use_fp8_prefill: bool,
    ) -> None:
        """Parent chunked-context path, then merge sink as leftmost prefix."""
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.prefill_backend is not None
        assert prefill_metadata.chunked_context is not None

        k, v = self._expand_prefill_kv(
            kv_c_normed, k_pe, prefill_metadata, use_fp8_prefill
        )
        suffix_output, suffix_lse = (
            prefill_metadata.prefill_backend.run_prefill_new_tokens(
                q=q,
                k=k,
                v=v,
                return_softmax_lse=True,
            )
        )
        if self.dcp_world_size > 1:
            context_output, context_lse = (
                self._context_parallel_compute_prefill_context(
                    q,
                    kv_c_and_k_pe_cache,
                    attn_metadata,
                    k_scale=None,
                    dcp_world_size=self.dcp_world_size,
                )
            )
        else:
            context_output, context_lse = self._compute_prefill_context(
                q, kv_c_and_k_pe_cache, attn_metadata, k_scale
            )

        context_suffix_output = torch.empty_like(suffix_output)
        context_suffix_lse = torch.empty_like(suffix_lse)
        merge_attn_states(
            output=context_suffix_output,
            output_lse=context_suffix_lse,
            prefix_output=context_output,
            prefix_lse=context_lse,
            suffix_output=suffix_output,
            suffix_lse=suffix_lse,
            prefill_tokens_with_context=(
                prefill_metadata.chunked_context.prefill_tokens_with_context
            ),
        )

        output_view = output.view(-1, self.num_heads, self.v_head_dim)
        merge_attn_states(
            output=output_view,
            output_lse=None,
            prefix_output=sink_o,
            prefix_lse=sink_lse,
            suffix_output=context_suffix_output,
            suffix_lse=context_suffix_lse,
        )

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        if self.sink_len == 0:
            super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                output_scale,
            )
            return
        assert output_scale is None, (
            "Fused FP8 output is not supported for FlashAttnStaticSinkMLAImpl yet"
        )

        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        use_fp8_prefill = prefill_metadata.q_data_type == current_platform.fp8_dtype()
        q_attn = q.to(prefill_metadata.q_data_type) if use_fp8_prefill else q

        has_context = prefill_metadata.chunked_context is not None
        use_sliding_window = self.window_size is not None and self.window_size != (
            -1,
            -1,
        )

        if use_sliding_window and has_context:
            self._forward_mha_merge_sink_with_swa_paged_attn(
                q_attn,
                kv_c_and_k_pe_cache,
                attn_metadata,
                output,
            )
        elif has_context:
            sink_o, sink_lse = self._compute_sink_prefill_attn(q_attn, attn_metadata)
            self._forward_mha_merge_sink_with_chunked_context(
                q_attn,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                sink_o,
                sink_lse,
                use_fp8_prefill,
            )
        elif use_sliding_window:
            sink_o, sink_lse = self._compute_sink_prefill_attn(q_attn, attn_metadata)
            self._forward_mha_merge_sink_with_sliding_window(
                q_attn,
                kv_c_normed,
                k_pe,
                attn_metadata,
                output,
                sink_o,
                sink_lse,
                use_fp8_prefill,
            )
        else:
            sink_o, sink_lse = self._compute_sink_prefill_attn(q_attn, attn_metadata)
            self._forward_mha_merge_sink_with_new_tokens(
                q_attn,
                kv_c_normed,
                k_pe,
                attn_metadata,
                output,
                sink_o,
                sink_lse,
                use_fp8_prefill,
            )

    def forward_mqa(
        self,
        q,
        kv_c_and_k_pe_cache,
        attn_metadata,
        layer,
    ):
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None
        assert self.sink_k_pe is not None and self.sink_compressed_kv is not None

        if type(q) is tuple:
            q_nope, q_pe = q
        else:
            q_nope, q_pe = torch.split(
                q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 MLA not yet supported")

        decode = attn_metadata.decode
        num_reqs = decode.seq_lens.shape[0]
        max_seqlen_q = max(decode.max_query_len, 1)

        # MQA attention over static sink KV (dense tensors, no block_table).
        sink_k_pe = self.sink_k_pe.repeat(num_reqs, 1, 1)
        sink_kv_c = self.sink_compressed_kv
        if sink_kv_c.dim() == 2:
            sink_kv_c = sink_kv_c.unsqueeze(1)
        sink_kv_c = sink_kv_c.repeat(num_reqs, 1, 1)
        cu_seqlens_k_sink = torch.arange(
            0,
            num_reqs * self.sink_len + 1,
            self.sink_len,
            device=q_pe.device,
            dtype=torch.int32,
        )

        sink_o, sink_lse = flash_attn_varlen_func(
            q=q_pe,
            k=sink_k_pe,
            v=sink_kv_c,
            q_v=q_nope,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=decode.query_start_loc,
            max_seqlen_k=self.sink_len,
            cu_seqlens_k=cu_seqlens_k_sink,
            softmax_scale=self.scale,
            causal=False,
            return_softmax_lse=True,
            fa_version=3,
            num_splits=0,
            cp_world_size=self.dcp_world_size,
            cp_rank=self.dcp_rank,
            cp_tot_seqused_k=decode.dcp_tot_seq_lens,
            window_size=None,
        )
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank].unsqueeze(-2)
        k_pe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank :].unsqueeze(-2)

        block_size = kv_c_and_k_pe_cache.size(1)
        window_seqlens = decode.seq_lens
        window_seqlens = torch.clamp(window_seqlens, min=0)
        max_window_seqlen = max(decode.max_seq_len, 1)
        window_scheduler_metadata = get_scheduler_metadata(
            batch_size=num_reqs,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_window_seqlen,
            num_heads_q=self.num_heads * self.dcp_world_size,
            num_heads_kv=1,
            headdim=self.qk_rope_head_dim,
            cache_seqlens=window_seqlens,
            qkv_dtype=kv_c_and_k_pe_cache.dtype,
            headdim_v=self.kv_lora_rank,
            page_size=block_size,
            cu_seqlens_q=decode.query_start_loc,
            causal=True,
            num_splits=0,
            window_size=self.window_size,
        )

        no_sink_o, no_sink_lse = flash_attn_varlen_func(
            q=q_pe,
            k=k_pe_cache,
            v=kv_c_cache,
            q_v=q_nope,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=decode.query_start_loc,
            max_seqlen_k=max_window_seqlen,
            seqused_k=window_seqlens,
            block_table=decode.block_table,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=True,
            fa_version=3,
            scheduler_metadata=window_scheduler_metadata,
            num_splits=0,
            cp_world_size=self.dcp_world_size,
            cp_rank=self.dcp_rank,
            cp_tot_seqused_k=decode.dcp_tot_seq_lens,
            window_size=self.window_size,
        )

        o = torch.empty_like(no_sink_o)
        lse = (
            torch.empty_like(no_sink_lse)
            if self.need_to_return_lse_for_decode
            else None
        )
        merge_attn_states(
            output=o,
            output_lse=lse,
            prefix_output=sink_o,
            prefix_lse=sink_lse,
            suffix_output=no_sink_o,
            suffix_lse=no_sink_lse,
        )
        return o, lse

    @staticmethod
    def _insert_tensor_by_start_loc(raw_tensor, insert_segment, start_loc):
        segment_len = insert_segment.shape[0]
        num_inserts = len(start_loc) - 1
        total_len = segment_len * num_inserts + raw_tensor.shape[0]
        result = torch.empty(
            total_len,
            *raw_tensor.shape[1:],
            dtype=raw_tensor.dtype,
            device=raw_tensor.device,
        )

        offset = 0
        for i in range(num_inserts):
            result[offset : offset + segment_len] = insert_segment.clone()
            offset += segment_len
            seg_len = start_loc[i + 1] - start_loc[i]
            result[offset : offset + seg_len] = raw_tensor[
                start_loc[i] : start_loc[i + 1]
            ]
            offset += seg_len

        return result
