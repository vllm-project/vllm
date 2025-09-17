# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with PagedAttention and Triton prefix prefill."""
from dataclasses import dataclass
from functools import cache
from typing import ClassVar, Optional

import torch

from vllm import envs
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode)
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey, kFp8StaticTensorSym)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.utils import (AttentionCGSupport,
                                              AttentionMetadataBuilder,
                                              CommonAttentionMetadata)
from vllm.v1.kv_cache_interface import AttentionSpec

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._ipex_ops import ipex_ops as ops

logger = init_logger(__name__)


@dataclass
class TritonAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: Optional[torch.Tensor]
    prefix_kv_lens: Optional[torch.Tensor]
    suffix_kv_lens: Optional[torch.Tensor]

    # Optional aot scheduling
    scheduler_metadata: Optional[torch.Tensor] = None
    prefix_scheduler_metadata: Optional[torch.Tensor] = None


class TritonAttentionMetadataBuilder(
        AttentionMetadataBuilder[TritonAttentionMetadata]):
    cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config)
        self.num_heads_kv = model_config.get_num_kv_heads(
            vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> TritonAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor([0, num_actual_tokens],
                                                dtype=torch.int32,
                                                device=self.device)
            prefix_kv_lens = torch.tensor([common_prefix_len],
                                          dtype=torch.int32,
                                          device=self.device)
            suffix_kv_lens = (common_attn_metadata.seq_lens_cpu -
                              common_prefix_len)
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None

        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
        )
        return attn_metadata


class TritonAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        supported_head_sizes = cls.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            attn_type = cls.__name__.removesuffix("Backend")
            raise ValueError(
                f"Head size {head_size} is not supported by {attn_type}. "
                f"Supported head sizes are: {supported_head_sizes}. "
                "Set VLLM_ATTENTION_BACKEND=FLEX_ATTENTION to use "
                "FlexAttention backend which supports all head sizes.")

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return TritonAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder


@cache
def use_aiter_unified_attention() -> bool:
    """Check if aiter unified attention should be used."""
    # VLLM_ROCM_USE_AITER_MHA needs to set to 0 as well as it is set
    # to 1 as default
    return envs.VLLM_ROCM_USE_AITER \
        and envs.VLLM_USE_AITER_UNIFIED_ATTENTION


class TritonAttentionImpl(AttentionImpl):

    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[int] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        TritonAttentionBackend.validate_head_size(head_size)

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "TritonAttentionImpl")

        self.fp8_dtype = current_platform.fp8_dtype()
        self.force_prefill_decode_attn = \
            envs.VLLM_V1_USE_PREFILL_DECODE_ATTENTION

        if not self.force_prefill_decode_attn:
            # If not using prefill decode attention, we use the Triton
            # unified attention implementation.
            if use_aiter_unified_attention():
                logger.info_once(
                    "Using aiter unified attention for TritonAttentionImpl")
                from aiter.ops.triton.unified_attention import (
                    unified_attention)
                self.unified_attention = unified_attention
            else:
                logger.info_once(
                    "Using vllm unified attention for TritonAttentionImpl")
                from vllm.attention.ops.triton_unified_attention import (
                    unified_attention)
                self.unified_attention = unified_attention

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}.")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl")

        if attn_metadata is None:
            # Profiling run.
            return output

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        use_prefill_decode_attn = self.force_prefill_decode_attn
        num_actual_tokens = attn_metadata.num_actual_tokens

        if use_prefill_decode_attn:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)
        else:
            key_cache, value_cache = kv_cache.unbind(0)

        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            if use_prefill_decode_attn:
                PagedAttention.write_to_paged_cache(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )
            else:
                ops.reshape_and_cache_flash(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )

        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            num_tokens, num_heads, head_size = query.shape
            assert layer._q_scale_float == 1.0, \
                "A non 1.0 q_scale is not currently supported."
            if current_platform.is_cuda():
                # Skip Q quantization on ROCm and XPU, enable this on cuda
                # only, since dequantizing back to f32 in the attention kernel
                # is not supported.
                query, _ = ops.scaled_fp8_quant(
                    query.reshape(
                        (num_tokens, num_heads * head_size)).contiguous(),
                    layer._q_scale)
                query = query.reshape((num_tokens, num_heads, head_size))

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        if use_prefill_decode_attn:
            # Compute attention and update output up to `num_actual_tokens`.
            chunked_prefill_paged_decode(
                query=query[:num_actual_tokens],
                key=key[:num_actual_tokens],
                value=value[:num_actual_tokens],
                output=output[:num_actual_tokens],
                kv_cache_dtype=self.kv_cache_dtype,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                query_start_loc=cu_seqlens_q,
                seq_lens=seqused_k,
                max_seq_len=max_seqlen_k,
                max_query_len=max_seqlen_q,
                k_scale=layer._k_scale,
                v_scale=layer._v_scale,
                alibi_slopes=self.alibi_slopes,
                sliding_window=self.sliding_window[0],
                sm_scale=self.scale,
                output_scale=output_scale,
                sinks=self.sinks,
            )

        else:
            descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

            self.unified_attention(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # Not supported
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
                sinks=self.sinks,
                output_scale=output_scale)

        return output
