"""Flash Attention 4 (CuTe DSL) backend for SM120 (consumer Blackwell).

Flash-attn-4 provides JIT-compiled attention kernels via NVIDIA CuTe DSL,
supporting SM80/SM90/SM100/SM120. This backend targets SM120 devices
(RTX PRO 6000, DGX Spark, etc.) where FA2/FA3 C++ cubins are unavailable.

The kernel interface (`flash_attn_varlen_func`) accepts paged KV cache
via `page_table`, matching vLLM's block table format directly:
  K: (num_blocks, block_size, num_kv_heads, head_dim)
  V: (num_blocks, block_size, num_kv_heads, head_dim)
  page_table: (batch_size, max_blocks_per_seq)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)

logger = init_logger(__name__)


@dataclass
class FlashAttention4Metadata:
    """Metadata for a batch of attention operations."""

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor  # (batch_size+1,)
    max_seq_len: int
    seq_lens: torch.Tensor  # (batch_size,)
    block_table: torch.Tensor  # (batch_size, max_blocks_per_seq)
    slot_mapping: torch.Tensor  # (num_actual_tokens,)
    causal: bool = True


class FlashAttention4Backend(AttentionBackend):
    """Attention backend using flash-attn-4 CuTe DSL kernels."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_4"

    @staticmethod
    def get_impl_cls() -> type[FlashAttention4Impl]:
        return FlashAttention4Impl

    @staticmethod
    def get_builder_cls() -> type[FlashAttention4MetadataBuilder]:
        return FlashAttention4MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # (K/V, num_blocks, block_size, num_kv_heads, head_dim)
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        return (0, 1, 2, 3, 4)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 96, 128, 256]

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in cls.get_supported_head_sizes()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # SM120 (consumer Blackwell) — primary target
        # Also supports SM80/SM90 but those have better backends
        return capability >= DeviceCapability(12, 0) and capability <= DeviceCapability(
            12, 1
        )

    @classmethod
    def supports_attn_type(cls, attn_type: AttentionType) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType) -> bool:
        return kv_cache_dtype in ("auto", "float16", "bfloat16")

    @classmethod
    def get_cudagraph_support(cls) -> AttentionCGSupport:
        # No CUDA graph support initially — enforce_eager required
        return AttentionCGSupport.NEVER


class FlashAttention4MetadataBuilder(AttentionMetadataBuilder[FlashAttention4Metadata]):
    """Build attention metadata from common batch metadata."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttention4Metadata:
        return FlashAttention4Metadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )


class FlashAttention4Impl(AttentionImpl):
    """Flash-attn-4 CuTe DSL attention implementation for SM120."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("FlashAttention4 only supports DECODER attention")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        if sliding_window is not None:
            self.sliding_window = (sliding_window - 1, 0)
        else:
            self.sliding_window = (-1, -1)

        if alibi_slopes is not None:
            logger.warning("FlashAttention4 does not support ALiBi. Ignoring.")

        self.softcap = logits_soft_cap or 0.0

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttention4Metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty_like(query)
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        from flash_attn.cute.interface import flash_attn_varlen_func

        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]

        # Separate K and V from paged cache
        # kv_cache: (2, num_blocks, block_size, num_kv_heads, head_dim)
        key_cache, value_cache = kv_cache.unbind(0)

        # flash_attn_varlen_func with page_table for paged KV
        out, *_ = flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            page_table=attn_metadata.block_table,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            window_size=self.sliding_window,
            softcap=self.softcap,
            return_lse=False,
        )
        output[:num_actual_tokens].copy_(out)
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # Use vLLM's standard reshape_and_cache for KV cache update
        from vllm._custom_ops import reshape_and_cache_flash

        key_cache, value_cache = kv_cache.unbind(0)
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            getattr(layer, "_k_scale", None),
            getattr(layer, "_v_scale", None),
        )
