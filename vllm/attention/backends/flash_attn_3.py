"""
Flash Attention 3 backend with sinks support for Blackwell architecture.
Extends Flash Attention with FA3 features including attention sinks.
"""
from typing import List, Optional, Tuple, Type

import torch

from vllm.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.logger import init_logger
from vllm.utils import is_hip

logger = init_logger(__name__)

try:
    import flash_attn
    
    # Check for FA3 support
    _flash_attn_version = getattr(flash_attn, "__version__", "0.0.0")
    _flash_attn_version_tuple = tuple(
        int(x) for x in _flash_attn_version.split(".")[:3]
    )
    HAS_FLASH_ATTN_3 = _flash_attn_version_tuple >= (3, 0, 0)
    
    if HAS_FLASH_ATTN_3:
        from flash_attn import flash_attn_func_v3, flash_attn_varlen_func_v3
        from flash_attn.flash_attn_interface import flash_attn_with_kvcache_v3
    else:
        # Fallback to regular FA2 functions
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.flash_attn_interface import flash_attn_with_kvcache
        
        flash_attn_func_v3 = flash_attn_func
        flash_attn_varlen_func_v3 = flash_attn_varlen_func
        flash_attn_with_kvcache_v3 = flash_attn_with_kvcache

except ImportError:
    HAS_FLASH_ATTN_3 = False
    logger.warning("Flash Attention 3 not available. Falling back to FA2.")


class FlashAttention3Backend(FlashAttentionBackend):
    """Flash Attention 3 backend with sinks support."""

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_3"

    @staticmethod
    def get_impl_cls() -> Type["FlashAttention3Impl"]:
        return FlashAttention3Impl

    @staticmethod
    def get_metadata_cls() -> Type["FlashAttention3Metadata"]:
        return FlashAttention3Metadata

    @staticmethod
    def get_builder_cls() -> Type["FlashAttention3MetadataBuilder"]:
        return FlashAttention3MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        """Get KV cache shape for FA3."""
        if HAS_FLASH_ATTN_3:
            # FA3 may have different optimal layout
            return (2, num_blocks, block_size, num_kv_heads, head_size)
        else:
            # Fallback to FA2 layout
            return (2, num_blocks, num_kv_heads, head_size, block_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        """Swap KV cache blocks for FA3."""
        # For now, use the same swapping logic as FA2
        # FA3 may optimize this in the future
        FlashAttentionBackend.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dsts: torch.Tensor,
    ) -> None:
        """Copy KV cache blocks for FA3."""
        # For now, use the same copying logic as FA2
        FlashAttentionBackend.copy_blocks(kv_caches, src_to_dsts)


class FlashAttention3Metadata(FlashAttentionMetadata):
    """Metadata for Flash Attention 3 with sinks support."""

    def __init__(
        self,
        num_prefills: int,
        slot_mapping: torch.Tensor,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        context_lens: Optional[torch.Tensor],
        prefill_seq_lens: Optional[List[int]],
        block_tables: Optional[torch.Tensor],
        # FA3-specific parameters
        use_sinks: bool = False,
        sink_size: int = 4,
        window_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            context_lens=context_lens,
            prefill_seq_lens=prefill_seq_lens,
            block_tables=block_tables,
            **kwargs,
        )
        
        # FA3-specific attributes
        self.use_sinks = use_sinks
        self.sink_size = sink_size
        self.window_size = window_size


class FlashAttention3MetadataBuilder(FlashAttentionMetadataBuilder):
    """Metadata builder for Flash Attention 3."""

    def __init__(
        self,
        input_builder: "ModelInputForGPUBuilder",
        # FA3-specific parameters
        use_sinks: bool = False,
        sink_size: int = 4,
        window_size: Optional[int] = None,
    ):
        super().__init__(input_builder)
        self.use_sinks = use_sinks
        self.sink_size = sink_size
        self.window_size = window_size

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build FA3 metadata."""
        
        # Build base metadata first
        metadata = super().build(seq_lens, query_lens, cuda_graph_pad_size, batch_size)
        
        # Convert to FA3 metadata
        fa3_metadata = FlashAttention3Metadata(
            num_prefills=metadata.num_prefills,
            slot_mapping=metadata.slot_mapping,
            num_prefill_tokens=metadata.num_prefill_tokens,
            num_decode_tokens=metadata.num_decode_tokens,
            context_lens=metadata.context_lens,
            prefill_seq_lens=metadata.prefill_seq_lens,
            block_tables=metadata.block_tables,
            # FA3-specific
            use_sinks=self.use_sinks,
            sink_size=self.sink_size,
            window_size=self.window_size,
        )
        
        # Copy other attributes
        for attr in dir(metadata):
            if not attr.startswith('_') and not hasattr(fa3_metadata, attr):
                setattr(fa3_metadata, attr, getattr(metadata, attr))
        
        return fa3_metadata


class FlashAttention3Impl(FlashAttentionImpl):
    """Flash Attention 3 implementation with sinks support."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[torch.Tensor],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_local_blocks: int,
        blocksparse_vert_stride: int,
        blocksparse_block_size: int,
        blocksparse_head_sliding_step: int,
        # FA3-specific parameters
        use_sinks: bool = False,
        sink_size: int = 4,
        use_fa3_features: bool = True,
    ):
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            blocksparse_local_blocks=blocksparse_local_blocks,
            blocksparse_vert_stride=blocksparse_vert_stride,
            blocksparse_block_size=blocksparse_block_size,
            blocksparse_head_sliding_step=blocksparse_head_sliding_step,
        )
        
        self.use_sinks = use_sinks and HAS_FLASH_ATTN_3
        self.sink_size = sink_size if self.use_sinks else 0
        self.use_fa3_features = use_fa3_features and HAS_FLASH_ATTN_3

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttention3Metadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        output: Optional[torch.Tensor] = None,
        attn_type: str = "DECODER",
    ) -> torch.Tensor:
        """Forward pass with FA3 features."""
        
        if not self.use_fa3_features or not HAS_FLASH_ATTN_3:
            # Fallback to regular Flash Attention
            return super().forward(
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                k_scale=k_scale,
                v_scale=v_scale,
                output=output,
                attn_type=attn_type,
            )
        
        # FA3-specific forward pass
        num_tokens, hidden_size = query.shape
        query = query.view(num_tokens, self.num_heads, self.head_size)
        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)

        if output is None:
            output = torch.empty_like(query)
        else:
            output = output.view(num_tokens, self.num_heads, self.head_size)

        if kv_cache.numel() > 0:
            key_cache, value_cache = kv_cache.unbind(0)
            
            # Use FA3 with KV cache if available
            if hasattr(attn_metadata, 'use_sinks') and attn_metadata.use_sinks:
                # FA3 with sinks
                output = flash_attn_with_kvcache_v3(
                    q=query,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    k=key,
                    v=value,
                    cache_seqlens=attn_metadata.context_lens,
                    block_table=attn_metadata.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                    sink_token_length=attn_metadata.sink_size if attn_metadata.use_sinks else 0,
                    window_size=attn_metadata.window_size,
                )
            else:
                # Regular FA3 without sinks
                output = flash_attn_with_kvcache_v3(
                    q=query,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    k=key,
                    v=value,
                    cache_seqlens=attn_metadata.context_lens,
                    block_table=attn_metadata.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )
        else:
            # No KV cache, use varlen function
            if self.use_sinks and hasattr(attn_metadata, 'use_sinks') and attn_metadata.use_sinks:
                output = flash_attn_varlen_func_v3(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=attn_metadata.seq_start_loc,
                    cu_seqlens_k=attn_metadata.seq_start_loc,
                    max_seqlen_q=attn_metadata.max_prefill_seq_len,
                    max_seqlen_k=attn_metadata.max_prefill_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    sink_token_length=attn_metadata.sink_size if attn_metadata.use_sinks else 0,
                    window_size=attn_metadata.window_size,
                )
            else:
                output = flash_attn_varlen_func_v3(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=attn_metadata.seq_start_loc,
                    cu_seqlens_k=attn_metadata.seq_start_loc,
                    max_seqlen_q=attn_metadata.max_prefill_seq_len,
                    max_seqlen_k=attn_metadata.max_prefill_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                )

        return output.view(num_tokens, hidden_size)


def create_fa3_backend_with_sinks(
    use_sinks: bool = True,
    sink_size: int = 4,
    window_size: Optional[int] = None,
) -> FlashAttention3Backend:
    """Create a Flash Attention 3 backend with sinks configuration."""
    
    if not HAS_FLASH_ATTN_3:
        logger.warning(
            "Flash Attention 3 not available. "
            "Please install flash-attn >= 3.0.0 for FA3 features."
        )
        return FlashAttentionBackend()  # Fallback to FA2
    
    backend = FlashAttention3Backend()
    backend.use_sinks = use_sinks
    backend.sink_size = sink_size
    backend.window_size = window_size
    
    return backend
