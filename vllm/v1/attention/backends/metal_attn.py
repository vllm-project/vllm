"""Metal attention backend for Apple Silicon (MPS)."""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                               AttentionMetadata, AttentionType)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# Import Metal ops (will be loaded from C++ extension)
try:
    import vllm._metal_C as metal_ops
except ImportError:
    logger.warning("Metal C extension not available, falling back to CPU")
    metal_ops = None


@dataclass
class MetalAttentionMetadata(AttentionMetadata):
    """Metadata for Metal attention backend."""

    # Attention type (prefill or decode)
    attention_type: AttentionType

    # Decode-specific metadata
    block_tables: Optional[torch.Tensor] = None  # [batch_size, max_blocks]
    seq_lens_tensor: Optional[torch.Tensor] = None  # [batch_size]
    max_decode_seq_len: int = 0

    # Prefill-specific metadata
    query_start_loc: Optional[torch.Tensor] = None  # [batch_size + 1]
    max_prefill_seq_len: int = 0

    # KV cache metadata
    slot_mapping: Optional[torch.Tensor] = None  # [num_tokens]

    # Model configuration
    num_kv_heads: int = 0
    scale: float = 1.0
    block_size: int = 16

    # Optional features
    alibi_slopes: Optional[torch.Tensor] = None
    kv_cache_dtype: Optional[str] = None


class MetalAttentionImpl(AttentionImpl[MetalAttentionMetadata]):
    """Metal attention implementation using native Metal kernels."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        if metal_ops is None:
            raise RuntimeError("Metal C extension not available")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        if sliding_window is not None:
            logger.warning("Sliding window attention not yet supported on Metal")

        if logits_soft_cap is not None:
            logger.warning("Logits soft cap not yet supported on Metal")

        if kv_sharing_target_layer_name is not None:
            logger.warning("KV sharing not yet supported on Metal")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Execute attention computation."""

        # Ensure tensors are on MPS device
        assert query.device.type == "mps", f"Expected MPS device, got {query.device}"

        batch_size = query.size(0)
        num_heads = self.num_heads
        head_size = self.head_size

        # Prepare output tensor
        if output is None:
            output = torch.empty_like(query)

        # Split KV cache into key and value caches
        # KV cache shape: [2, num_blocks, block_size * num_kv_heads * head_size]
        key_cache, value_cache = self._split_kv_cache(
            kv_cache, attn_metadata.block_size)

        # Determine if this is prefill or decode based on sequence lengths
        is_prefill = (attn_metadata.max_prefill_seq_len > 0 and
                      attn_metadata.query_start_loc is not None)

        logger.debug(f"Metal Attention forward: is_prefill={is_prefill}, max_prefill_seq_len={attn_metadata.max_prefill_seq_len}, query_start_loc={attn_metadata.query_start_loc}")

        if is_prefill:
            # Prefill: Write new K/V to cache and compute attention
            logger.debug("  Taking PREFILL path")
            self._prefill_attention(
                query=query,
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                attn_metadata=attn_metadata,
                output=output,
            )
        else:
            # Decode: Append new K/V and compute attention with paging
            logger.debug("  Taking DECODE path")
            self._decode_attention(
                query=query,
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        return output

    def _split_kv_cache(
        self, kv_cache: torch.Tensor, block_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split KV cache into key and value caches with correct layout."""
        # KV cache shape: [2, num_blocks, block_size * num_kv_heads * head_size]
        num_blocks = kv_cache.shape[1]

        # Reshape to separate key and value caches
        # Key cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
        # Value cache: [num_blocks, num_kv_heads, head_size, block_size]
        x = 16  # vectorization factor

        key_cache = kv_cache[0]
        key_cache = key_cache.view(
            num_blocks, self.num_kv_heads, self.head_size // x, block_size, x
        ).contiguous()

        value_cache = kv_cache[1]
        value_cache = value_cache.view(
            num_blocks, self.num_kv_heads, self.head_size, block_size
        ).contiguous()

        return key_cache, value_cache

    def _pytorch_reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        """PyTorch fallback for reshape_and_cache operation."""
        # Key shape: [num_tokens, num_kv_heads, head_size]
        # Value shape: [num_tokens, num_kv_heads, head_size]
        # Key cache shape: [num_blocks, num_kv_heads, head_size // x, block_size, x]
        # Value cache shape: [num_blocks, num_kv_heads, head_size, block_size]
        # Slot mapping: [num_tokens] - maps each token to a cache slot

        x = 16  # vectorization factor
        num_tokens = key.size(0)

        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            block_idx = slot // block_size
            position_in_block = slot % block_size

            # Write key (handle vectorization)
            key_reshaped = key[i].view(self.num_kv_heads, self.head_size // x, x)
            key_cache[block_idx, :, :, position_in_block, :] = key_reshaped

            # Write value
            value_cache[block_idx, :, :, position_in_block] = value[i]

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        """Prefill attention computation."""

        # For prefill, we need to:
        # 1. Write K/V to cache
        # 2. Compute attention (for now, fallback to PyTorch SDPA)

        # Determine tensor shapes
        num_tokens = key.size(0)

        # Reshape key and value for caching: [num_tokens, hidden_size] -> [num_tokens, num_kv_heads, head_size]
        key_reshaped = key.view(num_tokens, self.num_kv_heads, self.head_size).contiguous()
        value_reshaped = value.view(num_tokens, self.num_kv_heads, self.head_size).contiguous()

        # Write K/V to cache
        if attn_metadata.slot_mapping is not None:
            logger.debug(f"PREFILL reshape_and_cache: key_cache.shape={key_cache.shape}, value_cache.shape={value_cache.shape}, block_size={attn_metadata.block_size}")
            logger.debug(f"  key_reshaped.shape={key_reshaped.shape}, slot_mapping={attn_metadata.slot_mapping}")

            # Check if we should use Metal kernels or PyTorch fallback
            import os
            use_metal_kernels = os.environ.get("VLLM_USE_METAL_KERNELS", "1") == "1"

            if use_metal_kernels:
                metal_ops.reshape_and_cache(
                    key=key_reshaped,
                    value=value_reshaped,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    slot_mapping=attn_metadata.slot_mapping,
                )
            else:
                # PyTorch fallback for cache write
                logger.warning("Using PyTorch fallback for prefill cache write (slower)")
                self._pytorch_reshape_and_cache(
                    key=key_reshaped,
                    value=value_reshaped,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    slot_mapping=attn_metadata.slot_mapping,
                    block_size=attn_metadata.block_size,
                )

        # For prefill attention, use PyTorch SDPA as fallback
        # TODO: Implement Metal prefill attention kernel
        # Reshape for attention computation
        # Query: [num_tokens, hidden_size] -> [batch_size, seq_len, num_heads, head_size]
        # For now, assume batch_size=1 and seq_len=num_tokens
        batch_size = 1
        seq_len = num_tokens

        query = query.view(batch_size, seq_len, self.num_heads, self.head_size)
        query = query.transpose(1, 2)  # [batch, num_heads, seq_len, head_size]

        key_attn = key.view(batch_size, seq_len, self.num_kv_heads, self.head_size)
        key_attn = key_attn.transpose(1, 2)

        value_attn = value.view(batch_size, seq_len, self.num_kv_heads, self.head_size)
        value_attn = value_attn.transpose(1, 2)

        # Grouped query attention
        if self.num_kv_heads != self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            key_attn = key_attn.repeat_interleave(num_groups, dim=1)
            value_attn = value_attn.repeat_interleave(num_groups, dim=1)

        # Compute attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key_attn, value_attn, scale=self.scale
        )

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        hidden_size = self.num_heads * self.head_size
        output.copy_(attn_output.view(num_tokens, hidden_size))

    def _decode_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        """Decode attention computation with paged KV cache."""

        # Reshape key and value for caching: [batch_size, hidden_size] -> [batch_size, num_kv_heads, head_size]
        batch_size = key.size(0)
        key_reshaped = key.view(batch_size, self.num_kv_heads, self.head_size).contiguous()
        value_reshaped = value.view(batch_size, self.num_kv_heads, self.head_size).contiguous()

        # Write new K/V to cache
        if attn_metadata.slot_mapping is not None:
            logger.debug(f"DECODE reshape_and_cache: key_cache.shape={key_cache.shape}, value_cache.shape={value_cache.shape}")
            logger.debug(f"  key_reshaped.shape={key_reshaped.shape}, slot_mapping={attn_metadata.slot_mapping}")

            # Check if we should use Metal kernels or PyTorch fallback
            import os
            use_metal_kernels = os.environ.get("VLLM_USE_METAL_KERNELS", "1") == "1"

            if use_metal_kernels:
                metal_ops.reshape_and_cache(
                    key=key_reshaped,
                    value=value_reshaped,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    slot_mapping=attn_metadata.slot_mapping,
                )
            else:
                # PyTorch fallback for cache write
                logger.warning("Using PyTorch fallback for decode cache write (slower)")
                self._pytorch_reshape_and_cache(
                    key=key_reshaped,
                    value=value_reshaped,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    slot_mapping=attn_metadata.slot_mapping,
                    block_size=attn_metadata.block_size,
                )

        # Reshape query for paged attention
        query = query.view(batch_size, self.num_heads, self.head_size)

        # Use Metal paged attention kernels
        # Note: Env var VLLM_USE_METAL_KERNELS=0 can disable if issues arise
        import os
        use_metal_kernels = os.environ.get("VLLM_USE_METAL_KERNELS", "1") == "1"

        if not use_metal_kernels:
            # Fallback to PyTorch SDPA - slower but correct
            # Reconstruct K/V from paged cache and use PyTorch attention
            logger.warning("Using PyTorch SDPA fallback for Metal decode attention (slower)")
            self._decode_attention_pytorch_fallback(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                attn_metadata=attn_metadata,
                output=output,
            )
            return

        # Choose V1 or V2 kernel based on sequence length
        max_seq_len = attn_metadata.max_decode_seq_len
        block_size = attn_metadata.block_size
        max_num_partitions = (max_seq_len + 511) // 512

        use_v1 = max_seq_len <= 8192 and (
            max_num_partitions == 1 or batch_size * self.num_heads > 512
        )

        if use_v1:
            # Convert bfloat16 to float16 if needed (Metal doesn't support bfloat16)
            query_metal = query.to(torch.float16) if query.dtype == torch.bfloat16 else query
            output_metal = output if output.dtype != torch.bfloat16 else output.to(torch.float16)

            # Debug logging
            logger.debug(f"Metal V1: batch={batch_size}, num_heads={self.num_heads}, head_size={self.head_size}")
            logger.debug(f"  query shape: {query_metal.shape}, dtype: {query_metal.dtype}")
            logger.debug(f"  output shape: {output_metal.shape}, dtype: {output_metal.dtype}")
            logger.debug(f"  seq_lens: {attn_metadata.seq_lens_tensor}")
            logger.debug(f"  max_seq_len: {max_seq_len}, block_size: {block_size}")

            output_view = output_metal.view(batch_size, self.num_heads, self.head_size)
            logger.debug(f"  output_view shape: {output_view.shape}")

            # Use V1 kernel
            metal_ops.paged_attention_v1(
                out=output_view,
                query=query_metal,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=attn_metadata.block_tables,
                seq_lens=attn_metadata.seq_lens_tensor,
                num_kv_heads=self.num_kv_heads,
                scale=self.scale,
                block_size=block_size,
                max_seq_len=max_seq_len,
                alibi_slopes=self.alibi_slopes,
                kv_cache_scales=None,
            )

            # Debug: Check output after kernel
            logger.debug(f"  After kernel - output_view[0,0,:5]: {output_view[0,0,:5]}")
            logger.debug(f"  After kernel - output_view min: {output_view.min()}, max: {output_view.max()}, mean: {output_view.mean()}")

            # Convert back to bfloat16 if needed
            if output.dtype == torch.bfloat16:
                output.copy_(output_metal.to(torch.bfloat16))
        else:
            # Use V2 kernel with partitioning
            # Convert bfloat16 to float16 if needed (Metal doesn't support bfloat16)
            query_metal = query.to(torch.float16) if query.dtype == torch.bfloat16 else query
            output_metal = output if output.dtype != torch.bfloat16 else output.to(torch.float16)

            # Allocate temporary buffers
            exp_sums = torch.empty(
                (batch_size, self.num_heads, max_num_partitions),
                dtype=torch.float32,
                device=query.device,
            )
            max_logits = torch.empty(
                (batch_size, self.num_heads, max_num_partitions),
                dtype=torch.float32,
                device=query.device,
            )
            tmp_out = torch.empty(
                (batch_size, self.num_heads, max_num_partitions, self.head_size),
                dtype=torch.float16 if output.dtype == torch.bfloat16 else output.dtype,
                device=query.device,
            )

            metal_ops.paged_attention_v2(
                out=output_metal.view(batch_size, self.num_heads, self.head_size),
                exp_sums=exp_sums,
                max_logits=max_logits,
                tmp_out=tmp_out,
                query=query_metal,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=attn_metadata.block_tables,
                seq_lens=attn_metadata.seq_lens_tensor,
                num_kv_heads=self.num_kv_heads,
                scale=self.scale,
                block_size=block_size,
                max_seq_len=max_seq_len,
                alibi_slopes=self.alibi_slopes,
                kv_cache_scales=None,
            )

            # Convert back to bfloat16 if needed
            if output.dtype == torch.bfloat16:
                output.copy_(output_metal.to(torch.bfloat16))

    def _decode_attention_pytorch_fallback(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        """
        Fallback decode attention using PyTorch SDPA.
        Reconstructs K/V from paged cache - slower but correct.
        """
        # query shape: [batch_size, num_heads, head_size]
        batch_size = query.size(0)
        block_size = attn_metadata.block_size

        # Get sequence lengths and block tables
        seq_lens = attn_metadata.seq_lens_tensor  # [batch_size]
        block_tables = attn_metadata.block_tables  # [batch_size, max_blocks]

        # Reconstruct full K/V tensors from paged cache
        # key_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
        # value_cache: [num_blocks, num_kv_heads, head_size, block_size]

        max_seq_len = seq_lens.max().item()

        # Allocate buffers for reconstructed K/V
        # Shape: [batch_size, num_kv_heads, max_seq_len, head_size]
        reconstructed_keys = torch.zeros(
            batch_size, self.num_kv_heads, max_seq_len, self.head_size,
            dtype=query.dtype, device=query.device
        )
        reconstructed_values = torch.zeros(
            batch_size, self.num_kv_heads, max_seq_len, self.head_size,
            dtype=query.dtype, device=query.device
        )

        # Reconstruct K/V from paged cache
        x = 16  # vectorization factor
        for batch_idx in range(batch_size):
            seq_len = seq_lens[batch_idx].item()
            num_blocks_needed = (seq_len + block_size - 1) // block_size

            for block_idx in range(num_blocks_needed):
                # Get physical block ID
                physical_block_id = block_tables[batch_idx, block_idx].item()

                # Calculate token range in this block
                start_token_idx = block_idx * block_size
                end_token_idx = min(start_token_idx + block_size, seq_len)
                num_tokens_in_block = end_token_idx - start_token_idx

                # Extract keys from cache
                # key_cache shape: [num_blocks, num_kv_heads, head_size/x, block_size, x]
                block_keys = key_cache[physical_block_id]  # [num_kv_heads, head_size/x, block_size, x]
                # Reshape to [num_kv_heads, block_size, head_size]
                block_keys = block_keys.permute(0, 2, 1, 3).reshape(
                    self.num_kv_heads, block_size, self.head_size
                )

                # Extract values from cache
                # value_cache shape: [num_blocks, num_kv_heads, head_size, block_size]
                block_values = value_cache[physical_block_id]  # [num_kv_heads, head_size, block_size]
                # Reshape to [num_kv_heads, block_size, head_size]
                block_values = block_values.transpose(1, 2)  # [num_kv_heads, block_size, head_size]

                # Copy to reconstructed tensors
                reconstructed_keys[batch_idx, :, start_token_idx:end_token_idx, :] = \
                    block_keys[:, :num_tokens_in_block, :]
                reconstructed_values[batch_idx, :, start_token_idx:end_token_idx, :] = \
                    block_values[:, :num_tokens_in_block, :]

        # Prepare query, key, value for attention
        # query: [batch_size, num_heads, head_size] -> [batch_size, num_heads, 1, head_size]
        query_attn = query.unsqueeze(2)

        # Handle grouped query attention
        if self.num_kv_heads != self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            # Repeat K/V for each group
            keys_attn = reconstructed_keys.repeat_interleave(num_groups, dim=1)
            values_attn = reconstructed_values.repeat_interleave(num_groups, dim=1)
        else:
            keys_attn = reconstructed_keys
            values_attn = reconstructed_values

        # Create causal mask for each sequence
        # We only attend to tokens up to seq_len for each sequence
        attn_mask = torch.zeros(
            batch_size, 1, 1, max_seq_len,
            dtype=query.dtype, device=query.device
        )
        for batch_idx in range(batch_size):
            seq_len = seq_lens[batch_idx].item()
            if seq_len < max_seq_len:
                attn_mask[batch_idx, :, :, seq_len:] = float('-inf')

        # Compute attention using PyTorch SDPA
        # query_attn: [batch_size, num_heads, 1, head_size]
        # keys_attn: [batch_size, num_heads, max_seq_len, head_size]
        # values_attn: [batch_size, num_heads, max_seq_len, head_size]
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_attn,
            keys_attn,
            values_attn,
            attn_mask=attn_mask,
            scale=self.scale,
        )

        # Reshape output
        # attn_output: [batch_size, num_heads, 1, head_size] -> [batch_size, num_heads, head_size]
        attn_output = attn_output.squeeze(2)

        # Copy to output tensor
        # output expected shape: [batch_size, num_heads * head_size]
        output.copy_(attn_output.reshape(batch_size, self.num_heads * self.head_size))


class MetalAttentionMetadataBuilder(AttentionMetadataBuilder[MetalAttentionMetadata]):
    """Metadata builder for Metal attention backend."""

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.scheduler_config = vllm_config.scheduler_config
        self._init_reorder_batch_threshold(1, False)

        # Allocate temporary buffers for building metadata
        self.seq_start_loc_cpu = torch.zeros(
            vllm_config.scheduler_config.max_num_seqs + 1,
            dtype=torch.int32,
            device="cpu",
        )
        self.seq_start_loc_np = self.seq_start_loc_cpu.numpy()

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MetalAttentionMetadata:
        """Build Metal attention metadata from common attention metadata."""
        num_reqs = common_attn_metadata.num_reqs

        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        seq_lens_np = seq_lens_cpu.numpy()

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        max_prefill_seq_len = (
            seq_lens_np[num_decodes:num_reqs].max().item() if num_prefills > 0 else 0
        )
        max_decode_seq_len = (
            seq_lens_np[:num_decodes].max().item() if num_prefills < num_reqs else 0
        )

        # Determine attention type (use DECODER for standard LLMs)
        # Note: The distinction between prefill and decode is handled by num_prefill_tokens and num_decode_tokens
        attention_type = AttentionType.DECODER

        # Build metadata
        block_tables = None
        seq_lens_tensor = None
        query_start_loc = None

        if num_decodes > 0:
            # Decode metadata
            block_tables = common_attn_metadata.block_table_tensor[:num_decodes]
            # Move seq_lens to MPS device for Metal operations
            seq_lens_tensor = torch.tensor(
                seq_lens_cpu[:num_decodes].numpy(),
                dtype=torch.int32,
                device="mps"
            )

        if num_prefills > 0:
            # Prefill metadata
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            query_start_loc = query_start_loc_cpu[num_decodes : num_reqs + 1]

        # Create Metal attention metadata
        # Calculate attention scale (1 / sqrt(head_size))
        import math
        scale = 1.0 / math.sqrt(self.kv_cache_spec.head_size)

        attn_metadata = MetalAttentionMetadata(
            attention_type=attention_type,
            block_tables=block_tables,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc,
            max_prefill_seq_len=max_prefill_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_kv_heads=self.kv_cache_spec.num_kv_heads,
            scale=scale,
            block_size=self.kv_cache_spec.block_size,
        )

        return attn_metadata


class MetalAttentionBackend(AttentionBackend):
    """Metal attention backend for Apple Silicon."""

    @staticmethod
    def get_name() -> str:
        return "METAL"

    @staticmethod
    def get_impl_cls() -> Type[MetalAttentionImpl]:
        return MetalAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type[MetalAttentionMetadata]:
        return MetalAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type[MetalAttentionMetadataBuilder]:
        return MetalAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> Tuple[int, ...]:
        """Return KV cache shape: [2, num_blocks, block_size * num_kv_heads * head_size]"""
        return (2, num_blocks, block_size * num_kv_heads * head_size)

    @staticmethod
    def get_supported_head_sizes() -> Tuple[int, ...]:
        return (64, 80, 96, 112, 128, 256)

    @staticmethod
    def get_supported_dtypes() -> Tuple[torch.dtype, ...]:
        return (torch.float16, torch.float32)
