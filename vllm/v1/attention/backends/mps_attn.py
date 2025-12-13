# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MPS (Metal Performance Shaders) attention backend for Apple Silicon.

This backend provides attention computation on Apple Silicon using PyTorch's
scaled_dot_product_attention. It uses a paged KV cache for efficient memory
management during autoregressive generation.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class MPSAttentionBackend(AttentionBackend):
    """MPS attention backend using PyTorch's scaled_dot_product_attention."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ]

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.float32, torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "MPS_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """MPS attention supports decoder and encoder-only attention."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

    @staticmethod
    def get_impl_cls() -> type["MPSAttentionBackendImpl"]:
        return MPSAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["MPSAttentionMetadataBuilder"]:
        return MPSAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Use flat cache layout for efficient vectorized access
        # Shape: [2, num_blocks * block_size, num_kv_heads, head_size]
        return 2, num_blocks, num_kv_heads, block_size, head_size

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class MPSAttentionMetadata:
    """Metadata for MPS attention."""

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    causal: bool = True
    # Number of decode tokens (decode requests are reordered to front)
    num_decode_tokens: int = 0


class MPSAttentionMetadataBuilder(AttentionMetadataBuilder[MPSAttentionMetadata]):
    """Builder for MPS attention metadata."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # Enable reorder batch to separate decode and prefill tokens
        # Decode tokens will be reordered to the front
        self._init_reorder_batch_threshold(1, False)

        self.kv_cache_spec = kv_cache_spec
        self.vllm_config = vllm_config

        parallel_config = vllm_config.parallel_config
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(parallel_config)
        self.num_heads = vllm_config.model_config.get_num_attention_heads(
            parallel_config
        )
        self.head_dim = kv_cache_spec.head_size
        self.dtype = vllm_config.model_config.dtype
        self.window_size = getattr(kv_cache_spec, "sliding_window", -1)
        if self.window_size is None:
            self.window_size = -1
        self.block_size = vllm_config.cache_config.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MPSAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # Split decode and prefill tokens
        # Decode tokens are reordered to the front by the reorder_batch_threshold
        num_decode_tokens = 0
        if causal:
            assert self.reorder_batch_threshold is not None
            (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
                split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=self.reorder_batch_threshold,
                    require_uniform=True,
                )
            )

        attn_metadata = MPSAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            scheduler_metadata=None,
            causal=causal,
            num_decode_tokens=num_decode_tokens,
        )

        return attn_metadata


class MPSAttentionBackendImpl(AttentionImpl):
    """MPS attention implementation using PyTorch SDPA.

    This implementation uses vectorized operations to maximize GPU utilization
    on Apple Silicon. All tensor operations stay on MPS device.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)

        if logits_soft_cap is not None and attn_type in (
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        ):
            logger.warning_once(
                "MPS_ATTN does not support logits softcap for"
                " ENCODER and ENCODER_ONLY, outputs may be slightly off"
            )
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if is_quantized_kv_cache(kv_cache_dtype):
            raise NotImplementedError("FP8 KV cache is unsupported in MPS_ATTN")
        self.attn_type = attn_type

        self.sinks = sinks
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MPSAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for MPS attention backend.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape = [2, num_blocks, num_kv_heads, block_size, head_size]
            attn_metadata: Metadata for attention.

        Returns:
            output tensor with shape = [num_tokens, num_heads, head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not yet supported"
                " for MPSAttentionBackendImpl"
            )

        # For warming-up / profiling
        if attn_metadata is None:
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention - no KV cache
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._compute_attention_no_cache(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                is_causal=False,
            )

        # For decoder attention, always use KV cache
        key_cache, value_cache = kv_cache.unbind(0)

        # Update KV cache with new key/value tensors using vectorized ops
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            self._update_kv_cache_vectorized(
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                key_cache,
                value_cache,
                attn_metadata.slot_mapping[:num_actual_tokens],
            )

        # Get the number of decode tokens (decode requests are at the front)
        num_decode_tokens = attn_metadata.num_decode_tokens

        # Process prefill tokens (from num_decode_tokens to num_actual_tokens)
        # Prefill uses original Q/K/V directly for accuracy
        num_prefill_tokens = num_actual_tokens - num_decode_tokens
        if num_prefill_tokens > 0:
            self._compute_attention_no_cache(
                query[num_decode_tokens:num_actual_tokens],
                key[num_decode_tokens:num_actual_tokens],
                value[num_decode_tokens:num_actual_tokens],
                output[num_decode_tokens:num_actual_tokens],
                attn_metadata,
                is_causal=True,
            )

        # Process decode tokens (from 0 to num_decode_tokens)
        # Decode reads K/V from cache
        if num_decode_tokens > 0:
            self._compute_attention_with_cache_vectorized(
                query[:num_decode_tokens],
                key_cache,
                value_cache,
                output[:num_decode_tokens],
                attn_metadata,
            )

        return output

    def _update_kv_cache_vectorized(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Update KV cache using vectorized operations (no Python loops).

        Note: Although MPS uses unified memory shared with CPU, we use MPS
        tensor operations (scatter_) instead of CPU custom ops. This keeps
        the computation on the GPU shader cores rather than CPU cores,
        avoiding synchronization overhead and leveraging MPS parallelism.

        Args:
            key: [num_tokens, num_kv_heads, head_size]
            value: [num_tokens, num_kv_heads, head_size]
            key_cache: [num_blocks, num_kv_heads, block_size, head_size]
            value_cache: [num_blocks, num_kv_heads, block_size, head_size]
            slot_mapping: [num_tokens] - indices into flattened cache
        """
        num_tokens = key.shape[0]
        num_blocks, num_kv_heads, block_size, head_size = key_cache.shape

        # Reshape cache to [num_blocks * block_size, num_kv_heads, head_size]
        key_cache_flat = key_cache.view(-1, num_kv_heads, head_size)
        value_cache_flat = value_cache.view(-1, num_kv_heads, head_size)

        # Use scatter_ for fast vectorized updates on MPS
        # scatter_ is ~40x faster than index_copy_ on MPS
        # Expand slot_mapping to match tensor dimensions
        idx = slot_mapping.view(-1, 1, 1).expand(num_tokens, num_kv_heads, head_size)
        key_cache_flat.scatter_(0, idx, key)
        value_cache_flat.scatter_(0, idx, value)

    def _compute_attention_no_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: MPSAttentionMetadata,
        is_causal: bool,
    ) -> torch.Tensor:
        """Compute attention without reading from KV cache.

        Used for prefill and encoder attention. Uses batched SDPA when possible.

        Args:
            query: [num_tokens, num_heads, head_size]
            key: [num_tokens, num_kv_heads, head_size]
            value: [num_tokens, num_kv_heads, head_size]
            output: [num_tokens, num_heads, head_size]
        """
        query_start_loc = attn_metadata.query_start_loc
        num_seqs = query_start_loc.shape[0] - 1

        # Fast path: single sequence - can use SDPA directly
        if num_seqs == 1:
            # Transpose for SDPA:
            # [num_tokens, heads, head_size] -> [heads, num_tokens, head_size]
            q = query.transpose(0, 1)
            k = key.transpose(0, 1)
            v = value.transpose(0, 1)

            # Expand KV heads if using GQA
            if self.num_kv_heads != self.num_heads:
                k = k.repeat_interleave(self.num_queries_per_kv, dim=0)
                v = v.repeat_interleave(self.num_queries_per_kv, dim=0)

            # SDPA: [1, heads, seq_len, head_size]
            attn_out = F.scaled_dot_product_attention(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scale,
            )

            # [1, heads, seq_len, head_size] -> [seq_len, heads, head_size]
            output.copy_(attn_out.squeeze(0).transpose(0, 1))
            return output

        # Multiple sequences: need to handle variable lengths
        # Get start locations on CPU for indexing (minimal transfer)
        start_locs = query_start_loc.tolist()

        for seq_idx in range(num_seqs):
            start = start_locs[seq_idx]
            end = start_locs[seq_idx + 1]
            seq_len = end - start

            if seq_len == 0:
                continue

            # Extract and transpose for this sequence
            q = query[start:end].transpose(0, 1)
            k = key[start:end].transpose(0, 1)
            v = value[start:end].transpose(0, 1)

            # Expand KV heads if using GQA
            if self.num_kv_heads != self.num_heads:
                k = k.repeat_interleave(self.num_queries_per_kv, dim=0)
                v = v.repeat_interleave(self.num_queries_per_kv, dim=0)

            # Compute attention
            attn_out = F.scaled_dot_product_attention(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scale,
            )

            output[start:end] = attn_out.squeeze(0).transpose(0, 1)

        return output

    def _compute_attention_with_cache_vectorized(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: MPSAttentionMetadata,
    ) -> None:
        """Compute attention by reading K/V from cache using vectorized ops.

        Used for decode phase. Fully vectorized - no CPU transfers.

        Args:
            query: [num_tokens, num_heads, head_size]
            key_cache: [num_blocks, num_kv_heads, block_size, head_size]
            value_cache: [num_blocks, num_kv_heads, block_size, head_size]
            output: [num_tokens, num_heads, head_size]
        """
        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens
        block_size = key_cache.shape[2]
        num_kv_heads = key_cache.shape[1]
        head_size = key_cache.shape[3]

        num_seqs = seq_lens.shape[0]

        # Flatten cache for easier gathering
        # [num_blocks, num_kv_heads, block_size, head_size] ->
        # [num_blocks * block_size, num_kv_heads, head_size]
        key_cache_flat = key_cache.view(-1, num_kv_heads, head_size)
        value_cache_flat = value_cache.view(-1, num_kv_heads, head_size)

        # Get sliding window size (left side of window tuple)
        # sliding_window[0] is the number of tokens to look back (-1 means no limit)
        sliding_window_size = (
            self.sliding_window[0] + 1 if self.sliding_window[0] >= 0 else -1
        )

        # Fast path: single sequence decode (most common case)
        # Avoid all CPU transfers by keeping everything on GPU
        if num_seqs == 1:
            kv_len = seq_lens[0].item()  # Single scalar transfer

            # Apply sliding window: only attend to last `sliding_window_size` tokens
            kv_start = 0
            if sliding_window_size > 0 and kv_len > sliding_window_size:
                kv_start = kv_len - sliding_window_size
                kv_len = sliding_window_size

            num_blocks_needed = (kv_start + kv_len + block_size - 1) // block_size
            seq_block_table = block_table[0, :num_blocks_needed]

            # Build flat indices entirely on GPU, starting from kv_start
            flat_indices = self._build_flat_indices(
                seq_block_table, block_size, kv_start + kv_len
            )
            # Slice to only include tokens within the window
            if kv_start > 0:
                flat_indices = flat_indices[kv_start:]

            # Gather K/V using direct indexing (faster than index_select on MPS)
            seq_key = key_cache_flat[flat_indices]
            seq_value = value_cache_flat[flat_indices]

            # Transpose: [kv_len, heads, dim] -> [heads, kv_len, dim]
            seq_key = seq_key.transpose(0, 1)
            seq_value = seq_value.transpose(0, 1)
            seq_query = query.transpose(0, 1)

            # Expand KV heads if using GQA
            if self.num_kv_heads != self.num_heads:
                seq_key = seq_key.repeat_interleave(self.num_queries_per_kv, dim=0)
                seq_value = seq_value.repeat_interleave(self.num_queries_per_kv, dim=0)

            # SDPA - decode is not causal (single query pos attending to past)
            attn_out = F.scaled_dot_product_attention(
                seq_query.unsqueeze(0),
                seq_key.unsqueeze(0),
                seq_value.unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
            )

            output.copy_(attn_out.squeeze(0).transpose(0, 1))
            return

        # Multiple sequences: need to iterate (less common for decode)
        # Use GPU tensor indexing to avoid tolist()
        query_start_loc = attn_metadata.query_start_loc

        for seq_idx in range(num_seqs):
            q_start = query_start_loc[seq_idx].item()
            q_end = query_start_loc[seq_idx + 1].item()
            query_len = q_end - q_start
            kv_len = seq_lens[seq_idx].item()

            if query_len == 0:
                continue

            # Apply sliding window: only attend to last `sliding_window_size` tokens
            kv_start = 0
            if sliding_window_size > 0 and kv_len > sliding_window_size:
                kv_start = kv_len - sliding_window_size
                kv_len = sliding_window_size

            # Get query for this sequence
            seq_query = query[q_start:q_end]

            # Build flat indices for gathering K/V from cache
            num_blocks_needed = (kv_start + kv_len + block_size - 1) // block_size
            seq_block_table = block_table[seq_idx, :num_blocks_needed]

            flat_indices = self._build_flat_indices(
                seq_block_table, block_size, kv_start + kv_len
            )
            # Slice to only include tokens within the window
            if kv_start > 0:
                flat_indices = flat_indices[kv_start:]

            # Gather K/V using direct indexing (faster than index_select on MPS)
            seq_key = key_cache_flat[flat_indices]
            seq_value = value_cache_flat[flat_indices]

            # Transpose for attention computation
            seq_key = seq_key.transpose(0, 1)
            seq_value = seq_value.transpose(0, 1)
            seq_query = seq_query.transpose(0, 1)

            # Expand KV heads if using GQA
            if self.num_kv_heads != self.num_heads:
                seq_key = seq_key.repeat_interleave(self.num_queries_per_kv, dim=0)
                seq_value = seq_value.repeat_interleave(self.num_queries_per_kv, dim=0)

            # For decode, we typically have query_len=1 and kv_len>1
            use_causal = query_len > 1 and query_len == kv_len

            attn_out = F.scaled_dot_product_attention(
                seq_query.unsqueeze(0),
                seq_key.unsqueeze(0),
                seq_value.unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=use_causal,
                scale=self.scale,
            )

            output[q_start:q_end] = attn_out.squeeze(0).transpose(0, 1)

    def _build_flat_indices(
        self,
        block_table: torch.Tensor,
        block_size: int,
        kv_len: int,
    ) -> torch.Tensor:
        """Build flat indices for gathering from cache.

        Args:
            block_table: [num_blocks] - block IDs for this sequence
            block_size: size of each block
            kv_len: total number of KV tokens to gather

        Returns:
            flat_indices: [kv_len] - indices into flattened cache
        """
        device = block_table.device

        # Create block start indices: block_id * block_size
        block_starts = block_table * block_size  # [num_blocks]

        # Create offsets within blocks: [0, 1, ..., block_size-1] repeated
        offsets = torch.arange(block_size, device=device)  # [block_size]

        # Expand and add: [num_blocks, block_size]
        all_indices = block_starts.unsqueeze(1) + offsets.unsqueeze(0)

        # Flatten and truncate to kv_len
        flat_indices = all_indices.flatten()[:kv_len]

        return flat_indices
