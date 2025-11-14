# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Trainable Flash Attention module for research and fine-tuning.

This module provides a training-compatible wrapper around vLLM's optimized
flash attention implementation, enabling backpropagation for RL and fine-tuning
use cases.
"""

import itertools

import torch
import torch.nn as nn

from vllm.attention.utils.fa_utils import is_flash_attn_varlen_func_available

if is_flash_attn_varlen_func_available():
    from vllm.attention.utils.fa_utils import (
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import FullAttentionSpec


class TrainableFlashAttention(nn.Module, AttentionLayerBase):
    """
    Training-compatible flash attention module using vLLM's optimized kernels.

    This module wraps vLLM's flash attention forward pass and adds backward
    support for training scenarios like reinforcement learning and fine-tuning.

    Example:
        ```python
        # Create attention module
        attn = TrainableFlashAttention(hidden_size=768, num_heads=12, dropout=0.1)

        # Use in training
        attn.train()
        hidden_states = torch.randn(2, 16, 768, requires_grad=True)
        output = attn(hidden_states)

        # Backward pass works
        loss = output.sum()
        loss.backward()
        ```

    Args:
        hidden_size: Hidden dimension of the model
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA). Defaults to num_heads
        head_dim: Dimension per head. Defaults to hidden_size // num_heads
        dropout: Dropout probability during training. Defaults to 0.0
        scale: Attention scale factor. Defaults to 1/sqrt(head_dim)
        causal: Whether to use causal masking. Defaults to True
    """

    # Class variable for auto-generating unique layer names (thread-safe)
    _layer_counter = itertools.count()

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        scale: float | None = None,
        causal: bool = True,
    ):
        super().__init__()

        if not is_flash_attn_varlen_func_available():
            raise RuntimeError(
                "Flash attention is not available. "
                "Please install flash-attn: pip install flash-attn"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.dropout = dropout
        self.causal = causal

        if scale is None:
            self.scale = self.head_dim**-0.5
        else:
            self.scale = scale

        # QKV projection (column-wise output, split across heads)
        self.qkv = nn.Linear(
            hidden_size, (num_heads + 2 * self.num_kv_heads) * self.head_dim, bias=False
        )

        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # KV cache - will be populated by vLLM during model loading
        # For V1 engine, this is a list[torch.Tensor] indexed by virtual_engine
        self.kv_cache: list[torch.Tensor] | None = None

        # Auto-register for vLLM KV cache if in vLLM context
        self._auto_register_for_kv_cache()

    def _auto_register_for_kv_cache(self):
        """Automatically register this layer for vLLM KV cache allocation.

        This is called during __init__ and will register the layer if we're in
        a vLLM context. If not in vLLM context (e.g., pure PyTorch training),
        this silently does nothing.
        """
        # Initialize layer_name attribute
        self.layer_name: str | None = None

        try:
            from vllm.config import get_current_vllm_config

            config = get_current_vllm_config()
            compilation_config = config.compilation_config

            # Generate unique layer name using class counter
            # Format: "layers.{index}" for compatibility with extract_layer_index()
            layer_name = f"layers.{next(TrainableFlashAttention._layer_counter)}"

            # Register this layer in static forward context
            if layer_name in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {layer_name}")
            compilation_config.static_forward_context[layer_name] = self
            self.layer_name = layer_name

        except (ImportError, RuntimeError, AttributeError):
            # Not in vLLM context - this is fine!
            # Layer will work normally for training/inference without vLLM
            pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,  # Accept any additional vLLM-specific kwargs
    ) -> torch.Tensor:
        """
        Forward pass with flash attention.

        Supports both training (full sequences) and vLLM inference (with KV cache).

        Args:
            hidden_states: Input tensor of shape [total_tokens, hidden_size]
                          or [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask (not yet fully supported)
            **kwargs: Additional vLLM-specific kwargs (intermediate_tensors, etc.)

        Returns:
            output: Attention output of same shape as hidden_states
        """
        # Handle both batched [batch, seq, hidden] and flattened [total_tokens, hidden]
        input_is_batched = hidden_states.dim() == 3
        if input_is_batched:
            original_batch_size, original_seq_len, _ = hidden_states.shape
            hidden_states = hidden_states.view(-1, self.hidden_size)
        else:
            original_batch_size = None
            original_seq_len = None

        total_tokens = hidden_states.shape[0]

        # Project to Q, K, V
        qkv = self.qkv(hidden_states)

        # Split into Q, K, V
        # qkv shape: [total_tokens, (num_heads + 2*num_kv_heads) * head_dim]
        q_size = self.num_heads * self.head_dim
        k_size = self.num_kv_heads * self.head_dim
        v_size = self.num_kv_heads * self.head_dim

        q = qkv[:, :q_size]
        k = qkv[:, q_size : q_size + k_size]
        v = qkv[:, q_size + k_size : q_size + k_size + v_size]

        # Reshape for attention: [total_tokens, num_heads, head_dim]
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(total_tokens, self.num_kv_heads, self.head_dim)

        # Try vLLM KV-cached path first (for inference performance)
        if not self.training and self.kv_cache is not None:
            try:
                from vllm.forward_context import get_forward_context

                forward_ctx = get_forward_context()

                # Get attention metadata for this layer
                # V1 engine: attn_metadata is a dict[layer_name, metadata]
                # Get the first available metadata
                # (all layers share same metadata for this model)
                if (
                    isinstance(forward_ctx.attn_metadata, dict)
                    and len(forward_ctx.attn_metadata) > 0
                ):
                    attn_meta = next(iter(forward_ctx.attn_metadata.values()))
                    kv_cache = self.kv_cache[forward_ctx.virtual_engine]

                    # Cache K and V using vLLM's caching function
                    # For non-quantized cache, k_scale and v_scale are 1.0
                    from vllm.config import get_current_vllm_config

                    current_config = get_current_vllm_config()
                    kv_cache_dtype = current_config.cache_config.cache_dtype

                    reshape_and_cache_flash(
                        k,
                        v,
                        kv_cache[0],  # key cache
                        kv_cache[1],  # value cache
                        attn_meta.slot_mapping,
                        kv_cache_dtype,
                        k_scale=torch.tensor(1.0, dtype=torch.float32, device=k.device),
                        v_scale=torch.tensor(1.0, dtype=torch.float32, device=v.device),
                    )

                    # Use flash attention with KV cache
                    attn_output = torch.ops.vllm.flash_attn_varlen_func(
                        q=q,
                        k=kv_cache[0],  # Cached keys
                        v=kv_cache[1],  # Cached values
                        cu_seqlens_q=attn_meta.query_start_loc,
                        # For self-attention
                        cu_seqlens_k=attn_meta.query_start_loc,
                        max_seqlen_q=attn_meta.max_query_len,
                        max_seqlen_k=attn_meta.max_seq_len,
                        softmax_scale=self.scale,
                        causal=self.causal,
                        block_table=attn_meta.block_table,
                    )

                    # Flatten and project output
                    attn_output = attn_output.reshape(total_tokens, -1)
                    return self.o_proj(attn_output)
            except (ImportError, AssertionError, AttributeError):
                # Fall through to regular attention if vLLM context not available
                pass

        # Training mode or fallback: use regular flash attention (no KV cache)
        if not self.training and hidden_states.is_cuda:
            # Inference without KV cache: use flash attention varlen
            # Create simple cu_seqlens for single sequence
            cu_seqlens_q = torch.tensor(
                [0, total_tokens],
                dtype=torch.int32,
                device=hidden_states.device,
            )

            attn_output = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_q,
                max_seqlen_q=total_tokens,
                max_seqlen_k=total_tokens,
                softmax_scale=self.scale,
                causal=self.causal,
                dropout_p=0.0,
                fa_version=3,
            )
        else:
            # Training mode with CPU: use PyTorch SDPA
            batch_size = 1
            seq_len = total_tokens

            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

            q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Handle GQA by repeating k, v if needed
            if self.num_kv_heads != self.num_heads:
                num_repeats = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(num_repeats, dim=1)
                v = v.repeat_interleave(num_repeats, dim=1)

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal and attention_mask is None,
            )

            attn_output = attn_output.transpose(1, 2)  # [batch, seq_len, heads, dim]
            attn_output = attn_output.reshape(
                total_tokens, self.num_heads, self.head_dim
            )

        # Flatten heads and project output
        attn_output = attn_output.reshape(total_tokens, -1)
        output = self.o_proj(attn_output)

        # Restore original shape if input was batched
        if input_is_batched:
            output = output.view(
                original_batch_size, original_seq_len, self.hidden_size
            )

        return output

    def get_attn_backend(self):
        """
        Get the attention backend for this layer.

        For TrainableFlashAttention, we don't use a specific vLLM backend
        since we implement attention directly. Return None to indicate
        this layer manages its own attention computation.
        """
        # Import here to avoid circular dependency
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

        return FlashAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> FullAttentionSpec:
        """
        Return KV cache specification for V1 engine integration.

        This allows TrainableFlashAttention to work with vLLM's V1 engine
        by providing the necessary KV cache metadata.
        """
        block_size = vllm_config.cache_config.block_size
        # Determine the dtype for KV cache
        kv_cache_dtype = vllm_config.cache_config.cache_dtype
        if kv_cache_dtype == "auto":
            kv_cache_dtype = vllm_config.model_config.dtype

        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=kv_cache_dtype,
        )


__all__ = ["TrainableFlashAttention"]
