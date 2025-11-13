# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Trainable Flash Attention module for research and fine-tuning.

This module provides a training-compatible wrapper around vLLM's optimized
flash attention implementation, enabling backpropagation for RL and fine-tuning
use cases.
"""

import torch
import torch.nn as nn

from vllm.attention.utils.fa_utils import is_flash_attn_varlen_func_available

if is_flash_attn_varlen_func_available():
    from vllm.attention.utils.fa_utils import flash_attn_varlen_func

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with flash attention.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask (not yet fully supported)

        Returns:
            output: Attention output of shape [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        qkv = self.qkv(hidden_states)

        # Split into Q, K, V
        # qkv shape: [batch, seq_len, (num_heads + 2*num_kv_heads) * head_dim]
        q_size = self.num_heads * self.head_dim
        k_size = self.num_kv_heads * self.head_dim
        v_size = self.num_kv_heads * self.head_dim

        q = qkv[:, :, :q_size]
        k = qkv[:, :, q_size : q_size + k_size]
        v = qkv[:, :, q_size + k_size : q_size + k_size + v_size]

        # Reshape for attention: [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Use flash attention in training mode WITH CUDA
        if self.training and hidden_states.is_cuda:
            # Flatten batch dimension for varlen format
            # flash_attn_varlen_func expects [total_tokens, num_heads, head_dim]
            q_flat = q.reshape(batch_size * seq_len, self.num_heads, self.head_dim)
            k_flat = k.reshape(batch_size * seq_len, self.num_kv_heads, self.head_dim)
            v_flat = v.reshape(batch_size * seq_len, self.num_kv_heads, self.head_dim)

            # Create cumulative sequence lengths for varlen format
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=hidden_states.device,
            )

            # Call flash attention varlen func
            # This supports backprop automatically via autograd
            # Force FA v3 to avoid PTX compilation issues
            attn_output = flash_attn_varlen_func(
                q=q_flat,
                k=k_flat,
                v=v_flat,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=self.causal,
                dropout_p=self.dropout if self.training else 0.0,
                fa_version=3,
            )

            # Reshape back: [batch, seq_len, num_heads, head_dim]
            attn_output = attn_output.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            )

        else:
            # CPU or evaluation mode: use PyTorch's SDPA as fallback
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

            attn_output = attn_output.transpose(
                1, 2
            )  # Back to [batch, seq_len, heads, dim]

        # Flatten heads and project output
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output

    def get_attn_backend(self):
        """
        Get the attention backend for this layer.

        For TrainableFlashAttention, we don't use a specific vLLM backend
        since we implement attention directly. Return None to indicate
        this layer manages its own attention computation.
        """
        # Import here to avoid circular dependency
        from vllm.attention.backends.flash_attn import FlashAttentionBackend

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
