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

from vllm.attention import Attention
from vllm.attention.utils.fa_utils import is_flash_attn_varlen_func_available

if is_flash_attn_varlen_func_available():
    from vllm.attention.utils.fa_utils import (
        flash_attn_varlen_func,
    )

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import FullAttentionSpec


class TrainableFlashAttention(nn.Module, AttentionLayerBase):
    """
    Training-compatible flash attention module using vLLM's optimized kernels.

    This module wraps vLLM's flash attention forward pass and adds backward
    support for training scenarios like reinforcement learning and fine-tuning.

    Supports both fused QKV projections (efficient) and separate projections
    (for compatibility with TorchTitan models during module surgery).

    Example:
        ```python
        # Create attention module (fused, efficient)
        attn = TrainableFlashAttention(hidden_size=768, num_heads=12, dropout=0.1)

        # Create TorchTitan-compatible module (separate projections)
        attn = TrainableFlashAttention(
            hidden_size=768,
            num_heads=12,
            use_fused_qkv=False,  # Separate wq/wk/wv for compatibility
            use_qk_norm=True,  # QK normalization like Qwen3
        )

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
        use_fused_qkv: Use fused QKV projection (efficient). Set False for
            TorchTitan compatibility. Defaults to True.
        use_qk_norm: Apply RMSNorm to Q and K after projection (Qwen3 style).
            Defaults to False.
        norm_eps: Epsilon for RMSNorm if use_qk_norm=True. Defaults to 1e-6.
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
        use_fused_qkv: bool = True,
        use_qk_norm: bool = False,
        norm_eps: float = 1e-6,
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
        self.use_fused_qkv = use_fused_qkv
        self.use_qk_norm = use_qk_norm

        if scale is None:
            self.scale = self.head_dim**-0.5
        else:
            self.scale = scale

        # TODO(future optimization): Always use fused QKV for efficiency
        # Currently supporting separate projections for TorchTitan compatibility
        # during module surgery. Once we have weight conversion utilities,
        # we should always initialize with fused weights and convert TorchTitan
        # weights (wq, wk, wv) -> fused (qkv) during load_weights().
        # This will give us the best of both worlds: compatibility + efficiency.

        if use_fused_qkv:
            # Fused QKV projection (efficient - single matmul)
            self.qkv = nn.Linear(
                hidden_size,
                (num_heads + 2 * self.num_kv_heads) * self.head_dim,
                bias=False,
            )
        else:
            # Separate projections (TorchTitan compatibility)
            self.wq = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
            self.wk = nn.Linear(
                hidden_size, self.num_kv_heads * self.head_dim, bias=False
            )
            self.wv = nn.Linear(
                hidden_size, self.num_kv_heads * self.head_dim, bias=False
            )

        # Output projection (naming convention follows use_fused_qkv)
        if use_fused_qkv:
            self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        else:
            # TorchTitan uses 'wo' naming
            self.wo = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # Optional QK normalization (for Qwen3 and similar models)
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # Create vLLM Attention layer to handle KV cache automatically
        # This delegates all the complex KV cache logic to vLLM
        try:
            from vllm.config import get_current_vllm_config

            config = get_current_vllm_config()
            cache_config = (
                config.cache_config if hasattr(config, "cache_config") else None
            )

            # Generate unique prefix for this attention layer
            # vLLM expects format "layers.X" for layer index extraction
            layer_idx = next(TrainableFlashAttention._layer_counter)
            prefix = f"layers.{layer_idx}"

            self.vllm_attn = Attention(
                num_heads=num_heads,
                head_size=self.head_dim,
                scale=self.scale,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=None,
                prefix=prefix,
            )
        except (ImportError, RuntimeError, AttributeError):
            # Not in vLLM context - attention layer not needed
            self.vllm_attn = None

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
        freqs_cis: torch.Tensor
        | None = None,  # RoPE frequencies (TorchTitan compatibility)
        attention_mask: torch.Tensor | None = None,
        **kwargs,  # Accept any additional vLLM-specific kwargs
    ) -> torch.Tensor:
        """
        Forward pass with flash attention.

        Supports both training (full sequences) and vLLM inference (with KV cache).

        Args:
            hidden_states: Input tensor of shape [total_tokens, hidden_size]
                          or [batch, seq_len, hidden_size]
            freqs_cis: RoPE frequencies (for TorchTitan compatibility, currently unused)
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

        # Project to Q, K, V (supports both fused and separate modes)
        if self.use_fused_qkv:
            # Fused projection path (efficient)
            qkv = self.qkv(hidden_states)

            # Split into Q, K, V
            # qkv shape: [total_tokens, (num_heads + 2*num_kv_heads) * head_dim]
            q_size = self.num_heads * self.head_dim
            k_size = self.num_kv_heads * self.head_dim
            v_size = self.num_kv_heads * self.head_dim

            q = qkv[:, :q_size]
            k = qkv[:, q_size : q_size + k_size]
            v = qkv[:, q_size + k_size : q_size + k_size + v_size]
        else:
            # Separate projections (TorchTitan compatibility)
            q = self.wq(hidden_states)
            k = self.wk(hidden_states)
            v = self.wv(hidden_states)

        # Reshape for attention: [total_tokens, num_heads, head_dim]
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(total_tokens, self.num_kv_heads, self.head_dim)

        # Optional QK normalization (Qwen3 style)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # DEBUG: Log layer 0 values to compare with TorchTitan
        is_layer_0 = not hasattr(self, "_debug_logged")
        if is_layer_0 and total_tokens > 1 and total_tokens < 100:  # Skip warmup
            self._debug_logged = True
            print("\n[VLLM ATT DEBUG] Layer 0 - Input")
            print(f"  hidden_states.shape: {hidden_states.shape}")
            print(f"  total_tokens: {total_tokens}")
            print(f"  q (before RoPE)[0,0,:5]: {q[0, 0, :5]}")
            print(f"  k (before RoPE)[0,0,:5]: {k[0, 0, :5]}")

        # Apply RoPE if freqs_cis is provided (TorchTitan integration)
        if freqs_cis is not None:
            # Get positions from vLLM forward context
            try:
                from vllm.forward_context import get_forward_context

                forward_ctx = get_forward_context()

                # Try to get positions from custom attribute set by wrapper
                positions = None
                if hasattr(forward_ctx, "_torchtitan_positions"):
                    positions = forward_ctx._torchtitan_positions
                    # Debug: Log positions during generation, not just warmup
                    unique_pos = torch.unique(positions[: min(100, len(positions))])
                    # Skip warmup with all zeros
                    if (len(unique_pos) > 1 or unique_pos[0] != 0) and not hasattr(
                        self, "_rope_gen_debug"
                    ):
                        self._rope_gen_debug = True
                        print(f"\n[ROPE GEN] Got real positions: {unique_pos[:20]}")
                        print(
                            f"[ROPE GEN] total_tokens: {total_tokens}, "
                            f"freqs_cis.shape: {freqs_cis.shape}"
                        )
                else:
                    # Fallback to sequential positions
                    positions = torch.arange(total_tokens, device=q.device)

                # Index rope_cache by positions
                # freqs_cis shape: [max_seq_len, head_dim*2] (cos and sin concatenated)
                positions_flat = positions.flatten()

                # Ensure positions are within bounds
                max_pos = freqs_cis.shape[0] - 1
                positions_flat = torch.clamp(positions_flat[:total_tokens], 0, max_pos)

                cos_sin = freqs_cis.index_select(0, positions_flat)

                # Split into cos and sin
                head_dim = self.head_dim
                cos = cos_sin[..., :head_dim]
                sin = cos_sin[..., head_dim:]

                # Apply rotary embedding (same as TorchTitan's apply_rotary_emb)
                def rotate_half(x):
                    """Rotates half the hidden dims of the input."""
                    x1 = x[..., : x.shape[-1] // 2]
                    x2 = x[..., x.shape[-1] // 2 :]
                    return torch.cat((-x2, x1), dim=-1)

                # Reshape cos/sin for broadcast: [total_tokens, 1, head_dim]
                cos = cos.unsqueeze(1).to(dtype=q.dtype, device=q.device)
                sin = sin.unsqueeze(1).to(dtype=q.dtype, device=q.device)

                # Apply rotation
                q = (q * cos) + (rotate_half(q) * sin)
                k = (k * cos) + (rotate_half(k) * sin)

                # DEBUG: Log after RoPE
                if is_layer_0 and total_tokens > 1 and total_tokens < 100:
                    print(f"  RoPE applied with positions: {unique_pos[:10]}")
                    print(f"  freqs_cis.shape: {freqs_cis.shape}")
                    print(f"  q (after RoPE)[0,0,:5]: {q[0, 0, :5]}")
                    print(f"  k (after RoPE)[0,0,:5]: {k[0, 0, :5]}")

            except (ImportError, AttributeError, IndexError, AssertionError) as e:
                # If we can't get positions, fall through without RoPE
                # This will happen in pure training mode
                if not hasattr(self, "_rope_error"):
                    self._rope_error = True
                    print(f"\n[ROPE DEBUG] Error applying RoPE: {e}")
                pass

        # Delegate to vLLM's Attention layer if available
        # (handles KV cache automatically)
        if self.vllm_attn is not None and not self.training:
            # Let vLLM handle all KV cache logic
            # vllm_attn expects q,k,v in shape [total_tokens, num_heads*head_dim]
            # or [total_tokens, num_heads, head_dim]
            attn_output = self.vllm_attn(q, k, v)
            # vllm_attn returns [total_tokens, num_heads * head_dim]
        else:
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

                attn_output = attn_output.transpose(
                    1, 2
                )  # [batch, seq_len, heads, dim]
                attn_output = attn_output.reshape(
                    total_tokens, self.num_heads, self.head_dim
                )

        # Flatten heads and project output
        attn_output = attn_output.reshape(total_tokens, -1)
        if self.use_fused_qkv:
            output = self.o_proj(attn_output)
        else:
            output = self.wo(attn_output)

        # DEBUG: Log attention output for layer 0
        if is_layer_0 and total_tokens > 1 and total_tokens < 100:
            print(f"  attn_output (before o_proj)[0,:5]: {attn_output[0, :5]}")
            print(f"  output (after o_proj)[0,:5]: {output[0, :5]}")

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
