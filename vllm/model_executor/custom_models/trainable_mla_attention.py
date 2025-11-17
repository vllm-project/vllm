# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Trainable Multi-Head Latent Attention (MLA) for DeepSeek V3.

This module implements the MLA architecture used in DeepSeek V3, which uses:
1. Low-rank compression for Q and KV projections
2. Split Q/K into RoPE and non-RoPE parts
3. Shared K_PE (RoPE-encoded key) across all heads

Reference: https://github.com/deepseek-ai/DeepSeek-V3
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MLAConfig:
    """Configuration for Multi-Head Latent Attention.

    Args:
        hidden_size: Hidden dimension of the model
        num_heads: Number of attention heads
        q_lora_rank: LoRA rank for query projection. If 0, use direct projection.
        kv_lora_rank: LoRA rank for key-value projection
        qk_nope_head_dim: Dimension of Q/K without positional encoding
        qk_rope_head_dim: Dimension of Q/K with RoPE
        v_head_dim: Dimension of value projection per head
        norm_eps: Epsilon for RMSNorm layers
        dropout: Dropout probability during training
        scale: Attention scale factor. If None, defaults to 1/sqrt(qk_head_dim)
        causal: Whether to use causal masking
    """

    hidden_size: int
    num_heads: int
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    norm_eps: float = 1e-5
    dropout: float = 0.0
    scale: float | None = None
    causal: bool = True

    @property
    def qk_head_dim(self) -> int:
        """Total Q/K head dimension."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim


class TrainableMLA(nn.Module):
    """
    Training-compatible Multi-Head Latent Attention (MLA).

    This implements DeepSeek V3's MLA architecture:
    - Low-rank compression with intermediate RMSNorm
    - Split Q/K into RoPE and non-RoPE parts
    - Shared K_PE across all attention heads (memory efficient!)

    Example:
        ```python
        config = MLAConfig(
            hidden_size=2048,
            num_heads=16,
            q_lora_rank=512,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
        mla = TrainableMLA(config)

        # Forward pass
        hidden_states = torch.randn(2, 16, 2048)
        freqs_cis = torch.randn(16, config.qk_rope_head_dim // 2)
        output = mla(hidden_states, freqs_cis=freqs_cis)
        ```
    """

    def __init__(self, config: MLAConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.dropout = config.dropout
        self.causal = config.causal

        # Attention scale
        if config.scale is None:
            self.scale = self.qk_head_dim**-0.5
        else:
            self.scale = config.scale

        # Query projection
        if self.q_lora_rank == 0:
            # Direct projection without LoRA
            self.wq = nn.Linear(
                self.hidden_size, self.num_heads * self.qk_head_dim, bias=False
            )
            self.q_norm = None
        else:
            # Low-rank projection with intermediate norm
            self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=config.norm_eps)
            self.wq_b = nn.Linear(
                self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )

        # Key-value projection (always uses LoRA)
        # Note: wkv_a outputs kv_lora_rank + qk_rope_head_dim
        # The extra qk_rope_head_dim is for the shared K_PE
        self.wkv_a = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=config.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection
        self.wo = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )

        # Create vLLM MLAAttention for KV cache + optimized attention
        # We'll initialize it in a lazy way since we need vLLM config
        self.vllm_mla_attn = None
        self._init_vllm_mla_attention()

    def _init_vllm_mla_attention(self):
        """Initialize vLLM's MLAAttention for KV cache and optimized attention."""
        try:
            from vllm.attention.layer import MLAAttention
            from vllm.config import get_current_vllm_config
            from vllm.model_executor.layers.linear import ColumnParallelLinear

            # Get vLLM config if available
            try:
                vllm_config = get_current_vllm_config()
                cache_config = vllm_config.cache_config
                quant_config = vllm_config.quant_config
            except (RuntimeError, AttributeError):
                # Not in vLLM context - skip MLAAttention initialization
                return

            # Generate unique layer name for KV cache registration
            import itertools

            if not hasattr(TrainableMLA, "_layer_counter"):
                TrainableMLA._layer_counter = itertools.count()

            layer_name = f"layers.{next(TrainableMLA._layer_counter)}.attention"

            # Wrap wkv_b in ColumnParallelLinear (vLLM's parallel layer)
            # This allows vLLM to handle TP sharding properly
            kv_b_proj = ColumnParallelLinear(
                input_size=self.kv_lora_rank,
                output_size=self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
                quant_config=quant_config,
            )
            # Copy weights from our regular Linear layer
            kv_b_proj.weight.data.copy_(self.wkv_b.weight.data)

            # Create vLLM's MLAAttention
            self.vllm_mla_attn = MLAAttention(
                num_heads=self.num_heads,
                scale=self.scale,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                q_lora_rank=self.q_lora_rank if self.q_lora_rank > 0 else None,
                kv_lora_rank=self.kv_lora_rank,
                kv_b_proj=kv_b_proj,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=layer_name,
            )

            print(f"  ✓ Created vLLM MLAAttention for {layer_name}")

        except (ImportError, RuntimeError, AttributeError, AssertionError) as e:
            # vLLM not available or not in vLLM context - use manual implementation
            print(f"  ⚠️  Could not create vLLM MLAAttention: {e}")
            pass

    def _auto_register_for_kv_cache(self):
        """Automatically register this layer for vLLM KV cache allocation.

        This is called during __init__ and will register the layer if we're in
        a vLLM context. If not in vLLM context, this silently does nothing.
        """
        self.layer_name: str | None = None

        try:
            from vllm.config import get_current_vllm_config

            config = get_current_vllm_config()
            compilation_config = config.compilation_config

            # Generate unique layer name
            import itertools

            if not hasattr(TrainableMLA, "_layer_counter"):
                TrainableMLA._layer_counter = itertools.count()

            layer_name = f"layers.{next(TrainableMLA._layer_counter)}"

            # Register this layer in static forward context
            if layer_name in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {layer_name}")
            compilation_config.static_forward_context[layer_name] = self
            self.layer_name = layer_name

        except (ImportError, RuntimeError, AttributeError):
            # Not in vLLM context - this is fine!
            pass

    def apply_rotary_emb(
        self, x: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rotary positional embeddings to the input tensor.

        Args:
            x: Input tensor [total_tokens, heads, qk_rope_head_dim]
                or [bsz, seq_len, heads, qk_rope_head_dim]
            freqs_cis: Precomputed complex exponentials
                [max_seq_len, qk_rope_head_dim//2] (complex64/complex128)

        Returns:
            Tensor with rotary embeddings applied
        """
        # Determine if batched or flattened
        # [bsz, seq_len, heads, dim] or [total_tokens, heads, dim]
        seq_len = x.size(1) if x.dim() == 4 else x.size(0)

        # Slice freqs_cis to actual sequence length
        # freqs_cis is complex: [max_seq_len, qk_rope_head_dim//2]
        freqs = freqs_cis[:seq_len]  # [seq_len, qk_rope_head_dim//2]

        # Convert x to complex for rotation
        # x: [..., qk_rope_head_dim] -> [..., qk_rope_head_dim//2] complex
        x_complex = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2)
        )  # [..., qk_rope_head_dim//2]

        # Reshape freqs for broadcasting
        # Batched: [bsz, seq_len, heads, dim] -> freqs [1, seq_len, 1, dim//2]
        # Flattened: [total_tokens, heads, dim] -> freqs [seq_len, 1, dim//2]
        freqs = freqs.unsqueeze(0).unsqueeze(2) if x.dim() == 4 else freqs.unsqueeze(1)

        # Apply rotation: multiply by complex exponential
        x_rotated = x_complex * freqs

        # Convert back to real
        x_out = torch.view_as_real(x_rotated).flatten(-2)  # [..., qk_rope_head_dim]

        return x_out.to(x.dtype)

    def apply_rotary_emb_with_cos_sin(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rotary positional embeddings using cos and sin directly.

        Args:
            x: Input tensor [total_tokens, heads, qk_rope_head_dim]
            cos: Cosine values [total_tokens, qk_rope_head_dim//2]
            sin: Sine values [total_tokens, qk_rope_head_dim//2]

        Returns:
            Tensor with rotary embeddings applied
                [total_tokens, heads, qk_rope_head_dim]
        """
        # Expand cos/sin to match x's head dimension
        # cos/sin: [total_tokens, qk_rope_head_dim//2]
        #       -> [total_tokens, 1, qk_rope_head_dim//2]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # Repeat to full dimension: [total_tokens, 1, qk_rope_head_dim]
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)

        # Apply rotation using rotate_half (avoids complex operations)
        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # Apply RoPE: x_rotated = x * cos + rotate_half(x) * sin
        x_out = (x * cos) + (rotate_half(x) * sin)

        return x_out.to(x.dtype)

    def apply_rotary_emb_indexed(
        self, x: torch.Tensor, freqs_for_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        DEPRECATED: Use apply_rotary_emb_with_cos_sin instead.

        Apply rotary positional embeddings using pre-indexed frequencies.

        Args:
            x: Input tensor [total_tokens, heads, qk_rope_head_dim]
            freqs_for_tokens: Pre-indexed frequencies - complex or real format

        Returns:
            Tensor with rotary embeddings applied
                [total_tokens, heads, qk_rope_head_dim]
        """
        # Check if freqs_for_tokens is complex or already split into cos/sin
        if freqs_for_tokens.is_complex():
            # Extract cos and sin from complex frequencies
            # freqs_for_tokens is complex exponentials: e^(i*theta)
            # = cos(theta) + i*sin(theta)
            cos = freqs_for_tokens.real  # [total_tokens, qk_rope_head_dim//2]
            sin = freqs_for_tokens.imag  # [total_tokens, qk_rope_head_dim//2]
        elif freqs_for_tokens.shape[-1] == x.shape[-1] // 2:
            # Format: [total_tokens, qk_rope_head_dim//2]
            # complex stored as real
            # This happens after index_select on complex tensor
            # The tensor is complex data stored in real format
            # We need to extract real and imaginary parts
            # Actually this shouldn't happen, but handle it anyway
            print(
                f"[DEBUG] freqs_for_tokens shape: {freqs_for_tokens.shape}, "
                f"dtype: {freqs_for_tokens.dtype}"
            )
            print(f"[DEBUG] x shape: {x.shape}")
            # This format is ambiguous - assume it needs to be duplicated
            cos = freqs_for_tokens
            sin = freqs_for_tokens
        else:
            # freqs_for_tokens is already real, split it into cos and sin
            # Assume format: [total_tokens, qk_rope_head_dim]
            # where first half is cos, second is sin
            half_dim = freqs_for_tokens.shape[-1] // 2
            cos = freqs_for_tokens[
                ..., :half_dim
            ]  # [total_tokens, qk_rope_head_dim//2]
            sin = freqs_for_tokens[
                ..., half_dim:
            ]  # [total_tokens, qk_rope_head_dim//2]

        return self.apply_rotary_emb_with_cos_sin(x, cos, sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,  # vLLM provides positions
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for Multi-Head Latent Attention.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
                or [total_tokens, hidden_size]
            freqs_cis: Precomputed RoPE frequencies
                [max_seq_len, qk_rope_head_dim//2]
            attention_mask: Optional attention mask (not fully supported yet)
            positions: Per-token positions for RoPE indexing (from vLLM)
            **kwargs: Additional vLLM-specific kwargs

        Returns:
            Output tensor of same shape as hidden_states
        """
        # Handle both batched [batch, seq, hidden] and flattened [total_tokens, hidden]
        input_is_batched = hidden_states.dim() == 3
        if input_is_batched:
            bsz, seqlen, _ = hidden_states.shape
            hidden_states_flat = hidden_states.view(-1, self.hidden_size)
        else:
            # Flattened format (vLLM inference)
            hidden_states_flat = hidden_states
            bsz = 1
            seqlen = hidden_states.shape[0]

        total_tokens = hidden_states_flat.shape[0]

        # Get positions for RoPE indexing
        if positions is None:
            # Try to get from vLLM forward context
            try:
                from vllm.forward_context import get_forward_context

                forward_ctx = get_forward_context()
                if hasattr(forward_ctx, "_torchtitan_positions"):
                    positions = forward_ctx._torchtitan_positions
                else:
                    # Fallback: sequential positions
                    positions = torch.arange(
                        total_tokens, device=hidden_states_flat.device
                    )
            except (ImportError, AttributeError, AssertionError):
                # Training mode: sequential positions
                positions = torch.arange(total_tokens, device=hidden_states_flat.device)

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(hidden_states_flat)  # [total_tokens, n_heads * qk_head_dim]
        else:
            q = self.wq_a(hidden_states_flat)  # [total_tokens, q_lora_rank]
            assert self.q_norm is not None  # q_norm exists when q_lora_rank > 0
            q = self.wq_b(self.q_norm(q))  # [total_tokens, n_heads * qk_head_dim]

        # Reshape: [total_tokens, n_heads, qk_head_dim]
        q = q.view(total_tokens, self.num_heads, self.qk_head_dim)

        # Split Q into non-RoPE and RoPE parts
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Apply RoPE to q_pe using positions to index freqs_cis
        # Convert freqs_cis from complex to cos/sin BEFORE indexing
        # to avoid dtype issues
        if freqs_cis.is_complex():
            # Extract cos and sin from complex freqs_cis
            freqs_cos = freqs_cis.real  # [max_seq_len, qk_rope_head_dim//2]
            freqs_sin = freqs_cis.imag  # [max_seq_len, qk_rope_head_dim//2]
            # Concatenate for easier indexing
            freqs_real = torch.cat(
                [freqs_cos, freqs_sin], dim=-1
            )  # [max_seq_len, qk_rope_head_dim]
        else:
            freqs_real = freqs_cis

        # Index by positions
        positions_flat = positions.flatten()[:total_tokens]
        max_pos = freqs_real.shape[0] - 1
        positions_clamped = torch.clamp(positions_flat, 0, max_pos)
        freqs_for_tokens = freqs_real.index_select(
            0, positions_clamped
        )  # [total_tokens, qk_rope_head_dim]

        # Split into cos and sin
        half_dim = self.qk_rope_head_dim // 2
        cos_for_tokens = freqs_for_tokens[
            ..., :half_dim
        ]  # [total_tokens, qk_rope_head_dim//2]
        sin_for_tokens = freqs_for_tokens[
            ..., half_dim:
        ]  # [total_tokens, qk_rope_head_dim//2]

        # Apply RoPE to q_pe: [total_tokens, num_heads, qk_rope_head_dim]
        q_pe = self.apply_rotary_emb_with_cos_sin(q_pe, cos_for_tokens, sin_for_tokens)

        # Concatenate back: [total_tokens, n_heads, qk_head_dim]
        q = torch.cat([q_nope, q_pe], dim=-1)

        # Key-value projection
        kv = self.wkv_a(
            hidden_states_flat
        )  # [total_tokens, kv_lora_rank + qk_rope_head_dim]

        # Split into compressed KV and K_PE
        kv_c, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # Apply RoPE to k_pe: [total_tokens, qk_rope_head_dim]
        # Reshape to [total_tokens, 1, qk_rope_head_dim] for apply_rotary_emb
        k_pe = k_pe.unsqueeze(1)  # [total_tokens, 1, qk_rope_head_dim]
        k_pe = self.apply_rotary_emb_with_cos_sin(
            k_pe, cos_for_tokens, sin_for_tokens
        )  # [total_tokens, 1, qk_rope_head_dim]

        # Normalize compressed KV
        kv_c_normed = self.kv_norm(kv_c)  # [total_tokens, kv_lora_rank]

        # Delegate to vLLM's MLAAttention if available (handles KV cache automatically)
        if self.vllm_mla_attn is not None and not self.training:
            # Let vLLM handle all KV cache logic
            attn_output = self.vllm_mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=(total_tokens, self.num_heads * self.v_head_dim),
            )
        else:
            # Training mode or fallback: manual implementation
            # Decompress KV
            kv = self.wkv_b(
                kv_c_normed
            )  # [total_tokens, n_heads * (qk_nope_head_dim + v_head_dim)]
            kv = kv.view(
                total_tokens, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )

            # Split into K_nope and V
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )

            # Concatenate K_nope with broadcasted K_PE
            # k_pe shape: [total_tokens, 1, qk_rope_head_dim]
            # Expand to: [total_tokens, n_heads, qk_rope_head_dim]
            k = torch.cat(
                [k_nope, k_pe.expand(-1, self.num_heads, -1)], dim=-1
            )  # [total_tokens, n_heads, qk_head_dim]

            # Reshape for batched attention: [bsz, seqlen, n_heads, head_dim]
            q = q.view(bsz, seqlen, self.num_heads, self.qk_head_dim)
            k = k.view(bsz, seqlen, self.num_heads, self.qk_head_dim)
            v = v.view(bsz, seqlen, self.num_heads, self.v_head_dim)

            # Transpose for attention: [bsz, n_heads, seqlen, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Use PyTorch SDPA (supports different head dims for Q/K vs V)
            # Flash attention doesn't support qk_head_dim != v_head_dim, so we use SDPA
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal and attention_mask is None,
                scale=self.scale,
            )

            # Transpose back and reshape: [total_tokens, n_heads * v_head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(
                total_tokens, self.num_heads * self.v_head_dim
            )

        # Output projection: [total_tokens, hidden_size]
        output = self.wo(attn_output)

        # Restore original shape if input was batched
        if input_is_batched:
            output = output.view(bsz, seqlen, self.hidden_size)

        return output
