# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA Attention layer that implements AttentionLayerBase."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from vllm.attention.selector import get_attn_backend
from vllm.config import CacheConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.quantization import QuantizationConfig


@dataclass
class MLAModules:
    """Modules used in MLA."""
    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: Optional[torch.nn.Module]
    kv_a_proj_with_mqa: Optional[torch.nn.Module]
    q_a_layernorm: Optional[torch.nn.Module]
    q_b_proj: Optional[torch.nn.Module]
    q_proj: Optional[torch.nn.Module]


class MLAAttention(nn.Module, AttentionLayerBase):
    """
    MLA (Multi-Head Latent Attention) layer that implements AttentionLayerBase.
    
    This class provides a dedicated attention layer for MLA that is separate
    from the standard MHA/GQA/MQA attention mechanisms. It uses the existing
    MultiHeadLatentAttention CustomOp for the actual computation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.prefix = prefix

        # Store MLA modules
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj

        # Create the underlying MLA attention using the existing CustomOp
        # In the MLA backend, kv_cache includes both k_c and
        # pe (i.e. decoupled position embeddings). In particular,
        # the concat_and_cache_mla op requires
        #     k_c.size(1) + k_pe.size(1) == kv_cache.size(2)
        # i.e.
        #     kv_lora_rank + qk_rope_head_dim == head_size

        # Store scale for attention computation
        self.scale = scale

        # Get the MLA backend for this layer
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            is_attention_free = cache_config.is_attention_free
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            is_attention_free = False

        dtype = torch.get_default_dtype()
        self.mla_backend = get_attn_backend(
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            block_size=block_size,
            is_attention_free=is_attention_free,
            use_mla=True,
            has_sink=False,
        )

        # Parse debug layer index from prefix (e.g., "layers.0.attn" -> 0)
        prefix_parts = self.prefix.split(".")
        if len(prefix_parts) >= 2:
            try:
                self.debug_layer_idx = int(prefix_parts[-2])
            except ValueError:
                self.debug_layer_idx = 0  # Default to 0 if not a number
        else:
            self.debug_layer_idx = 0  # Default to 0 if prefix format is unexpected

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for MLA attention.
        
        Args:
            positions: Position indices tensor
            hidden_states: Input hidden states tensor
            
        Returns:
            Output tensor after MLA attention computation
        """
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, \
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            assert self.q_a_layernorm is not None, \
                "q_a_layernorm is required when q_lora_rank is not None"
            assert self.q_b_proj is not None, \
                "q_b_proj is required when q_lora_rank is not None"
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, \
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            assert self.q_proj is not None, \
                "q_proj is required when q_lora_rank is None"
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim],
                                   dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim:], k_pe)

        # Use the MLA backend directly for attention computation
        attn_out = self.mla_backend.forward(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0],
                          self.num_heads * self.v_head_dim))
        return self.o_proj(attn_out)[0]

    def get_attn_backend(self) -> type:
        """Get the attention backend class for this MLA layer."""
        return self.mla_backend
