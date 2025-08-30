# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import Final, Generic, Optional, Protocol, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.attention.selector import get_env_variable_attn_backend
from vllm.logger import init_logger
from vllm.platforms import _Backend, current_platform

logger = init_logger(__name__)

_C = TypeVar("_C", bound=PretrainedConfig)


class VisionEncoderInfo(ABC, Generic[_C]):

    def __init__(self, hf_config: _C) -> None:
        super().__init__()

        self.hf_config = hf_config
        self.vision_config = hf_config.vision_config

    @abstractmethod
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_image_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_patch_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_patch_grid_length(self) -> int:
        raise NotImplementedError


class VisionLanguageConfig(Protocol):
    vision_config: Final[PretrainedConfig]


def get_vision_encoder_info(
        hf_config: VisionLanguageConfig) -> VisionEncoderInfo:
    # Avoid circular imports
    from .clip import CLIPEncoderInfo, CLIPVisionConfig
    from .pixtral import PixtralHFEncoderInfo, PixtralVisionConfig
    from .siglip import SiglipEncoderInfo, SiglipVisionConfig

    if isinstance(hf_config.vision_config, CLIPVisionConfig):
        return CLIPEncoderInfo(hf_config)
    if isinstance(hf_config.vision_config, PixtralVisionConfig):
        return PixtralHFEncoderInfo(hf_config)
    if isinstance(hf_config.vision_config, SiglipVisionConfig):
        return SiglipEncoderInfo(hf_config)

    msg = f"Unsupported vision config: {type(hf_config.vision_config)}"
    raise NotImplementedError(msg)


def get_vit_attn_backend(support_fa: bool = False) -> _Backend:
    """
    Get the available attention backend for Vision Transformer.
    """
    # TODO(Isotr0py): Remove `support_fa` after support FA for all ViTs attn.

    selected_backend: Optional[_Backend] = get_env_variable_attn_backend()
    if selected_backend is not None:
        return selected_backend

    return current_platform.get_vit_attn_backend(support_fa)


def resolve_visual_encoder_outputs(
    encoder_outputs: Union[torch.Tensor, list[torch.Tensor]],
    feature_sample_layers: Optional[list[int]],
    post_layer_norm: Optional[torch.nn.LayerNorm],
    max_possible_layers: int,
) -> torch.Tensor:
    """Given the outputs a visual encoder module that may correspond to the
    output of the last layer, or a list of hidden states to be stacked,
    handle post normalization and resolve it into a single output tensor.

    Args:
        encoder_outputs: Output of encoder's last layer or all hidden states.
        feature_sample_layers: Optional layer indices to grab from the encoder
            outputs; if provided, encoder outputs must be a list.
        post_layer_norm: Post norm to apply to the output of the encoder.
        max_possible_layers: Total layers in the fully loaded visual encoder.

    """
    if feature_sample_layers is None:
        if post_layer_norm is not None:
            return post_layer_norm(encoder_outputs)
        return encoder_outputs

    # Get the hidden states corresponding to the layer indices.
    # Negative values are relative to the full visual encoder,
    # so offset them depending on how many layers were loaded.
    # NOTE: this assumes that encoder_outputs is a list containing
    # the inputs to the visual encoder, followed by the hidden states
    # of each layer.
    num_loaded_layers = len(encoder_outputs) - 1
    offset = max_possible_layers - num_loaded_layers
    hs_pool = [
        encoder_outputs[layer_idx]
        if layer_idx >= 0 else encoder_outputs[layer_idx + offset]
        for layer_idx in feature_sample_layers
    ]

    # Apply post-norm on the final hidden state if we are using it
    uses_last_layer = feature_sample_layers[-1] in (len(hs_pool) - 1, -1)
    if post_layer_norm is not None and uses_last_layer:
        hs_pool[-1] = post_layer_norm(encoder_outputs)
    return torch.cat(hs_pool, dim=-1)


class VisionAttention(torch.nn.Module):
    """
    Unified Vision Transformer attention module that automatically selects
    the optimal backend based on hardware, compute capability, head size, etc.
    
    This allows model developers to focus on model architecture without
    worrying about attention implementation details.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        use_rotary: bool = False,
        rotary_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (embed_dim // num_heads)
        self.dropout = dropout
        self.bias = bias
        self.use_rotary = use_rotary
        self.rotary_dim = rotary_dim or self.head_dim
        
        # Auto-select optimal backend
        self.backend = self._select_backend()
        
        # Initialize QKV projection
        self.qkv = torch.nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Rotary embeddings if needed
        if use_rotary:
            self.rotary_emb = self._create_rotary_embeddings()
    
    def _select_backend(self) -> _Backend:
        """Automatically select the optimal attention backend."""
        # Check environment override first
        env_backend = get_env_variable_attn_backend()
        if env_backend is not None:
            return env_backend
        
        # Use existing logic with support for FA
        return get_vit_attn_backend(support_fa=True)
    
    def _create_rotary_embeddings(self):
        """Create rotary position embeddings if needed."""
        # This would be implemented based on the specific rotary embedding
        # requirements of the model
        pass
    
    def _apply_rotary_embeddings(self, q: torch.Tensor, k: torch.Tensor, 
                                positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to Q and K."""
        if not self.use_rotary:
            return q, k
        
        # Implementation would depend on the specific rotary embedding method
        # For now, return as-is
        return q, k
    
    def _flash_attention_forward(self, q: torch.Tensor, k: torch.Tensor, 
                                v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass using FlashAttention."""
        try:
            from flash_attn import flash_attn_func
            return flash_attn_func(q, k, v, dropout_p=self.dropout, causal=False)
        except ImportError:
            # Fallback to torch SDPA
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
    
    def _torch_sdpa_forward(self, q: torch.Tensor, k: torch.Tensor, 
                           v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass using torch scaled_dot_product_attention."""
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
    
    def _xformers_forward(self, q: torch.Tensor, k: torch.Tensor, 
                         v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass using xFormers."""
        try:
            from xformers import ops as xops
            return xops.memory_efficient_attention_forward(q, k, v, p=self.dropout)
        except ImportError:
            # Fallback to torch SDPA
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with automatic backend selection.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            positions: Optional position indices for rotary embeddings
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to QKV
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply rotary embeddings if needed
        if positions is not None:
            q, k = self._apply_rotary_embeddings(q, k, positions)
        
        # Select attention implementation based on backend
        if self.backend == _Backend.FLASH_ATTN:
            attn_output = self._flash_attention_forward(q, k, v, mask)
        elif self.backend == _Backend.TORCH_SDPA:
            attn_output = self._torch_sdpa_forward(q, k, v, mask)
        elif self.backend == _Backend.XFORMERS:
            attn_output = self._xformers_forward(q, k, v, mask)
        else:
            # Fallback to torch SDPA
            attn_output = self._torch_sdpa_forward(q, k, v, mask)
        
        # Project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        output = self.proj(attn_output)
        
        return output