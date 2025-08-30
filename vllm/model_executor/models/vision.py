# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import Final, Generic, Optional, Protocol, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.attention.layer import MultiHeadAttention
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
    Unified Vision Transformer attention module using MultiHeadAttention.
    
    This simplified version uses the unified MultiHeadAttention implementation
    while maintaining the same interface for backward compatibility.
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
        
        # Initialize QKV projection
        self.qkv = torch.nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Use unified MultiHeadAttention with Flash Attention support
        self.attn = MultiHeadAttention(num_heads, self.head_dim, self.head_dim**-0.5)
        
        # Rotary embeddings if needed
        if use_rotary:
            self.rotary_emb = self._create_rotary_embeddings()
    

    
    def _create_rotary_embeddings(self):
        """Create rotary position embeddings if needed."""
        if not self.use_rotary:
            return None
        
        # Create rotary embeddings based on head dimension
        try:
            from vllm.model_executor.layers.rotary_embedding import get_rope
            return get_rope(head_size=self.rotary_dim)
        except ImportError:
            # Fallback to basic implementation
            return None
    
    def _apply_rotary_embeddings(self, q: torch.Tensor, k: torch.Tensor, 
                                positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to Q and K."""
        if not self.use_rotary or self.rotary_emb is None:
            return q, k
        
        try:
            # Apply rotary embeddings using vLLM's implementation
            q = self.rotary_emb(q, positions)
            k = self.rotary_emb(k, positions)
            return q, k
        except Exception:
            # Fallback: return as-is if rotary embedding fails
            return q, k
    

    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
拉取        Forward pass using unified MultiHeadAttention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask (not used in current implementation)
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
        
        # Reshape for MultiHeadAttention: (batch, seq, hidden_size)
        q_reshaped = q.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        k_reshaped = k.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        v_reshaped = v.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Use unified MultiHeadAttention
        attn_output = self.attn(q_reshaped, k_reshaped, v_reshaped)
        
        # Project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        output = self.proj(attn_output)
        
        return output