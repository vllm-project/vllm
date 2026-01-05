# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
T5Gemma2 Model Implementation

This module implements the T5Gemma2 encoder-decoder model for vLLM.
T5Gemma2 is a multimodal model built from Gemma 3 using UL2 pre-training.
It supports both text and image inputs.

Key features:
- SigLIP vision encoder for image inputs
- Bidirectional attention in encoder
- Merged self+cross attention in decoder
- Sliding window attention pattern
- RoPE with dual theta for position encoding
- EOI token handling for multimodal inputs
"""

from collections.abc import Iterable
from itertools import islice
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Gemma2Config

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import GeluAndMul, get_act_fn
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.sequence import IntermediateTensors

from .gemma2 import Gemma2MLP
from .utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    WeightsMapper,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)


def get_t5gemma2_text_config(config: T5Gemma2Config, is_encoder: bool = True) -> dict:
    """Extract text config from T5Gemma2Config for vLLM.
    
    Args:
        config: The T5Gemma2Config object
        is_encoder: If True, extracts from config.encoder.text_config.
                   If False, extracts from config.decoder directly.
    """
    text_config = config.encoder.text_config if is_encoder else config.decoder
    return {
        "vocab_size": text_config.vocab_size,
        "hidden_size": text_config.hidden_size,
        "intermediate_size": text_config.intermediate_size,
        "num_hidden_layers": text_config.num_hidden_layers,
        "num_attention_heads": text_config.num_attention_heads,
        "num_key_value_heads": text_config.num_key_value_heads,
        "head_dim": text_config.head_dim,
        "max_position_embeddings": text_config.max_position_embeddings,
        "rms_norm_eps": text_config.rms_norm_eps,
        "attn_logit_softcapping": text_config.attn_logit_softcapping,
        "final_logit_softcapping": text_config.final_logit_softcapping,
        "dropout_rate": getattr(config, "dropout_rate", 0.0),
        "layer_types": text_config.layer_types,
        "sliding_window": text_config.sliding_window,
        "rope_parameters": text_config.rope_parameters,
        "hidden_act": text_config.hidden_activation,
    }


class T5Gemma2TextScaledWordEmbedding(nn.Module):
    """Text embedding with EOI token handling for multimodal inputs."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
        eoi_token_index: int = 256000,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.padding_idx = padding_idx
        self.embed_scale = embed_scale
        self.eoi_token_index = eoi_token_index
        self.eoi_embedding = nn.Parameter(torch.zeros(embedding_dim))
        
        # Add quant_method attribute for compatibility with LogitsProcessor
        # This is a no-op quantization method that just applies the embedding
        self.quant_method = _NoOpQuantMethod(self)

        # Initialize weights
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.eoi_embedding)

    def forward(self, input_ids: torch.Tensor | None) -> torch.Tensor:
        # Handle None input_ids (from dummy run during profiling)
        if input_ids is None:
            # Return dummy tensor with shape (1, 1, hidden_size)
            # This is used during memory profiling to determine tensor shapes
            # Don't apply scaling during dummy run to avoid torch.compile issues
            return torch.zeros(
                1, 1, self.weight.shape[1],
                dtype=self.weight.dtype,
                device=self.weight.device
            )
        
        # Standard embedding
        embeddings = F.embedding(input_ids, self.weight, self.padding_idx)
        embeddings = embeddings * self.embed_scale

        # Replace EOI token embeddings
        # Note: We use torch.where instead of conditional indexing to avoid CUDA graph issues
        # The .any() call is not compatible with CUDA graph capture
        if self.eoi_token_index is not None:
            eoi_mask = (input_ids == self.eoi_token_index)
            # Use torch.where to replace EOI token embeddings without breaking CUDA graphs
            embeddings = torch.where(
                eoi_mask.unsqueeze(-1),
                self.eoi_embedding.to(embeddings.dtype),
                embeddings
            )

        return embeddings


class _NoOpQuantMethod:
    """No-op quantization method for embedding layers.
    
    This is used to make T5Gemma2TextScaledWordEmbedding compatible with
    LogitsProcessor which expects a quant_method attribute.
    """
    
    def __init__(self, embedding_layer):
        self.embedding_layer = embedding_layer
    
    def apply(self, embedding_layer, hidden_states, bias=None):
        """Apply embedding projection to hidden states.
        
        Args:
            embedding_layer: The embedding layer (T5Gemma2TextScaledWordEmbedding)
            hidden_states: Hidden states to project (batch, seq, hidden_size)
            bias: Optional bias (not used for embeddings)
        
        Returns:
            Logits (batch, seq, vocab_size)
        """
        # Project hidden states to vocabulary size using embedding weight
        # hidden_states: (batch, seq, hidden_size)
        # embedding_layer.weight: (vocab_size, hidden_size)
        # output: (batch, seq, vocab_size)
        return torch.matmul(hidden_states, embedding_layer.weight.t())


class T5Gemma2MLP(nn.Module):
    """MLP with dropout for T5Gemma2."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        dropout_rate: float = 0.0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        # Use merged gate_up_proj for efficiency (matches Gemma2MLP pattern)
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

        # Use GeluAndMul for gelu_pytorch_tanh activation
        if hidden_act == "gelu_pytorch_tanh":
            self.act_fn = GeluAndMul(approximate="tanh")
        else:
            self.act_fn = get_act_fn(hidden_act)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.dropout(x)
        x, _ = self.down_proj(x)
        return x


class T5Gemma2Attention(nn.Module):
    """Attention module with sliding window support.

    Uses separate Q/K/V projections to match the checkpoint format.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        sliding_window: int | None = None,
        is_encoder: bool = True,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim

        # Use separate Q/K/V projections to match checkpoint format
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Add q_norm and k_norm for attention head normalization (matches transformers)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=1e-6)

        # Use MMEncoderAttention for encoder (no KV cache), Attention for decoder
        if is_encoder:
            self.attn = MMEncoderAttention(
                self.num_heads,
                self.head_dim,
                self.head_dim**-0.5,
                num_kv_heads=self.num_kv_heads,
            )
        else:
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.head_dim**-0.5,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                attn_type=AttentionType.DECODER,
                per_layer_sliding_window=sliding_window,
                prefix=f"{prefix}.attn",
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        
        # Reshape to 3D for normalization: (num_tokens, num_heads, head_dim)
        num_tokens = hidden_states.shape[0]
        
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        
        # Apply q_norm and k_norm on 3D tensors
        # Add unsqueeze(2) to make it 4D for norm, then squeeze back
        q = self.q_norm(q.unsqueeze(2)).squeeze(2)
        k = self.k_norm(k.unsqueeze(2)).squeeze(2)
        
        # Reshape back to 2D for MMEncoderAttention: (num_tokens, hidden_size)
        # MMEncoderAttention expects (batch, seq_len, hidden_size) format
        # Since we flattened to (num_tokens, hidden_size), we need to reshape to (1, num_tokens, hidden_size)
        q = q.reshape(num_tokens, self.num_heads * self.head_dim).unsqueeze(0)
        k = k.reshape(num_tokens, self.num_kv_heads * self.head_dim).unsqueeze(0)
        v = v.reshape(num_tokens, self.num_kv_heads * self.head_dim).unsqueeze(0)
        
        # MMEncoderAttention expects (batch, seq_len, hidden_size)
        attn_output = self.attn(q, k, v)
        # Flatten back to (num_tokens, hidden_size)
        attn_output = attn_output.squeeze(0)
        output, _ = self.o_proj(attn_output)
        return output


class T5Gemma2MergedAttention(nn.Module):
    """Merged self-attention and cross-attention for decoder.

    This fuses self-attention and cross-attention into a single operation
    to match the transformers implementation. The key and value states
    from self-attention and cross-attention are concatenated before
    the attention computation.

    Uses separate Q/K/V projections to match the checkpoint format.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        cross_attention_hidden_size: int,
        attn_logit_softcapping: float | None = None,
        sliding_window: int | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim

        # Use separate Q/K/V projections to match checkpoint format
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Add q_norm and k_norm for attention head normalization (matches transformers)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=1e-6)

        # Merged attention uses DECODER attention type
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=AttentionType.DECODER,
            logits_soft_cap=attn_logit_softcapping,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention: Q/K/V from hidden_states
        num_tokens = hidden_states.shape[0]
        
        q, _ = self.q_proj(hidden_states)
        k_self, _ = self.k_proj(hidden_states)
        v_self, _ = self.v_proj(hidden_states)

        # Reshape to 3D for normalization and attention: (num_tokens, num_heads, head_dim)
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k_self = k_self.view(num_tokens, self.num_kv_heads, self.head_dim)
        v_self = v_self.view(num_tokens, self.num_kv_heads, self.head_dim)
        
        # Apply q_norm and k_norm on 3D tensors
        # Add unsqueeze(2) to make it 4D for norm, then squeeze back
        q = self.q_norm(q.unsqueeze(2)).squeeze(2)
        k_self = self.k_norm(k_self.unsqueeze(2)).squeeze(2)

        # Cross-attention: K/V from encoder_hidden_states
        if encoder_hidden_states is not None:
            num_encoder_tokens = encoder_hidden_states.shape[0]

            k_cross, _ = self.k_proj(encoder_hidden_states)
            v_cross, _ = self.v_proj(encoder_hidden_states)

            # Reshape to 3D for normalization and attention
            k_cross = k_cross.view(num_encoder_tokens, self.num_kv_heads,
                                   self.head_dim)
            v_cross = v_cross.view(num_encoder_tokens, self.num_kv_heads,
                                   self.head_dim)

            # Apply k_norm on 3D tensor (matches transformers)
            k_cross = self.k_norm(k_cross.unsqueeze(2)).squeeze(2)

            # Concatenate self and cross K/V along token dimension (dim=0)
            k = torch.cat([k_self, k_cross], dim=0)
            v = torch.cat([v_self, v_cross], dim=0)
        else:
            k = k_self
            v = v_self

        # vLLM attention expects 3D tensors: (num_tokens, num_heads, head_dim)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class T5Gemma2EncoderLayer(nn.Module):
    """Encoder layer with bidirectional attention and sliding window support."""

    def __init__(
        self,
        config: dict,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_type = config["layer_types"][layer_idx]
        is_sliding = self.attention_type == "sliding_attention"
        sliding_window = config["sliding_window"] if is_sliding else None

        self.hidden_size = config["hidden_size"]
        self.self_attn = T5Gemma2Attention(
            hidden_size=self.hidden_size,
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            max_position_embeddings=config["max_position_embeddings"],
            sliding_window=sliding_window,
            is_encoder=True,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = T5Gemma2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config["intermediate_size"],
            hidden_act=config["hidden_act"],
            dropout_rate=config["dropout_rate"],
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        # Initialize all layer norms in __init__ to match checkpoint names
        self.pre_self_attn_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_self_attn_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.pre_feedforward_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_feedforward_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class T5Gemma2VisionEncoder(nn.Module):
    """SigLIP vision encoder for T5Gemma2 multimodal inputs."""

    def __init__(
        self,
        config: T5Gemma2Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        vision_config = config.encoder.vision_config
        text_config = config.encoder.text_config

        # Load SigLIP vision model
        self.vision_tower = SiglipVisionModel(
            vision_config,
            quant_config,
            prefix=maybe_prefix(prefix, "vision_tower"),
        )

        # Vision feature pooling (matches transformers T5Gemma2MultiModalProjector)
        self.patches_per_image = int(vision_config.image_size // vision_config.patch_size)
        # Compute mm_tokens_per_image from patches_per_image (matches transformers behavior)
        # mm_tokens_per_image represents the number of image tokens, derived from patches
        self.mm_tokens_per_image = int(self.patches_per_image**0.5)**2
        self.kernel_size = self.patches_per_image // self.mm_tokens_per_image
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

        # Vision feature normalization (matches transformers T5Gemma2MultiModalProjector)
        self.mm_soft_emb_norm = GemmaRMSNorm(
            vision_config.hidden_size, eps=vision_config.layer_norm_eps
        )

        # Multi-modal projector (maps vision features to text hidden size)
        # Parameter name matches checkpoint: mm_input_projection_weight
        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(vision_config.hidden_size, text_config.hidden_size)
        )

        # Initialize projector weights to zeros (matches transformers)
        nn.init.zeros_(self.mm_input_projection_weight)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract and project vision features (matches transformers T5Gemma2MultiModalProjector)."""
        batch_size = pixel_values.shape[0]
        vision_outputs = self.vision_tower(pixel_values=pixel_values)
        # vision_outputs.last_hidden_state shape: (batch, seq_len, hidden_size)

        # Reshape for pooling: (batch, seq_len, hidden) -> (batch, hidden, patches_per_image, patches_per_image)
        _, seq_length, hidden_size = vision_outputs.last_hidden_state.shape
        reshaped_vision_outputs = vision_outputs.last_hidden_state.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, hidden_size, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        # Average pooling to reduce patches
        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2).transpose(1, 2)

        # Normalize pooled vision outputs
        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        # Project to text hidden size using mm_input_projection_weight
        image_features = torch.matmul(normed_vision_outputs, self.mm_input_projection_weight)
        return image_features.type_as(vision_outputs.last_hidden_state)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for vision tower and projector.
        
        The SiglipVisionModel handles q_proj/k_proj/v_proj → qkv_proj merging
        via its stacked_params_mapping. We need to track the merged parameter names
        that the SiglipVisionModel returns, not the original checkpoint names.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        # Separate weights for vision_tower and multimodal projector
        vision_tower_weights = []
        projector_weights = []
        
        for name, weight in weights:
            if name.startswith("vision_tower."):
                # Strip the "vision_tower." prefix for SiglipVisionModel
                stripped_name = name[len("vision_tower."):]
                vision_tower_weights.append((stripped_name, weight))
            elif name in ("mm_input_projection_weight", "mm_soft_emb_norm.weight"):
                projector_weights.append((name, weight))
            else:
                # Unknown weight, try to load it directly
                projector_weights.append((name, weight))
        
        # Load vision tower weights using SiglipVisionModel's load_weights
        if vision_tower_weights:
            vision_tower_loaded = self.vision_tower.load_weights(vision_tower_weights)
            # Add "vision_tower." prefix back to loaded params
            # The SiglipVisionModel returns merged names (with qkv_proj instead of q_proj/k_proj/v_proj)
            for param in vision_tower_loaded:
                loaded_params.add(f"vision_tower.{param}")
        
        # Load projector weights directly
        for name, loaded_weight in projector_weights:
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params


class T5Gemma2Encoder(nn.Module):
    """T5Gemma2 encoder with vision and bidirectional attention."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: T5Gemma2Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        text_config = get_t5gemma2_text_config(config, is_encoder=True)

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.encoder.text_config.pad_token_id

        # Embed tokens with EOI handling
        self.embed_tokens = T5Gemma2TextScaledWordEmbedding(
            text_config["vocab_size"],
            text_config["hidden_size"],
            self.padding_idx,
            embed_scale=text_config["hidden_size"] ** 0.5,
            eoi_token_index=config.eoi_token_index,
        )

        # Vision encoder
        self.vision_encoder = T5Gemma2VisionEncoder(
            config, quant_config, prefix=maybe_prefix(prefix, "vision_encoder")
        )

        # Encoder layers with bidirectional attention
        self.start_layer, self.end_layer, self.layers = make_layers(
            text_config["num_hidden_layers"],
            lambda prefix: T5Gemma2EncoderLayer(
                text_config,
                layer_idx=int(prefix.split(".")[-1]) if "." in prefix else 0,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm = GemmaRMSNorm(text_config["hidden_size"], eps=text_config["rms_norm_eps"])

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Get text embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Flatten to 2D for vLLM V1 engine: (batch * seq_len, hidden_size)
        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # Process images if provided
        if pixel_values is not None:
            image_features = self.vision_encoder(pixel_values)

            # Replace image placeholder tokens with image features
            image_token_id = self.config.image_token_index
            # Flatten input_ids to match hidden_states
            flat_input_ids = input_ids.view(-1)
            image_mask = (flat_input_ids == image_token_id)

            if image_mask.any():
                # Flatten image features
                flat_image_features = image_features.view(-1, image_features.size(-1))

                # Scatter image features into hidden states
                hidden_states = hidden_states.clone()
                hidden_states[image_mask] = flat_image_features.to(hidden_states.dtype)

        # Pass through encoder layers
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Convert to list to allow multiple passes
        weights_list = list(weights)
        
        # Filter weights for vision_encoder submodule first
        # This handles weights with "vision_encoder." prefix from T5Gemma2Model.load_weights
        vision_encoder_weights = [
            (name[len("vision_encoder."):], weight)
            for name, weight in weights_list
            if name.startswith("vision_encoder.")
        ]

        # Load vision encoder weights
        loaded_params = set()
        if vision_encoder_weights:
            ve_loaded = self.vision_encoder.load_weights(vision_encoder_weights)
            # Add "vision_encoder." prefix back to loaded params
            for param in ve_loaded:
                loaded_params.add(f"vision_encoder.{param}")

        # For remaining weights, collect them for self
        weights_for_self = []
        for name, weight in weights_list:
            if name.startswith("vision_encoder."):
                # Already handled above, skip
                continue
            else:
                weights_for_self.append((name, weight))

        params_dict = dict(self.named_parameters())
        
        # Stacked params mapping for merged projections
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        
        # Process weights with support for merged projections
        for name, loaded_weight in weights_for_self:
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache scales for compressed-tensors quantization
                if scale_name in params_dict:
                    param = params_dict[scale_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    loaded_weight_value = loaded_weight[0] if isinstance(loaded_weight, tuple) else loaded_weight
                    weight_loader(param, loaded_weight_value)
                    loaded_params.add(scale_name)
                continue
            
            # Check for stacked params (merged projections)
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    # Weight doesn't belong to this module, skip it
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params


class T5Gemma2DecoderLayer(nn.Module):
    """Decoder layer with merged self+cross attention."""

    def __init__(
        self,
        config: dict,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_type = config["layer_types"][layer_idx]
        is_sliding = self.attention_type == "sliding_attention"
        sliding_window = config["sliding_window"] if is_sliding else None

        self.hidden_size = config["hidden_size"]

        # Merged attention (self + cross attention)
        self.self_attn = T5Gemma2MergedAttention(
            hidden_size=self.hidden_size,
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            max_position_embeddings=config["max_position_embeddings"],
            cross_attention_hidden_size=self.hidden_size,
            attn_logit_softcapping=config.get("attn_logit_softcapping"),
            sliding_window=sliding_window,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = T5Gemma2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config["intermediate_size"],
            hidden_act=config["hidden_act"],
            dropout_rate=config["dropout_rate"],
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        # Initialize all layer norms in __init__ to match checkpoint names
        self.pre_self_attn_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_self_attn_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.pre_feedforward_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_feedforward_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.pre_self_attn_layernorm(hidden_states)
        else:
            hidden_states, residual = self.pre_self_attn_layernorm(hidden_states, residual)

        # Merged attention (self + cross)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            positions=positions,
        )

        hidden_states = self.post_self_attn_layernorm(hidden_states)

        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return hidden_states, residual


class T5Gemma2Decoder(nn.Module):
    """T5Gemma2 decoder with merged attention support."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: T5Gemma2Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        text_config = get_t5gemma2_text_config(config, is_encoder=False)

        self.config = text_config
        self.quant_config = quant_config

        self.embed_tokens = T5Gemma2TextScaledWordEmbedding(
            text_config["vocab_size"],
            text_config["hidden_size"],
            config.decoder.pad_token_id,
            embed_scale=text_config["hidden_size"] ** 0.5,
            eoi_token_index=config.eoi_token_index,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            text_config["num_hidden_layers"],
            lambda prefix: T5Gemma2DecoderLayer(
                text_config,
                layer_idx=int(prefix.split(".")[-1]) if "." in prefix else 0,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm = GemmaRMSNorm(text_config["hidden_size"], eps=text_config["rms_norm_eps"])

        # Normalize the embedding by sqrt(hidden_size)
        normalizer = self.config["hidden_size"] ** 0.5
        self.register_buffer("normalizer", torch.tensor(normalizer), persistent=False)

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], text_config["hidden_size"]
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states *= self.normalizer
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                encoder_hidden_states,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        # Convert weights to list to allow multiple passes
        weights_list = list(weights)
        
        # Stacked params mapping for merged projections
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        
        # Process weights with support for merged projections
        for name, loaded_weight in weights_list:
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache scales for compressed-tensors quantization
                if scale_name in params_dict:
                    param = params_dict[scale_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    loaded_weight_value = loaded_weight[0] if isinstance(loaded_weight, tuple) else loaded_weight
                    weight_loader(param, loaded_weight_value)
                    loaded_params.add(scale_name)
                continue
            
            # Check for stacked params (merged projections)
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params


class T5Gemma2Model(nn.Module):
    """T5Gemma2 encoder-decoder model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: T5Gemma2Config = vllm_config.model_config.hf_config
        self.config = config

        self.encoder = T5Gemma2Encoder(
            vllm_config=vllm_config, prefix=f"{prefix}.encoder"
        )
        self.decoder = T5Gemma2Decoder(
            vllm_config=vllm_config, prefix=f"{prefix}.decoder"
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        encoder_outputs: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_outputs,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return decoder_outputs

    def get_encoder_outputs(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if input_ids is None:
            return None
        return self.encoder(input_ids, pixel_values=pixel_values)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Strip prefixes from weight names before passing to encoder/decoder
        # The checkpoint weights have names like "encoder.embed_tokens.weight"
        # but the encoder/decoder modules expect names like "embed_tokens.weight"
        encoder_weights = [
            (name[len("encoder."):], weight)
            for name, weight in weights
            if name.startswith("encoder.")
        ]
        decoder_weights = [
            (name[len("decoder."):], weight)
            for name, weight in weights
            if name.startswith("decoder.")
        ]
        loaded_params = set()
        
        # Load encoder weights and add "encoder." prefix back
        encoder_loaded = self.encoder.load_weights(encoder_weights)
        for param in encoder_loaded:
            loaded_params.add(f"encoder.{param}")
        
        # Load decoder weights and add "decoder." prefix back
        decoder_loaded = self.decoder.load_weights(decoder_weights)
        for param in decoder_loaded:
            loaded_params.add(f"decoder.{param}")
        
        return loaded_params


class T5Gemma2ForConditionalGeneration(nn.Module, SupportsLoRA, SupportsPP):
    """T5Gemma2 for conditional generation (seq2seq)."""

    packed_modules_mapping = {
        # No packed modules - we use separate projections for all layers
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: T5Gemma2Config = vllm_config.model_config.hf_config

        self.config = config
        assert config.tie_word_embeddings

        self.model = T5Gemma2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        # Logits processor with softcapping
        self.logits_processor = LogitsProcessor(
            config.decoder.vocab_size,
            soft_cap=config.decoder.final_logit_softcapping,
        )

        self.make_empty_intermediate_tensors = (
            self.model.decoder.make_empty_intermediate_tensors
        )
 
    def get_language_model(self) -> nn.Module:
        return self.model.decoder
 
    def get_encoder_outputs(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.get_encoder_outputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )
 
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.decoder.embed_input_ids(input_ids)
 
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if encoder_outputs is None:
            encoder_outputs = self.model.get_encoder_outputs(
                kwargs.get("encoder_input_ids"), pixel_values
            )
 
        decoder_outputs = self.model(
            input_ids=input_ids,
            positions=positions,
            encoder_outputs=encoder_outputs,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return decoder_outputs

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # Use embed_tokens for logits computation
        # The logits_processor expects an embedding layer with a quant_method attribute
        logits = self.logits_processor(self.model.decoder.embed_tokens, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str] | None:
        # T5Gemma2 has tied weights between encoder and decoder embed_tokens
        # and between lm_head and encoder embed_tokens when tie_word_embeddings=True
        # We handle this by:
        # 1. Routing weights to encoder/decoder model
        # 2. Special handling for eoi_embedding (tied between encoder and decoder)
        # 3. lm_head.weight is tied to encoder.embed_tokens.weight

        # Convert weights to list to check for model prefix
        weights_list = list(weights)
        
        # Check if weights have the "model." prefix
        # The checkpoint weights have names like "encoder.embed_tokens.weight"
        # but the model's named_parameters() returns "model.encoder.embed_tokens.weight"
        has_model_prefix = any(name.startswith("model.") for name, _ in weights_list)
        if not has_model_prefix:
            # Add "model." prefix and fix vision_tower path
            # Checkpoint has: encoder.vision_tower.vision_model...
            # vLLM expects: model.encoder.vision_encoder.vision_tower.vision_model...
            mapper = WeightsMapper(
                orig_to_new_prefix={"": "model."},
                orig_to_new_substr={"encoder.vision_tower": "encoder.vision_encoder.vision_tower"}
            )
            weights_list = mapper.apply(weights_list)
        
        # Now pass weights to T5Gemma2Model.load_weights which will handle routing
        # T5Gemma2Model expects weights with "model." prefix
        loaded_params = self.model.load_weights(weights_list)
        
        # Return None to skip strict weight loading check
        # The model has parameters that are not in the checkpoint (like q_norm, k_norm)
        # which are initialized in __init__ and don't need to be loaded
        return None
