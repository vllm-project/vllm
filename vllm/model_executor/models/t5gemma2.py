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

from vllm.attention.layer import Attention
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

        # Initialize weights
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.eoi_embedding)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Standard embedding
        embeddings = F.embedding(input_ids, self.weight, self.padding_idx)
        embeddings = embeddings * self.embed_scale

        # Replace EOI token embeddings
        if self.eoi_token_index is not None:
            eoi_mask = (input_ids == self.eoi_token_index)
            if eoi_mask.any():
                embeddings = embeddings.clone()  # Avoid in-place operation
                embeddings[eoi_mask] = self.eoi_embedding.to(embeddings.dtype)

        return embeddings


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
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
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
        gate, up = gate_up.chunk(2, dim=-1)
        x = self.act_fn(gate) * up
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

        # Use separate Q, K, V projections to match checkpoint format
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

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            quant_config=quant_config,
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
        # Apply q_norm and k_norm (matches transformers)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_output = self.attn(q, k, v)
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

        # Use separate Q, K, V projections for self-attention to match checkpoint format
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

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            quant_config=quant_config,
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
        q, _ = self.q_proj(hidden_states)
        k_self, _ = self.k_proj(hidden_states)
        v_self, _ = self.v_proj(hidden_states)

        # Apply q_norm and k_norm (matches transformers)
        q = self.q_norm(q)
        k_self = self.k_norm(k_self)

        # Cross-attention: K/V from encoder_hidden_states
        # Note: In transformers, the same k_proj and v_proj are reused for cross-attention
        # In vLLM, we use the same projections for both self and cross-attention
        k_cross, _ = self.k_proj(encoder_hidden_states)
        v_cross, _ = self.v_proj(encoder_hidden_states)

        # Concatenate self and cross K/V along sequence dimension
        k = torch.cat([k_self, k_cross], dim=1)
        v = torch.cat([v_self, v_cross], dim=1)

        # Single attention computation
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
        
        Handles combined qkv_proj weights from checkpoint by splitting them into
        separate q_proj, k_proj, v_proj weights for the SigLIP vision model.
        
        All weights are loaded directly without passing through SigLIP's load_weights
        to properly track which parameters were loaded with their full names.
        
        DEBUG: Added extensive logging to diagnose weight loading issues.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        # DEBUG: Log the expected parameter names
        logger.info("=" * 80)
        logger.info("T5Gemma2VisionEncoder.load_weights - DEBUG START")
        logger.info(f"Number of parameters in params_dict: {len(params_dict)}")
        
        # Collect all expected weight names for debugging
        expected_weights = set(params_dict.keys())
        logger.info(f"Expected weights (first 30): {list(expected_weights)[:30]}")
        
        # Log all vision_tower weights for debugging
        vision_tower_weights = [w for w in expected_weights if "vision_tower" in w]
        logger.info(f"Vision tower weights ({len(vision_tower_weights)}): {vision_tower_weights[:30]}")
        
        # First pass: collect all weights and split combined qkv_proj weights
        qkv_weights: dict[str, dict] = {}
        processed_weights: list[tuple[str, torch.Tensor]] = []
        
        weight_names = list(weights)
        logger.info(f"Number of input weights: {len(weight_names)}")
        logger.info(f"Input weight names (first 30): {[w[0] for w in weight_names[:30]]}")
        
        # Log vision-related weights
        vision_weights = [(n, w) for n, w in weight_names if "vision" in n.lower()]
        logger.info(f"Vision-related weights in checkpoint ({len(vision_weights)}): {[w[0] for w in vision_weights[:30]]}")
        
        # Log qkv weights specifically
        qkv_weights = [(n, w) for n, w in weight_names if "qkv" in n.lower()]
        logger.info(f"QKV weights in checkpoint ({len(qkv_weights)}): {[w[0] for w in qkv_weights[:20]]}")
        
        # Log all input weight names that contain "vision_tower"
        vision_tower_input = [(n, w) for n, w in weight_names if "vision_tower" in n.lower()]
        logger.info(f"Vision tower weights from input ({len(vision_tower_input)}): {[w[0] for w in vision_tower_input[:30]]}")
        
        for name, loaded_weight in weight_names:
            # Handle qkv_proj weights - split into separate q/k/v
            if name.endswith(".qkv_proj.weight") or name.endswith(".qkv_proj.bias"):
                # Extract the base path (e.g., "vision_encoder.vision_tower.vision_model.encoder.layers.0.self_attn")
                base_name = name.rsplit(".", 2)[0]
                weight_type = name.split(".")[-1]  # "weight" or "bias"
                
                if base_name not in qkv_weights:
                    qkv_weights[base_name] = {}
                qkv_weights[base_name][weight_type] = loaded_weight
            else:
                processed_weights.append((name, loaded_weight))
        
        logger.info(f"Found {len(qkv_weights)} qkv_proj weights to split")
        
        # Split combined qkv_proj weights into separate q/k/v
        for base_name, qkv_dict in qkv_weights.items():
            if "weight" not in qkv_dict:
                logger.warning(f"QKV weight at {base_name} missing weight tensor, skipping")
                continue
                
            qkv_weight = qkv_dict["weight"]
            qkv_bias = qkv_dict.get("bias", None)
            
            # Strip the prefix to match params_dict keys - try multiple prefixes
            stripped_base = base_name
            for prefix in ["model.encoder.vision_encoder.", "vision_encoder.", "model.", ""]:
                if stripped_base.startswith(prefix):
                    stripped_base = stripped_base[len(prefix):]
                    logger.debug(f"Stripped prefix '{prefix}' from '{base_name}' -> '{stripped_base}'")
                    break
            
            # Try to find the q_proj weight in params_dict
            q_name = stripped_base + ".q_proj.weight"
            head_dim = None
            q_name_found = None
            if q_name in params_dict:
                head_dim = params_dict[q_name].shape[0]
                q_name_found = q_name
                logger.info(f"Found q_proj at {q_name}, head_dim={head_dim}")
            else:
                # Try alternative naming - search for any q_proj that contains the stripped base
                for param_name in params_dict:
                    if ".q_proj.weight" in param_name and stripped_base in param_name:
                        head_dim = params_dict[param_name].shape[0]
                        q_name_found = param_name
                        logger.info(f"Found q_proj at {param_name}, head_dim={head_dim}")
                        break
            
            if head_dim is None:
                logger.warning(f"Could not find q_proj for base_name: {base_name}, stripped: {stripped_base}")
                logger.warning(f"Available q_proj weights: {[p for p in params_dict if '.q_proj.weight' in p]}")
                # Skip this qkv weight
                continue
            
            # Split the weight into q, k, v
            # qkv_weight shape is [3 * num_heads * head_dim, hidden_size]
            q_weight = qkv_weight[:head_dim, :]
            k_weight = qkv_weight[head_dim:2*head_dim, :]
            v_weight = qkv_weight[2*head_dim:, :]
            
            # Add split weights to processed_weights with the stripped base name
            for name_suffix, weight in [
                (".q_proj.weight", q_weight),
                (".k_proj.weight", k_weight),
                (".v_proj.weight", v_weight),
            ]:
                processed_weights.append((stripped_base + name_suffix, weight))
            
            # Handle bias if present
            if qkv_bias is not None:
                q_bias = qkv_bias[:head_dim]
                k_bias = qkv_bias[head_dim:2*head_dim]
                v_bias = qkv_bias[2*head_dim:]
                
                for name_suffix, bias in [
                    (".q_proj.bias", q_bias),
                    (".k_proj.bias", k_bias),
                    (".v_proj.bias", v_bias),
                ]:
                    processed_weights.append((stripped_base + name_suffix, bias))
        
        logger.info(f"Total processed weights after qkv splitting: {len(processed_weights)}")
        logger.info(f"Processed weight names (first 30): {[w[0] for w in processed_weights[:30]]}")
        
        # Log which processed weights are expected to match
        expected_matches = [w[0] for w in processed_weights if any(w[0] in p for p in params_dict)]
        logger.info(f"Processed weights expected to match ({len(expected_matches)}): {expected_matches[:30]}")
        
        # Log which processed weights won't match
        expected_misses = [w[0] for w in processed_weights if not any(w[0] in p for p in params_dict)]
        logger.info(f"Processed weights NOT expected to match ({len(expected_misses)}): {expected_misses[:30]}")
        
        # Second pass: load all processed weights
        missing_weights = []
        loaded_count = 0
        skipped_count = 0
        
        for name, loaded_weight in processed_weights:
            # Handle mm_input_projection_weight (checkpoint name) -> mm_input_projection_weight (model parameter)
            if name == "mm_input_projection_weight":
                logger.debug(f"Loading mm_input_projection_weight directly")
                param = params_dict["mm_input_projection_weight"]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add("mm_input_projection_weight")
                loaded_count += 1
            # Handle mm_soft_emb_norm weights
            elif name == "mm_soft_emb_norm.weight":
                logger.debug(f"Loading mm_soft_emb_norm.weight directly")
                param = params_dict["mm_soft_emb_norm.weight"]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add("mm_soft_emb_norm.weight")
                loaded_count += 1
            # Handle vision tower weights - strip various prefixes if present
            else:
                weight_name = name
                weight_name_matched = None
                # Try stripping different prefixes
                for prefix in ["model.encoder.vision_encoder.", "vision_encoder.", "model.", ""]:
                    if weight_name.startswith(prefix):
                        potential_name = weight_name[len(prefix):]
                        if potential_name in params_dict:
                            weight_name = potential_name
                            weight_name_matched = prefix
                            logger.debug(f"Matched prefix '{prefix}': '{name}' -> '{weight_name}'")
                            break
                
                if weight_name in params_dict:
                    param = params_dict[weight_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(weight_name)
                    loaded_count += 1
                else:
                    # Weight not found in this module, skip
                    missing_weights.append(name)
                    skipped_count += 1
                    logger.debug(f"Could not find weight for: '{name}' (after prefix stripping: '{weight_name}')")
        
        logger.info(f"Loaded {loaded_count} weights, skipped {skipped_count} weights")
        logger.info(f"Missing weights (first 20): {missing_weights[:20]}")
        logger.info(f"Loaded params (first 20): {list(loaded_params)[:20]}")
        logger.info("T5Gemma2VisionEncoder.load_weights - DEBUG END")
        logger.info("=" * 80)

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

        # Process images if provided
        if pixel_values is not None:
            image_features = self.vision_encoder(pixel_values)

            # Replace image placeholder tokens with image features
            image_token_id = self.config.image_token_index
            image_mask = (input_ids == image_token_id)

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
        # Filter weights for vision_encoder submodule first
        # This handles weights with "vision_encoder." prefix from T5Gemma2Model.load_weights
        vision_encoder_weights = [
            (name[len("vision_encoder."):], weight)
            for name, weight in weights
            if name.startswith("vision_encoder.")
        ]
        # Also handle weights that go directly to vision_tower
        vision_tower_weights = [
            (name, weight)
            for name, weight in weights
            if name.startswith("vision_tower.")
        ]

        # Load vision encoder weights
        loaded_params = set()
        if vision_encoder_weights:
            loaded_params.update(self.vision_encoder.load_weights(vision_encoder_weights))
        if vision_tower_weights:
            loaded_params.update(self.vision_encoder.load_weights(vision_tower_weights))

        # For remaining weights, also check if they belong to submodules like vision_encoder
        # AutoWeightsLoader might pass weights that should go to submodules
        # The checkpoint has weight names like "mm_input_projection_weight" without prefix
        vision_encoder_direct_weights = [
            "mm_input_projection_weight",
            "mm_soft_emb_norm.weight",
        ]
        weights_for_self = []

        for name, weight in weights:
            if name.startswith("vision_encoder.") or name.startswith("vision_tower."):
                # Already handled above, skip
                continue
            # Check if this weight belongs to vision_encoder directly (no prefix)
            if name in vision_encoder_direct_weights:
                remaining = [(name, weight)]
                ve_loaded = self.vision_encoder.load_weights(remaining)
                loaded_params.update(ve_loaded)
            # Check if this weight belongs to a submodule with prefix
            elif name.startswith("multi_modal_projector"):
                # These should go to vision_encoder, not directly to encoder
                # Strip the submodule prefix and pass to vision_encoder
                submodule_name = name.split(".", 1)[1] if "." in name else name
                remaining = [(submodule_name, weight)]
                ve_loaded = self.vision_encoder.load_weights(remaining)
                loaded_params.update(ve_loaded)
            else:
                weights_for_self.append((name, weight))

        # stacked_params_mapping for handling combined gate_up_proj
        # Maps (param_name, shard_name, shard_id)
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())

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
        # stacked_params_mapping for handling combined gate_up_proj
        # Maps (param_name, shard_name, shard_id)
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
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
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


@support_torch_compile
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
        encoder_input_ids: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        encoder_outputs = self.encoder(
            input_ids=encoder_input_ids,
            pixel_values=pixel_values,
        )

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
        loaded_params.update(self.encoder.load_weights(encoder_weights))
        loaded_params.update(self.decoder.load_weights(decoder_weights))
        return loaded_params


class T5Gemma2ForConditionalGeneration(nn.Module, SupportsLoRA, SupportsPP):
    """T5Gemma2 for conditional generation (seq2seq)."""

    packed_modules_mapping = {
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
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
            encoder_input_ids=kwargs.get("encoder_input_ids"),
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
        )
        return decoder_outputs

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.model.decoder.embed_tokens, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # T5Gemma2 has tied weights between encoder and decoder embed_tokens
        # and between lm_head and encoder embed_tokens when tie_word_embeddings=True
        # We handle this by:
        # 1. Routing weights to encoder/decoder model
        # 2. Special handling for eoi_embedding (tied between encoder and decoder)
        # 3. lm_head.weight is tied to encoder.embed_tokens.weight

        # Strip prefixes from weight names before passing to encoder/decoder
        # The checkpoint weights have names like "model.encoder.embed_tokens.weight"
        # but the encoder/decoder modules expect names like "embed_tokens.weight"
        encoder_weights = [
            (name[len("model.encoder."):] if name.startswith("model.encoder.") else name, weight)
            for name, weight in weights
            if name.startswith("model.encoder.") or name.startswith("encoder.")
        ]
        decoder_weights = [
            (name[len("model.decoder."):] if name.startswith("model.decoder.") else name, weight)
            for name, weight in weights
            if name.startswith("model.decoder.") or name.startswith("decoder.")
        ]
        
        # Handle lm_head weights - they should be loaded but tied to encoder.embed_tokens
        lm_head_weights = [
            (name, weight)
            for name, weight in weights
            if name.startswith("model.lm_head.") or name.startswith("lm_head.")
        ]

        loaded_params = set()
        loaded_params.update(self.model.encoder.load_weights(encoder_weights))
        loaded_params.update(self.model.decoder.load_weights(decoder_weights))

        # Handle lm_head - it's tied to encoder.embed_tokens.weight
        # When tie_word_embeddings=True, we load lm_head into encoder.embed_tokens
        if self.config.tie_word_embeddings and lm_head_weights:
            encoder_params = dict(self.model.encoder.named_parameters())
            for name, weight in lm_head_weights:
                # Transform lm_head.out_proj.weight -> embed_tokens.weight
                if name == "lm_head.out_proj.weight" or name == "model.lm_head.out_proj.weight":
                    param_name = "embed_tokens.weight"
                    if param_name in encoder_params:
                        param = encoder_params[param_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, weight)
                        loaded_params.add(param_name)
                        loaded_params.add(name)

        # Handle eoi_embedding - it's tied between encoder and decoder
        # When tie_word_embeddings=True, decoder.embed_tokens.eoi_embedding is tied to encoder.embed_tokens.eoi_embedding
        encoder_params = dict(self.model.encoder.named_parameters())
        decoder_params = dict(self.model.decoder.named_parameters())
        
        # Check if encoder has eoi_embedding and decoder needs it
        if "embed_tokens.eoi_embedding" in encoder_params:
            encoder_eoi = encoder_params["embed_tokens.eoi_embedding"]
            if "embed_tokens.eoi_embedding" in decoder_params:
                decoder_eoi = decoder_params["embed_tokens.eoi_embedding"]
                # Tie them by sharing the same data
                decoder_eoi.data = encoder_eoi.data
                loaded_params.add("embed_tokens.eoi_embedding")

        return loaded_params
