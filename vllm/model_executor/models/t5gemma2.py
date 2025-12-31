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
from vllm.attention.layers.cross_attention import CrossAttention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import GeluAndMul, get_act_fn
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.siglip import load_siglip_vision_model
from vllm.sequence import IntermediateTensors

from .gemma2 import Gemma2Attention, Gemma2MLP
from .utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def get_t5gemma2_text_config(config: T5Gemma2Config) -> dict:
    """Extract text config from T5Gemma2Config for vLLM."""
    text_config = config.text_config
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
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,
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
    """Attention module with sliding window support."""

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
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

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
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class T5Gemma2MergedAttention(nn.Module):
    """Merged self-attention and cross-attention for decoder.

    This fuses self-attention and cross-attention into a single operation
    to match the transformers implementation. The key and value states
    from self-attention and cross-attention are concatenated before
    the attention computation.
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
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Self-attention QKV projection
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        # Cross-attention K/V projections (from encoder hidden states)
        self.k_proj = nn.Linear(
            cross_attention_hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            cross_attention_hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

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
        qkv, _ = self.qkv_proj(hidden_states)
        q, k_self, v_self = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Cross-attention: K/V from encoder_hidden_states
        k_cross = self.k_proj(encoder_hidden_states)
        v_cross = self.v_proj(encoder_hidden_states)

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
        self.input_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = GemmaRMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class T5Gemma2VisionEncoder(nn.Module):
    """SigLIP vision encoder for T5Gemma2 multimodal inputs."""

    def __init__(self, config: T5Gemma2Config, quant_config: QuantizationConfig | None = None):
        super().__init__()
        self.config = config
        vision_config = config.vision_config

        # Load SigLIP vision model
        self.vision_tower = load_siglip_vision_model(vision_config)

        # Multi-modal projector (maps vision features to text hidden size)
        self.multi_modal_projector = nn.Linear(
            vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=False,
        )

        # Initialize projector weights to zeros (matches transformers)
        nn.init.zeros_(self.multi_modal_projector.weight)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract and project vision features."""
        vision_outputs = self.vision_tower(pixel_values=pixel_values)
        image_features = self.multi_modal_projector(vision_outputs.last_hidden_state)
        return image_features

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for vision tower and projector."""
        # Delegate to vision tower's load_weights
        return self.vision_tower.load_weights(weights)


class T5Gemma2Encoder(nn.Module):
    """T5Gemma2 encoder with vision and bidirectional attention."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: T5Gemma2Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        text_config = get_t5gemma2_text_config(config)

        self.config = config
        self.padding_idx = config.text_config.pad_token_id

        # Embed tokens with EOI handling
        self.embed_tokens = T5Gemma2TextScaledWordEmbedding(
            text_config["vocab_size"],
            text_config["hidden_size"],
            self.padding_idx,
            embed_scale=text_config["hidden_size"] ** 0.5,
            eoi_token_index=config.eoi_token_index,
        )

        # Vision encoder
        self.vision_encoder = T5Gemma2VisionEncoder(config, quant_config)

        # Encoder layers with bidirectional attention
        self.start_layer, self.end_layer, self.layers = make_layers(
            text_config["num_hidden_layers"],
            lambda p: T5Gemma2EncoderLayer(
                text_config, layer_idx=p.split(".")[-1] if "." in p else 0,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                prefix=p,
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
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Handle vision encoder weights
            if name.startswith("vision_encoder."):
                try:
                    loaded = self.vision_encoder.load_weights([(name, loaded_weight)])
                    loaded_params.update(loaded)
                except Exception:
                    pass

            # Handle text encoder weights
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
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

        self.input_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = GemmaRMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"]
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Merged attention (self + cross)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            positions=positions,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return hidden_states, residual

    def _init_layernorms(self, config: dict):
        """Initialize layer norms."""
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"]
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"]
        )


class T5Gemma2Decoder(nn.Module):
    """T5Gemma2 decoder with merged attention support."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: T5Gemma2Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        text_config = get_t5gemma2_text_config(config)

        self.config = text_config

        self.embed_tokens = T5Gemma2TextScaledWordEmbedding(
            text_config["vocab_size"],
            text_config["hidden_size"],
            config.text_config.pad_token_id,
            embed_scale=text_config["hidden_size"] ** 0.5,
            eoi_token_index=config.eoi_token_index,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            text_config["num_hidden_layers"],
            lambda p: T5Gemma2DecoderLayer(
                text_config,
                layer_idx=int(p.split(".")[-1]) if "." in p else 0,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                prefix=p,
            ),
            prefix=f"{prefix}.layers",
        )

        # Initialize layer norms for each decoder layer
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer._init_layernorms(text_config)

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
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
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
        encoder_weights = [
            (name, weight) for name, weight in weights if name.startswith("encoder.")
        ]
        decoder_weights = [
            (name, weight) for name, weight in weights if name.startswith("decoder.")
        ]
        loaded_params = set()
        loaded_params.update(self.encoder.load_weights(encoder_weights))
        loaded_params.update(self.decoder.load_weights(decoder_weights))
        return loaded_params


class T5Gemma2ForConditionalGeneration(nn.Module, SupportsLoRA, SupportsPP):
    """T5Gemma2 for conditional generation (seq2seq)."""

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
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
            config.text_config.vocab_size,
            soft_cap=config.text_config.final_logit_softcapping,
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
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
