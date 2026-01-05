# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Iterable, Sequence
from typing import cast

import torch
import torch.nn as nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_AUDIO_LEN_S = 655
DEFAULT_MERGE_FACTOR = 4
# Default convolution parameters: (padding, kernel_size, stride)
# These correspond to the two conv layers in GlmAsrEncoder
DEFAULT_CONV_PARAMS = [(1, 3, 1), (1, 3, 2)]


class _GlmAsrEncoderOutput:
    """Simple output container compatible with transformers' BaseModelOutput."""

    __slots__ = ("last_hidden_state",)

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


def _calculate_conv_output_length(
    input_length: torch.Tensor, padding: int, kernel_size: int, stride: int
) -> torch.Tensor:
    """Calculate Conv1d output length using standard formula."""
    # Standard formula: floor((input + 2*padding - kernel_size) / stride) + 1
    return (input_length + 2 * padding - kernel_size) // stride + 1


def _as_list_chunk_counts(
    chunk_counts: torch.Tensor | list[int] | list[torch.Tensor],
) -> list[int]:
    if isinstance(chunk_counts, torch.Tensor):
        return chunk_counts.tolist()
    if chunk_counts and isinstance(chunk_counts[0], torch.Tensor):
        tensor_counts = cast(list[torch.Tensor], chunk_counts)
        return [int(c.item()) for c in tensor_counts]
    return [int(c) for c in chunk_counts]


def _normalize_chunk_counts(
    chunk_counts: torch.Tensor | list[int] | list[torch.Tensor] | None,
    num_chunks: int,
) -> list[int]:
    if chunk_counts is None:
        return [1] * num_chunks
    return _as_list_chunk_counts(chunk_counts)


def _get_audio_output_lengths_from_lengths(
    audio_lengths: torch.Tensor,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> torch.Tensor:
    for padding, kernel_size, stride in conv_params:
        audio_lengths = _calculate_conv_output_length(
            audio_lengths, padding, kernel_size, stride
        )
    return (audio_lengths - merge_factor) // merge_factor + 1


def _get_audio_output_lengths_from_mask(
    mask: torch.Tensor,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> torch.Tensor:
    audio_lengths = mask.sum(-1)
    return _get_audio_output_lengths_from_lengths(
        audio_lengths, merge_factor, conv_params
    )


def _get_audio_output_lengths_for_tower(
    audio_tower: nn.Module,
    audio_lengths: torch.Tensor,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> torch.Tensor:
    """
    Calculate the output lengths after audio processing.

    The output length accounts for:
    1. Convolution layers (downsampling)
    2. Merge factor (further downsampling during projection)

    Args:
        audio_tower: The audio encoder module
        audio_lengths: Input feature lengths [batch_size]
        merge_factor: Factor for merging adjacent features
        conv_params: List of (padding, kernel_size, stride) for each conv layer

    Returns:
        Output lengths after all processing [batch_size]
    """
    # First, calculate the output length after convolutions
    if hasattr(audio_tower, "_get_feat_extract_output_lengths"):
        _, conv_output_lengths = audio_tower._get_feat_extract_output_lengths(
            audio_lengths
        )
    else:
        conv_output_lengths = audio_lengths
        for padding, kernel_size, stride in conv_params:
            conv_output_lengths = _calculate_conv_output_length(
                conv_output_lengths, padding, kernel_size, stride
            )

    # Then, apply merge_factor to get final output length
    # Formula: (conv_output_lengths - merge_factor) // merge_factor + 1
    return (conv_output_lengths - merge_factor) // merge_factor + 1


def _flatten_audio_features_by_length(
    audio_features: torch.Tensor,
    audio_output_lengths: torch.Tensor,
) -> torch.Tensor:
    num_chunks, max_audio_tokens, embed_dim = audio_features.shape
    audio_output_lengths = audio_output_lengths.unsqueeze(1)
    audio_features_mask = (
        torch.arange(max_audio_tokens)
        .expand(num_chunks, max_audio_tokens)
        .to(audio_output_lengths.device)
        < audio_output_lengths
    )
    return audio_features[audio_features_mask].view(-1, embed_dim)


def _group_audio_embeddings(
    chunk_embeddings: Sequence[torch.Tensor],
    chunk_counts: Sequence[int],
) -> tuple[torch.Tensor, ...]:
    grouped_embeddings = []
    current_idx = 0
    for count in chunk_counts:
        audio_chunks = chunk_embeddings[current_idx : current_idx + count]
        grouped_embeddings.append(torch.cat(audio_chunks, dim=0))
        current_idx += count
    return tuple(grouped_embeddings)


def _normalize_to_tensor(mask: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Convert mask to tensor, handling both list and tensor formats."""
    if isinstance(mask, list):
        return (
            torch.stack(mask)
            if mask and isinstance(mask[0], torch.Tensor)
            else torch.tensor(mask)
        )
    return mask


def _extract_mask_for_item(
    feature_attention_mask: torch.Tensor | list[torch.Tensor],
    chunk_counts: torch.Tensor | list[int] | None,
    item_idx: int,
) -> torch.Tensor:
    """Extract attention mask for a specific audio item."""
    if chunk_counts is None:
        # Single item per audio
        mask = feature_attention_mask[item_idx]
        if isinstance(feature_attention_mask, torch.Tensor):
            return mask.unsqueeze(0)
        return _normalize_to_tensor(mask)

    # Multiple chunks per audio: calculate slice indices
    counts = _as_list_chunk_counts(chunk_counts)
    start_idx = sum(counts[:item_idx])
    end_idx = start_idx + counts[item_idx]

    # Extract slice
    if isinstance(feature_attention_mask, torch.Tensor):
        return feature_attention_mask[start_idx:end_idx]
    mask_slice = feature_attention_mask[start_idx:end_idx]
    return _normalize_to_tensor(mask_slice)


def _get_num_features_for_item(
    feature_attention_mask: torch.Tensor | None,
    chunk_counts: torch.Tensor | list[int] | None,
    item_idx: int,
    audio_embeds: list[torch.Tensor] | None,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> int:
    """Get number of features for a specific audio item."""
    if feature_attention_mask is not None:
        mask = _extract_mask_for_item(feature_attention_mask, chunk_counts, item_idx)
        audio_output_lengths = _get_audio_output_lengths_from_mask(
            mask, merge_factor, conv_params
        )
        return audio_output_lengths.sum().item()
    if audio_embeds is not None:
        return audio_embeds[item_idx].shape[0]
    raise ValueError("Either feature_attention_mask or audio_embeds must be provided")


# ============================================================================
# Optimized vLLM Native GlmAsrEncoder Implementation
# ============================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Follows transformers' apply_rotary_pos_emb exactly.
    Supports partial rotary where only the first rotary_dim of head_dim is rotated.

    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        cos: [batch, seq_len, rotary_dim]
        sin: [batch, seq_len, rotary_dim]
    """
    # unsqueeze_dim=1 to add head dimension: [batch, 1, seq_len, rotary_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Get the rotary dimension from cos/sin
    rotary_dim = cos.shape[-1]

    # Split into rotary and pass-through parts
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


class GlmAsrRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding for GLM-ASR encoder.

    Optimized with pre-computed cos/sin cache for better performance.
    Falls back to dynamic computation only when sequence length exceeds cache.
    """

    def __init__(self, config, device: torch.device | None = None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings

        # Compute inverse frequencies following transformers implementation
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        # Handle rope_parameters if present (for compatibility with transformers config)
        if hasattr(config, "rope_parameters") and config.rope_parameters:
            base = config.rope_parameters.get("rope_theta", 10000.0)
            partial_rotary_factor = config.rope_parameters.get(
                "partial_rotary_factor", 1.0
            )
            dim = int(head_dim * partial_rotary_factor)
            self.attention_scaling = config.rope_parameters.get(
                "attention_scaling", 1.0
            )
        else:
            base = getattr(config, "rope_theta", 10000.0)
            dim = head_dim
            self.attention_scaling = 1.0

        self.dim = dim
        self.base = base

        # Compute the inverse frequencies exactly as transformers does
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin cache for efficiency
        self._set_cos_sin_cache(self.max_seq_len_cached, device)

    def _set_cos_sin_cache(
        self, seq_len: int, device: torch.device | None = None
    ) -> None:
        """Pre-compute cos and sin cache for given sequence length."""
        self.max_seq_len_cached = seq_len

        # Create position indices
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        # Compute frequencies: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        # Double the frequencies: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        # Compute and cache cos/sin
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings with caching optimization.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            position_ids: Position indices [batch_size, seq_len]

        Returns:
            Tuple of (cos, sin) tensors with shape [batch_size, seq_len, rotary_dim]
        """
        seq_len = position_ids.shape[-1]

        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device)

        # Use cached values - index with position_ids for correctness
        # For encoder, position_ids is typically [0, 1, 2, ..., seq_len-1]
        # so we can directly slice the cache
        cos = self.cos_cached[:seq_len].unsqueeze(0)  # [1, seq_len, dim]
        sin = self.sin_cached[:seq_len].unsqueeze(0)  # [1, seq_len, dim]

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors for Grouped Query Attention.

    Args:
        hidden_states: [batch, num_kv_heads, seq_len, head_dim]
        n_rep: Number of repetitions

    Returns:
        [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class GlmAsrAttention(nn.Module):
    """
    Optimized Multi-headed Grouped Query Attention for GLM-ASR.
    Uses vLLM's QKVParallelLinear for better performance.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_rank = self.num_heads // self.tp_size
        self.num_kv_heads_per_rank = max(1, self.num_kv_heads // self.tp_size)

        # Use QKVParallelLinear for fused QKV projection
        # Note: GLM-ASR uses bias on Q and V, but not K
        # For simplicity with QKVParallelLinear, we use bias=True for all
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) for RoPE

        Returns:
            [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection - fused for efficiency
        qkv, _ = self.qkv_proj(hidden_states)

        # Split into q, k, v
        q_size = self.num_heads_per_rank * self.head_dim
        kv_size = self.num_kv_heads_per_rank * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Reshape and transpose
        # [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
        q = q.view(
            batch_size, seq_len, self.num_heads_per_rank, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, seq_len, self.num_kv_heads_per_rank, self.head_dim
        ).transpose(1, 2)
        # v doesn't go through RoPE, so make it contiguous now for SDPA
        v = (
            v.view(batch_size, seq_len, self.num_kv_heads_per_rank, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # Apply rotary position embeddings
        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # Handle GQA: repeat k/v if needed
        if self.num_kv_groups > 1:
            k = _repeat_kv(k, self.num_kv_groups)
            v = _repeat_kv(v, self.num_kv_groups)

        # Ensure contiguous for optimal SDPA/Flash Attention performance
        # Non-contiguous tensors can cause fallback to slower implementations
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


class GlmAsrMLP(nn.Module):
    """
    Optimized MLP for GLM-ASR encoder.
    Uses vLLM's parallel linear layers for better performance.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )

        self.act_fn = get_act_fn(config.hidden_act)

        self.fc2 = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class GlmAsrEncoderLayer(nn.Module):
    """
    Optimized Transformer encoder layer for GLM-ASR.
    Combines attention and MLP with residual connections and layer norms.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GlmAsrAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = GlmAsrMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        layer_norm_eps = getattr(config, "layer_norm_eps", 1e-5)
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            self.hidden_size, eps=layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) for RoPE

        Returns:
            [batch_size, seq_len, hidden_size]
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GlmAsrEncoder(nn.Module):
    """
    Optimized GLM-ASR Audio Encoder with vLLM native implementation.

    This encoder processes audio features through convolutional layers
    followed by transformer layers with rotary position embeddings.
    Optimized for performance with:
    - QKVParallelLinear for fused attention projections
    - Tensor parallelism support via ColumnParallelLinear/RowParallelLinear
    - Quantization support
    - Flash Attention (SDPA)
    """

    # Mapping for weight loading: transformers uses separate q/k/v, we use fused qkv
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        # Convolutional feature extraction layers
        self.conv1 = nn.Conv1d(
            config.num_mel_bins,
            config.hidden_size,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                GlmAsrEncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Final layer norm
        layer_norm_eps = getattr(config, "layer_norm_eps", 1e-5)
        self.norm = nn.LayerNorm(config.hidden_size, eps=layer_norm_eps)

        # Rotary position embeddings
        self.rotary_emb = GlmAsrRotaryEmbedding(config)

        # Pre-register position_ids buffer for efficiency
        # This avoids creating a new tensor on every forward pass
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

    def _get_feat_extract_output_lengths(
        self, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the output length after convolutions.

        Args:
            input_lengths: Input sequence lengths [batch_size]

        Returns:
            Tuple of (output after conv1, output after conv2)
        """
        # Conv1: kernel=3, stride=1, padding=1
        output_lengths = (input_lengths + 2 * 1 - 3) // 1 + 1

        # Conv2: kernel=3, stride=2, padding=1
        output_lengths = (output_lengths + 2 * 1 - 3) // 2 + 1

        return input_lengths, output_lengths

    def forward(self, input_features: torch.Tensor):
        """
        Forward pass through the encoder.

        Args:
            input_features: [batch_size, num_mel_bins, seq_len]

        Returns:
            Object with .last_hidden_state attribute containing
            [batch_size, seq_len', hidden_size] where seq_len' is
            the sequence length after convolutions
        """
        # Apply convolutional layers with GELU activation
        hidden_states = torch.nn.functional.gelu(self.conv1(input_features))
        hidden_states = torch.nn.functional.gelu(self.conv2(hidden_states))

        # Transpose to [batch_size, seq_len, hidden_size]
        hidden_states = hidden_states.transpose(1, 2)
        output_seq_len = hidden_states.shape[1]

        # Use pre-registered position_ids buffer (slice to actual seq_len)
        position_ids = self.position_ids[:, :output_seq_len]

        # Get position embeddings - uses pre-computed cache
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Apply transformer layers
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, position_embeddings)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Return in a format compatible with transformers' BaseModelOutput
        return _GlmAsrEncoderOutput(last_hidden_state=hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Custom weight loading to handle q_proj/k_proj/v_proj -> qkv_proj mapping."""
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Default weight loading for non-stacked params
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
