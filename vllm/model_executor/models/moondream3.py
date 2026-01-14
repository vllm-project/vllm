# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Moondream3 model implementation."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import BatchFeature

from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, BaseDummyOptions
from vllm.sequence import IntermediateTensors

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class Moondream3TextConfig:
    """Configuration for Moondream3 text decoder."""

    dim: int = 2048
    ff_dim: int = 8192
    n_layers: int = 24
    vocab_size: int = 51200
    max_context: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 32
    prefix_attn: int = 730  # BOS + 729 vision tokens
    rope_theta: float = 1500000.0
    # MoE configuration
    moe_start_layer: int = 4
    moe_num_experts: int = 64
    moe_experts_per_token: int = 8
    moe_expert_inner_dim: int = 1024

    @classmethod
    def from_dict(cls, d: dict) -> "Moondream3TextConfig":
        text_cfg = d.get("text", d)
        return cls(
            dim=text_cfg.get("dim", 2048),
            ff_dim=text_cfg.get("ff_dim", 8192),
            n_layers=text_cfg.get("n_layers", 24),
            vocab_size=text_cfg.get("vocab_size", 51200),
            max_context=text_cfg.get("max_context", 4096),
            n_heads=text_cfg.get("n_heads", 32),
            n_kv_heads=text_cfg.get("n_kv_heads", 32),
            prefix_attn=text_cfg.get("prefix_attn", 730),
            rope_theta=text_cfg.get("rope_theta", 1500000.0),
            moe_start_layer=text_cfg.get("moe", {}).get("start_layer", 4),
            moe_num_experts=text_cfg.get("moe", {}).get("n_experts", 64),
            moe_experts_per_token=text_cfg.get("moe", {}).get("n_experts_per_tok", 8),
            moe_expert_inner_dim=text_cfg.get("moe", {}).get("expert_inner_dim", 1024),
        )


@dataclass
class Moondream3VisionConfig:
    """Configuration for Moondream3 vision encoder."""

    enc_dim: int = 1152
    enc_patch_size: int = 14
    enc_n_layers: int = 27
    enc_ff_dim: int = 4304
    enc_n_heads: int = 16
    proj_inner_dim: int = 8192
    crop_size: int = 378
    max_crops: int = 12
    overlap_margin: int = 4

    @classmethod
    def from_dict(cls, d: dict) -> "Moondream3VisionConfig":
        vision_cfg = d.get("vision", d)
        return cls(
            enc_dim=vision_cfg.get("enc_dim", 1152),
            enc_patch_size=vision_cfg.get("enc_patch_size", 14),
            enc_n_layers=vision_cfg.get("enc_n_layers", 27),
            enc_ff_dim=vision_cfg.get("enc_ff_dim", 4304),
            enc_n_heads=vision_cfg.get("enc_n_heads", 16),
            proj_inner_dim=vision_cfg.get("proj_inner_dim", 8192),
            crop_size=vision_cfg.get("crop_size", 378),
            max_crops=vision_cfg.get("max_crops", 12),
            overlap_margin=vision_cfg.get("overlap_margin", 4),
        )


@dataclass
class Moondream3Config:
    """Combined configuration for Moondream3 model."""

    text: Moondream3TextConfig
    vision: Moondream3VisionConfig

    @classmethod
    def from_dict(cls, d: dict) -> "Moondream3Config":
        return cls(
            text=Moondream3TextConfig.from_dict(d),
            vision=Moondream3VisionConfig.from_dict(d),
        )


# ============================================================================
# Image Processing Utilities
# ============================================================================


def select_tiling(
    height: int, width: int, crop_size: int, max_crops: int
) -> tuple[int, int]:
    """Determine the optimal number of tiles to cover an image."""
    if height <= crop_size or width <= crop_size:
        return (1, 1)

    min_h = math.ceil(height / crop_size)
    min_w = math.ceil(width / crop_size)

    if min_h * min_w > max_crops:
        ratio = math.sqrt(max_crops / (min_h * min_w))
        return (max(1, math.floor(min_h * ratio)), max(1, math.floor(min_w * ratio)))

    h_tiles = math.floor(math.sqrt(max_crops * height / width))
    w_tiles = math.floor(math.sqrt(max_crops * width / height))

    h_tiles = max(h_tiles, min_h)
    w_tiles = max(w_tiles, min_w)

    if h_tiles * w_tiles > max_crops:
        if w_tiles > h_tiles:
            w_tiles = math.floor(max_crops / h_tiles)
        else:
            h_tiles = math.floor(max_crops / w_tiles)

    return (max(1, h_tiles), max(1, w_tiles))


def overlap_crop_image(
    image: np.ndarray,
    overlap_margin: int,
    max_crops: int,
    base_size: tuple[int, int] = (378, 378),
    patch_size: int = 14,
) -> dict:
    """Process an image using overlap-and-resize cropping strategy."""
    original_h, original_w = image.shape[:2]
    margin_pixels = patch_size * overlap_margin
    total_margin_pixels = margin_pixels * 2

    crop_patches = base_size[0] // patch_size
    crop_window_patches = crop_patches - (2 * overlap_margin)
    crop_window_size = crop_window_patches * patch_size

    tiling = select_tiling(
        original_h - total_margin_pixels,
        original_w - total_margin_pixels,
        crop_window_size,
        max_crops,
    )

    n_crops = tiling[0] * tiling[1] + 1
    crops = np.zeros(
        (n_crops, base_size[0], base_size[1], image.shape[2]), dtype=np.uint8
    )

    target_size = (
        tiling[0] * crop_window_size + total_margin_pixels,
        tiling[1] * crop_window_size + total_margin_pixels,
    )

    # Use PIL for resizing
    pil_img = Image.fromarray(image)
    resized = pil_img.resize(
        (int(target_size[1]), int(target_size[0])),
        resample=Image.Resampling.LANCZOS,
    )
    image = np.asarray(resized)

    # Create global crop
    global_pil = pil_img.resize(
        (int(base_size[1]), int(base_size[0])), resample=Image.Resampling.LANCZOS
    )
    crops[0] = np.asarray(global_pil)

    for i in range(tiling[0]):
        for j in range(tiling[1]):
            y0 = i * crop_window_size
            x0 = j * crop_window_size
            y_end = min(y0 + base_size[0], image.shape[0])
            x_end = min(x0 + base_size[1], image.shape[1])

            crop_region = image[y0:y_end, x0:x_end]
            crops[
                1 + i * tiling[1] + j, : crop_region.shape[0], : crop_region.shape[1]
            ] = crop_region

    return {"crops": crops, "tiling": tiling}


def reconstruct_from_crops(
    crops: torch.Tensor,
    tiling: tuple[int, int],
    overlap_margin: int,
    patch_size: int = 14,
) -> torch.Tensor:
    """Reconstruct features from overlapping crops."""
    tiling_h, tiling_w = tiling
    crop_height, crop_width = crops[0].shape[:2]
    margin_pixels = overlap_margin * patch_size

    output_h = (crop_height - 2 * margin_pixels) * tiling_h + 2 * margin_pixels
    output_w = (crop_width - 2 * margin_pixels) * tiling_w + 2 * margin_pixels

    reconstructed = torch.zeros(
        (output_h, output_w, crops[0].shape[2]),
        device=crops[0].device,
        dtype=crops[0].dtype,
    )

    for i, crop in enumerate(crops):
        tile_y = i // tiling_w
        tile_x = i % tiling_w

        x_start = 0 if tile_x == 0 else margin_pixels
        x_end = crop_width if tile_x == tiling_w - 1 else crop_width - margin_pixels
        y_start = 0 if tile_y == 0 else margin_pixels
        y_end = crop_height if tile_y == tiling_h - 1 else crop_height - margin_pixels

        out_x = tile_x * (crop_width - 2 * margin_pixels)
        out_y = tile_y * (crop_height - 2 * margin_pixels)

        reconstructed[
            out_y + y_start : out_y + y_end, out_x + x_start : out_x + x_end
        ] = crop[y_start:y_end, x_start:x_end]

    return reconstructed


# ============================================================================
# Vision Encoder Components
# ============================================================================


class Moondream3VisionMLP(nn.Module):
    """MLP for vision encoder blocks."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.act = get_act_fn("gelu_pytorch_tanh")
        self.fc2 = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x


class Moondream3VisionAttention(nn.Module):
    """Self-attention for vision encoder (bidirectional).

    Uses native PyTorch scaled_dot_product_attention to avoid
    dependency on vLLM forward context during memory profiling.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = num_heads // tp_size
        self.scale = self.head_dim**-0.5

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass using native PyTorch SDPA.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            output: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        q = q.transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        k = k.transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        v = v.transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention (bidirectional, no mask)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Reshape back to (batch, seq_len, hidden_size)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, -1)

        out, _ = self.out_proj(out)
        return out


class Moondream3VisionBlock(nn.Module):
    """Transformer block for vision encoder."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.attn = Moondream3VisionAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.ln2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = Moondream3VisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Moondream3VisionEncoder(nn.Module):
    """Vision encoder (SigLIP-style ViT)."""

    def __init__(
        self,
        config: Moondream3VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_emb = nn.Linear(
            config.enc_patch_size * config.enc_patch_size * 3,
            config.enc_dim,
            bias=True,
        )

        # Position embeddings (27x27 = 729 patches for 378x378 / 14)
        num_patches = (config.crop_size // config.enc_patch_size) ** 2
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches, config.enc_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Moondream3VisionBlock(
                    hidden_size=config.enc_dim,
                    intermediate_size=config.enc_ff_dim,
                    num_heads=config.enc_n_heads,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{i}",
                )
                for i in range(config.enc_n_layers)
            ]
        )

        self.post_ln = nn.LayerNorm(config.enc_dim, eps=1e-5)

    def create_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to patch embeddings.

        Args:
            images: (batch, channels, height, width)

        Returns:
            patches: (batch, num_patches, patch_dim)
        """
        patch_size = self.config.enc_patch_size
        batch, channels, height, width = images.shape
        patches_h = height // patch_size
        patches_w = width // patch_size

        # Unfold into patches
        patches = images.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size
        )
        # (batch, channels, patches_h, patches_w, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # (batch, patches_h, patches_w, channels, patch_size, patch_size)
        patches = patches.view(batch, patches_h * patches_w, -1)
        # (batch, num_patches, channels * patch_size * patch_size)

        return patches

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images.

        Args:
            pixel_values: (batch, channels, height, width)

        Returns:
            features: (batch, num_patches, hidden_size)
        """
        # Create patches and embed
        patches = self.create_patches(pixel_values)
        x = self.patch_emb(patches)

        # Add position embeddings
        x = x + self.pos_emb

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.post_ln(x)

        return x


class Moondream3VisionProjection(nn.Module):
    """Projects vision features to text embedding dimension."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        output_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        # Input is concatenated global and local features (2 * input_dim)
        self.fc1 = ColumnParallelLinear(
            input_dim * 2,
            inner_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.act = get_act_fn("gelu_pytorch_tanh")
        self.fc2 = RowParallelLinear(
            inner_dim,
            output_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x


# ============================================================================
# Text Decoder Components
# ============================================================================


class Moondream3TextMLP(nn.Module):
    """Standard MLP for non-MoE layers (layers 0-3)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.act = get_act_fn("gelu_pytorch_tanh")
        self.fc2 = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x


class Moondream3TextMoE(nn.Module):
    """Mixture of Experts layer for layers 4+ using FusedMoE.

    Uses FusedMoE for efficient expert parallelism across GPUs.
    Moondream3 checkpoint format:
    - fc1.weight: [num_experts, expert_inner_dim * 2, hidden_size] (gate+up)
    - fc2.weight: [num_experts, hidden_size, expert_inner_dim] (down)
    - router.weight: [num_experts, hidden_size]
    - router.bias: [num_experts]
    """

    def __init__(
        self,
        hidden_size: int,
        expert_inner_dim: int,
        num_experts: int,
        experts_per_token: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_inner_dim = expert_inner_dim
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token

        # Router (gate) - use ReplicatedLinear for compatibility
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        # Use FusedMoE for expert parallelism
        self.experts = FusedMoE(
            num_experts=num_experts,
            top_k=experts_per_token,
            hidden_size=hidden_size,
            intermediate_size=expert_inner_dim,
            reduce_results=True,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            activation="gelu_pytorch_tanh",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get router logits
        router_logits, _ = self.gate(x)  # [num_tokens, num_experts]

        # Use FusedMoE forward
        return self.experts(hidden_states=x, router_logits=router_logits)


class Moondream3Attention(nn.Module):
    """Decoder attention with RoPE."""

    def __init__(
        self,
        config: Moondream3TextConfig,
        layer_idx: int,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.dim
        self.num_heads = config.n_heads
        self.num_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = self.num_heads // tp_size
        self.num_kv_heads_per_partition = max(1, self.num_kv_heads // tp_size)

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )

        self.out_proj = RowParallelLinear(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )

        # Moondream uses 32-dim rotation out of 64-dim head (partial_rotary_factor=0.5)
        rope_parameters = {
            "rope_theta": config.rope_theta,
            "partial_rotary_factor": 32 / self.head_dim,  # 32/64 = 0.5
        }
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_context,
            rope_parameters=rope_parameters,
            is_neox_style=False,  # Moondream uses interleaved RoPE
        )

        self.scaling = self.head_dim**-0.5
        self.attn = Attention(
            num_heads=self.num_heads_per_partition,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads_per_partition,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [
                self.num_heads_per_partition * self.head_dim,
                self.num_kv_heads_per_partition * self.head_dim,
                self.num_kv_heads_per_partition * self.head_dim,
            ],
            dim=-1,
        )

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class Moondream3DecoderLayer(nn.Module):
    """Decoder layer with attention + MLP/MoE."""

    def __init__(
        self,
        config: Moondream3TextConfig,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx

        self.ln = nn.LayerNorm(config.dim, eps=1e-5, bias=True)

        self.attn = Moondream3Attention(
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        # Use MoE for layers >= moe_start_layer, standard MLP otherwise
        if layer_idx >= config.moe_start_layer:
            self.mlp = Moondream3TextMoE(
                hidden_size=config.dim,
                expert_inner_dim=config.moe_expert_inner_dim,
                num_experts=config.moe_num_experts,
                experts_per_token=config.moe_experts_per_token,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Moondream3TextMLP(
                hidden_size=config.dim,
                intermediate_size=config.ff_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-norm architecture
        normed = self.ln(hidden_states)
        attn_out = self.attn(positions, normed)
        mlp_out = self.mlp(normed)
        hidden_states = hidden_states + attn_out + mlp_out
        return hidden_states


class Moondream3TextModel(nn.Module):
    """Text decoder model."""

    def __init__(
        self,
        config: Moondream3TextConfig,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.dim,
            prefix=f"{prefix}.wte",
        )

        blocks_prefix = maybe_prefix(prefix, "blocks")
        self.start_layer, self.end_layer, self.blocks = make_layers(
            config.n_layers,
            lambda layer_prefix: Moondream3DecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=layer_prefix,
            ),
            prefix=blocks_prefix,
        )

        self.post_ln = nn.LayerNorm(config.dim, eps=1e-5, bias=True)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.dim
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        pp_group = get_pp_group()
        if pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.blocks, self.start_layer, self.end_layer):
            hidden_states = layer(positions, hidden_states)

        if not pp_group.is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.post_ln(hidden_states)
        return hidden_states


@dataclass(frozen=True)
class Moondream3ImageInput:
    """Container holding per-image inputs for embedding."""

    pixel_values: torch.Tensor
    tiling: tuple[int, int] | None


# ============================================================================
# Multimodal Processing
# ============================================================================


class Moondream3ProcessingInfo(BaseProcessingInfo):
    """Processing info for Moondream3."""

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        from vllm.transformers_utils.processors.moondream3 import Moondream3Processor

        return self.ctx.get_hf_processor(Moondream3Processor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        # Moondream3 produces 729 vision tokens (27x27)
        return 729

    def get_image_size_with_most_features(self) -> ImageSize:
        return ImageSize(width=378, height=378)

    def get_max_image_tokens(self) -> int:
        return 729


class Moondream3DummyInputsBuilder(BaseDummyInputsBuilder[Moondream3ProcessingInfo]):
    """Dummy inputs builder for profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<image>\n\nQuestion: What is this?\n\nAnswer:"

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        return {
            "image": self._get_dummy_images(
                width=378,
                height=378,
                num_images=num_images,
            )
        }


class Moondream3MultiModalProcessor(BaseMultiModalProcessor[Moondream3ProcessingInfo]):
    """Multimodal processor for Moondream3."""

    image_placeholder: str = "<image>"

    @cached_property
    def image_placeholder_tokens(self) -> list[int]:
        tokenizer = self.info.get_tokenizer()
        token_ids = tokenizer.encode(
            self.image_placeholder,
            add_special_tokens=False,
        )
        if not token_ids:
            raise ValueError(
                f"Tokenizer could not encode placeholder {self.image_placeholder!r}."
            )
        return token_ids

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "tilings": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptUpdate]:
        image_size = self.info.get_image_size_with_most_features()
        num_tokens = self.info.get_num_image_tokens(
            image_width=image_size.width,
            image_height=image_size.height,
        )
        return [
            PromptReplacement(
                modality="image",
                target=self.image_placeholder_tokens,
                replacement=self.image_placeholder_tokens * num_tokens,
            ),
        ]


# ============================================================================
# Main Model
# ============================================================================


@MULTIMODAL_REGISTRY.register_processor(
    Moondream3MultiModalProcessor,
    info=Moondream3ProcessingInfo,
    dummy_inputs=Moondream3DummyInputsBuilder,
)
class Moondream3ForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    """Moondream3 multimodal model for causal language modeling."""

    supports_multimodal = True
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        hf_config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        # Parse config from HuggingFace config
        config_dict = hf_config.config if hasattr(hf_config, "config") else {}

        self.config = Moondream3Config.from_dict(config_dict)

        # Vision encoder
        self.vision = Moondream3VisionEncoder(
            config=self.config.vision,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision"),
        )

        # Vision projection
        self.vision_proj = Moondream3VisionProjection(
            input_dim=self.config.vision.enc_dim,
            inner_dim=self.config.vision.proj_inner_dim,
            output_dim=self.config.text.dim,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision_proj"),
        )

        # Text decoder
        self.text = Moondream3TextModel(
            config=self.config.text,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "text"),
        )

        # LM head
        self.lm_head = ParallelLMHead(
            self.config.text.vocab_size,
            self.config.text.dim,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        self.logits_processor = LogitsProcessor(self.config.text.vocab_size)
        self.make_empty_intermediate_tensors = self.text.make_empty_intermediate_tensors

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<image>"
        return None

    def get_language_model(self) -> nn.Module:
        return self.text

    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int:
        return num_image_tokens

    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int:
        return num_vision_tokens

    def _split_pixel_values(
        self,
        pixel_values: object,
    ) -> list[torch.Tensor]:
        if isinstance(pixel_values, torch.Tensor):
            if pixel_values.dim() == 5:
                return [pv.contiguous() for pv in pixel_values]
            if pixel_values.dim() == 4:
                return [pixel_values.contiguous()]
            if pixel_values.dim() == 3:
                return [pixel_values.unsqueeze(0).contiguous()]
            raise ValueError(
                f"Unsupported pixel_values shape {tuple(pixel_values.shape)}."
            )

        if isinstance(pixel_values, (list, tuple)):
            tensors: list[torch.Tensor] = []
            for value in pixel_values:
                tensor_value = (
                    value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                )
                if tensor_value.dim() == 3:
                    tensor_value = tensor_value.unsqueeze(0)
                elif tensor_value.dim() != 4:
                    raise ValueError(
                        f"Unsupported pixel_values element shape "
                        f"{tuple(tensor_value.shape)}."
                    )
                tensors.append(tensor_value.contiguous())
            return tensors

        raise TypeError(
            "pixel_values must be a tensor or a sequence of tensors, "
            f"got {type(pixel_values)!r}."
        )

    def _split_tilings(
        self,
        tilings: object,
        expected: int,
    ) -> list[tuple[int, int] | None]:
        if tilings is None:
            return [None] * expected

        if isinstance(tilings, torch.Tensor):
            tiling_items = tilings.tolist()
        elif isinstance(tilings, (list, tuple)):
            tiling_items = list(tilings)
        else:
            raise TypeError(
                "tilings must be None, a tensor or a sequence of tuples, "
                f"got {type(tilings)!r}."
            )

        if len(tiling_items) != expected:
            raise ValueError(
                "Mismatch between the number of pixel_values entries "
                f"({expected}) and tilings ({len(tiling_items)})."
            )

        normalized: list[tuple[int, int] | None] = []
        for tiling in tiling_items:
            if tiling is None:
                normalized.append(None)
                continue
            if isinstance(tiling, torch.Tensor):
                tiling = tiling.tolist()
            if isinstance(tiling, (list, tuple)) and len(tiling) == 2:
                normalized.append((int(tiling[0]), int(tiling[1])))
            else:
                raise ValueError(
                    f"Each tiling entry must be a pair of integers, got {tiling!r}."
                )
        return normalized

    def _parse_image_inputs(self, **kwargs: object) -> list[Moondream3ImageInput]:
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is None:
            return []

        pixel_values_list = self._split_pixel_values(pixel_values)
        tilings_list = self._split_tilings(
            kwargs.get("tilings"), len(pixel_values_list)
        )

        image_inputs: list[Moondream3ImageInput] = []
        for value, tiling in zip(pixel_values_list, tilings_list):
            if value.dim() != 4:
                raise ValueError(
                    f"Expected 4D tensor for crops, got {tuple(value.shape)}."
                )
            image_inputs.append(Moondream3ImageInput(pixel_values=value, tiling=tiling))
        return image_inputs

    def _encode_image_input(self, image_input: Moondream3ImageInput) -> torch.Tensor:
        pixel_values = image_input.pixel_values
        if pixel_values.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor for crops, got {tuple(pixel_values.shape)}."
            )

        device = self.vision.patch_emb.weight.device
        dtype = self.vision.patch_emb.weight.dtype
        pixel_values = pixel_values.to(device=device, dtype=dtype)

        features = self.vision(pixel_values)
        grid_size = self.config.vision.enc_n_layers
        enc_dim = self.config.vision.enc_dim
        global_features = features[0]

        if features.shape[0] > 1:
            if image_input.tiling is None:
                raise ValueError(
                    "Missing tiling metadata for multi-crop Moondream image."
                )
            local = features[1:].contiguous().view(-1, grid_size, grid_size, enc_dim)
            reconstructed = reconstruct_from_crops(
                local,
                image_input.tiling,
                overlap_margin=self.config.vision.overlap_margin,
                patch_size=1,
            )
        else:
            reconstructed = global_features.view(grid_size, grid_size, enc_dim)

        recon = reconstructed.permute(2, 0, 1).contiguous()
        recon = F.adaptive_avg_pool2d(recon, output_size=(grid_size, grid_size))
        recon = recon.permute(1, 2, 0).contiguous().view(-1, enc_dim)

        combined = torch.cat([global_features, recon], dim=-1).unsqueeze(0)
        projected = self.vision_proj(combined).squeeze(0)
        return projected

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Generate embeddings from multimodal inputs."""
        image_inputs = self._parse_image_inputs(**kwargs)
        if not image_inputs:
            return []

        return [self._encode_image_input(image_input) for image_input in image_inputs]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.text(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with remapping from HuggingFace format."""

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Get expert intermediate size for fc1 splitting
        expert_inner_dim = self.config.text.moe_expert_inner_dim

        for name, loaded_weight in weights:
            # Skip tau weights (not implemented in MVP)
            if ".tau." in name:
                continue

            # Skip region weights (not implemented in MVP)
            if ".region." in name:
                continue

            # Map from HF naming to vLLM naming
            # model.vision.* -> vision.*
            # model.text.* -> text.*
            if name.startswith("model."):
                name = name[6:]  # Remove "model." prefix

            # Specific name mappings
            # Vision projection: vision.proj_mlp.fc1 -> vision_proj.fc1
            name = name.replace("vision.proj_mlp.", "vision_proj.")

            # Text embedding: text.wte (no suffix) -> text.wte.weight
            if name == "text.wte":
                name = "text.wte.weight"

            # LM head: text.lm_head -> lm_head
            name = name.replace("text.lm_head.", "lm_head.")

            # Attention mapping
            name = name.replace(".attn.qkv.", ".attn.qkv_proj.")
            name = name.replace(".attn.proj.", ".attn.out_proj.")

            # MoE router mapping: mlp.router -> mlp.gate
            name = name.replace(".mlp.router.", ".mlp.gate.")

            # Handle MoE expert weights for layers 4+ (FusedMoE format)
            # fc1.weight: [n_experts, gate_up_dim, hidden] -> w13 (split gate/up)
            # fc2.weight: [n_experts, hidden, expert_dim] -> w2
            # Note: Only 3D weights are MoE, 2D weights are standard MLP
            if ".mlp.fc1.weight" in name and loaded_weight.dim() == 3:
                # fc1 is combined gate+up, need to split and load via FusedMoE
                # Shape: [n_experts, expert_inner_dim * 2, hidden_size]
                layer_name = name.replace(".mlp.fc1.weight", "")
                experts_name = f"{layer_name}.mlp.experts"

                # Find the FusedMoE module
                module = self
                for part in experts_name.split("."):
                    if part:
                        module = getattr(module, part)

                # Split fc1 into gate (w1) and up (w3)
                # fc1[:, :expert_inner_dim, :] is gate
                # fc1[:, expert_inner_dim:, :] is up
                gate_weight = loaded_weight[:, :expert_inner_dim, :]
                up_weight = loaded_weight[:, expert_inner_dim:, :]

                # Load gate (w1) for each expert
                for expert_id in range(loaded_weight.shape[0]):
                    module.weight_loader(
                        module.w13_weight,
                        gate_weight[expert_id],
                        f"{experts_name}.w13_weight",
                        "w1",
                        expert_id,
                    )
                    module.weight_loader(
                        module.w13_weight,
                        up_weight[expert_id],
                        f"{experts_name}.w13_weight",
                        "w3",
                        expert_id,
                    )

                loaded_params.add(f"{layer_name}.mlp.experts.w13_weight")
                continue

            if ".mlp.fc2.weight" in name and loaded_weight.dim() == 3:
                # fc2 is down projection
                # Shape: [n_experts, hidden_size, expert_inner_dim]
                layer_name = name.replace(".mlp.fc2.weight", "")
                experts_name = f"{layer_name}.mlp.experts"

                # Find the FusedMoE module
                module = self
                for part in experts_name.split("."):
                    if part:
                        module = getattr(module, part)

                # Load w2 for each expert
                for expert_id in range(loaded_weight.shape[0]):
                    module.weight_loader(
                        module.w2_weight,
                        loaded_weight[expert_id],
                        f"{experts_name}.w2_weight",
                        "w2",
                        expert_id,
                    )

                loaded_params.add(f"{layer_name}.mlp.experts.w2_weight")
                continue

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params
