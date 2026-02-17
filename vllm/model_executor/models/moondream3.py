# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Moondream3 model implementation."""

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import Attention
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
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
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
    prefix_lm_left_padding: int = 1  # include BOS in prefix-lm span
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
            prefix_lm_left_padding=text_cfg.get("prefix_lm_left_padding", 1),
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
class Moondream3RegionConfig:
    """Configuration for Moondream3 region module (point/detect)."""

    dim: int = 2048
    coord_feat_dim: int = 256
    coord_out_dim: int = 1024
    size_feat_dim: int = 512
    size_out_dim: int = 2048

    @classmethod
    def from_dict(cls, d: dict) -> "Moondream3RegionConfig":
        region_cfg = d.get("region", d)
        return cls(
            dim=region_cfg.get("dim", 2048),
            coord_feat_dim=region_cfg.get("coord_feat_dim", 256),
            coord_out_dim=region_cfg.get("coord_out_dim", 1024),
            size_feat_dim=region_cfg.get("size_feat_dim", 512),
            size_out_dim=region_cfg.get("size_out_dim", 2048),
        )


@dataclass
class Moondream3Config:
    """Combined configuration for Moondream3 model."""

    text: Moondream3TextConfig
    vision: Moondream3VisionConfig
    region: Moondream3RegionConfig

    @classmethod
    def from_dict(cls, d: dict) -> "Moondream3Config":
        return cls(
            text=Moondream3TextConfig.from_dict(d),
            vision=Moondream3VisionConfig.from_dict(d),
            region=Moondream3RegionConfig.from_dict(d),
        )


# ============================================================================
# Image Processing Utilities
# ============================================================================


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


class Moondream3RegionModule(nn.Module):
    """Region module for coordinate encoding/decoding (point/detect).

    This module handles Fourier feature encoding of coordinates and sizes
    for the point and detect capabilities. The weights are loaded for
    completeness, but point/detect generation requires a custom decoding
    loop that is incompatible with vLLM's standard serving pipeline.

    The module is small (~14M params) and uses plain nn.Linear layers
    (replicated on all TP ranks, no parallelization needed).
    """

    def __init__(self, config: Moondream3RegionConfig, prefix: str = ""):
        super().__init__()
        self._config = config
        # Fourier frequency matrices for coordinate/size encoding
        self.coord_features = nn.Parameter(torch.empty(1, config.coord_feat_dim // 2))
        self.size_features = nn.Parameter(torch.empty(2, config.size_feat_dim // 2))

        # Coordinate encoder/decoder
        self.coord_encoder = nn.Linear(config.coord_feat_dim, config.dim)
        self.coord_decoder = nn.Linear(config.dim, config.coord_out_dim)

        # Size encoder/decoder
        self.size_encoder = nn.Linear(config.size_feat_dim, config.dim)
        self.size_decoder = nn.Linear(config.dim, config.size_out_dim)

        # Layer norm
        self.ln = nn.LayerNorm(config.dim)

    def _fourier_features(
        self, x: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """Fourier feature mapping: x @ w -> cat(cos, sin)."""
        f = 2 * math.pi * x @ w
        return torch.cat([f.cos(), f.sin()], dim=-1)

    def encode_coordinate(self, coord_value: torch.Tensor) -> torch.Tensor:
        """Encode a coordinate value into an embedding.

        Args:
            coord_value: Scalar or tensor of shape [..., 1] with float
                coordinate values in [0, 1].

        Returns:
            Embedding of shape [..., dim].
        """
        if coord_value.dim() == 0:
            coord_value = coord_value.reshape(1, 1)
        elif coord_value.dim() == 1:
            coord_value = coord_value.unsqueeze(-1)
        feat = self._fourier_features(coord_value, self.coord_features)
        return self.coord_encoder(feat)

    def decode_coordinate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Decode coordinate logits from hidden states.

        Args:
            hidden_states: Tensor of shape [..., dim].

        Returns:
            Logits of shape [..., coord_out_dim] (1024 bins).
        """
        return self.coord_decoder(self.ln(hidden_states))

    def encode_size(self, w: float, h: float,
                    device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
        """Encode width and height into an embedding.

        Args:
            w: Width value (float).
            h: Height value (float).
            device: Target device.
            dtype: Target dtype.

        Returns:
            Embedding of shape [1, dim].
        """
        size = torch.tensor([w, h], device=device, dtype=dtype)
        feat = self._fourier_features(size, self.size_features)
        return self.size_encoder(feat.unsqueeze(0))

    def decode_size(
        self, hidden_states: torch.Tensor
    ) -> tuple[float, float]:
        """Decode size (width, height) from hidden states.

        Applies ln + size_decoder, argmax on 2x1024 bins, then converts
        from log-scale bins to float values.

        Args:
            hidden_states: Tensor of shape [..., dim].

        Returns:
            Tuple (w_float, h_float).
        """
        logits = self.size_decoder(self.ln(hidden_states)).view(2, -1)
        w_bin = torch.argmax(logits[0], dim=-1)
        h_bin = torch.argmax(logits[1], dim=-1)
        w = torch.pow(2.0, (w_bin.float() / 1023.0) * 10.0 - 10.0)
        h = torch.pow(2.0, (h_bin.float() / 1023.0) * 10.0 - 10.0)
        return w.item(), h.item()


# ============================================================================
# Detect/Point State Machine
# ============================================================================

# Maximum number of objects per detect/point request.
_DEFAULT_MAX_OBJECTS = 150


@dataclass
class DetectPointState:
    """Per-request state for detect/point generation.

    Tracks the current step in the state machine, pending embedding
    information for the next decode step, the object currently being
    constructed, and the accumulated results.
    """

    mode: Literal["detect", "point"]
    step: Literal["decode_x_or_stop", "decode_y", "decode_size"] = (
        "decode_x_or_stop"
    )

    # Embedding info for the *next* decode step (set in compute_logits,
    # consumed in forward to replace the token embedding).
    pending_embed_type: str | None = None  # "coord" or "size"
    pending_embed_coord: float | None = None
    pending_embed_w: float | None = None
    pending_embed_h: float | None = None

    # Current object being constructed.
    current_x: float | None = None
    current_y: float | None = None

    # Accumulated results.
    objects: list[dict] = field(default_factory=list)
    finished: bool = False
    max_objects: int = _DEFAULT_MAX_OBJECTS


class DetectPointStateManager:
    """Manages per-request detect/point state for the model."""

    def __init__(
        self,
        coord_id: int = 5,
        size_id: int = 6,
        eos_id: int = 0,
    ) -> None:
        self._states: dict[str, DetectPointState] = {}
        self.coord_id = coord_id
        self.size_id = size_id
        self.eos_id = eos_id

    def register_request(
        self,
        req_id: str,
        mode: Literal["detect", "point"],
        max_objects: int = _DEFAULT_MAX_OBJECTS,
    ) -> None:
        self._states[req_id] = DetectPointState(
            mode=mode, max_objects=max_objects
        )

    def get_state(self, req_id: str) -> DetectPointState | None:
        return self._states.get(req_id)

    def has_active_requests(self) -> bool:
        return bool(self._states)

    def remove_request(self, req_id: str) -> None:
        self._states.pop(req_id, None)

    def get_json_result(self, req_id: str) -> str | None:
        """Serialize the accumulated objects for a finished request."""
        state = self._states.get(req_id)
        if state is None:
            return None
        if state.mode == "detect":
            return json.dumps({"objects": state.objects})
        return json.dumps({"points": state.objects})

    def update_after_sample(
        self, req_id: str, sampled_token_id: int
    ) -> None:
        """Transition state after a token is sampled."""
        state = self._states.get(req_id)
        if state is None:
            return

        tok = sampled_token_id

        if state.step == "decode_x_or_stop":
            if tok == self.eos_id:
                state.finished = True
            else:
                # coord token sampled → move to decode_y
                state.step = "decode_y"

        elif state.step == "decode_y":
            if state.mode == "point":
                # For point: y was decoded, object is complete.
                state.objects.append(
                    {"x": state.current_x, "y": state.current_y}
                )
                state.step = "decode_x_or_stop"
            else:
                # For detect: y decoded, need size next.
                state.step = "decode_size"

        elif state.step == "decode_size":
            # Size decoded, compute bbox from center + size.
            x, y = state.current_x, state.current_y
            w, h = state.pending_embed_w, state.pending_embed_h
            state.objects.append(
                {
                    "x_min": x - w / 2,
                    "y_min": y - h / 2,
                    "x_max": x + w / 2,
                    "y_max": y + h / 2,
                }
            )
            state.step = "decode_x_or_stop"

        # Check max objects limit.
        if len(state.objects) >= state.max_objects:
            state.finished = True


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
    """Mixture of Experts layer for layers 4+ with expert parallelism.

    Moondream3 uses a custom GeGLU activation: gelu(h) * (g + 1)
    where fc1 outputs [gate, up] and the activation is gelu(gate) * (up + 1).

    Uses expert parallelism where each GPU stores num_experts/tp_size experts.
    Routing and communication handled via all-to-all or replicated computation.

    Checkpoint format:
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

        # Expert parallelism: each GPU stores a subset of experts
        self.tp_size = get_tensor_model_parallel_world_size()
        self.experts_per_rank = num_experts // self.tp_size
        self.num_local_experts = self.experts_per_rank

        # Router (gate) - use ReplicatedLinear for compatibility
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        # Local expert weights (only store experts_per_rank experts)
        # fc1: [experts_per_rank, expert_inner_dim * 2, hidden_size]
        # fc2: [experts_per_rank, hidden_size, expert_inner_dim]
        self.fc1_weight = nn.Parameter(
            torch.empty(self.num_local_experts, expert_inner_dim * 2, hidden_size)
        )
        self.fc2_weight = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, expert_inner_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with expert parallelism and custom GeGLU activation."""

        tp_rank = get_tensor_model_parallel_rank()

        # Get router logits and compute top-k
        router_logits, _ = self.gate(x)  # [num_tokens, num_experts]
        topk_logits, topk_idxs = torch.topk(
            router_logits, self.experts_per_token, dim=-1
        )
        # Softmax over selected experts
        topk_weights = F.softmax(topk_logits, dim=-1, dtype=torch.float32).to(x.dtype)

        # Compute local expert range
        local_expert_start = tp_rank * self.experts_per_rank

        # Compute MoE output using loop over local experts
        out = x.new_zeros(x.shape)

        for local_expert_idx in range(self.num_local_experts):
            global_expert_id = local_expert_start + local_expert_idx

            # Find tokens assigned to this expert
            token_pos, which_k = (topk_idxs == global_expert_id).nonzero(as_tuple=True)
            if token_pos.numel() == 0:
                continue

            # Get tokens and their routing weights
            x_tok = x.index_select(0, token_pos)  # [n_tokens, hidden_size]
            gate_tok = topk_weights[token_pos, which_k]  # [n_tokens]

            # fc1: [expert_inner_dim * 2, hidden_size]
            # h_full: [n_tokens, expert_inner_dim * 2]
            h_full = F.linear(x_tok, self.fc1_weight[local_expert_idx])

            # GeGLU with (g + 1): h, g = split; output = gelu(h) * (g + 1)
            # HF MoE uses exact GELU (not tanh approximation).
            h, g = h_full.chunk(2, dim=-1)  # Each [n_tokens, expert_inner_dim]
            h = F.gelu(h) * (g + 1.0)

            # fc2: [hidden_size, expert_inner_dim]
            # y: [n_tokens, hidden_size]
            y = F.linear(h, self.fc2_weight[local_expert_idx])

            # Apply routing weight
            y = y * gate_tok.unsqueeze(-1)

            # Accumulate output
            out.index_add_(0, token_pos, y)

        # All-reduce to combine results from all experts across GPUs
        out = tensor_model_parallel_all_reduce(out)

        return out


class Moondream3Attention(nn.Module):
    """Decoder attention with RoPE and tau scaling.

    Moondream3 uses a tau attention mechanism that scales Q and V
    based on both token content and position.
    """

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
        # HF Moondream uses non-interleaved RoPE (split by half)
        # In vLLM, is_neox_style=True means split by half (GPT-NeoX style)
        rope_parameters = {
            "rope_theta": config.rope_theta,
            "partial_rotary_factor": 32 / self.head_dim,  # 32/64 = 0.5
        }
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_context,
            rope_parameters=rope_parameters,
            is_neox_style=True,  # Moondream uses split-by-half (GPT-NeoX) style
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

        # Tau scaling parameters for position-dependent attention
        # These are learned during training to modulate attention based on position
        # tau_wq and tau_wv need full qkv_dim for correct computation
        # Only heads are partitioned, qkv dimension is kept full for all-gather
        qkv_dim = self.hidden_size * 3  # Q + K + V dimension (full)
        self.tau_alpha = nn.Parameter(torch.zeros(self.num_heads_per_partition))
        self.tau_wq = nn.Parameter(torch.zeros(self.num_heads_per_partition, qkv_dim))
        self.tau_wv = nn.Parameter(torch.zeros(self.num_heads_per_partition, qkv_dim))
        self.tp_size = tp_size

        # Prefix-LM attention length: BOS (1) + vision patches (729) = 730
        self._prefix_attn_len = config.prefix_attn  # 730

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

        # Apply tau scaling to Q and V
        # Tau scaling has two components:
        # 1. Token-based: tok_q = tanh(gelu(qkv) @ tau_wq.T)
        # 2. Position-based: tau_pos = 1 + (sigmoid(alpha * log(pos+1)) - 0.5)
        # Final: tau = tok + tau_pos
        #
        # For TP, tau weights are sharded by head, but qkv_dim is kept full

        # Get full qkv for tau computation
        # With TP, reconstruct qkv in correct layout [q_full, k_full, v_full]
        # (all-gather would produce [q_0, k_0, v_0, q_1, k_1, v_1] - wrong)
        if self.tp_size > 1:
            from vllm.distributed import tensor_model_parallel_all_gather

            # All-gather q, k, v separately and concatenate in correct order
            q_full = tensor_model_parallel_all_gather(q.contiguous())
            k_full = tensor_model_parallel_all_gather(k.contiguous())
            v_full = tensor_model_parallel_all_gather(v.contiguous())
            qkv_full = torch.cat([q_full, k_full, v_full], dim=-1).contiguous()
        else:
            qkv_full = qkv

        # Compute tau scaling factors matching HF implementation exactly:
        # tok_feat = gelu(qkv)
        # tok_q = tanh(tok_feat @ tau_wq.T)  # [num_tokens, num_heads]
        # tau_pos = 1 + (sigmoid(alpha * log(pos+1)) - 0.5)  # [num_heads, num_tokens]
        # tau = (tok_q.T + tau_pos).T  # [num_tokens, num_heads]
        num_tokens = qkv_full.shape[0]
        orig_dtype = q.dtype

        # Token-based component
        tok_feat = F.gelu(qkv_full)  # Apply GELU activation
        tok_q = torch.tanh(tok_feat @ self.tau_wq.t())  # [N, H_per_partition]
        tok_v = torch.tanh(tok_feat @ self.tau_wv.t())  # [N, H_per_partition]

        # Position-based component
        # tau_pos = 1 + (sigmoid(alpha * log(pos+1)) - 0.5)
        # positions is [num_tokens], need to compute for each head
        # tau_alpha: [num_heads_per_partition]
        pos_float = (positions.to(orig_dtype) + 1.0).clamp(min=1e-6)
        pos_log = pos_float.log()  # [num_tokens]
        # alpha[:, None] * pos_log[None, :] -> [num_heads, num_tokens]
        tau_pos = 1.0 + (
            torch.sigmoid(self.tau_alpha[:, None] * pos_log[None, :]) - 0.5
        )  # [H_per_partition, N]

        # Combine token and position components
        tau_q = (tok_q + tau_pos.t()).to(orig_dtype)  # [N, H_per_partition]
        tau_v = (tok_v + tau_pos.t()).to(orig_dtype)  # [N, H_per_partition]

        # Reshape q and v to apply per-head tau scaling
        q = q.view(num_tokens, self.num_heads_per_partition, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads_per_partition, self.head_dim)

        # Apply tau scaling
        q = q * tau_q.unsqueeze(-1)
        v = v * tau_v[:, : self.num_kv_heads_per_partition].unsqueeze(-1)

        # Reshape back
        q = q.view(num_tokens, -1)
        v = v.view(num_tokens, -1)

        q, k = self.rotary_emb(positions, q, k)

        # ---- SDPA prefill override ----
        # During prefill (num_tokens >= prefix_attn_len), use PyTorch SDPA
        # with an explicit bidirectional mask for the prefix-LM region.
        # This produces hidden states that closely match the HF reference
        # (which also uses SDPA for prefill).  Decode tokens and short
        # sequences fall through to the configured attention backend (e.g.
        # FLEX_ATTENTION) for KV-cache-aware decoding.
        H = self.num_heads_per_partition
        Hkv = self.num_kv_heads_per_partition
        D = self.head_dim
        P = self._prefix_attn_len  # 730

        if num_tokens > 1 and num_tokens >= P:
            q_4d = q.view(num_tokens, H, D).transpose(0, 1).unsqueeze(0)
            k_4d = k.view(num_tokens, Hkv, D).transpose(0, 1).unsqueeze(0)
            v_4d = v.view(num_tokens, Hkv, D).transpose(0, 1).unsqueeze(0)

            # Causal mask with bidirectional prefix region
            bool_mask = torch.tril(
                torch.ones(1, 1, num_tokens, num_tokens,
                           dtype=torch.bool, device=q.device)
            )
            bool_mask[:, :, :P, :P] = True  # Positions 0-729 bidir

            attn_output = F.scaled_dot_product_attention(
                q_4d, k_4d, v_4d,
                attn_mask=bool_mask,
                scale=self.scaling,
            )
            attn_output = (
                attn_output.squeeze(0).transpose(0, 1).contiguous()
                .view(num_tokens, H * D)
            )

            # Still call self.attn to populate the KV cache
            _ = self.attn(q, k, v)
        else:
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
            lambda prefix: Moondream3DecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
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

        for i, layer in enumerate(
            islice(self.blocks, self.start_layer, self.end_layer)
        ):
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
        # 729 = 27x27 patches from 378x378 crop / 14 patch size
        return 729

    def get_image_size_with_most_features(self) -> ImageSize:
        return ImageSize(width=378, height=378)

    def get_max_image_tokens(self) -> int:
        return 729


class Moondream3DummyInputsBuilder(BaseDummyInputsBuilder[Moondream3ProcessingInfo]):
    """Dummy inputs builder for profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return (
            "<|endoftext|><image><|md_reserved_0|>query<|md_reserved_1|>"
            "What is this image?<|md_reserved_2|>"
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
        mm_processor_kwargs: Mapping[str, object] | None = None,
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

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # Moondream3 HF processor does NOT expand placeholder tokens.
        # vLLM should apply prompt updates to expand <image> to 729 tokens.
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptUpdate]:
        image_size = self.info.get_image_size_with_most_features()
        num_image_tokens = self.info.get_num_image_tokens(
            image_width=image_size.width,
            image_height=image_size.height,
        )
        replacement_token = self.image_placeholder_tokens[0]
        return [
            PromptReplacement(
                modality="image",
                target=self.image_placeholder_tokens,
                replacement=[replacement_token] * num_image_tokens,
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
    """Moondream3 multimodal model for causal language modeling.

    Moondream3 has four capabilities:

    - **query**: Visual QA.
    - **caption**: Image description.
    - **detect**: Object detection (bounding boxes).
    - **point**: Object pointing (x, y coordinates).

    All four capabilities are supported. Query and caption use standard
    autoregressive generation. Detect and point use a custom state machine
    that intercepts ``compute_logits`` and ``forward`` to decode
    coordinates from hidden states and feed Fourier-encoded coordinate
    embeddings back as the next input.

    Detect/point mode is activated by setting
    ``SamplingParams(extra_args={"moondream3_task": "detect"})``
    (or ``"point"``).
    """

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

        with self._mark_tower_model(vllm_config, "image"):
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

        with self._mark_language_model(vllm_config):
            # Text decoder
            self.text = Moondream3TextModel(
                config=self.config.text,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "text"),
            )

            # LM head (with bias - Moondream3 has lm_head bias)
            self.lm_head = ParallelLMHead(
                self.config.text.vocab_size,
                self.config.text.dim,
                bias=True,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        # Region module for point/detect coordinate encoding/decoding.
        self.region = Moondream3RegionModule(
            config=self.config.region,
            prefix=maybe_prefix(prefix, "region"),
        )

        self.logits_processor = LogitsProcessor(self.config.text.vocab_size)
        self.make_empty_intermediate_tensors = self.text.make_empty_intermediate_tensors

        # Detect/point token IDs (from config, with defaults).
        self._coord_id = getattr(hf_config, "coord_token_id", 5)
        self._size_id = getattr(hf_config, "size_token_id", 6)
        self._eos_id = getattr(hf_config, "region_eos_token_id", 0)

        # Detect/point state management.  The model runner sets
        # ``_dp_row_states`` (list mapping logits row → DetectPointState)
        # before calling ``compute_logits``, and ``_dp_embed_data``
        # (dict mapping input position → embed info) before ``forward``.
        self.detect_point_manager = DetectPointStateManager(
            coord_id=self._coord_id,
            size_id=self._size_id,
            eos_id=self._eos_id,
        )
        self._dp_row_states: list[DetectPointState | None] | None = None
        self._dp_embed_data: dict[int, dict] | None = None

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

        # Grid size = crop_size / patch_size (e.g., 378 / 14 = 27)
        grid_size = self.config.vision.crop_size // self.config.vision.enc_patch_size
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
        pooled_size = self.config.vision.enc_n_layers
        recon = F.adaptive_avg_pool2d(recon, output_size=(pooled_size, pooled_size))
        recon = recon.permute(1, 2, 0).contiguous().view(-1, enc_dim)

        combined = torch.cat([global_features, recon], dim=-1).unsqueeze(0)
        projected = self.vision_proj(combined).squeeze(0)

        # Note: Vision embeddings are already synchronized across TP ranks
        # because the vision projection uses RowParallelLinear which performs
        # all-reduce internally, ensuring identical outputs on all ranks.

        return projected

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Generate 729 vision embeddings per image (27x27 patches)."""
        image_inputs = self._parse_image_inputs(**kwargs)
        if not image_inputs:
            return []

        return [
            self._encode_image_input(image_input)
            for image_input in image_inputs
        ]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        # Replace embeddings for detect/point decode positions with
        # Fourier-encoded coordinate/size embeddings from the region module.
        dp_embed_data = self._dp_embed_data
        if dp_embed_data and inputs_embeds is not None:
            for pos, info in dp_embed_data.items():
                if pos >= inputs_embeds.shape[0]:
                    continue
                if info["type"] == "coord":
                    coord_t = torch.tensor(
                        [[info["value"]]],
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    )
                    emb = self.region.encode_coordinate(coord_t)  # [1, dim]
                    inputs_embeds[pos] = emb.squeeze(0)
                elif info["type"] == "size":
                    emb = self.region.encode_size(
                        info["w"],
                        info["h"],
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    )  # [1, dim]
                    inputs_embeds[pos] = emb.squeeze(0)
            # Clear after use.
            self._dp_embed_data = None

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
    ) -> torch.Tensor | None:
        row_states = self._dp_row_states
        if row_states is None or not any(
            s is not None for s in row_states
        ):
            # No detect/point requests — standard path.
            return self.logits_processor(self.lm_head, hidden_states)

        # Compute standard logits for ALL rows (needed for text requests
        # and for the continue/stop sparse check in decode_x_or_stop).
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return logits

        for i, state in enumerate(row_states):
            if state is None:
                continue
            h = hidden_states[i: i + 1]  # [1, dim]

            if state.step == "decode_x_or_stop":
                # Check continue/stop via sparse lm_head logits.
                coord_score = logits[i, self._coord_id].item()
                eos_score = logits[i, self._eos_id].item()

                if state.finished or eos_score > coord_score:
                    # Model wants to stop — force EOS.
                    logits[i] = float("-inf")
                    logits[i, self._eos_id] = 0.0
                else:
                    # Decode x coordinate from the *same* hidden state.
                    coord_logits = self.region.decode_coordinate(h)
                    x_bin = torch.argmax(coord_logits, dim=-1)
                    x_val = (
                        x_bin.float().item() / coord_logits.shape[-1]
                    )
                    state.current_x = x_val
                    state.pending_embed_type = "coord"
                    state.pending_embed_coord = x_val
                    # Force COORD_ID token.
                    logits[i] = float("-inf")
                    logits[i, self._coord_id] = 0.0

            elif state.step == "decode_y":
                coord_logits = self.region.decode_coordinate(h)
                y_bin = torch.argmax(coord_logits, dim=-1)
                y_val = y_bin.float().item() / coord_logits.shape[-1]
                state.current_y = y_val
                state.pending_embed_type = "coord"
                state.pending_embed_coord = y_val
                logits[i] = float("-inf")
                logits[i, self._coord_id] = 0.0

            elif state.step == "decode_size":
                w, h_val = self.region.decode_size(h)
                state.pending_embed_type = "size"
                state.pending_embed_w = w
                state.pending_embed_h = h_val
                logits[i] = float("-inf")
                logits[i, self._size_id] = 0.0

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with remapping from HuggingFace format."""

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Get expert intermediate size for fc1 splitting

        for name, loaded_weight in weights:
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

            # Tau attention scaling weights
            # HF format: .attn.tau.alpha -> .attn.tau_alpha
            name = name.replace(".attn.tau.alpha", ".attn.tau_alpha")
            name = name.replace(".attn.tau.wq", ".attn.tau_wq")
            name = name.replace(".attn.tau.wv", ".attn.tau_wv")

            # MoE router mapping: mlp.router -> mlp.gate
            name = name.replace(".mlp.router.", ".mlp.gate.")

            # Handle MoE expert weights for layers 4+ with expert parallelism
            # fc1.weight: [n_experts, expert_inner_dim * 2, hidden_size] (gate+up)
            # fc2.weight: [n_experts, hidden_size, expert_inner_dim] (down)
            # Each GPU stores n_experts/tp_size experts
            # Note: Only 3D weights are MoE, 2D weights are standard MLP
            if ".mlp.fc1.weight" in name and loaded_weight.dim() == 3:
                from vllm.distributed import get_tensor_model_parallel_rank

                tp_size = get_tensor_model_parallel_world_size()
                tp_rank = get_tensor_model_parallel_rank()
                num_experts = loaded_weight.shape[0]
                experts_per_rank = num_experts // tp_size
                expert_start = tp_rank * experts_per_rank
                expert_end = expert_start + experts_per_rank
                # Shard by expert dimension
                loaded_weight = loaded_weight[expert_start:expert_end].contiguous()
                # Map to our custom MoE format: mlp.fc1_weight
                name = name.replace(".mlp.fc1.weight", ".mlp.fc1_weight")

            if ".mlp.fc2.weight" in name and loaded_weight.dim() == 3:
                from vllm.distributed import get_tensor_model_parallel_rank

                tp_size = get_tensor_model_parallel_world_size()
                tp_rank = get_tensor_model_parallel_rank()
                num_experts = loaded_weight.shape[0]
                experts_per_rank = num_experts // tp_size
                expert_start = tp_rank * experts_per_rank
                expert_end = expert_start + experts_per_rank
                # Shard by expert dimension
                loaded_weight = loaded_weight[expert_start:expert_end].contiguous()
                # Map to our custom MoE format: mlp.fc2_weight
                name = name.replace(".mlp.fc2.weight", ".mlp.fc2_weight")

            # Handle tau weights with tensor parallelism
            # tau_alpha: [num_heads] -> [num_heads/tp]
            # tau_wq: [num_heads, qkv_dim] -> [num_heads/tp, qkv_dim/tp]
            # tau_wv: [num_heads, qkv_dim] -> [num_heads/tp, qkv_dim/tp]
            if ".tau_alpha" in name:
                from vllm.distributed import get_tensor_model_parallel_rank

                tp_size = get_tensor_model_parallel_world_size()
                tp_rank = get_tensor_model_parallel_rank()
                num_heads = loaded_weight.shape[0]
                heads_per_partition = num_heads // tp_size
                start = tp_rank * heads_per_partition
                end = start + heads_per_partition
                loaded_weight = loaded_weight[start:end].contiguous()

            if ".tau_wq" in name or ".tau_wv" in name:
                from vllm.distributed import get_tensor_model_parallel_rank

                tp_size = get_tensor_model_parallel_world_size()
                tp_rank = get_tensor_model_parallel_rank()
                num_heads, qkv_dim = loaded_weight.shape
                heads_per_partition = num_heads // tp_size
                # Only shard by head dimension, keep full qkv_dim for all-gather
                head_start = tp_rank * heads_per_partition
                head_end = head_start + heads_per_partition
                loaded_weight = loaded_weight[head_start:head_end, :].contiguous()

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params
