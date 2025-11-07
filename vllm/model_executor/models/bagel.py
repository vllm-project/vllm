# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
"""Inference-only BAGEL model compatible with HuggingFace weights.

BAGEL is a unified multimodal model for image understanding and generation.
For vLLM, we focus on the image understanding (vision-to-text) capabilities.
"""

from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Literal, TypeAlias

import torch
import torch.nn as nn
from PIL import Image
from transformers import PretrainedConfig, SiglipVisionConfig
from transformers.models.qwen2 import Qwen2Config

from vllm.attention.backends.registry import _Backend
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    ImageItem,
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from .siglip import SiglipVisionModel

logger = init_logger(__name__)


# === Configuration === #


class BagelConfig(PretrainedConfig):
    """Configuration class for BAGEL model."""

    model_type = "bagel"

    def __init__(
        self,
        visual_gen: bool = True,
        visual_und: bool = True,
        llm_config: dict | Qwen2Config | None = None,
        vit_config: dict | SiglipVisionConfig | None = None,
        vae_config: dict | None = None,
        latent_patch_size: int = 2,
        max_latent_size: int = 32,
        vit_max_num_patch_per_side: int = 70,
        connector_act: str = "gelu_pytorch_tanh",
        interpolate_pos: bool = False,
        timestep_shift: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und

        # Convert dict configs to proper config objects
        if isinstance(llm_config, dict):
            self.llm_config = Qwen2Config(**llm_config)
        else:
            self.llm_config = llm_config or Qwen2Config()

        if isinstance(vit_config, dict):
            self.vit_config = SiglipVisionConfig(**vit_config)
        else:
            self.vit_config = vit_config or SiglipVisionConfig()

        self.vae_config = vae_config or {"z_channels": 16, "downsample": 8}
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift

    @property
    def hidden_size(self) -> int:
        """Return the hidden size of the language model."""
        return self.llm_config.hidden_size


# === Vision Inputs === #


class BagelImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - c: Number of channels (3)
        - h: Height of each image
        - w: Width of each image
    """

    type: Literal["pixel_values"]
    pixel_values: torch.Tensor  # Shape: (bn, 3, h, w)


BagelImageInputs: TypeAlias = BagelImagePixelInputs


# === Vision Encoder === #


class BagelVisionMLP(nn.Module):
    """MLP connector for vision features."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: str = "gelu_pytorch_tanh",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.act = get_act_fn(act_layer)
        self.fc2 = RowParallelLinear(
            hidden_features,
            out_features,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x


class PositionEmbedding(nn.Module):
    """2D position embedding for vision tokens using sin-cos embeddings."""

    def __init__(self, max_num_patch_per_side: int, hidden_size: int):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size

        # Create learnable 2D position embeddings (frozen sin-cos)
        pos_embed = self._get_2d_sincos_pos_embed(
            hidden_size, max_num_patch_per_side
        )
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(pos_embed).float(),
            persistent=False,
        )

    @staticmethod
    def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int):
        """Generate 2D sin-cos position embeddings."""
        import numpy as np

        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # w goes first
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = PositionEmbedding._get_2d_sincos_pos_embed_from_grid(
            embed_dim, grid
        )
        return pos_embed

    @staticmethod
    def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid):
        """Generate 2D sin-cos position embeddings from grid."""
        import numpy as np

        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = PositionEmbedding._get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, grid[0]
        )
        emb_w = PositionEmbedding._get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, grid[1]
        )
        emb = np.concatenate([emb_h, emb_w], axis=1)
        return emb

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
        """Generate 1D sin-cos position embeddings."""
        import numpy as np

        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega

        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position_ids: Flattened position IDs, shape (N,) where each ID
                         corresponds to a position in the flattened grid
        Returns:
            Position embeddings of shape (N, hidden_size)
        """
        return self.pos_embed[position_ids]


# === Multimodal Processing === #


class BagelProcessingInfo(BaseProcessingInfo):
    """Processing information for BAGEL model."""

    def get_hf_config(self) -> BagelConfig:
        return self.ctx.get_hf_config(BagelConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_config = self.get_hf_config()
        # Calculate max tokens per image
        # For BAGEL: (vit_max_num_patch_per_side) ** 2
        max_num_patches = hf_config.vit_max_num_patch_per_side**2
        return {"image": max_num_patches}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vit_config = hf_config.vit_config
        patch_size = vit_config.patch_size

        # Calculate number of patches
        num_patches_h = image_height // patch_size
        num_patches_w = image_width // patch_size
        return num_patches_h * num_patches_w


class BagelDummyInputsBuilder(BaseDummyInputsBuilder[BagelProcessingInfo]):
    """Build dummy inputs for BAGEL model profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        # Use a simple placeholder for each image
        return "<image>" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        hf_config = self.info.get_hf_config()
        vit_config = hf_config.vit_config

        # Use the configured image size
        image_size = vit_config.image_size
        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=image_size,
                height=image_size,
                num_images=num_images,
                overrides=image_overrides,
            ),
        }


class BagelMultiModalProcessor(BaseMultiModalProcessor[BagelProcessingInfo]):
    """Multimodal processor for BAGEL model."""

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        """Replace image placeholders with the correct number of tokens."""
        hf_config = self.info.get_hf_config()
        vit_config = hf_config.vit_config
        patch_size = vit_config.patch_size

        def get_replacement_bagel(item_idx: int):
            # For BAGEL, we need to calculate based on actual image size
            # Default to max patches
            num_tokens = hf_config.vit_max_num_patch_per_side**2

            # BAGEL uses a special image token
            # Token ID 151655 is <|image_pad|>
            return [151655] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=["<image>"],
                replacement=get_replacement_bagel,
            )
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
        }


# === Main Model === #


@MULTIMODAL_REGISTRY.register_processor(
    BagelMultiModalProcessor,
    info=BagelProcessingInfo,
    dummy_inputs=BagelDummyInputsBuilder,
)
class BagelForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP
):
    """
    BAGEL: A unified multimodal model for image understanding and generation.

    For vLLM, we focus on the image understanding (vision-to-text) capabilities.
    The image generation part is not supported in vLLM.
    """

    # Weight mapping from HF to vLLM
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.": "language_model.",
            "vit_model.": "vit_model.",
            "connector.": "connector.",
            "vit_pos_embed.": "vit_pos_embed.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        # Ensure we have a BagelConfig
        if not isinstance(config, BagelConfig):
            raise ValueError(
                f"Expected BagelConfig, got {type(config).__name__}. "
                "Make sure the model config is properly loaded."
            )

        self.config = config
        self.multimodal_config = multimodal_config

        # Initialize language model (Qwen2)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        # Initialize vision model (SigLIP) if visual understanding is enabled
        if config.visual_und:
            self.vit_model = SiglipVisionModel(
                config=config.vit_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vit_model"),
            )

            # Initialize connector (MLP)
            vit_hidden_size = config.vit_config.hidden_size
            llm_hidden_size = config.llm_config.hidden_size

            self.connector = BagelVisionMLP(
                in_features=vit_hidden_size,
                hidden_features=llm_hidden_size,
                out_features=llm_hidden_size,
                act_layer=config.connector_act,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "connector"),
            )

            # Position embedding for vision tokens
            self.vit_pos_embed = PositionEmbedding(
                max_num_patch_per_side=config.vit_max_num_patch_per_side,
                hidden_size=llm_hidden_size,
            )
        else:
            self.vit_model = None
            self.connector = None
            self.vit_pos_embed = None

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> BagelImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)

        if pixel_values is None:
            return None

        return BagelImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
        )

    def _process_image_input(
        self, image_input: BagelImageInputs
    ) -> tuple[torch.Tensor, ...]:
        """Process image inputs through vision encoder and connector."""
        pixel_values = image_input["pixel_values"]

        # Get vision features from SigLIP
        # pixel_values shape: (batch_size * num_images, 3, H, W)
        vision_features = self.vit_model(pixel_values)

        # vision_features shape: (batch_size * num_images, num_patches, hidden_size)
        # where num_patches = (H / patch_size) * (W / patch_size)

        # Pass through connector
        vision_embeds = self.connector(vision_features)

        # Add position embeddings
        batch_size, num_patches, hidden_size = vision_embeds.shape
        patch_size = self.config.vit_config.patch_size
        image_size = self.config.vit_config.image_size

        # Calculate grid dimensions
        num_patches_per_side = image_size // patch_size

        # Create flattened position IDs (0 to num_patches-1)
        # For BAGEL, we use extrapolate mode by default
        h_coords = torch.arange(num_patches_per_side, device=vision_embeds.device)
        w_coords = torch.arange(num_patches_per_side, device=vision_embeds.device)
        position_ids = (
            h_coords[:, None] * self.config.vit_max_num_patch_per_side + w_coords
        ).flatten()
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1).flatten()

        # Add position embeddings
        pos_embeds = self.vit_pos_embed(position_ids)
        pos_embeds = pos_embeds.reshape(batch_size, num_patches, hidden_size)
        vision_embeds = vision_embeds + pos_embeds

        # Split by image
        return tuple(vision_embeds)

    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        """Get multimodal embeddings from input."""
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for BAGEL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a batch.
            positions: Flattened (concatenated) position ids corresponding to a batch.
            intermediate_tensors: Intermediate tensors from prior forward pass.
            inputs_embeds: Optional tensor of input embeddings.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
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
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from checkpoint."""
        skip_prefixes = []
        if self.vit_model is None:
            skip_prefixes.extend(["vit_model.", "connector.", "vit_pos_embed."])

        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """Get placeholder string for a given modality."""
        if modality == "image":
            return "<image>"
        return None
