# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from NVIDIA Eagle2.5-VL model
# https://huggingface.co/nvidia/Eagle2.5-8B

from collections.abc import Iterable
from typing import Annotated, Literal, TypeAlias

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import PromptUpdateDetails
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import TokenizerLike
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .internvl import (
    IMG_CONTEXT,
    IMG_END,
    IMG_START,
    BaseInternVLDummyInputsBuilder,
    BaseInternVLMultiModalProcessor,
    BaseInternVLProcessingInfo,
    BaseInternVLProcessor,
)
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix


class Eagle2_5_VLImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - bnp: Batch size * number of images * (1 + num_patches)
        - c: Number of channels (3)
        - h: Height of each image patch
        - w: Width of each image patch
    """

    type: Literal["pixel_values"]
    pixel_values_flat: Annotated[torch.Tensor, TensorShape("bnp", 3, "h", "w")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]


class Eagle2_5_VLImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - n: Number of images
        - f: Total image feature size
        - h: Hidden size (must match the hidden size of language model backbone)
    """

    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("n", "f", "h")]


Eagle2_5_VLImageInputs: TypeAlias = (
    Eagle2_5_VLImagePixelInputs | Eagle2_5_VLImageEmbeddingInputs
)


class Eagle2_5_VLProcessor(BaseInternVLProcessor):
    """
    Custom processor for Eagle2.5-VL model.
    Extends BaseInternVLProcessor with Eagle-specific token handling.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> None:
        # Skip super().__init__() to avoid config manipulation
        # Directly initialize all required attributes
        self.config = config
        self.tokenizer = tokenizer

        # Image size with force_image_size override
        image_size: int = config.vision_config.image_size
        if hasattr(config, "force_image_size") and config.force_image_size:
            image_size = config.force_image_size

        patch_size: int = config.vision_config.patch_size
        downsample_ratio: float = getattr(config, "downsample_ratio", 0.5)

        # Compute num_image_token
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (downsample_ratio**2)
        )
        self.image_size = image_size

        # Dynamic patch settings with defaults
        self.min_dynamic_patch = (
            min_dynamic_patch
            if min_dynamic_patch is not None
            else getattr(config, "min_dynamic_patch", 1)
        )
        self.max_dynamic_patch = (
            max_dynamic_patch
            if max_dynamic_patch is not None
            else getattr(config, "max_dynamic_patch", 12)
        )
        self.dynamic_image_size = (
            dynamic_image_size
            if dynamic_image_size is not None
            else getattr(config, "dynamic_image_size", True)
        )
        self.use_thumbnail: bool = getattr(config, "use_thumbnail", True)

    @property
    def image_token_id(self) -> int:
        """Get the image token ID from config or tokenizer."""
        if hasattr(self.config, "image_token_index"):
            return self.config.image_token_index
        # Fallback to tokenizer vocab - use <IMG_CONTEXT> (ID: 151667)
        vocab = self.tokenizer.get_vocab()
        if IMG_CONTEXT in vocab:
            return vocab[IMG_CONTEXT]
        raise ValueError(f"Cannot find image token '{IMG_CONTEXT}' in vocabulary")

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: int | None,
    ) -> PromptUpdateDetails[str]:
        """Get image replacement string for prompt."""
        repl_features = IMG_CONTEXT * feature_size
        repl_full = IMG_START + repl_features + IMG_END

        return PromptUpdateDetails.select_text(repl_full, IMG_CONTEXT)


class Eagle2_5_VLProcessingInfo(BaseInternVLProcessingInfo):
    """Processing info for Eagle2.5-VL model."""

    def get_hf_processor(self, **kwargs) -> Eagle2_5_VLProcessor:
        return self.ctx.init_processor(
            Eagle2_5_VLProcessor,
            config=self.ctx.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )


class Eagle2_5_VLDummyInputsBuilder(
    BaseInternVLDummyInputsBuilder[Eagle2_5_VLProcessingInfo]
):
    """Dummy inputs builder for Eagle2.5-VL model."""

    pass


class Eagle2_5_VLMultiModalProcessor(
    BaseInternVLMultiModalProcessor[Eagle2_5_VLProcessingInfo]
):
    """Multi-modal processor for Eagle2.5-VL model."""

    pass


@MULTIMODAL_REGISTRY.register_processor(
    Eagle2_5_VLMultiModalProcessor,
    info=Eagle2_5_VLProcessingInfo,
    dummy_inputs=Eagle2_5_VLDummyInputsBuilder,
)
class Eagle2_5_VLForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA
):
    """
    Eagle2.5-VL model for conditional generation.

    Architecture:
        - Vision Encoder: SigLIP
        - Language Model: Qwen2
        - Projection: MLP with pixel shuffle downsampling
    """

    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        # Image configuration
        image_size = (
            getattr(config, "force_image_size", None) or config.vision_config.image_size
        )
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.downsample_ratio = getattr(config, "downsample_ratio", 0.5)
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (self.downsample_ratio**2)
        )

        self.select_layer = getattr(config, "select_layer", -1)

        # Vision encoder (SigLIP)
        self.vision_model = self._init_vision_model(
            config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision_model"),
        )

        # Language model (Qwen2)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # MLP projection
        self.mlp1 = self._init_mlp1(config)

        self.img_context_token_id = None

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ):
        """Initialize SigLIP vision model."""
        vision_config = config.vision_config

        # Determine number of hidden layers based on select_layer
        vision_feature_layer = self.select_layer
        if vision_feature_layer < 0:
            num_hidden_layers = (
                vision_config.num_hidden_layers + vision_feature_layer + 1
            )
        else:
            num_hidden_layers = vision_feature_layer + 1

        # Disable the pooling head - Eagle2.5 needs all patch tokens,
        # not a single pooled output
        vision_config.vision_use_head = False

        return SiglipVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            prefix=prefix,
        )

    def _init_mlp1(self, config: PretrainedConfig) -> nn.Module:
        """Initialize MLP projection layer."""
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        return nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        """
        Pixel shuffle operation for downsampling vision features.

        Args:
            x: Input tensor of shape (n, w, h, c)
            scale_factor: Downsampling factor

        Returns:
            Downsampled tensor
        """
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from pixel values.

        Args:
            pixel_values: Input pixel values of shape (batch, channels, height, width)

        Returns:
            Visual embeddings
        """
        vit_embeds = self.vision_model(pixel_values=pixel_values)

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Eagle2_5_VLImageInputs | None:
        """Parse and validate image inputs."""
        pixel_values_flat = kwargs.pop("pixel_values_flat", None)
        image_num_patches = kwargs.pop("image_num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values_flat is None and image_embeds is None:
            return None

        if image_embeds is not None:
            return Eagle2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        image_token_id = kwargs.get("image_token_id")
        if image_token_id is not None:
            if isinstance(image_token_id, torch.Tensor):
                image_token_id = image_token_id.flatten().unique().item()
            assert isinstance(image_token_id, int)
            self.img_context_token_id = image_token_id

        if pixel_values_flat is not None:
            image_size = getattr(self.config, "force_image_size", None)
            if image_size is None:
                image_size = self.config.vision_config.image_size
            expected_h = expected_w = image_size
            resolve_bindings = {"h": expected_h, "w": expected_w}

            return Eagle2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values_flat=pixel_values_flat,
                num_patches=image_num_patches,
                resolve_bindings=resolve_bindings,
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: Eagle2_5_VLImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        """Process image input to get embeddings."""
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_model is not None

        image_embeds = self.extract_feature(image_input["pixel_values_flat"])

        num_patches = image_input["num_patches"]

        # Only one image in the current batch
        if len(num_patches) == 1:
            return (image_embeds.view(-1, self.config.text_config.hidden_size),)

        # Split embeddings by image
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1, self.config.text_config.hidden_size)
        image_feature_sizes = [
            num_patches * feature_size for num_patches in num_patches
        ]
        return image_embeds.split(image_feature_sizes)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Embed multimodal inputs."""
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        image_embeddings = self._process_image_input(image_input)
        return tuple(image_embeddings)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        """Embed input IDs with optional multimodal embeddings."""
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        """Forward pass through the model."""
        if intermediate_tensors is not None:
            inputs_embeds = None

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }

        hidden_states = self.language_model.model(**forward_kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states."""
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load model weights."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get the module prefix mapping for multimodal models."""
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="mlp1",
            tower_model="vision_model",
        )
