# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OpenVLA model implementation for vLLM inference.

OpenVLA (Open Vision-Language-Action) is a 7B VLA model for robotic manipulation.
Architecture: DINOv2 + SigLIP (fused) -> MLP Projector -> Llama-2-7B -> Action Tokens

References:
    - Paper: https://arxiv.org/abs/2406.09246
    - Model: https://huggingface.co/openvla/openvla-7b
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import OpenVLAConfig
from vllm.utils.torch_utils import set_default_torch_dtype

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)


class PrismaticVisionBackbone(nn.Module):
    """Fused vision backbone combining DINOv2 and SigLIP using timm.

    OpenVLA uses a fused dual-encoder vision backbone that concatenates
    features from DINOv2 (structural/geometric) and SigLIP (semantic).
    Uses timm models for proven correctness.

    Note: Image normalization (DINOv2 uses ImageNet stats, SigLIP uses 0.5)
    is handled in OpenVLAMultiModalProcessor._preprocess_image_6channel().
    """

    def __init__(
        self,
        image_sizes: list[int],
        use_fused_vision_backbone: bool = True,
        timm_model_ids: list[str] | None = None,
    ):
        super().__init__()
        self.use_fused = use_fused_vision_backbone
        self.image_sizes = image_sizes
        self.timm_model_ids = timm_model_ids or [
            "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_so400m_patch14_siglip_224",
        ]

        # Will be initialized when weights are loaded
        self.featurizer = None
        self.fused_featurizer = None
        self.embed_dim = 2176 if use_fused_vision_backbone else 1024

    def _init_timm_models(self):
        """Initialize timm models. Called during weight loading."""
        try:
            import timm
        except ImportError as err:
            raise ImportError("timm is required for OpenVLA: pip install timm") from err

        img_size = self.image_sizes[0] if self.image_sizes else 224

        # Create models with float16 dtype, then convert to default dtype
        with set_default_torch_dtype(torch.float16):
            # DINOv2 encoder
            self.featurizer = timm.create_model(
                self.timm_model_ids[0],
                pretrained=False,
                num_classes=0,
                img_size=img_size,
            )

            # SigLIP encoder (fused)
            if self.use_fused:
                self.fused_featurizer = timm.create_model(
                    self.timm_model_ids[1],
                    pretrained=False,
                    num_classes=0,
                    img_size=img_size,
                )

        # Convert to default dtype and move to device immediately
        device = current_platform.device_type
        self.featurizer = self.featurizer.to(
            device=device, dtype=torch.get_default_dtype()
        )
        if self.fused_featurizer is not None:
            self.fused_featurizer = self.fused_featurizer.to(
                device=device, dtype=torch.get_default_dtype()
            )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract and fuse features from both vision encoders.

        Args:
            pixel_values: Images of shape (batch, 6, height, width) where
                          channels 0-2: DINOv2 normalized (ImageNet)
                          channels 3-5: SigLIP normalized

        Returns:
            Patch features of shape (batch, num_patches, embed_dim).
        """
        if self.featurizer is None:
            raise RuntimeError(
                "Vision backbone not initialized. Call _init_timm_models first."
            )

        # Input must be 6-channel (preprocessed by OpenVLAMultiModalProcessor)
        if pixel_values.shape[1] != 6:
            raise ValueError(
                f"Expected 6-channel input from processor, got {pixel_values.shape[1]}"
            )

        # Split 6-channel input into DINOv2 and SigLIP normalized images
        dinov2_pixels = pixel_values[:, :3, :, :]
        siglip_pixels = pixel_values[:, 3:, :, :]

        # Get features from DINOv2 using second-to-last layer
        n_blocks = len(self.featurizer.blocks)
        features = self.featurizer.get_intermediate_layers(
            dinov2_pixels, n={n_blocks - 2}
        )[0]

        # Fuse with SigLIP features
        if self.use_fused and self.fused_featurizer is not None:
            n_fused_blocks = len(self.fused_featurizer.blocks)
            fused_features = self.fused_featurizer.get_intermediate_layers(
                siglip_pixels, n={n_fused_blocks - 2}
            )[0]
            features = torch.cat([features, fused_features], dim=-1)

        return features


class PrismaticProjector(nn.Module):
    """MLP projector to align vision features with LLM embedding space.

    For fused vision backbone (OpenVLA default):
        vision_dim -> 4*vision_dim -> llm_dim -> llm_dim
        with GELU activations between layers.
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        use_fused: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.use_fused = use_fused

        if use_fused:
            # Fused projector: 3-layer MLP
            intermediate_dim = 4 * vision_dim
            self.fc1 = ColumnParallelLinear(
                vision_dim,
                intermediate_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1",
            )
            self.act_fn1 = get_act_fn("gelu")
            self.fc2 = RowParallelLinear(
                intermediate_dim,
                llm_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2",
            )
            self.act_fn2 = get_act_fn("gelu")
            self.fc3 = ColumnParallelLinear(
                llm_dim,
                llm_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc3",
            )
        else:
            # Simple 2-layer MLP
            self.fc1 = ColumnParallelLinear(
                vision_dim,
                llm_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1",
            )
            self.act_fn1 = get_act_fn("gelu")
            self.fc2 = RowParallelLinear(
                llm_dim,
                llm_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2",
            )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project vision features to LLM embedding space.

        Args:
            vision_features: Shape (batch, num_patches, vision_dim).

        Returns:
            Projected features of shape (batch, num_patches, llm_dim).
        """
        x, _ = self.fc1(vision_features)
        x = self.act_fn1(x)
        x, _ = self.fc2(x)

        if self.use_fused:
            x = self.act_fn2(x)
            x, _ = self.fc3(x)

        return x


@dataclass
class OpenVLAImagePixelInputs:
    """Schema for OpenVLA image pixel inputs."""

    type: Literal["pixel_values"] = "pixel_values"
    # Shape: (batch * num_images, 6, height, width) - 6 channels for dual norm
    pixel_values: torch.Tensor = field(default=None)


class OpenVLAProcessingInfo(BaseProcessingInfo):
    """Processing info for OpenVLA model."""

    def get_hf_config(self) -> OpenVLAConfig:
        return self.ctx.get_hf_config(OpenVLAConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}  # OpenVLA typically uses single image

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        # OpenVLA uses 224x224 images with 14x14 patches = 256 tokens
        return (224 // 14) ** 2  # 256 patches

    def get_image_size_with_most_features(self) -> ImageSize:
        return ImageSize(width=224, height=224)

    def get_max_image_tokens(self) -> int:
        return self.get_num_image_tokens(image_width=224, image_height=224)

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        """Return the maximum number of tokens per image.

        OpenVLA has a fixed image size (224x224) with 14x14 patches = 256 tokens.
        Returning this directly avoids the profiling flow.
        """
        return {"image": self.get_max_image_tokens()}  # 256


class OpenVLADummyInputsBuilder(BaseDummyInputsBuilder[OpenVLAProcessingInfo]):
    """Builds dummy inputs for profiling OpenVLA."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # Empty string - image tokens are inserted at prefix, not as replacement
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        return {
            "image": self._get_dummy_images(
                width=224,
                height=224,
                num_images=num_images,
                overrides=mm_options.get("image") if mm_options else None,
            )
        }


class OpenVLAMultiModalProcessor(BaseMultiModalProcessor[OpenVLAProcessingInfo]):
    """Multi-modal processor for OpenVLA.

    Uses custom 6-channel preprocessing to match HuggingFace exactly:
    - Channels 0-2: DINOv2 normalized (HF's quantized ImageNet stats)
    - Channels 3-5: SigLIP normalized (0.5 mean/std)

    This avoids bfloat16 precision loss during runtime conversion.
    """

    # HF's exact quantized normalization values for bfloat16 compatibility
    IMAGENET_MEAN = [0.484375, 0.455078125, 0.40625]
    IMAGENET_STD = [0.228515625, 0.2236328125, 0.224609375]
    SIGLIP_MEAN = [0.5, 0.5, 0.5]
    SIGLIP_STD = [0.5, 0.5, 0.5]

    def _preprocess_image_6channel(self, image) -> torch.Tensor:
        """Preprocess image into 6-channel tensor with both normalizations.

        Args:
            image: PIL Image to process.

        Returns:
            Tensor of shape (6, 224, 224) with DINOv2 and SigLIP normalizations.
        """
        # Ensure RGB and resize to 224x224
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        image = image.resize((224, 224), PIL.Image.BILINEAR)

        # Convert to float32 numpy array normalized to [0, 1]
        raw = np.array(image, dtype=np.float32) / 255.0

        # DINOv2 normalization (HF's exact quantized ImageNet stats)
        dinov2_mean = np.array(self.IMAGENET_MEAN, dtype=np.float32)
        dinov2_std = np.array(self.IMAGENET_STD, dtype=np.float32)
        dinov2_pixels = (raw - dinov2_mean) / dinov2_std
        dinov2_pixels = dinov2_pixels.transpose(2, 0, 1)  # HWC -> CHW

        # SigLIP normalization
        siglip_mean = np.array(self.SIGLIP_MEAN, dtype=np.float32)
        siglip_std = np.array(self.SIGLIP_STD, dtype=np.float32)
        siglip_pixels = (raw - siglip_mean) / siglip_std
        siglip_pixels = siglip_pixels.transpose(2, 0, 1)  # HWC -> CHW

        # Stack into 6-channel tensor: [DINOv2(3), SigLIP(3)]
        pixel_values = np.concatenate([dinov2_pixels, siglip_pixels], axis=0)

        return torch.from_numpy(pixel_values)

    def _apply_hf_processor_text_only(
        self,
        prompt_text: str,
        tokenization_kwargs: Mapping[str, object],
    ) -> list[int]:
        """Tokenize text without using the HF processor.

        OpenVLA's PrismaticProcessor requires images even for text tokenization,
        so we use the tokenizer directly for text-only processing.
        """
        tokenizer = self.info.ctx.tokenizer
        # Use add_special_tokens=True to include BOS token for prefix matching
        return tokenizer.encode(prompt_text, add_special_tokens=True)

    def _apply_hf_processor_text_mm(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> tuple[list[int], BatchFeature, bool]:
        """Apply custom 6-channel preprocessing for OpenVLA.

        This overrides the base implementation to use our custom preprocessing
        that precomputes both DINOv2 and SigLIP normalizations in float32.
        """
        # Tokenize the text
        tokenizer = self.info.ctx.tokenizer
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)

        # Process images with 6-channel preprocessing
        mm_counts = mm_items.get_all_counts()
        num_images = mm_counts.get("image", 0)

        if num_images == 0:
            # Create dummy for profiling
            dummy_image = PIL.Image.new("RGB", (224, 224), color=(128, 128, 128))
            images = [dummy_image]
        else:
            image_items = mm_items.get("image")
            if image_items is not None:
                images = list(image_items)
            else:
                dummy_image = PIL.Image.new("RGB", (224, 224), color=(128, 128, 128))
                images = [dummy_image]

        # Process each image with 6-channel preprocessing
        pixel_values_list = []
        for img in images:
            pixel_values = self._preprocess_image_6channel(img)
            pixel_values_list.append(pixel_values)

        # Stack into batch tensor: (num_images, 6, 224, 224)
        pixel_values = torch.stack(pixel_values_list, dim=0)

        processed_data = BatchFeature({"pixel_values": pixel_values})

        # OpenVLA doesn't apply prompt updates via HF processor
        is_update_applied = False

        return prompt_ids, processed_data, is_update_applied

    def _apply_hf_processor_mm_only(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Process multimodal data with custom 6-channel preprocessing.

        Uses custom preprocessing to create 6-channel tensors with both
        DINOv2 and SigLIP normalizations precomputed in float32.
        This matches HuggingFace exactly and avoids bfloat16 precision loss.
        """
        mm_counts = mm_items.get_all_counts()
        num_images = mm_counts.get("image", 0)

        # Handle no images case (profiling)
        if num_images == 0:
            num_images = self.allowed_mm_limits.get("image", 1)
            dummy_image = PIL.Image.new("RGB", (224, 224), color=(128, 128, 128))
            images = [dummy_image]
        else:
            # Extract images from mm_items
            image_items = mm_items.get("image")
            if image_items is not None:
                images = list(image_items)
            else:
                dummy_image = PIL.Image.new("RGB", (224, 224), color=(128, 128, 128))
                images = [dummy_image]

        # Process each image with 6-channel preprocessing
        pixel_values_list = []
        for img in images:
            pixel_values = self._preprocess_image_6channel(img)
            pixel_values_list.append(pixel_values)

        # Stack into batch tensor: (num_images, 6, 224, 224)
        pixel_values = torch.stack(pixel_values_list, dim=0)

        # Return as BatchFeature-like dict
        return BatchFeature({"pixel_values": pixel_values})

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        # Use image_token_index (32000) as placeholder for image features
        image_token_id = getattr(hf_config, "image_token_index", 32000)
        # Get BOS token from tokenizer
        tokenizer = self.info.ctx.tokenizer
        bos_token_id = tokenizer.bos_token_id

        def get_insertion(item_idx: int):
            num_image_tokens = self.info.get_num_image_tokens(
                image_width=224,
                image_height=224,
            )
            image_tokens = [image_token_id] * num_image_tokens

            # Return with proper token selection for embeddings
            return PromptUpdateDetails.select_token_id(
                image_tokens,
                embed_token_id=image_token_id,
            )

        # Insert image tokens at the start of the prompt (after BOS if present)
        return [
            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.prefix(
                    [bos_token_id] if bos_token_id is not None else []
                ),
                insertion=get_insertion,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    OpenVLAMultiModalProcessor,
    info=OpenVLAProcessingInfo,
    dummy_inputs=OpenVLADummyInputsBuilder,
)
class OpenVLAForActionPrediction(nn.Module, SupportsMultiModal, SupportsPP):
    """OpenVLA model for action prediction via vLLM.

    Architecture follows Prismatic VLM with fused vision backbone:
    - Vision: DINOv2 + SigLIP (concatenated features)
    - Projector: 3-layer MLP with GELU
    - LLM: Llama-2-7B

    Action prediction:
    - 7D action space: [dx, dy, dz, drx, dry, drz, gripper]
    - 256 bins per dimension (tokens 32000-32255 in Llama vocab)
    - Autoregressive generation of 7 action tokens
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.model.": "language_model.model.",
            "language_model.lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            # Return None because image tokens are inserted at prefix,
            # not replaced from a placeholder string in the prompt
            return None
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # Vision backbone config
        default_timm_ids = [
            "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_so400m_patch14_siglip_224",
        ]
        self.timm_model_ids = getattr(config, "timm_model_ids", default_timm_ids)
        self.image_sizes = getattr(config, "image_sizes", [224, 224])
        self.use_fused_vision_backbone = getattr(
            config, "use_fused_vision_backbone", True
        )

        # Vision backbone
        has_image_input = (
            multimodal_config is not None
            and multimodal_config.get_limit_per_prompt("image")
        )
        if has_image_input:
            self.vision_backbone = PrismaticVisionBackbone(
                image_sizes=self.image_sizes,
                use_fused_vision_backbone=self.use_fused_vision_backbone,
            )
            # Get LLM hidden dim from text_config
            text_config = getattr(config, "text_config", None)
            llm_dim = getattr(text_config, "hidden_size", 4096) if text_config else 4096
            # Create projector with known dimensions
            self.projector = PrismaticProjector(
                vision_dim=self.vision_backbone.embed_dim,  # 2176 for fused
                llm_dim=llm_dim,
                use_fused=self.use_fused_vision_backbone,
                prefix="projector",
            )
        else:
            self.vision_backbone = None
            self.projector = None

        # Language model
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config if hasattr(config, "text_config") else config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # Image token handling
        self.image_token_id = getattr(config, "image_token_index", 32000)

        # Action token config
        self.n_action_bins = getattr(config, "n_action_bins", 256)
        self.action_dim = 7  # 6 DoF pose + gripper

        # Number of image patches (224/14)^2 = 256
        self.num_patches = (self.image_sizes[0] // 14) ** 2

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> OpenVLAImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)

        if pixel_values is None:
            return None

        return OpenVLAImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
        )

    def _process_image_input(
        self,
        image_input: OpenVLAImagePixelInputs,
    ) -> torch.Tensor:
        """Process image through vision backbone and projector."""
        if self.vision_backbone is None or self.projector is None:
            raise RuntimeError("Vision components not initialized")

        pixel_values = image_input.pixel_values
        vision_features = self.vision_backbone(pixel_values)
        return self.projector(vision_features)

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Compute multimodal embeddings from image inputs."""
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass for OpenVLA.

        Args:
            input_ids: Flattened input_ids for the batch.
            positions: Position indices for input tokens.
            intermediate_tensors: Intermediate tensors from prior forward pass.
            inputs_embeds: Optional tensor of input embeddings with multimodal
                features already merged.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load OpenVLA weights from HuggingFace checkpoint.

        Weight mapping:
        - vision_backbone.featurizer.* -> DINOv2 weights (timm)
        - vision_backbone.fused_featurizer.* -> SigLIP weights (timm)
        - projector.fc1/fc2/fc3.* -> Projector weights
        - language_model.* -> Llama weights

        Note: HF checkpoint uses .scale_factor for LayerScale, timm uses .gamma
        """
        # Initialize timm models for vision backbone
        if self.vision_backbone is not None:
            self.vision_backbone._init_timm_models()

        # Transform weights to handle LayerScale naming difference
        def transform_weights():
            for name, weight in weights:
                # HF uses .scale_factor, timm uses .gamma
                if ".ls1.scale_factor" in name or ".ls2.scale_factor" in name:
                    name = name.replace(".scale_factor", ".gamma")
                yield name, weight

        # Use AutoWeightsLoader for proper handling of packed modules
        loader = AutoWeightsLoader(self)
        return loader.load_weights(transform_weights(), mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self):
        """Get the module prefix in multimodal models."""
        from .module_mapping import MultiModelKeys

        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="projector",
            tower_model="vision_backbone",
        )

    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int:
        """Returns number of multi-modal encoder tokens."""
        return num_image_tokens

    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int:
        """Returns number of multi-modal connector tokens."""
        return num_vision_tokens
