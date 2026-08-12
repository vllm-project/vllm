# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal

import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import OpenVLAConfig
from vllm.transformers_utils.processors.openvla import (
    OpenVLAImageProcessor,
    OpenVLAProcessor,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .module_mapping import MultiModelKeys
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix

# openvla/openvla-7b uses 224x224 images with ViT patch size 14, yielding a
# 16x16 image-token grid.
_OPENVLA_IMAGE_SIZE = 224
_OPENVLA_PATCH_SIZE = 14
_OPENVLA_TIMM_MODEL_IDS = (
    "vit_large_patch14_reg4_dinov2.lvd142m",
    "vit_so400m_patch14_siglip_224",
)
_OPENVLA_TIMM_OVERRIDE_ACT_LAYERS = (None, None)
_OPENVLA_IMAGE_SIZES = (_OPENVLA_IMAGE_SIZE, _OPENVLA_IMAGE_SIZE)


def _get_num_image_tokens(image_size: int) -> int:
    return (image_size // _OPENVLA_PATCH_SIZE) ** 2


class OpenVLAImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - c: Number of channels (6)
        - h: Height
        - w: Width
    """

    type: Literal["pixel_values"] = "pixel_values"
    data: Annotated[torch.Tensor, TensorShape("bn", 6, "h", "w")]


class PrismaticVisionBackbone(nn.Module):
    """OpenVLA's fused DINOv2 + SigLIP vision backbone."""

    def __init__(
        self,
        *,
        image_sizes: Sequence[int],
        timm_model_ids: Sequence[str],
        timm_override_act_layers: Sequence[str | None],
        use_fused_vision_backbone: bool,
    ) -> None:
        super().__init__()
        if not use_fused_vision_backbone:
            raise ValueError(
                "OpenVLA currently supports only the fused DINOv2 + SigLIP "
                "vision backbone."
            )
        if tuple(image_sizes) != _OPENVLA_IMAGE_SIZES:
            raise ValueError(
                "OpenVLA currently supports only 224x224 image inputs, "
                f"got image_sizes={list(image_sizes)}."
            )
        if tuple(timm_model_ids) != _OPENVLA_TIMM_MODEL_IDS:
            raise ValueError(
                "OpenVLA currently supports only the dinosiglip-vit-so-224px "
                "vision backbone, got "
                f"timm_model_ids={list(timm_model_ids)}."
            )
        if tuple(timm_override_act_layers) != _OPENVLA_TIMM_OVERRIDE_ACT_LAYERS:
            raise ValueError(
                "OpenVLA currently supports only the default timm activation "
                "layers, got "
                f"timm_override_act_layers={list(timm_override_act_layers)}."
            )

        self.image_size = image_sizes[0]
        self.use_fused_vision_backbone = use_fused_vision_backbone

        self.embed_dim = 2176 if use_fused_vision_backbone else 1024

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "Please install timm to use OpenVLA. OpenVLA verification "
                "used timm==0.9.10."
            ) from e

        self.dinov2_featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=self.image_size,
            act_layer=timm_override_act_layers[0],
        )
        self.siglip_featurizer = (
            timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=self.image_size,
                act_layer=timm_override_act_layers[1],
            )
            if use_fused_vision_backbone
            else None
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.dinov2_featurizer is None:
            raise RuntimeError("OpenVLA vision backbone is not initialized.")

        if self.use_fused_vision_backbone and pixel_values.shape[1] != 6:
            raise ValueError(
                "OpenVLA fused DINOv2 + SigLIP backbone expects 6-channel "
                "image inputs: 3 DINOv2-normalized channels followed by 3 "
                "SigLIP-normalized channels, "
                f"got {pixel_values.shape[1]} channels."
            )

        dinov2_pixels = pixel_values[:, :3]

        num_dinov2_blocks = len(self.dinov2_featurizer.blocks)
        dinov2_features = self.dinov2_featurizer.get_intermediate_layers(
            dinov2_pixels, n={num_dinov2_blocks - 2}
        )[0]

        if self.siglip_featurizer is not None:
            siglip_pixels = pixel_values[:, 3:]
            num_siglip_blocks = len(self.siglip_featurizer.blocks)
            siglip_features = self.siglip_featurizer.get_intermediate_layers(
                siglip_pixels, n={num_siglip_blocks - 2}
            )[0]
            return torch.cat([dinov2_features, siglip_features], dim=-1)

        return dinov2_features


class PrismaticProjector(nn.Module):
    """Project Prismatic vision features into the language-model hidden size."""

    def __init__(
        self,
        *,
        vision_dim: int,
        text_dim: int,
        use_fused_vision_backbone: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        if use_fused_vision_backbone:
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
                text_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2",
            )
            self.act_fn2 = get_act_fn("gelu")
            self.fc3 = ReplicatedLinear(
                text_dim,
                text_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc3",
            )
        else:
            self.fc1 = ColumnParallelLinear(
                vision_dim,
                text_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1",
            )
            self.act_fn1 = get_act_fn("gelu")
            self.fc2 = RowParallelLinear(
                text_dim,
                text_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2",
            )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(image_features)
        hidden_states = self.act_fn1(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        if self.use_fused_vision_backbone:
            hidden_states = self.act_fn2(hidden_states)
            hidden_states, _ = self.fc3(hidden_states)

        return hidden_states


class OpenVLAProcessingInfo(BaseProcessingInfo):
    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)
        self.hf_processor = OpenVLAProcessor(
            image_processor=OpenVLAImageProcessor(
                image_size=self.get_hf_config().image_sizes[0],
            ),
            tokenizer=self.get_tokenizer(),
        )

    def get_hf_config(self) -> OpenVLAConfig:
        return self.ctx.get_hf_config(OpenVLAConfig)

    def get_hf_processor(self, **kwargs: object) -> OpenVLAProcessor:
        return self.hf_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        image_size = self.get_hf_config().image_sizes[0]
        return _get_num_image_tokens(image_size)

    def get_image_size_with_most_features(self) -> ImageSize:
        image_size = self.get_hf_config().image_sizes[0]
        return ImageSize(width=image_size, height=image_size)

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        image_size = self.get_hf_config().image_sizes[0]
        return {"image": _get_num_image_tokens(image_size)}


class OpenVLADummyInputsBuilder(BaseDummyInputsBuilder[OpenVLAProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        image_overrides = mm_options.get("image")
        image_size = self.info.get_image_size_with_most_features()

        return {
            "image": self._get_dummy_images(
                width=image_size.width,
                height=image_size.height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class OpenVLAMultiModalProcessor(BaseMultiModalProcessor[OpenVLAProcessingInfo]):
    """Processor contract for OpenVLA image inputs.

    OpenVLA feeds the same RGB image to DINOv2 and SigLIP after different
    normalizations. The processor exposes this as one 6-channel tensor:
    channels 0-2 are DINOv2-normalized and channels 3-5 are SigLIP-normalized.
    """

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"))

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index

        tokenizer = self.info.get_tokenizer()
        bos_token_id = tokenizer.bos_token_id

        def get_insertion(item_idx: int) -> PromptUpdateDetails[list[int]]:
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )
            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

            image_tokens = [image_token_id] * num_image_tokens
            return PromptUpdateDetails.select_token_id(
                image_tokens,
                embed_token_id=image_token_id,
            )

        return [
            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.prefix(
                    [bos_token_id] if bos_token_id is not None else []
                ),
                insertion=get_insertion,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    OpenVLAMultiModalProcessor,
    info=OpenVLAProcessingInfo,
    dummy_inputs=OpenVLADummyInputsBuilder,
)
class OpenVLAForActionPrediction(nn.Module, SupportsMultiModal, SupportsPP):
    """OpenVLA wrapper with vLLM language-model execution wired in."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.image_token_id = config.image_token_index
        self.n_action_bins = config.n_action_bins
        self.num_patches = _get_num_image_tokens(config.image_sizes[0])

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_backbone = PrismaticVisionBackbone(
                image_sizes=config.image_sizes,
                timm_model_ids=config.timm_model_ids,
                timm_override_act_layers=config.timm_override_act_layers,
                use_fused_vision_backbone=config.use_fused_vision_backbone,
            )
            self.projector = PrismaticProjector(
                vision_dim=self.vision_backbone.embed_dim,
                text_dim=config.text_config.hidden_size,
                use_fused_vision_backbone=config.use_fused_vision_backbone,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "projector"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> OpenVLAImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None

        return OpenVLAImagePixelInputs(
            type="pixel_values",
            data=pixel_values,
            resolve_bindings={
                "h": self.config.image_sizes[0],
                "w": self.config.image_sizes[0],
            },
        )

    def _process_image_input(
        self,
        image_input: OpenVLAImagePixelInputs,
    ) -> torch.Tensor:
        if self.vision_backbone.dinov2_featurizer is None:
            raise RuntimeError("OpenVLA vision backbone is not initialized.")

        pixel_values = image_input["data"].to(
            dtype=self.vision_backbone.dinov2_featurizer.patch_embed.proj.weight.dtype
        )
        vision_features = self.vision_backbone(pixel_values)
        return self.projector(vision_features)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        return self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="projector",
            tower_model="vision_backbone",
        )

    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int:
        return num_image_tokens

    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int:
        return num_vision_tokens

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def maybe_rename_vision_weights(
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if name.startswith("vision_backbone.featurizer."):
                    name = name.replace(
                        "vision_backbone.featurizer.",
                        "vision_backbone.dinov2_featurizer.",
                        1,
                    )
                elif name.startswith("vision_backbone.fused_featurizer."):
                    name = name.replace(
                        "vision_backbone.fused_featurizer.",
                        "vision_backbone.siglip_featurizer.",
                        1,
                    )
                # HF uses .scale_factor, timm uses .gamma
                if ".ls1.scale_factor" in name or ".ls2.scale_factor" in name:
                    name = name.replace(".scale_factor", ".gamma")
                yield name, weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(maybe_rename_vision_weights(weights))
