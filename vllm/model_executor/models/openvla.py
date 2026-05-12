# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
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
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import OpenVLAConfig
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.utils.torch_utils import set_default_torch_dtype

from .module_mapping import MultiModelKeys
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix

_OPENVLA_IMAGE_SIZE = 224
_OPENVLA_PATCH_SIZE = 14
_OPENVLA_NUM_IMAGE_TOKENS = (_OPENVLA_IMAGE_SIZE // _OPENVLA_PATCH_SIZE) ** 2


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
        self.image_sizes = list(image_sizes)
        self.timm_model_ids = list(timm_model_ids)
        self.timm_override_act_layers = list(timm_override_act_layers)
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.embed_dim = 2176 if use_fused_vision_backbone else 1024
        self.featurizer: nn.Module | None = None
        self.fused_featurizer: nn.Module | None = None

    def init_timm_models(self) -> None:
        if self.featurizer is not None:
            return

        try:
            import timm
        except ImportError as e:
            raise ImportError("Please install timm to use OpenVLA.") from e

        image_size = self.image_sizes[0]
        with set_default_torch_dtype(torch.float16):
            self.featurizer = timm.create_model(
                self.timm_model_ids[0],
                pretrained=False,
                num_classes=0,
                img_size=image_size,
                act_layer=self.timm_override_act_layers[0],
            )
            if self.use_fused_vision_backbone:
                self.fused_featurizer = timm.create_model(
                    self.timm_model_ids[1],
                    pretrained=False,
                    num_classes=0,
                    img_size=image_size,
                    act_layer=self.timm_override_act_layers[1],
                )

        self.featurizer = self.featurizer.to(
            device=current_platform.device_type,
            dtype=torch.get_default_dtype(),
        )
        if self.fused_featurizer is not None:
            self.fused_featurizer = self.fused_featurizer.to(
                device=current_platform.device_type, dtype=torch.get_default_dtype()
            )

    def get_input_dtype(self) -> torch.dtype:
        if self.featurizer is None:
            raise RuntimeError("OpenVLA vision backbone is not initialized.")
        return self.featurizer.patch_embed.proj.weight.dtype

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.featurizer is None:
            raise RuntimeError("OpenVLA vision backbone is not initialized.")
        if pixel_values.shape[1] != 6:
            raise ValueError(
                "OpenVLA expects 6-channel image inputs, "
                f"got {pixel_values.shape[1]} channels."
            )

        dinov2_pixels = pixel_values[:, :3]
        siglip_pixels = pixel_values[:, 3:]

        n_blocks = len(self.featurizer.blocks)
        features = self.featurizer.get_intermediate_layers(
            dinov2_pixels, n={n_blocks - 2}
        )[0]

        if self.fused_featurizer is not None:
            n_fused_blocks = len(self.fused_featurizer.blocks)
            fused_features = self.fused_featurizer.get_intermediate_layers(
                siglip_pixels, n={n_fused_blocks - 2}
            )[0]
            features = torch.cat([features, fused_features], dim=-1)

        return features


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
    def get_hf_config(self) -> OpenVLAConfig:
        return self.ctx.get_hf_config(OpenVLAConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        return _OPENVLA_NUM_IMAGE_TOKENS

    def get_image_size_with_most_features(self) -> ImageSize:
        return ImageSize(width=_OPENVLA_IMAGE_SIZE, height=_OPENVLA_IMAGE_SIZE)

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        return {"image": _OPENVLA_NUM_IMAGE_TOKENS}


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

        return {
            "image": self._get_dummy_images(
                width=_OPENVLA_IMAGE_SIZE,
                height=_OPENVLA_IMAGE_SIZE,
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

    IMAGENET_MEAN = np.array([0.484375, 0.455078125, 0.40625], dtype=np.float32)
    IMAGENET_STD = np.array([0.228515625, 0.2236328125, 0.224609375], dtype=np.float32)
    SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    SIGLIP_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    @staticmethod
    def _to_rgb_image(image: object) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError(
                    "OpenVLA image input must have 3 dimensions, "
                    f"got shape {image.shape}"
                )

            if image.shape[0] in (1, 3):
                image = np.moveaxis(image, 0, -1)

            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            elif image.shape[-1] != 3:
                raise ValueError(
                    "OpenVLA image input must have 1 or 3 channels, "
                    f"got shape {image.shape}"
                )

            if image.dtype != np.uint8:
                image = image.astype(np.float32)
                if image.max(initial=0.0) <= 1.0:
                    image = image * 255.0
                image = np.clip(image, 0, 255).astype(np.uint8)

            return Image.fromarray(image).convert("RGB")

        raise TypeError(
            "OpenVLA image input must be a PIL image, numpy array, or torch tensor; "
            f"got {type(image)}"
        )

    def _preprocess_image(self, image: object) -> torch.Tensor:
        image = self._to_rgb_image(image)
        image = image.resize(
            (_OPENVLA_IMAGE_SIZE, _OPENVLA_IMAGE_SIZE),
            Image.Resampling.BICUBIC,
        )

        raw = np.asarray(image, dtype=np.float32) / 255.0
        dinov2_pixels = ((raw - self.IMAGENET_MEAN) / self.IMAGENET_STD).transpose(
            2, 0, 1
        )
        siglip_pixels = ((raw - self.SIGLIP_MEAN) / self.SIGLIP_STD).transpose(2, 0, 1)
        pixel_values = np.concatenate([dinov2_pixels, siglip_pixels], axis=0)
        return torch.from_numpy(pixel_values)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, **tok_kwargs)

        images = mm_data.get("images")
        if images is None:
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
            images = [images]
        if len(images) == 0:
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        pixel_values = torch.stack(
            [self._preprocess_image(image) for image in images],
            dim=0,
        )
        return BatchFeature(
            dict(input_ids=[prompt_ids], pixel_values=pixel_values),
            tensor_type="pt",
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"))

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

    embed_input_ids = SupportsMultiModal.embed_input_ids

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
        self.num_patches = _OPENVLA_NUM_IMAGE_TOKENS

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
                "h": _OPENVLA_IMAGE_SIZE,
                "w": _OPENVLA_IMAGE_SIZE,
            },
        )

    def _process_image_input(
        self,
        image_input: OpenVLAImagePixelInputs,
    ) -> torch.Tensor:
        pixel_values = image_input["data"].to(
            dtype=self.vision_backbone.get_input_dtype()
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
        self.vision_backbone.init_timm_models()

        def maybe_rename_layerscale(
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if ".ls1.scale_factor" in name or ".ls2.scale_factor" in name:
                    name = name.replace(".scale_factor", ".gamma")
                yield name, weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(maybe_rename_layerscale(weights))
