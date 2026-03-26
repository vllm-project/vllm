# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM support for microsoft/Phi-4-reasoning-vision-15B.

Architecture: Siglip2 vision tower + MLP projector + Phi3 language model.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from transformers import BatchFeature, PretrainedConfig, Siglip2VisionConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
)
from vllm.sequence import IntermediateTensors

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .lfm2_siglip2 import Siglip2Model
from .llava import LlavaMultiModalProjector
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

logger = init_logger(__name__)

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


class Phi4VisionRProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def _get_vision_config(self) -> dict:
        return self.get_hf_config().vision_config  # type: ignore[attr-defined]

    def _get_patch_size(self) -> int:
        vc = self._get_vision_config()
        if isinstance(vc, dict):
            return vc.get("patch_size", 16)
        return getattr(vc, "patch_size", 16)

    def _get_max_num_patches(self) -> int:
        return getattr(self.get_hf_config(), "max_num_patches", 3600)

    def _get_min_num_patches(self) -> int:
        return getattr(self.get_hf_config(), "min_num_patches", 256)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        patch_size = self._get_patch_size()
        min_patches = self._get_min_num_patches()
        max_patches = self._get_max_num_patches()

        num_patches_h = image_height // patch_size
        num_patches_w = image_width // patch_size
        num_patches = max(num_patches_h * num_patches_w, 1)
        num_patches = max(min(num_patches, max_patches), min_patches)
        return num_patches

    def get_image_size_with_most_features(self) -> ImageSize:
        patch_size = self._get_patch_size()
        max_patches = self._get_max_num_patches()
        side = int(math.sqrt(max_patches)) * patch_size
        return ImageSize(width=side, height=side)

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        return {"image": self._get_max_num_patches()}


class Phi4VisionRDummyInputsBuilder(
    BaseDummyInputsBuilder[Phi4VisionRProcessingInfo],
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return DEFAULT_IMAGE_TOKEN * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=num_images,
                overrides=mm_options.get("image"),
            ),
        }


class Phi4VisionRMultiModalProcessor(
    BaseMultiModalProcessor[Phi4VisionRProcessingInfo],
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        input_ids = processed["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        # The HF processor inserts IMAGE_TOKEN_INDEX (-200) as placeholder.
        # Replace with a real token id that vLLM can handle during
        # prompt-update matching. We use 0 as a temporary stand-in;
        # _get_prompt_updates will expand these into the correct count.
        input_ids = input_ids.clone()
        input_ids.masked_fill_(input_ids == IMAGE_TOKEN_INDEX, 0)
        processed["input_ids"] = input_ids

        return processed

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            pixel_attention_mask=MultiModalFieldConfig.batched("image"),
            spatial_shapes=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)
            num_tokens = self.info.get_num_image_tokens(
                image_width=image_size.width,
                image_height=image_size.height,
            )
            return [0] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=DEFAULT_IMAGE_TOKEN,
                replacement=get_replacement,
            ),
        ]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    Phi4VisionRMultiModalProcessor,
    info=Phi4VisionRProcessingInfo,
    dummy_inputs=Phi4VisionRDummyInputsBuilder,
)
class Phi4ForCausalLMV(nn.Module, SupportsMultiModal, SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_tower.vision_tower.": "vision_tower.",
            "model.mm_projector.0.": "multi_modal_projector.linear_1.",
            "model.mm_projector.2.": "multi_modal_projector.linear_2.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return DEFAULT_IMAGE_TOKEN
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config: PretrainedConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        vision_config_dict: dict = getattr(config, "vision_config", {})
        if isinstance(vision_config_dict, dict):
            if "patch_size" not in vision_config_dict:
                vision_config_dict["patch_size"] = 16
            siglip2_config = Siglip2VisionConfig(**vision_config_dict)
        else:
            siglip2_config = vision_config_dict

        vision_hidden_size: int = config.mm_hidden_size  # type: ignore[attr-defined]
        text_hidden_size: int = config.hidden_size  # type: ignore[attr-defined]

        with self._mark_tower_model(vllm_config, "image"):
            layer_idx = -2
            num_hidden_layers = siglip2_config.num_hidden_layers + layer_idx + 1

            self.vision_tower = Siglip2Model(
                siglip2_config,
                quant_config=quant_config,
                num_hidden_layers_override=num_hidden_layers,
                require_post_norm=False,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.multi_modal_projector = LlavaMultiModalProjector(
                vision_hidden_size=vision_hidden_size,
                text_hidden_size=text_hidden_size,
                projector_hidden_act="gelu",
                multimodal_projector_bias=True,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "multi_modal_projector"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Phi3ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        self.configure_mm_token_handling(
            vocab_size=config.vocab_size,  # type: ignore[attr-defined]
            mm_token_ids=[],
        )

    def _packed_from_padded(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert padded NaFlex tensors to packed format for Siglip2Model."""
        valid_counts = pixel_attention_mask.sum(dim=1).to(torch.int32)
        pixel_values_packed = pixel_values[pixel_attention_mask.bool()]
        cu_seqlens = torch.zeros(
            len(valid_counts) + 1,
            dtype=torch.int32,
            device=pixel_values.device,
        )
        cu_seqlens[1:] = valid_counts.cumsum(0)
        max_seqlen = valid_counts.max()
        return pixel_values_packed, spatial_shapes, cu_seqlens, max_seqlen

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return []

        assert isinstance(pixel_values, torch.Tensor)
        pixel_attention_mask = kwargs.pop("pixel_attention_mask")
        spatial_shapes = kwargs.pop("spatial_shapes")
        assert isinstance(pixel_attention_mask, torch.Tensor)
        assert isinstance(spatial_shapes, torch.Tensor)

        (
            pixel_values_packed,
            spatial_shapes_packed,
            cu_seqlens,
            max_seqlen,
        ) = self._packed_from_padded(pixel_values, pixel_attention_mask, spatial_shapes)

        vision_features = self.vision_tower(
            pixel_values_packed=pixel_values_packed,
            spatial_shapes=spatial_shapes_packed,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            select_layers=[-2],
        )

        image_features = self.multi_modal_projector(vision_features)

        valid_counts = pixel_attention_mask.sum(dim=1).tolist()
        per_image = torch.split(image_features, [int(c) for c in valid_counts])
        return per_image  # type: ignore[return-value]

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

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
