# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM support for microsoft/Phi-4-reasoning-vision-15B.

Architecture: Siglip2 vision tower + MLP projector + Phi3 language model.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal

import torch
import torch.nn as nn
from transformers import BatchFeature, PretrainedConfig, Siglip2VisionConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
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
from vllm.utils.tensor_schema import TensorSchema, TensorShape

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

# The HF processor replaces "<image>" with IMAGE_TOKEN_INDEX (-200) in input_ids.
# Negative token IDs cause OverflowError during decoding, so we remap to a real
# in-vocabulary token.  The Phi-4-reasoning-vision tokenizer ships with reserved
# dummy tokens (<|dummy_0|> … <|dummy_83|>); we reuse the first one as the
# image placeholder.  This mirrors how Phi-3-vision uses its dedicated <|image|>
# token (ID 32044).
_IMAGE_TOKEN_ID = 100256  # <|dummy_0|> in the Phi-4 tokenizer


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


class Phi4SiglipProcessingInfo(BaseProcessingInfo):
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


class Phi4SiglipDummyInputsBuilder(
    BaseDummyInputsBuilder[Phi4SiglipProcessingInfo],
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


class Phi4SiglipMultiModalProcessor(
    BaseMultiModalProcessor[Phi4SiglipProcessingInfo],
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

        # The HF processor's tokenizer_image_token() replaces the "<image>"
        # string with IMAGE_TOKEN_INDEX (-200) in input_ids.  This breaks
        # vLLM's prompt-replacement pipeline which needs to find "<image>"
        # as normal sub-tokens.  Re-tokenize with the plain tokenizer so
        # that "<image>" stays as sub-tokens and can be located by
        # PromptReplacement.
        # NOTE: tokenizer.__call__() (not .encode()) must be used so that
        # added/special tokens like <|user|>, <|end|> are kept as single IDs.
        tokenizer = self.info.get_tokenizer()
        new_ids = tokenizer(prompt).input_ids
        processed["input_ids"] = torch.tensor([new_ids])

        return processed

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # The HF processor replaces "<image>" with a single -200 placeholder
        # but does NOT expand it into N vision-encoder tokens.  Since we also
        # re-tokenize the prompt (see _call_hf_processor), prompt updates are
        # never applied by the HF processor — vLLM handles the expansion via
        # _apply_prompt_updates.
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            pixel_attention_mask=MultiModalFieldConfig.batched("image"),
            spatial_shapes=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        def get_replacement(item_idx: int):
            # Read the actual patch grid from the NaFlex processor's
            # spatial_shapes output (same pattern as LFM2-VL).  This avoids
            # predicting from raw image dimensions, which can diverge from
            # the NaFlex resize/tile logic.
            out_item = out_mm_kwargs["image"][item_idx]
            spatial_shapes = out_item["spatial_shapes"].data
            assert isinstance(spatial_shapes, torch.Tensor)
            num_tokens = int(spatial_shapes.prod().item())
            return [_IMAGE_TOKEN_ID] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=DEFAULT_IMAGE_TOKEN,
                replacement=get_replacement,
            ),
        ]


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class Phi4SiglipImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - d: Max number of patches (padded across images in the batch)
        - fd: Features per patch (patch_size * patch_size * channels)
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[torch.Tensor, TensorShape("bn", "d", "fd")]
    pixel_attention_mask: Annotated[torch.Tensor, TensorShape("bn", "d")]
    spatial_shapes: Annotated[torch.Tensor, TensorShape("bn", 2)]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    Phi4SiglipMultiModalProcessor,
    info=Phi4SiglipProcessingInfo,
    dummy_inputs=Phi4SiglipDummyInputsBuilder,
)
class Phi4ForCausalLMV(nn.Module, SupportsMultiModal, SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_tower.vision_tower.vision_model.head.": None,
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
            mm_token_ids=[_IMAGE_TOKEN_ID],
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
        return (
            pixel_values_packed,
            spatial_shapes,
            cu_seqlens,
            max_seqlen,
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Phi4SiglipImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        pixel_attention_mask = kwargs.pop("pixel_attention_mask", None)
        spatial_shapes = kwargs.pop("spatial_shapes", None)
        if pixel_values is None:
            return None

        return Phi4SiglipImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )

    def _process_image_input(
        self, image_input: Phi4SiglipImagePixelInputs
    ) -> MultiModalEmbeddings:
        pixel_values = image_input["pixel_values"]
        pixel_attention_mask = image_input["pixel_attention_mask"]
        spatial_shapes = image_input["spatial_shapes"]

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

        if vision_features.dim() == 3:
            vision_features = vision_features.squeeze(0)

        image_features = self.multi_modal_projector(vision_features)

        valid_counts = pixel_attention_mask.sum(dim=1).tolist()
        return torch.split(image_features, [int(c) for c in valid_counts])

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
