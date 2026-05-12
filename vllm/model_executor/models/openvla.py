# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
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
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import OpenVLAConfig

_OPENVLA_IMAGE_SIZE = 224
_OPENVLA_PATCH_SIZE = 14
_OPENVLA_NUM_IMAGE_TOKENS = (_OPENVLA_IMAGE_SIZE // _OPENVLA_PATCH_SIZE) ** 2


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
            Image.Resampling.BILINEAR,
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

        images = mm_data.get("images", [])
        if not images:
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
            images = [images]

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
    """Registration and processor skeleton for OpenVLA.

    The executable OpenVLA model is implemented in later phases.  Keeping this
    class importable lets config, architecture registration, and multimodal
    token accounting be verified independently before adding vision towers,
    projector, language model, and action decoding.
    """

    embed_input_ids = SupportsMultiModal.embed_input_ids

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config

    def get_language_model(self) -> nn.Module:
        raise NotImplementedError("OpenVLA execution is implemented in a later phase")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        raise NotImplementedError("OpenVLA execution is implemented in a later phase")

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        raise NotImplementedError("OpenVLA execution is implemented in a later phase")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        raise NotImplementedError("OpenVLA execution is implemented in a later phase")

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        raise NotImplementedError("OpenVLA execution is implemented in a later phase")

    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int:
        return num_image_tokens

    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int:
        return num_vision_tokens

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        raise NotImplementedError("OpenVLA execution is implemented in a later phase")
