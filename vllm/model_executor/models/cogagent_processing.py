# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from transformers import BatchFeature

    from vllm.config.multimodal import BaseDummyOptions

from PIL import Image

from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.inputs import (
    MultiModalEncDecInputs,
    MultiModalFieldConfig,
    MultiModalKwargs,
    MultiModalUUIDDict,
)
from vllm.multimodal.processing import (
    EncDecMultiModalProcessor,
    MultiModalDataItems,
    PromptIndexTargets,
    PromptInsertion,
    PromptReplacement,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, BaseProcessingInfo
from vllm.transformers_utils.configs.cogagent import (
    CogAgentConfig,
    EVACLIPVisionConfig,
    EVALargeVisionConfig,
)
from vllm.transformers_utils.processors.cogagent import CogAgentProcessor
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils.tensor_schema import TensorSchema, TensorShape

ImageData = Union[
    list[Image.Image | np.ndarray | torch.Tensor], np.ndarray, torch.Tensor
]


def get_max_image_tokens(hf_config: EVACLIPVisionConfig | EVALargeVisionConfig) -> int:
    image_size = hf_config.image_size
    patch_size = hf_config.patch_size

    return (image_size // patch_size) ** 2 + 2


class CogAgentImagePixelInputs(TensorSchema):
    """
    images: bn, C, H, W. Images
    cross_images: bn, C, H, W. Resized Images passed to the Large Encoder.
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[torch.Tensor, TensorShape("bn", 3, "side", "side")]
    cross_pixel_values: Annotated[
        torch.Tensor, TensorShape("bn", 3, "cross_side", "cross_side")
    ]


class CogAgentImageEmbeddingInputs(TensorSchema):
    """
    image_embeds: bn, L, D. Combined Embeddings of EVAVisionEncoder and EVALargeEncoder.
    """

    type: Literal["image_embeds"] = "image_embeds"
    image_embeds: Annotated[torch.Tensor, TensorShape("bn", "L", "HD")]


class CogAgentProcessorConfig(TypedDict, total=True):
    """
    template_version: version to use when a chat template is not added.
    image_size: EVACLIPVisionConfig Image Size.
        This is the smaller of the two image sizes.
    cross_image_size: EVALargeVisionConfig Image Size
    """

    template_version: Literal["base", "chat", "chat_old"]
    image_size: int
    cross_image_size: int
    image_token: str
    dtype: torch.dtype


class CogAgentProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> CogAgentConfig:
        return self.ctx.get_hf_config(CogAgentConfig)

    def get_image_processor(self) -> Callable[[ImageData], dict[str, torch.Tensor]]:
        return self.get_hf_processor()._process_images

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        # max number of tokens generated per request is large encoder + small encoder
        # only tokens from the small encoder are embedded with text.
        
        num_tokens = get_max_image_tokens(self.get_hf_config().vision_config)
        num_tokens += get_max_image_tokens(self.get_hf_config().cross_vision_config) - 2

        return {"image": num_tokens}

    def get_hf_processor(self, **kwargs):
        hf_config = self.get_hf_config()

        # allow passthrough from model config to processor config
        defaults = CogAgentProcessorConfig(
            image_size=hf_config.image_size,
            cross_image_size=hf_config.cross_image_size,
            template_version=hf_config.template_version,
            image_token=hf_config.image_token,
            dtype=hf_config.dtype,
        )
        kwargs = {**defaults, **kwargs}

        return self.ctx.get_hf_processor(CogAgentProcessor, **kwargs)


class CogAgentDummyInputsBuilder(BaseDummyInputsBuilder[CogAgentProcessingInfo]):
    def get_dummy_text(self, mm_counts):
        num_images = mm_counts.get("image", 0)

        return " " * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, "BaseDummyOptions"] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        cfg = self.info.get_hf_config()
        image_size = cfg.image_size

        mm_data = {
            "image": self._get_dummy_images(
                width=image_size,
                height=image_size,
                num_images=num_images,
                overrides=mm_options.get("image") if mm_options else None,
            )
        }

        return mm_data


class CogAgentMultiModalProcessor(EncDecMultiModalProcessor[CogAgentProcessingInfo]):
    """
    Creates MultiModal Output for Cogagent in the form
    {
        encoder: image_tokens
        decoder: image_tokens + prompt
    }
    """

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return True

    def create_encoder_prompt(self, prompt, mm_data):
        config = self.info.get_hf_config()
        num_encoder_tokens = get_max_image_tokens(config.cross_vision_config) - 2

        # For similar reasons as Whisper, we ignore this prompt.
        return [0] * num_encoder_tokens

    def create_decoder_prompt(self, prompt, mm_data):
        return prompt

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        tokenizer = self.info.get_tokenizer()  # type: LlamaTokenizer

        image_token = self.info.get_hf_processor().image_token
        bos_token: list[int] = [tokenizer.bos_token_id]

        num_image_tokens = get_max_image_tokens(self.info.get_hf_config().vision_config)

        return [
            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.prefix(bos_token),
                insertion=image_token * num_image_tokens,
            )
        ]

    def _get_mm_fields_config(
        self, hf_inputs: "BatchFeature", hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            cross_pixel_values=MultiModalFieldConfig.batched("image"),
            # image_embeds=MultiModalFieldConfig.batched("image"),
            # cross_embeds=MultiModalFieldConfig.batched("image")
        )

    def apply(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> MultiModalEncDecInputs:
        """
        Process multi-modal inputs to be used in vLLM.
        The main processing steps are modified to fit encoder-decoder model:
        1. Create encoder prompt from input prompt text.
        2. Apply the HF processor on encoder prompt.
        3. Copy the input prompt text as decoder prompt inputs.
        """
        encoder_prompt = self.create_encoder_prompt(prompt, mm_data)
        decoder_prompt = self.create_decoder_prompt(prompt, mm_data)

        # decoder mixes the smaller image model embeds during prefill.
        # so we need to ensure the position information exists by applying
        # the processor onto it instead.
        mm_inputs = super(EncDecMultiModalProcessor, self).apply(
            decoder_prompt,
            mm_data,
            hf_processor_mm_kwargs,
            tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

        # Skip over _get_enc_dec_inputs as result is fixed
        mm_inputs = MultiModalEncDecInputs(
            encoder_prompt_token_ids=encoder_prompt, **mm_inputs
        )

        return mm_inputs
