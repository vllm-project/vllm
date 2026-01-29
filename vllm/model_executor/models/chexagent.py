# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)

from .blip2 import (
    Blip2DummyInputsBuilder,
    Blip2ForConditionalGeneration,
    Blip2ImageEmbeddingInputs,
    Blip2ImageInputs,
    Blip2ImagePixelInputs,
    Blip2MultiModalProcessor,
    Blip2ProcessingInfo,
)
from .utils import maybe_prefix

# Hugging Face tokenizer for CheXagent uses this sentinel for image slots.
_IMAGE_TOKEN_ID = 50265

CheXagentImagePixelInputs = Blip2ImagePixelInputs
CheXagentImageEmbeddingInputs = Blip2ImageEmbeddingInputs
CheXagentImageInputs = Blip2ImageInputs


class CheXagentProcessingInfo(Blip2ProcessingInfo):
    def get_hf_config(self):
        # CheXagent config is compatible with BLIP-2 but does not subclass the
        # exact HF type, so we return the stored config directly.
        return self.ctx.model_config.hf_config


class CheXagentDummyInputsBuilder(Blip2DummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return "<image>" * num_images


class CheXagentMultiModalProcessor(Blip2MultiModalProcessor):
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        num_image_tokens = self.info.get_num_image_tokens()
        image_tokens = [_IMAGE_TOKEN_ID] * num_image_tokens

        return [
            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.start(),
                insertion=image_tokens,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    CheXagentMultiModalProcessor,
    info=CheXagentProcessingInfo,
    dummy_inputs=CheXagentDummyInputsBuilder,
)
class CheXagentForConditionalGeneration(Blip2ForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        old_projection = self.language_projection
        params_dtype = old_projection.weight.dtype
        self.language_projection = ReplicatedLinear(
            config.qformer_config.hidden_size,
            config.text_config.hidden_size,
            bias=True,
            params_dtype=params_dtype,
            prefix=maybe_prefix(prefix, "language_projection"),
        )

        with torch.no_grad():
            self.language_projection.weight.copy_(old_projection.weight)
            old_bias = getattr(old_projection, "bias", None)
            new_bias = getattr(self.language_projection, "bias", None)
            if old_bias is not None and new_bias is not None:
                new_bias.copy_(old_bias)

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> CheXagentImageInputs | None:
        return super()._parse_and_validate_image_input(**kwargs)
