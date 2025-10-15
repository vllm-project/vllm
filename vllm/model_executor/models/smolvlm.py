# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from transformers import SmolVLMProcessor

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from .idefics3 import Idefics3DummyInputsBuilder as SmolVLMDummyInputsBuilder
from .idefics3 import Idefics3ForConditionalGeneration, Idefics3ProcessingInfo
from .idefics3 import Idefics3MultiModalProcessor as SmolVLMMultiModalProcessor


class SmolVLMProcessingInfo(Idefics3ProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> SmolVLMProcessor:
        return self.ctx.get_hf_processor(SmolVLMProcessor, **kwargs)

    def _get_image_token(self, processor: SmolVLMProcessor | None) -> tuple[str, str]:
        if processor is None:
            processor = self.get_hf_processor()
        image_token = processor.image_token
        fake_image_token = processor.fake_image_token
        global_image_token = processor.global_image_token
        return image_token, fake_image_token, global_image_token


@MULTIMODAL_REGISTRY.register_processor(
    SmolVLMMultiModalProcessor,
    info=SmolVLMProcessingInfo,
    dummy_inputs=SmolVLMDummyInputsBuilder,
)
class SmolVLMForConditionalGeneration(Idefics3ForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
        )
