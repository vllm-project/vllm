# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

from transformers import SmolVLMProcessor

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

# yapf: disable
from .idefics3 import Idefics3DummyInputsBuilder as SmolVLMDummyInputsBuilder
from .idefics3 import Idefics3ForConditionalGeneration
from .idefics3 import Idefics3MultiModalProcessor as SmolVLMMultiModalProcessor
from .idefics3 import Idefics3ProcessingInfo

# yapf: enable


class SmolVLMProcessingInfo(Idefics3ProcessingInfo):

    def get_hf_processor(
        self,
        *,
        max_image_size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ) -> SmolVLMProcessor:
        if max_image_size is not None:
            kwargs["max_image_size"] = max_image_size

        return self.ctx.get_hf_processor(SmolVLMProcessor, **kwargs)

    def _get_image_token(
            self, processor: Optional[SmolVLMProcessor]) -> tuple[str, str]:
        if processor is None:
            processor = self.get_hf_processor()
        image_token = processor.image_token
        fake_image_token = processor.fake_image_token
        global_image_token = processor.global_image_token
        return image_token, fake_image_token, global_image_token


@MULTIMODAL_REGISTRY.register_processor(SmolVLMMultiModalProcessor,
                                        info=SmolVLMProcessingInfo,
                                        dummy_inputs=SmolVLMDummyInputsBuilder)
class SmolVLMForConditionalGeneration(Idefics3ForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
        )
