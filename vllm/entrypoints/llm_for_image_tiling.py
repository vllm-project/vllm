# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM
from vllm.inputs.data import MultiModalPromptType
from vllm.outputs import ImageRequestOutput
from vllm.plugins.multimodal_data_processors import (
    get_multimodal_data_processor)


class LLMForImageTiling(LLM):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.multimodal_processor = (get_multimodal_data_processor(
            self.llm_engine.vllm_config))

    def predict(self, prompt: MultiModalPromptType) -> ImageRequestOutput:

        # At the momend we generate images (ab-)using pooling models
        # Here we first extract the prompts for the pooling model
        pooling_prompts = (self.multimodal_processor.pre_process(prompt, ))

        pooling_output = self.encode(pooling_prompts)

        output = (self.multimodal_processor.post_process(
            model_out=pooling_output))

        return output
