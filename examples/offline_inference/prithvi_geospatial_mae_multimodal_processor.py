# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import os

import torch

from vllm import LLM
from vllm.plugins.multimodal_data_processors import get_multimodal_data_processor
from vllm.plugins.multimodal_data_processors.types import (
    ImagePrompt,
    ImageRequestOutput,
    MultiModalPromptType,
)

# This example shows how to perform an offline inference that generates
# multimodal data. In this specific case this example will take a geotiff
# image as input, process it using the multimodal data processor, and
# perform inference.
# Reuirement - install plugin at:
#   https://github.com/christian-pinto/prithvi_multimodal_processor_plugin


class LLMForImageTiling(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.multimodal_processor = get_multimodal_data_processor(
            self.llm_engine.vllm_config
        )

    def predict(self, prompt: MultiModalPromptType) -> ImageRequestOutput:
        # At the momend we generate images (ab-)using pooling models
        # Here we first extract the prompts for the pooling model
        pooling_prompts = self.multimodal_processor.pre_process(prompt)

        pooling_output = self.encode(pooling_prompts)

        output = self.multimodal_processor.post_process(
            model_out=pooling_output, out_format="path"
        )

        assert isinstance(output, ImageRequestOutput)

        return output


def main():
    torch.set_default_dtype(torch.float16)

    image_url = "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/India_900498_S2Hand.tif"  # noqa: E501

    img_prompt = ImagePrompt(
        data=image_url,
        data_format="url",
        image_format="tiff",
        out_format="b64_json",
    )
    prompt = {
        "prompt_token_ids": [1],
        "multi_modal_data": {"image": dict(img_prompt)},
    }

    llm = LLM(
        model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        enforce_eager=True,
        # Limit the maximum number of parallel requests
        # to avoid the model going OOM.
        # The maximum number depends on the available GPU memory
        max_num_seqs=32,
    )

    output = llm.encode_with_mm_data_plugin(prompt)

    print(output)
    decoded_data = base64.b64decode(output[0].task_output.data)

    file_path = os.path.join(os.getcwd(), "offline_prediction.tiff")
    with open(file_path, "wb") as f:
        f.write(decoded_data)

    print(f"Output file path: {file_path}")


if __name__ == "__main__":
    main()
