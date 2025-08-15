# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm import LLM
from vllm.inputs.data import ImagePrompt, MultiModalPromptType
from vllm.outputs import ImageRequestOutput
from vllm.plugins.multimodal_data_processors import get_multimodal_data_processor

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

    image_url = "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/resolve/main/examples/India_900498_S2Hand.tif"  # noqa: E501

    llm = LLMForImageTiling(
        model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        enforce_eager=True,
        # Limit the maximum number of parallel requests
        # to avoid the model going OOM
        max_num_seqs=32,
    )

    prompt = ImagePrompt(
        data_format="url",
        image_format="tiff",
        data=image_url,
    )

    output = llm.predict(prompt)

    print(output)
    print(f"Output file path: {output.data}")


if __name__ == "__main__":
    main()
