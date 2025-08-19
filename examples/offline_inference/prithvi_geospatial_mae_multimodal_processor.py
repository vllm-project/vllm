# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import os

import torch

from vllm import LLM
from vllm.plugins.multimodal_data_processors.types import (
    ImagePrompt,
)
from vllm.pooling_params import PoolingParams

# This example shows how to perform an offline inference that generates
# multimodal data. In this specific case this example will take a geotiff
# image as input, process it using the multimodal data processor, and
# perform inference.
# Reuirement - install plugin at:
#   https://github.com/christian-pinto/prithvi_multimodal_processor_plugin


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

    pooling_params = PoolingParams(task="encode", softmax=False)

    output = llm.encode_with_mm_data_plugin(
        prompt,
        pooling_params=pooling_params,
    )

    print(output)
    decoded_data = base64.b64decode(output[0].task_output.data)

    file_path = os.path.join(os.getcwd(), "offline_prediction.tiff")
    with open(file_path, "wb") as f:
        f.write(decoded_data)

    print(f"Output file path: {file_path}")


if __name__ == "__main__":
    main()
