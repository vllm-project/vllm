# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import os

import torch

from vllm import LLM
from vllm.pooling_params import PoolingParams

# This example shows how to perform an offline inference that generates
# multimodal data. In this specific case this example will take a geotiff
# image as input, process it using the multimodal data processor, and
# perform inference.
# Requirement - install plugin at:
#   https://github.com/christian-pinto/prithvi_io_processor_plugin


def main():
    torch.set_default_dtype(torch.float16)
    image_url = "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/India_900498_S2Hand.tif"  # noqa: E501

    img_prompt = dict(
        data=image_url,
        data_format="url",
        image_format="tiff",
        out_data_format="b64_json",
    )

    llm = LLM(
        model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        enforce_eager=True,
        # Limit the maximum number of parallel requests
        # to avoid the model going OOM.
        # The maximum number depends on the available GPU memory
        max_num_seqs=32,
        io_processor_plugin="prithvi_to_tiff_india",
        model_impl="terratorch",
    )

    pooling_params = PoolingParams(task="encode", softmax=False)
    pooler_output = llm.encode(
        img_prompt,
        pooling_params=pooling_params,
    )
    output = pooler_output[0].outputs

    print(output)
    decoded_data = base64.b64decode(output.data)

    file_path = os.path.join(os.getcwd(), "offline_prediction.tiff")
    with open(file_path, "wb") as f:
        f.write(decoded_data)

    print(f"Output file path: {file_path}")


if __name__ == "__main__":
    main()
