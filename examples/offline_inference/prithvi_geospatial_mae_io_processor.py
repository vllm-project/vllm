# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import os

import torch

from vllm import LLM

# This example shows how to perform an offline inference that generates
# multimodal data. In this specific case this example will take a geotiff
# image as input, process it using the multimodal data processor, and
# perform inference.
# Requirements:
# - install TerraTorch v1.1 (or later):
#   pip install terratorch>=v1.1


def main():
    torch.set_default_dtype(torch.float16)
    image_url = "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff"  # noqa: E501

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
        io_processor_plugin="terratorch_segmentation",
        model_impl="terratorch",
        enable_mm_embeds=True,
    )

    pooler_output = llm.encode(img_prompt, pooling_task="plugin")
    output = pooler_output[0].outputs

    print(output)
    decoded_data = base64.b64decode(output.data)

    file_path = os.path.join(os.getcwd(), "offline_prediction.tiff")
    with open(file_path, "wb") as f:
        f.write(decoded_data)

    print(f"Output file path: {file_path}")


if __name__ == "__main__":
    main()
