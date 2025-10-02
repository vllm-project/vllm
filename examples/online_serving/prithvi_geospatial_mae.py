# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import os

import requests

# This example shows how to perform an online inference that generates
# multimodal data. In this specific case this example will take a geotiff
# image as input, process it using the multimodal data processor, and
# perform inference.
# Requirements :
# - install plugin at:
#   https://github.com/christian-pinto/prithvi_io_processor_plugin
# - start vllm in serving mode with the below args
#   --model='christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM'
#   --model-impl terratorch
#   --task embed --trust-remote-code
#   --skip-tokenizer-init --enforce-eager
#   --io-processor-plugin prithvi_to_tiff


def main():
    image_url = "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff"  # noqa: E501
    server_endpoint = "http://localhost:8000/pooling"

    request_payload_url = {
        "data": {
            "data": image_url,
            "data_format": "url",
            "image_format": "tiff",
            "out_data_format": "b64_json",
        },
        "priority": 0,
        "model": "christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
        "softmax": False,
    }

    ret = requests.post(server_endpoint, json=request_payload_url)

    print(f"response.status_code: {ret.status_code}")
    print(f"response.reason:{ret.reason}")

    response = ret.json()

    decoded_image = base64.b64decode(response["data"]["data"])

    out_path = os.path.join(os.getcwd(), "online_prediction.tiff")

    with open(out_path, "wb") as f:
        f.write(decoded_image)


if __name__ == "__main__":
    main()
