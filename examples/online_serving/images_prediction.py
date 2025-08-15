# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64

import requests

# This example shows how to perform an online inference that generates
# multimodal data. In this specific case this example will take a geotiff
# image as input, process it using the multimodal data processor, and
# perform inference.
# Reuirements :
# - install plugin at:
#   https://github.com/christian-pinto/prithvi_multimodal_processor_plugin
# - start vllm in serving mode with the below args
#   --model='christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM'
#   --task embed --trust-remote-code
#   --skip-tokenizer-init --enforce-eager


def main():
    image_url = "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/resolve/main/examples/India_900498_S2Hand.tif"  # noqa: E501
    server_endpoint = "http://localhost:8000/v1/images/prediction"

    request_payload_url = {
        "image": {"data": image_url, "data_format": "url"},
        "image_format": "tiff",
        "model": "christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
        "response_format": "b64_json",
        "priority": 0,
    }

    ret = requests.post(server_endpoint, json=request_payload_url)

    response = ret.json()

    decoded_image = base64.b64decode(response["image"]["data"])

    with open("/workspace/vllm_max/online.tiff", "wb") as f:
        f.write(decoded_image)


if __name__ == "__main__":
    main()
