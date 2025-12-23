# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64

import numpy as np
import pytest
import requests
import torch

from vllm.utils.serial_utils import tensor2base64

from ...utils import RemoteOpenAIServer


def _terratorch_dummy_messages():
    pixel_values = torch.full((6, 512, 512), 1.0, dtype=torch.float16)
    location_coords = torch.full((1, 2), 1.0, dtype=torch.float16)

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_embeds",
                    "image_embeds": {
                        "pixel_values": tensor2base64(pixel_values),
                        "location_coords": tensor2base64(location_coords),
                    },
                }
            ],
        }
    ]


@pytest.mark.parametrize(
    "model_name", ["ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"]
)
def test_single_request(model_name: str):
    args = [
        "--runner",
        "pooling",
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--enforce-eager",
        "--trust-remote-code",
        "--max-num-seqs",
        "32",
        "--model-impl",
        "terratorch",
        "--skip-tokenizer-init",
        "--enable-mm-embeds",
    ]

    with RemoteOpenAIServer(model_name, args) as server:
        response = requests.post(
            server.url_for("pooling"),
            json={
                "model": model_name,
                "messages": _terratorch_dummy_messages(),
                "encoding_format": "base64",
            },
        )
        response.raise_for_status()

        output = response.json()["data"][0]["data"]

        np_response = np.frombuffer(base64.b64decode(output), dtype=np.float32)
        assert len(np_response) == 524288
