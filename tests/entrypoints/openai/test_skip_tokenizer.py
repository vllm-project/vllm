# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import io

import numpy as np
import pytest
import requests
import torch

from ...utils import RemoteOpenAIServer

MODEL_NAME = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
DTYPE = "float16"


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "pooling",
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        DTYPE,
        "--enforce-eager",
        "--trust-remote-code",
        "--skip-tokenizer-init",
        "--max-num-seqs",
        "32",
        "--model-impl",
        "terratorch"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_single_request(server: RemoteOpenAIServer, model_name: str):

    pixel_values = torch.full((6, 512, 512), 1.0, dtype=torch.float16)
    location_coords = torch.full((1, 2), 1.0, dtype=torch.float16)

    buffer_tiff = io.BytesIO()
    torch.save(pixel_values, buffer_tiff)
    buffer_tiff.seek(0)
    binary_data = buffer_tiff.read()
    base64_tensor_embedding = base64.b64encode(binary_data).decode('utf-8')

    buffer_coord = io.BytesIO()
    torch.save(location_coords, buffer_coord)
    buffer_coord.seek(0)
    binary_data = buffer_coord.read()
    base64_coord_embedding = base64.b64encode(binary_data).decode('utf-8')

    prompt = {
        "model":
        model_name,
        "additional_data": {
            "prompt_token_ids": [1]
        },
        "encoding_format":
        "base64",
        "messages": [{
            "role":
            "user",
            "content": [{
                "type": "image_embeds",
                "image_embeds": {
                    "pixel_values": base64_tensor_embedding,
                    "location_coords": base64_coord_embedding,
                },
            }],
        }]
    }

    # test single pooling
    response = requests.post(server.url_for("pooling"), json=prompt)
    response.raise_for_status()

    output = response.json()["data"][0]['data']

    np_response = np.frombuffer(base64.b64decode(output), dtype=np.float32)

    assert len(np_response) == 524288
