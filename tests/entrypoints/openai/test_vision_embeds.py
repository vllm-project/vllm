# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64

import numpy as np
import pytest
import requests
import torch

from vllm.utils.serial_utils import tensor2base64

from ...utils import RemoteOpenAIServer


@pytest.mark.parametrize(
    "model_name", ["ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"]
)
def test_single_content(model_name: str):
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
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_embeds",
                                "image_embeds": {
                                    "pixel_values": tensor2base64(
                                        torch.ones((6, 512, 512), dtype=torch.float16)
                                    ),
                                    "location_coords": tensor2base64(
                                        torch.ones((1, 2), dtype=torch.float16)
                                    ),
                                },
                            },
                        ],
                    }
                ],
                "encoding_format": "base64",
            },
        )
        response.raise_for_status()

        output = response.json()["data"][0]["data"]

        np_response = np.frombuffer(base64.b64decode(output), dtype=np.float32)
        assert len(np_response) == 524288


@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-VL-2B-Instruct"])
def test_multi_content(model_name: str):
    args = [
        "--enforce-eager",
        "--max-num-seqs",
        "32",
        "--max-model-len",
        "8192",
        "--enable-mm-embeds",
    ]

    with RemoteOpenAIServer(model_name, args) as server:
        client = server.get_client()

        # Image only
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_embeds",
                            "image_embeds": {
                                "image_embeds": tensor2base64(torch.zeros(220, 8192)),
                                "image_grid_thw": tensor2base64(
                                    torch.tensor([1, 22, 40])
                                ),
                            },
                        },
                        {
                            "type": "image_embeds",
                            "image_embeds": {
                                "image_embeds": tensor2base64(torch.zeros(220, 8192)),
                                "image_grid_thw": tensor2base64(
                                    torch.tensor([1, 22, 40])
                                ),
                            },
                        },
                    ],
                }
            ],
            max_tokens=5,
        )

        assert chat_completion.id is not None
        assert len(chat_completion.choices) == 1

        # Interleaved text and image
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_embeds",
                            "image_embeds": {
                                "image_embeds": tensor2base64(torch.zeros(220, 8192)),
                                "image_grid_thw": tensor2base64(
                                    torch.tensor([1, 22, 40])
                                ),
                            },
                        },
                        {"type": "text", "text": "OCR:"},
                        {
                            "type": "image_embeds",
                            "image_embeds": {
                                "image_embeds": tensor2base64(torch.zeros(220, 8192)),
                                "image_grid_thw": tensor2base64(
                                    torch.tensor([1, 22, 40])
                                ),
                            },
                        },
                    ],
                }
            ],
            max_tokens=5,
        )

        assert chat_completion.id is not None
        assert len(chat_completion.choices) == 1
