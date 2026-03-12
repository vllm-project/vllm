# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test: ``/tokenize`` must expand image placeholders for VLM models.

Fixed by PR #34560 ("Move InputPreprocessor into Renderer (2/2)").
Before that change, ``/tokenize`` returned ~26 tokens for a message with an
image instead of the expected 1451.  Confirmed broken on 0.15.1 and 0.16.0.
"""

import json

import pytest
import requests

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "5",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"image": 1}),
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def test_tokenize_chat_expands_image_placeholders(
    server: RemoteOpenAIServer,
    local_asset_server,
):
    image_url = local_asset_server.url_for("stop_sign.jpg")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    response = requests.post(
        server.url_for("tokenize"),
        json={"model": MODEL_NAME, "messages": messages},
    )
    response.raise_for_status()

    # stop_sign.jpg (1300x876) produces 1451 tokens after expansion.
    # Without expansion the count would be ~26 (text + one placeholder).
    assert response.json()["count"] == 1451
