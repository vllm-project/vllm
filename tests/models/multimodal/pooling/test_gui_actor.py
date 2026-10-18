# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GUI-Actor served through the structured decisions API.

GUI-Actor answers a `box` or a `ground` question from the pointer head's
attention over the image's patches, so a served answer must be the reference
decode of the same per-patch distribution. The decode itself is pinned by
`test_spatial.py`; these tests cover the served path, which needs a GPU and the
checkpoint.

The 3B checkpoint is used because this shard runs on an 18GB MIG slice: its
7.5GB of weights plus activations and the KV cache fit, where the 7B
checkpoint's 15.6GB of weights alone would not.
"""

import pytest
import requests
import torch

from tests.utils import RemoteOpenAIServer
from vllm.assets.image import ImageAsset
from vllm.multimodal.utils import encode_image_url

MODEL = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
STATE = "A screenshot of a checkout page."
QUESTIONS = {
    "pay": {"type": "box", "instructions": "the Pay button"},
    "cart": {"type": "ground", "instructions": "the cart icon"},
}

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GUI-Actor needs the CUDA kernels"
    ),
]


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "pooling",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        # Leaves room for a 18GB MIG slice: 7.5GB of weights, ~1GB of
        # activations, and the rest for the KV cache.
        "--gpu-memory-utilization",
        "0.4",
        "--enforce-eager",
    ]
    with RemoteOpenAIServer(MODEL, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def screenshot() -> str:
    image = ImageAsset("stop_sign").pil_image
    return encode_image_url(image)


def post(server, body):
    return requests.post(server.url_for("v1/systemone"), json=body)


def test_box_and_ground_answers(server, screenshot):
    response = post(
        server,
        {
            "model": MODEL,
            "state": STATE,
            "questions": QUESTIONS,
            "images": [screenshot],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert list(body["answers"]) == ["pay", "cart"]

    box = body["answers"]["pay"]
    assert box["type"] == "box"
    assert 0.0 <= box["box"]["left"] < box["box"]["right"] <= 1.0
    assert 0.0 <= box["box"]["top"] < box["box"]["bottom"] <= 1.0
    assert 0.0 <= box["confidence"] <= 1.0

    ground = body["answers"]["cart"]
    assert ground["type"] == "ground"
    assert all(0 <= coordinate <= 999 for coordinate in ground["point"])
    assert 0.0 <= ground["confidence"] <= 1.0

    # One prefill per question reads the prompt; nothing is generated.
    assert body["usage"]["output_tokens"] == 0
    assert body["usage"]["input_tokens"] > 0


def test_box_needs_an_image(server):
    response = post(server, {"model": MODEL, "state": STATE, "questions": QUESTIONS})
    assert response.status_code == 400, response.text
    assert "needs image input" in response.json()["error"]["message"]


def test_label_questions_are_unsupported(server, screenshot):
    # A patch distribution cannot answer a choice, so the request is refused
    # rather than misread; the same request answers against another model.
    response = post(
        server,
        {
            "model": MODEL,
            "state": STATE,
            "questions": {"team": {"type": "choice", "criteria": {"a": None}}},
            "images": [screenshot],
        },
    )
    assert response.status_code == 501, response.text
    assert "box, ground" in response.json()["error"]["message"]
