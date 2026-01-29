# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.classify.protocol import ClassificationResponse
from vllm.utils.mm_utils import encode_base64_content_from_url

MODEL_NAME = "muziyongshixin/Qwen2.5-VL-7B-for-VideoCls"
MAXIMUM_VIDEOS = 1

HF_OVERRIDES = {
    "text_config": {
        "architectures": ["Qwen2_5_VLForSequenceClassification"],
    },
}
input_text = "This product was excellent and exceeded my expectations"
image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/cat_snow.jpg"
image_base64 = encode_base64_content_from_url(image_url)
video_url = "https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4"


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "pooling",
        "--max-model-len",
        "5000",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"video": MAXIMUM_VIDEOS}),
    ]

    with RemoteOpenAIServer(
        MODEL_NAME, args, override_hf_configs=HF_OVERRIDES
    ) as remote_server:
        yield remote_server


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_chat_text_request(server: RemoteOpenAIServer, model_name: str):
    messages = [
        {
            "role": "assistant",
            "content": "Please classify this text request.",
        },
        {
            "role": "user",
            "content": input_text,
        },
    ]

    response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "messages": messages},
    )
    response.raise_for_status()

    output = ClassificationResponse.model_validate(response.json())

    assert output.object == "list"
    assert output.model == model_name
    assert len(output.data) == 1
    assert len(output.data[0].probs) == 2
    assert output.usage.prompt_tokens == 35


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_chat_image_url_request(server: RemoteOpenAIServer, model_name: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please classify this image."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "messages": messages},
    )
    response.raise_for_status()

    output = ClassificationResponse.model_validate(response.json())

    assert output.object == "list"
    assert output.model == model_name
    assert len(output.data) == 1
    assert len(output.data[0].probs) == 2
    assert output.usage.prompt_tokens == 47


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_chat_image_base64_request(server: RemoteOpenAIServer, model_name: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please classify this image."},
                {"type": "image_url", "image_url": image_base64},
            ],
        }
    ]

    response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "messages": messages},
    )
    response.raise_for_status()

    output = ClassificationResponse.model_validate(response.json())

    assert output.object == "list"
    assert output.model == model_name
    assert len(output.data) == 1
    assert len(output.data[0].probs) == 2
    assert output.usage.prompt_tokens == 47


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_chat_video_url_request(server: RemoteOpenAIServer, model_name: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please classify this video."},
                {"type": "video_url", "video_url": {"url": video_url}},
            ],
        }
    ]

    response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "messages": messages},
    )
    response.raise_for_status()

    output = ClassificationResponse.model_validate(response.json())

    assert output.object == "list"
    assert output.model == model_name
    assert len(output.data) == 1
    assert len(output.data[0].probs) == 2
    assert output.usage.prompt_tokens == 4807
