# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.openai.protocol import ClassificationResponse

VLM_MODEL_NAME = "google/gemma-3-4b-it"
DTYPE = "float32"
MAXIMUM_VIDEOS = 1
TEST_VIDEO_URL = (
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
)


@pytest.fixture(scope="module")
def server_vlm_classify():
    args = [
        "--runner",
        "pooling",
        "--task",
        "classify",
        "--convert",
        "classify",
        "--max-model-len",
        "512",
        "--dtype",
        DTYPE,
        "--enforce-eager",
        "--pooler-config",
        json.dumps({"pooling_type": "LAST"}),
        "--limit-mm-per-prompt",
        json.dumps({"video": MAXIMUM_VIDEOS}),
    ]

    with RemoteOpenAIServer(VLM_MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.parametrize("model_name", [VLM_MODEL_NAME])
def test_classify_accepts_chat_text_only(
    server_vlm_classify: RemoteOpenAIServer, model_name: str
) -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please classify this text request."},
            ],
        }
    ]

    response = requests.post(
        server_vlm_classify.url_for("classify"),
        json={"model": model_name, "messages": messages},
    )
    response.raise_for_status()

    output = ClassificationResponse.model_validate(response.json())

    assert output.object == "list"
    assert output.model == model_name
    assert len(output.data) == 1
    assert hasattr(output.data[0], "probs")
    assert output.usage is not None


@pytest.mark.parametrize("model_name", [VLM_MODEL_NAME])
def test_classify_accepts_chat_video_url(
    server_vlm_classify: RemoteOpenAIServer, model_name: str
) -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please classify this video."},
                {"type": "video_url", "video_url": {"url": TEST_VIDEO_URL}},
            ],
        }
    ]

    response = requests.post(
        server_vlm_classify.url_for("classify"),
        json={"model": model_name, "messages": messages},
    )
    response.raise_for_status()

    output = ClassificationResponse.model_validate(response.json())

    assert output.object == "list"
    assert output.model == model_name
    assert len(output.data) == 1
    assert hasattr(output.data[0], "probs")
    assert output.usage is not None
