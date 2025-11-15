# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.openai.protocol import ClassificationResponse

VLM_MODEL_NAME = "muziyongshixin/Qwen2.5-VL-7B-for-VideoCls"
MAXIMUM_VIDEOS = 1
TEST_VIDEO_URL = "https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4"

HF_OVERRIDES = {
    "text_config": {
        "architectures": ["Qwen2_5_VLForSequenceClassification"],
    },
}


@pytest.fixture(scope="module")
def server_vlm_classify():
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
        VLM_MODEL_NAME, args, override_hf_configs=HF_OVERRIDES
    ) as remote_server:
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
    assert len(output.data[0].probs) == 2
    assert output.usage.prompt_tokens == 22


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
    assert len(output.data[0].probs) == 2
    assert output.usage.prompt_tokens == 4807
