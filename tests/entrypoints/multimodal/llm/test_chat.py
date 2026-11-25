# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref
from typing import Any

import pytest
import requests

from tests.conftest import VllmRunner
from tests.entrypoints.multimodal.conftest import TEST_IMAGE_ASSETS
from vllm.connections import MediaFetchDeadlineExceededError
from vllm.exceptions import VLLMUnprocessableEntityError


@pytest.fixture(scope="function")
def vision_llm(vllm_runner):
    with vllm_runner(
        "microsoft/Phi-3.5-vision-instruct",
        max_model_len=4096,
        max_num_seqs=5,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 2},
        seed=0,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while VllmRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


@pytest.mark.parametrize(
    "image_urls", [[TEST_IMAGE_ASSETS[0], TEST_IMAGE_ASSETS[1]]], indirect=True
)
def test_chat_multi_image(vision_llm, image_urls: list[str]):
    messages = [
        {
            "role": "user",
            "content": [
                *(
                    {"type": "image_url", "image_url": {"url": image_url}}
                    for image_url in image_urls
                ),
                {"type": "text", "text": "What's in this image?"},
            ],
        }
    ]
    outputs = vision_llm.chat(messages)
    assert len(outputs) >= 0


def test_chat_video_deadline_reports_video_url(
    vllm_runner: type[VllmRunner], monkeypatch: pytest.MonkeyPatch
) -> None:
    url = "https://media.invalid/clip.mp4"
    messages = [
        {
            "role": "user",
            "content": [{"type": "video_url", "video_url": {"url": url}}],
        }
    ]
    send = requests.Session.send

    def expire_video_fetch(
        session: requests.Session, request: requests.PreparedRequest, **kwargs: Any
    ) -> requests.Response:
        if request.url == url:
            raise MediaFetchDeadlineExceededError("Media fetch exceeded its time limit")
        return send(session, request, **kwargs)

    with (
        vllm_runner(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            dtype="half",
            enforce_eager=True,
            max_model_len=2048,
            max_num_seqs=1,
            limit_mm_per_prompt={"video": 1},
        ) as runner,
        monkeypatch.context() as patch,
    ):
        patch.setattr(requests.Session, "send", expire_video_fetch)
        with pytest.raises(
            VLLMUnprocessableEntityError, match="exceeded its time limit"
        ) as exc_info:
            runner.llm.chat(messages)

    assert exc_info.value.parameter == "video_url"
