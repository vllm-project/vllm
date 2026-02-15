# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import io

from PIL import Image

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat


def _make_data_url(image_size: tuple[int, int], *, strip_padding: bool = False) -> str:
    image = Image.new("RGB", image_size, color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    if strip_padding:
        b64 = b64.rstrip("=")

    return f"data:image/png;base64,{b64}"


def test_extract_florence2_od_image_size_accepts_whitespace_and_missing_padding():
    serving = object.__new__(OpenAIServingChat)

    data_url = _make_data_url((9, 5), strip_padding=True)
    head, payload = data_url.split(",", 1)
    payload_with_whitespace = f"\n  {payload[:32]}\n{payload[32:]}  \n"

    request = ChatCompletionRequest(
        model="dummy",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{head},{payload_with_whitespace}"},
                    }
                ],
            }
        ],
    )

    assert serving._extract_florence2_od_image_size(request) == (9, 5)


def test_extract_florence2_od_image_size_invalid_data_returns_none():
    serving = object.__new__(OpenAIServingChat)

    request = ChatCompletionRequest(
        model="dummy",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,not-base64!!!"},
                    }
                ],
            }
        ],
    )

    assert serving._extract_florence2_od_image_size(request) is None


def test_request_contains_od_task_supports_text_and_input_text_parts():
    serving = object.__new__(OpenAIServingChat)

    req_from_text = ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "please do <od> now"}],
    )
    assert serving._request_contains_od_task(req_from_text)

    req_from_input_text = ChatCompletionRequest(
        model="dummy",
        messages=[
            {
                "role": "user",
                "content": [{"type": "input_text", "input_text": "Run <Od> task"}],
            }
        ],
    )
    assert serving._request_contains_od_task(req_from_input_text)

    req_no_od = ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "normal caption request"}],
    )
    assert not serving._request_contains_od_task(req_no_od)
