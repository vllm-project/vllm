# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64

import requests

from vllm.entrypoints.utils import sanitize_message


def test_sanitize_message():
    assert (
        sanitize_message("<_io.BytesIO object at 0x7a95e299e750>")
        == "<_io.BytesIO object>"
    )


def encode_base64_content_from_url(content_url: str) -> dict[str, str]:
    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return {"url": f"data:image/jpeg;base64,{result}"}
