# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import requests
import base64

DEFAULT_HEADERS = {"accept": "application/json", "Content-Type": "application/json"}

def encode_base64_content_from_url(content_url: str, headers:dict[str, str] | None=None) -> dict[str, str]:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url, headers=headers or DEFAULT_HEADERS) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return {"url": f"data:image/jpeg;base64,{result}"}
