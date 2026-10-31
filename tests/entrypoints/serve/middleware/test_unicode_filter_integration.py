# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end tests for ``UnicodeFilterMiddleware`` against a real
``vllm serve`` process.

The unit tests cover the byte filtering in isolation; these tests check the
middleware inside the real API server stack, where the generation handlers
are wrapped in ``with_cancellation`` and stream through ``StreamingResponse``.
Dummy weights keep startup fast: the assertions only depend on tokenization,
not on model output.
"""

import json
from collections.abc import Callable

import httpx
import pytest

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"
# Characters from the Unicode "Tags" block (U+E0020 - U+E007F).
TAG_CHARS = "".join(chr(c) for c in (0xE0020, 0xE0024, 0xE0041, 0xE007F))
CLEAN_PROMPT = "Tell me a story about a dragon"
TAGGED_PROMPT = f"Tell me a {TAG_CHARS}story about a dragon"

# Encoders for the two ways a JSON client can send tag characters.
ENCODERS: dict[str, Callable[[dict], bytes]] = {
    "raw-utf8": lambda payload: json.dumps(payload, ensure_ascii=False).encode(),
    "json-escaped": lambda payload: json.dumps(payload).encode(),
}


@pytest.fixture(scope="module")
def server():
    args = [
        "--load-format",
        "dummy",
        "--max-model-len",
        "512",
        "--enforce-eager",
        "--max-num-seqs",
        "2",
        "--gpu-memory-utilization",
        "0.3",
    ]
    env_dict = {"VLLM_ENABLE_UNICODE_FILTERING_MIDDLEWARE": "true"}
    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env_dict) as remote_server:
        yield remote_server


def _post(server: RemoteOpenAIServer, path: str, body: bytes) -> httpx.Response:
    return httpx.post(
        server.url_for(path),
        content=body,
        headers={"content-type": "application/json"},
        timeout=60.0,
    )


def _token_count(server: RemoteOpenAIServer, prompt: str) -> int:
    """Count prompt tokens through ``/tokenize``, which is not filtered."""
    response = _post(
        server, "tokenize", json.dumps({"model": MODEL_NAME, "prompt": prompt})
    )
    assert response.status_code == 200, response.text
    return response.json()["count"]


def test_tag_characters_are_tokenized_when_not_filtered(server: RemoteOpenAIServer):
    """Control: without filtering, the tag characters cost extra tokens, so a
    matching token count on filtered routes proves they were stripped."""
    assert _token_count(server, TAGGED_PROMPT) > _token_count(server, CLEAN_PROMPT)


@pytest.mark.parametrize("encoding", ENCODERS)
def test_chat_completion_strips_tag_characters(
    server: RemoteOpenAIServer, encoding: str
):
    encode = ENCODERS[encoding]

    def chat(content: str) -> dict:
        response = _post(
            server,
            "v1/chat/completions",
            encode(
                {
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 1,
                }
            ),
        )
        assert response.status_code == 200, response.text
        return response.json()

    clean, tagged = chat(CLEAN_PROMPT), chat(TAGGED_PROMPT)
    assert tagged["choices"] and tagged["choices"][0]["message"] is not None
    assert tagged["usage"]["prompt_tokens"] == clean["usage"]["prompt_tokens"]


def test_completion_echo_returns_stripped_prompt(server: RemoteOpenAIServer):
    response = _post(
        server,
        "v1/completions",
        ENCODERS["raw-utf8"](
            {
                "model": MODEL_NAME,
                "prompt": TAGGED_PROMPT,
                "max_tokens": 1,
                "echo": True,
            }
        ),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    echoed = body["choices"][0]["text"]
    assert not any(ch in echoed for ch in TAG_CHARS), repr(echoed)
    assert body["usage"]["prompt_tokens"] == _token_count(server, CLEAN_PROMPT)


def test_streaming_chat_completion_runs_to_completion(server: RemoteOpenAIServer):
    """The replayed request body must not look like a client disconnect, or
    the stream is cancelled before the first chunk."""
    body = ENCODERS["raw-utf8"](
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": TAGGED_PROMPT}],
            "max_tokens": 3,
            "stream": True,
        }
    )
    events = []
    with httpx.stream(
        "POST",
        server.url_for("v1/chat/completions"),
        content=body,
        headers={"content-type": "application/json"},
        timeout=60.0,
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(line.removeprefix("data: "))

    assert events and events[-1] == "[DONE]"
    assert all("choices" in json.loads(event) for event in events[:-1])


@pytest.mark.parametrize(
    ("path", "build_payload", "usage_key"),
    [
        (
            "v1/responses",
            lambda text: {"model": MODEL_NAME, "input": text, "max_output_tokens": 1},
            "input_tokens",
        ),
        (
            "v1/messages",
            lambda text: {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 1,
            },
            "input_tokens",
        ),
    ],
    ids=["responses", "anthropic-messages"],
)
def test_other_generation_apis_strip_tag_characters(
    server: RemoteOpenAIServer,
    path: str,
    build_payload: Callable[[str], dict],
    usage_key: str,
):
    def usage(text: str) -> int:
        response = _post(server, path, ENCODERS["raw-utf8"](build_payload(text)))
        assert response.status_code == 200, response.text
        return response.json()["usage"][usage_key]

    assert usage(TAGGED_PROMPT) == usage(CLEAN_PROMPT)
