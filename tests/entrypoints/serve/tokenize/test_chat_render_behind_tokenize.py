# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test: a chat request whose template render waits in the renderer
pool behind `/tokenize` work for longer than
`VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT` must succeed.

The renderer pool has one worker by default and serves chat-template renders
and prompt tokenization. The render timeout covers the render, not its wait in
the pool queue.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
RENDER_TIMEOUT_S = 1.0
# One tokenization of this prompt keeps the pool worker busy for a few tenths
# of a second, so the requests below queue several timeouts of work.
NUM_TOKENIZE_REQUESTS = 16
TOKENIZE_PROMPT = " ".join(f"tok{(i * 7919) % 100003}" for i in range(50_000))
# The tokenization requests reach the pool queue before the chat request.
CHAT_DELAY_S = 0.3


@pytest.fixture(scope="module")
def server():
    env_dict = {"VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT": str(RENDER_TIMEOUT_S)}
    with RemoteOpenAIServer(
        MODEL_NAME, ["--enforce-eager"], env_dict=env_dict
    ) as remote_server:
        yield remote_server


def test_chat_render_queued_behind_tokenize_succeeds(server: RemoteOpenAIServer):
    def tokenize() -> int:
        response = requests.post(
            server.url_for("tokenize"),
            json={"model": MODEL_NAME, "prompt": TOKENIZE_PROMPT},
        )
        return response.status_code

    def chat() -> tuple[requests.Response, float]:
        start = time.monotonic()
        response = requests.post(
            server.url_for("v1/chat/completions"),
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Reply with one word."}],
                "max_tokens": 1,
                "temperature": 0.0,
            },
        )
        return response, time.monotonic() - start

    with ThreadPoolExecutor(max_workers=NUM_TOKENIZE_REQUESTS + 1) as client_pool:
        tokenize_statuses = [
            client_pool.submit(tokenize) for _ in range(NUM_TOKENIZE_REQUESTS)
        ]
        time.sleep(CHAT_DELAY_S)
        chat_response, chat_latency_s = client_pool.submit(chat).result()
        assert [status.result() for status in tokenize_statuses] == (
            [200] * NUM_TOKENIZE_REQUESTS
        )

    # The request waited in the pool queue for longer than the render timeout;
    # without that, this test does not exercise the queue.
    assert chat_latency_s > RENDER_TIMEOUT_S
    assert chat_response.status_code == 200, chat_response.text
