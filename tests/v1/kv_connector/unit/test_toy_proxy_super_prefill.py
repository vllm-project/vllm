# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`--super-prefill` in the NIXL toy proxy, against fake prefill and decode engines.

The engines are external services to the proxy, so two small fakes stand in for
them: a deterministic model whose output at position i is token 1000 + i with
text "t{i} ", and a KV pull on the decoder that takes PULL_SECONDS. Like vLLM's
detokenizer inside a multi-byte character, the fakes hold back the text of
every token at a position i % 4 == 1 until the next token, even when the
stream ends there.
"""

import asyncio
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.utils.network_utils import get_open_port

pytestmark = pytest.mark.cpu_test

PROXY = Path(__file__).parents[1] / "nixl_integration" / "toy_proxy_server.py"
PROMPT_IDS = list(range(7))
KV_PARAMS = {"do_remote_prefill": True, "remote_block_ids": [0]}
TOKEN_SECONDS = 0.005
PULL_SECONDS = 1.0


def expected_text(num_tokens: int) -> str:
    return "".join(f"t{i} " for i in range(num_tokens))


def fake_engine() -> FastAPI:
    app = FastAPI()
    app.state.requests = []

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        app.state.requests.append(body)
        prompt = body["prompt"]
        start = len(prompt) - len(PROMPT_IDS) if isinstance(prompt, list) else 0
        positions = range(start, start + body["max_tokens"])
        kv_params = body.get("kv_transfer_params") or {}
        meta = {"id": "cmpl-fake", "created": 1, "model": "fake"}

        if not body.get("stream"):
            if kv_params.get("do_remote_prefill"):
                await asyncio.sleep(PULL_SECONDS)
            choice = {
                "index": 0,
                "text": "".join(f"t{i} " for i in positions),
                "finish_reason": "length",
                "token_ids": [1000 + i for i in positions],
                "prompt_token_ids": PROMPT_IDS,
            }
            out = {**meta, "choices": [choice]}
            if kv_params.get("do_remote_decode"):
                out["kv_transfer_params"] = KV_PARAMS
            return JSONResponse(out)

        async def stream():
            held = ""
            for i in positions:
                await asyncio.sleep(TOKEN_SECONDS)
                held += f"t{i} "
                last = i == positions[-1]
                text, held = ("", held) if i % 4 == 1 else (held, "")
                choice = {
                    "index": 0,
                    "text": text,
                    "finish_reason": "length" if last else None,
                    "token_ids": [1000 + i],
                }
                yield f"data: {json.dumps({**meta, 'choices': [choice]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def wait_until(condition, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met before timeout")
        time.sleep(0.05)


@pytest.fixture(scope="module")
def engines():
    apps: dict[str, tuple[FastAPI, int]] = {}
    servers: list[uvicorn.Server] = []
    for name in ("prefill", "decode"):
        app, port = fake_engine(), get_open_port()
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        threading.Thread(target=server.run, daemon=True).start()
        wait_until(lambda server=server: server.started)
        apps[name] = (app, port)
        servers.append(server)
    yield apps
    for server in servers:
        server.should_exit = True


def start_proxy(engines, *extra_args: str):
    port = get_open_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            str(PROXY),
            "--port",
            str(port),
            "--prefiller-host",
            "127.0.0.1",
            "--prefiller-port",
            str(engines["prefill"][1]),
            "--decoder-host",
            "127.0.0.1",
            "--decoder-port",
            str(engines["decode"][1]),
            *extra_args,
        ]
    )

    def healthy() -> bool:
        assert proc.poll() is None, "proxy exited"
        try:
            return httpx.get(f"http://127.0.0.1:{port}/healthcheck").is_success
        except httpx.TransportError:
            return False

    wait_until(healthy)
    return proc, f"http://127.0.0.1:{port}/v1/completions"


@pytest.fixture(scope="module")
def proxy(engines):
    proc, url = start_proxy(engines, "--super-prefill")
    yield url
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="module")
def capped_proxy(engines):
    proc, url = start_proxy(
        engines, "--super-prefill", "--super-prefill-max-tokens", "6"
    )
    yield url
    proc.terminate()
    proc.wait()


@pytest.fixture(autouse=True)
def clear_requests(engines):
    for app, _ in engines.values():
        app.state.requests.clear()


def stream_completion(url: str, max_tokens: int) -> tuple[str, list[dict], bool]:
    body = {
        "model": "fake",
        "prompt": "hello",
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    text, chunks, done = "", [], False
    with httpx.stream("POST", url, json=body, timeout=60) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            data = line.removeprefix("data:").strip()
            if not line.startswith("data:"):
                continue
            if data == "[DONE]":
                done = True
                continue
            chunk = json.loads(data)
            chunks.append(chunk)
            text += "".join(c["text"] for c in chunk["choices"])
    return text, chunks, done


def decode_continuations(engines) -> list[dict]:
    return [r for r in engines["decode"][0].state.requests if r.get("stream")]


def test_decoder_takes_over_mid_answer(engines, proxy):
    """The client gets the exact answer: P's tokens first, then D's."""
    text, chunks, done = stream_completion(proxy, max_tokens=300)

    assert text == expected_text(300)
    assert done
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"]["completion_tokens"] == 300
    [continuation] = decode_continuations(engines)
    on_prefill = len(continuation["prompt"]) - len(PROMPT_IDS)
    assert 1 <= on_prefill < 300
    assert continuation["max_tokens"] == 300 - on_prefill


def test_non_streaming_answer_is_complete(engines, proxy):
    body = {"model": "fake", "prompt": "hello", "max_tokens": 300}
    out = httpx.post(proxy, json=body, timeout=60).json()

    assert out["choices"][0]["text"] == expected_text(300)
    assert out["choices"][0]["finish_reason"] == "length"
    assert out["usage"]["completion_tokens"] == 300
    assert len(decode_continuations(engines)) == 1


def test_answer_shorter_than_pull_finishes_on_prefill(engines, proxy):
    text, chunks, _ = stream_completion(proxy, max_tokens=5)

    assert text == expected_text(5)
    assert chunks[-1]["usage"]["completion_tokens"] == 5
    assert decode_continuations(engines) == []


def test_cap_hands_over_and_regenerates_held_back_token(engines, capped_proxy):
    """With a 6-token cap, the text of P's last token (position 5) is held
    back. D must continue from position 5, not 6, or "t5 " is lost."""
    text, chunks, _ = stream_completion(capped_proxy, max_tokens=40)

    assert text == expected_text(40)
    assert chunks[-1]["usage"]["completion_tokens"] == 40
    [continuation] = decode_continuations(engines)
    assert len(continuation["prompt"]) == len(PROMPT_IDS) + 5


def test_unsupported_request_takes_serial_path(engines, proxy):
    """Stop strings cannot carry across the switch, so the proxy hands the
    request to D after one token, as without --super-prefill."""
    body = {"model": "fake", "prompt": "hello", "max_tokens": 20, "stop": ["zz"]}
    out = httpx.post(proxy, json=body, timeout=60).json()

    assert out["choices"][0]["text"] == expected_text(20)
    [prefill_request] = engines["prefill"][0].state.requests
    assert prefill_request["max_tokens"] == 1
    assert "return_token_ids" not in prefill_request
    [decode_request] = engines["decode"][0].state.requests
    assert decode_request["prompt"] == "hello"
    assert decode_request["kv_transfer_params"] == KV_PARAMS
