# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the vLLM RL pause/resume lifecycle."""

import json
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest
import requests

from tests.entrypoints.serve.dev.rlhf.conftest import (
    gen,
    is_paused,
    ok,
    pause,
    resume,
    server,
)


@pytest.fixture(scope="module")
def server_url():
    with server(
        extra_args=[
            "--enable-prefix-caching",
            "--enable-prompt-tokens-details",
        ]
    ) as url:
        yield url


@pytest.fixture(autouse=True)
def restore_unpaused_state(server_url):
    assert resume(server_url) == 200
    yield
    assert resume(server_url) == 200


@dataclass
class _StreamResult:
    started: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    error: Exception | None = None


def _stream_completion(url: str, result: _StreamResult, max_tokens: int) -> None:
    try:
        with requests.post(
            f"{url}/v1/completions",
            json={
                "model": "m",
                "prompt": "Count upward slowly: one, two, three,",
                "max_tokens": max_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
            },
            stream=True,
            timeout=(5, 60),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or line == "data: [DONE]":
                    continue
                assert line.startswith("data: ")
                chunk = json.loads(line.removeprefix("data: "))
                result.chunks.append(chunk)
                choice = chunk["choices"][0]
                if choice.get("text"):
                    result.started.set()
                if choice.get("finish_reason") is not None:
                    result.finish_reason = choice["finish_reason"]
    except Exception as error:
        result.error = error
    finally:
        result.done.set()


def _start_stream(url: str, max_tokens: int) -> tuple[_StreamResult, threading.Thread]:
    result = _StreamResult()
    thread = threading.Thread(
        target=_stream_completion,
        args=(url, result, max_tokens),
    )
    thread.start()
    started = result.started.wait(timeout=10)
    if not started or result.done.is_set():
        pause(url, mode="abort")
        resume(url)
        thread.join(timeout=10)
    assert started, "request did not start generating"
    assert not result.done.is_set(), "request completed before it could be paused"
    return result, thread


def _completion_with_cache_details(url: str, prompt: str) -> dict[str, Any]:
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": "m",
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": 0,
            "logprobs": 1,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _golden_output(response: dict[str, Any]) -> dict[str, Any]:
    choice = response["choices"][0]
    usage = response["usage"]
    return {
        "text": choice["text"],
        "finish_reason": choice["finish_reason"],
        "tokens": choice["logprobs"]["tokens"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
    }


def _cached_tokens(response: dict[str, Any]) -> int:
    return response["usage"]["prompt_tokens_details"]["cached_tokens"]


class TestPauseResume:
    def test_state_and_idempotency_across_cycles(self, server_url):
        assert not is_paused(server_url)

        assert resume(server_url) == 200
        assert resume(server_url) == 200
        assert not is_paused(server_url)

        for _ in range(2):
            for mode in ("abort", "wait", "keep"):
                assert pause(server_url, mode=mode) == 200
                assert pause(server_url, mode=mode) == 200
                assert is_paused(server_url)

                assert resume(server_url) == 200
                assert resume(server_url) == 200
                assert not is_paused(server_url)

    def test_invalid_mode_preserves_state(self, server_url):
        for paused in (False, True):
            if paused:
                assert pause(server_url) == 200
            assert is_paused(server_url) is paused

            response = requests.post(
                f"{server_url}/pause",
                params={"mode": "invalid"},
                timeout=10,
            )
            assert response.status_code == 400
            assert response.json()["error"]["param"] == "query.mode"
            assert is_paused(server_url) is paused

            assert resume(server_url) == 200

    @pytest.mark.parametrize(
        ("mode", "max_tokens", "inflight_finish_reason"),
        [
            pytest.param("abort", 256, "abort", id="abort"),
            pytest.param("wait", 256, "length", id="wait"),
            pytest.param("keep", 256, "length", id="keep"),
        ],
    )
    def test_mode_request_lifecycle(
        self,
        server_url,
        mode,
        max_tokens,
        inflight_finish_reason,
    ):
        inflight, inflight_thread = _start_stream(server_url, max_tokens)
        new_result: dict[str, Any] = {}
        new_done = threading.Event()

        def _new_request():
            new_result["response"] = gen(server_url, max_tokens=4, timeout=60)
            new_done.set()

        new_thread = threading.Thread(target=_new_request)
        try:
            assert pause(server_url, mode=mode) == 200
            assert is_paused(server_url)

            if mode in ("abort", "wait"):
                assert inflight.done.is_set()
            else:
                chunks_after_pause = len(inflight.chunks)
                assert not inflight.done.wait(timeout=5)
                assert len(inflight.chunks) == chunks_after_pause, (
                    "in-flight request continued generating in keep mode"
                )

            new_thread.start()
            assert not new_done.wait(timeout=0.3), (
                "new request completed while generation was paused"
            )
        finally:
            assert resume(server_url) == 200
            inflight_thread.join(timeout=30)
            if new_thread.ident is not None:
                new_thread.join(timeout=30)

        assert not inflight_thread.is_alive()
        assert inflight.error is None
        assert inflight.finish_reason == inflight_finish_reason
        assert not new_thread.is_alive()
        assert ok(new_result.get("response"))

    def test_clear_cache_preserves_output_and_controls_prefix_cache(
        self, server_url
    ):
        prompt = (
            "Paris is the capital of France. "
            "Berlin is the capital of Germany. "
        ) * 20

        assert pause(server_url, clear_cache=True) == 200
        assert resume(server_url) == 200
        baseline = _completion_with_cache_details(server_url, prompt)
        warmed = _completion_with_cache_details(server_url, prompt)

        assert _cached_tokens(baseline) == 0
        assert _cached_tokens(warmed) > 0

        assert pause(server_url, clear_cache=False) == 200
        assert resume(server_url) == 200
        preserved = _completion_with_cache_details(server_url, prompt)
        assert _golden_output(preserved) == _golden_output(baseline)
        assert _cached_tokens(preserved) > 0

        assert pause(server_url, clear_cache=True) == 200
        assert resume(server_url) == 200
        cleared = _completion_with_cache_details(server_url, prompt)
        assert _golden_output(cleared) == _golden_output(baseline)
        assert _cached_tokens(cleared) == 0
