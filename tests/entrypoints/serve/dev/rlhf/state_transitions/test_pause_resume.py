# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the vLLM RL pause/resume lifecycle."""

import os
import threading
from typing import Any
from unittest.mock import patch

import pytest
import requests

from tests.entrypoints.serve.dev.rlhf.conftest import (
    cached_tokens,
    completion_with_cache_details,
    gen,
    golden_output,
    is_paused,
    ok,
    pause,
    resume,
    server,
    start_stream,
)


@pytest.fixture(scope="module", params=[False, True], ids=["MRV1", "MRV2"])
def use_v2(request):
    return request.param


@pytest.fixture(scope="module")
def server_url(use_v2):
    env_vars = {
        "VLLM_USE_V2_MODEL_RUNNER": "1" if use_v2 else "0",
    }

    with (
        patch.dict(os.environ, env_vars),
        server(
            extra_args=[
                "--enable-prefix-caching",
                "--enable-prompt-tokens-details",
            ]
        ) as url,
    ):
        yield url


@pytest.fixture(autouse=True)
def restore_unpaused_state(server_url):
    assert resume(server_url) == 200
    yield
    assert resume(server_url) == 200


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
        inflight, inflight_thread = start_stream(server_url, max_tokens)
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

    def test_clear_cache_preserves_output_and_controls_prefix_cache(self, server_url):
        prompt = (
            "Paris is the capital of France. Berlin is the capital of Germany. "
        ) * 20

        assert pause(server_url, clear_cache=True) == 200
        assert resume(server_url) == 200
        baseline = completion_with_cache_details(server_url, prompt)
        warmed = completion_with_cache_details(server_url, prompt)

        assert cached_tokens(baseline) == 0
        assert cached_tokens(warmed) > 0

        assert pause(server_url, clear_cache=False) == 200
        assert resume(server_url) == 200
        preserved = completion_with_cache_details(server_url, prompt)
        assert golden_output(preserved) == golden_output(baseline)
        assert cached_tokens(preserved) > 0

        assert pause(server_url, clear_cache=True) == 200
        assert resume(server_url) == 200
        cleared = completion_with_cache_details(server_url, prompt)
        assert golden_output(cleared) == golden_output(baseline)
        assert cached_tokens(cleared) == 0
