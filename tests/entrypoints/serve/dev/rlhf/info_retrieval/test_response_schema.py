# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP response contracts for the RL sleep and wake-up endpoints.

This module covers the endpoints whose responses changed from an empty 200
response to structured JSON. Dummy weights suffice because this module does
not generate after level-2 sleep. Request gating and restored output are
covered by state_transitions/test_sleep_wake.py, not inferred from JSON here.
"""

import math

import pytest
import requests

from tests.entrypoints.serve.dev.rlhf.conftest import (
    ensure_awake,
    is_sleeping,
    reusable_server,
    sleep_response,
    wake_response,
)

# Parent cleanup runs once after the shared server has fully shut down.
pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(scope="module")
def server_url():
    with reusable_server(
        dummy_weights=True, env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1"}
    ) as url:
        try:
            yield url
        finally:
            ensure_awake(url)


@pytest.fixture(autouse=True)
def restore_awake_state(server_url):
    ensure_awake(server_url)
    try:
        yield
    finally:
        ensure_awake(server_url)


def assert_elapsed_ms(value):
    assert type(value) in (int, float)
    assert math.isfinite(value) and value >= 0


class TestSleepResponseSchema:
    @pytest.mark.parametrize("level", [0, 1, 2])
    def test_sleep_response_schema(self, server_url, level):
        """Verify /sleep reports the applied level and resulting state."""
        response = sleep_response(server_url, level=level)

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"status", "level", "elapsed_ms"}
        assert body["status"] == "sleeping"
        assert body["level"] == level
        assert_elapsed_ms(body["elapsed_ms"])
        assert is_sleeping(server_url)

    def test_is_sleeping_response_schema(self, server_url):
        """Verify /is_sleeping reports both awake and sleeping states."""
        response = requests.get(f"{server_url}/is_sleeping", timeout=5)
        assert response.status_code == 200
        assert response.json() == {"is_sleeping": False}

        sleep_response(server_url, level=1).raise_for_status()
        response = requests.get(f"{server_url}/is_sleeping", timeout=5)
        assert response.status_code == 200
        assert response.json() == {"is_sleeping": True}

    @pytest.mark.parametrize(
        ("params", "invalid_param"),
        [
            pytest.param({"level": "invalid"}, "query.level", id="non-integer-level"),
            pytest.param({"level": -1}, "query.level", id="negative-level"),
            pytest.param({"level": 3}, "query.level", id="unsupported-level"),
            pytest.param({"mode": "invalid"}, "query.mode", id="invalid-mode"),
        ],
    )
    @pytest.mark.parametrize("sleeping", [False, True], ids=["awake", "sleeping"])
    def test_invalid_parameters_preserve_state(
        self, server_url, params, invalid_param, sleeping
    ):
        """Verify invalid sleep parameters identify the field without sleeping."""
        if sleeping:
            sleep_response(server_url, level=1).raise_for_status()
        assert is_sleeping(server_url) is sleeping

        response = requests.post(f"{server_url}/sleep", params=params, timeout=15)

        assert response.status_code == 400
        assert response.json()["error"]["param"] == invalid_param
        assert is_sleeping(server_url) is sleeping


class TestWakeResponseSchema:
    def test_full_wake_response_schema(self, server_url):
        """Verify a full wake reports all allocations and scheduling as awake."""
        sleep_response(server_url, level=1).raise_for_status()

        response = wake_response(server_url)

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"status", "tags_woken", "elapsed_ms"}
        assert body["status"] == "awake"
        assert body["tags_woken"] is None
        assert_elapsed_ms(body["elapsed_ms"])
        assert not is_sleeping(server_url)

    @pytest.mark.parametrize("tags", [("weights", "kv_cache"), ("kv_cache", "weights")])
    def test_partial_wake_response_tracks_engine_state(self, server_url, tags):
        """Verify staged wake responses track the engine's remaining sleep tags."""
        sleep_response(server_url, level=1).raise_for_status()

        weights_response = wake_response(server_url, tags=[tags[0]])
        assert weights_response.status_code == 200
        weights_body = weights_response.json()
        assert set(weights_body) == {"status", "tags_woken", "elapsed_ms"}
        assert weights_body["status"] == "sleeping"
        assert weights_body["tags_woken"] == [tags[0]]
        assert_elapsed_ms(weights_body["elapsed_ms"])
        assert is_sleeping(server_url)

        cache_response = wake_response(server_url, tags=[tags[1]])
        assert cache_response.status_code == 200
        cache_body = cache_response.json()
        assert set(cache_body) == {"status", "tags_woken", "elapsed_ms"}
        assert cache_body["status"] == "awake"
        assert cache_body["tags_woken"] == [tags[1]]
        assert_elapsed_ms(cache_body["elapsed_ms"])
        assert not is_sleeping(server_url)
