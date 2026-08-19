# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the RL weight-version information endpoints."""

import os
from unittest.mock import patch

import pytest
import requests

from tests.entrypoints.serve.dev.rlhf.conftest import (
    gen,
    ok,
    server,
    update_weight_version_response,
    weight_info_response,
)


@pytest.fixture(scope="module")
def server_state():
    with (
        patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"}),
        server(dummy_weights=True) as url,
    ):
        initial_response = weight_info_response(url)
        initial_response.raise_for_status()
        try:
            yield url, initial_response.json()
        finally:
            update_weight_version_response(url, "default").raise_for_status()


@pytest.fixture(scope="module")
def server_url(server_state):
    return server_state[0]


@pytest.fixture(autouse=True)
def restore_default_version(server_url):
    update_weight_version_response(server_url, "default").raise_for_status()
    try:
        yield
    finally:
        update_weight_version_response(server_url, "default").raise_for_status()


class TestWeightInfoEndpoint:
    def test_initial_schema(self, server_state):
        """Verify a fresh server exposes the default weight version."""
        _, initial_weight_info = server_state
        assert initial_weight_info == {"weight_version": "default"}

    def test_does_not_affect_generation(self, server_url):
        """Verify version reads and updates do not change model output."""
        before = gen(server_url)
        assert ok(before)

        update_weight_version_response(
            server_url, "generation-check"
        ).raise_for_status()
        assert weight_info_response(server_url).json() == {
            "weight_version": "generation-check"
        }

        after = gen(server_url)
        assert ok(after)
        assert after["choices"][0]["text"] == before["choices"][0]["text"]


class TestUpdateWeightVersion:
    def test_update_is_visible(self, server_url):
        """Verify a version update is immediately visible through /weight_info."""
        response = update_weight_version_response(server_url, "step-42")

        assert response.status_code == 200
        assert response.json() == {"success": True, "new_version": "step-42"}

        info_response = weight_info_response(server_url)
        assert info_response.status_code == 200
        assert info_response.json() == {"weight_version": "step-42"}

    def test_version_can_be_overwritten(self, server_url):
        """Verify later version updates replace the previously reported value."""
        for version in ("step-1", "checkpoint/final"):
            response = update_weight_version_response(server_url, version)
            assert response.status_code == 200
            assert response.json() == {"success": True, "new_version": version}

        assert weight_info_response(server_url).json() == {
            "weight_version": "checkpoint/final"
        }

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({}, id="missing-version"),
            pytest.param({"weight_version": "wrong-field"}, id="wrong-field"),
            pytest.param({"new_version": ["not", "a", "string"]}, id="wrong-type"),
        ],
    )
    def test_invalid_update_preserves_state(self, server_url, payload):
        """Verify invalid version updates leave the committed version unchanged."""
        update_weight_version_response(server_url, "stable").raise_for_status()

        response = requests.post(
            f"{server_url}/update_weight_version",
            json=payload,
            timeout=5,
        )

        assert response.status_code == 400
        assert response.json()["error"]["param"] == "body.new_version"
        assert weight_info_response(server_url).json() == {"weight_version": "stable"}
