# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the POST /weight_checker development endpoint."""

import os

import pytest
import requests

from tests.entrypoints.serve.dev.rlhf.conftest import (
    collective_rpc,
    gen,
    health,
    ok,
    server,
    weight_checker,
)


def _mode(tp: int, dp: int = 1, ep: bool = False, real_weights: bool = False):
    key = f"tp{tp}"
    if dp > 1:
        key += f"dp{dp}"
    if ep:
        key += "ep"
    value = {
        "tp": tp,
        "dp": dp,
        "ep": ep,
        "name": key,
        "real_weights": real_weights,
    }
    return pytest.param(value, id=key)


_MODE_TP1 = _mode(1)
_MODE_TP2DP2EP = _mode(2, dp=2, ep=True, real_weights=True)


def _requested_modes(available: list) -> list:
    want = os.environ.get("VLLM_TEST_MODES", "")
    if not want:
        return available
    requested = set(want.split(","))
    return [mode for mode in available if mode.id in requested]


_API_MODES = _requested_modes([_MODE_TP1])
_DISTRIBUTED_MODES = _requested_modes([_MODE_TP2DP2EP])
_PORT_BY_MODE = {"tp1": 8830, "tp2dp2ep": 10830}


def _mode_args(mode: dict) -> list[str]:
    args: list[str] = []
    if mode["tp"] > 1:
        args += ["--tensor-parallel-size", str(mode["tp"])]
    if mode["dp"] > 1:
        args += [
            "--data-parallel-size",
            str(mode["dp"]),
            "--data-parallel-size-local",
            str(mode["dp"]),
        ]
    if mode["ep"]:
        args += ["--enable-expert-parallel"]
    return args


@pytest.fixture(scope="class")
def wc_server(request):
    """Start one server for each test class and parallel mode."""
    mode = request.param
    port = _PORT_BY_MODE[mode["name"]]
    with server(
        port=port,
        timeout=900,
        dummy_weights=not mode["real_weights"],
        extra_args=_mode_args(mode),
    ) as url:
        yield mode, url


@pytest.mark.parametrize("wc_server", _API_MODES, indirect=True)
class TestWeightCheckerAPI:
    """API and state semantics that only need a single-engine server."""

    @pytest.fixture(autouse=True)
    def consume_baseline_after_test(self, wc_server):
        """Prevent a failed test from leaking its baseline into the next one."""
        _, url = wc_server
        before = weight_checker(url, "compare")
        assert before.status_code == 400, (
            f"weight-checker baseline leaked from the previous test: {before.text}"
        )

        yield

        after = weight_checker(url, "compare")
        assert after.status_code in (200, 400), after.text

    def test_compare_without_baseline_returns_400(self, wc_server):
        mode, url = wc_server
        response = weight_checker(url, "compare")
        assert response.status_code == 400, (
            f"[{mode['name']}] expected 400, got "
            f"{response.status_code}: {response.text}"
        )
        assert health(url) == 200

    @pytest.mark.parametrize(
        "payload", [{}, {"action": "snapshot"}, {"action": "frobnicate"}]
    )
    def test_invalid_action_returns_400(self, wc_server, payload):
        mode, url = wc_server
        response = requests.post(
            f"{url}/weight_checker", json=payload, timeout=10
        )
        assert response.status_code == 400, (
            f"[{mode['name']}] expected 400, got "
            f"{response.status_code}: {response.text}"
        )
        assert health(url) == 200

    def test_checksum_is_stable(self, wc_server):
        mode, url = wc_server
        first = weight_checker(url, "checksum")
        assert first.status_code == 200, first.text
        checksums = first.json()["checksums"]

        second = weight_checker(url, "checksum")
        assert second.status_code == 200, second.text
        assert checksums == second.json()["checksums"], (
            f"[{mode['name']}] checksum changed while weights were unchanged"
        )
        comparison = weight_checker(url, "compare")
        assert comparison.status_code == 200, comparison.text
        assert comparison.json() == {"match": True, "mismatches": []}

    def test_checksum_compare_is_one_shot(self, wc_server):
        mode, url = wc_server
        assert weight_checker(url, "checksum").status_code == 200

        first = weight_checker(url, "compare")
        assert first.status_code == 200, first.text
        assert first.json() == {"match": True, "mismatches": []}

        second = weight_checker(url, "compare")
        assert second.status_code == 400, (
            f"[{mode['name']}] expected 400 for a consumed baseline, got "
            f"{second.status_code}: {second.text}"
        )

    def test_reset_changes_weights(self, wc_server):
        """Run last because reset destructively randomizes model weights."""
        mode, url = wc_server
        assert weight_checker(url, "checksum").status_code == 200

        response = weight_checker(url, "reset")
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "reset"

        comparison = weight_checker(url, "compare")
        assert comparison.status_code == 200, comparison.text
        body = comparison.json()
        assert body["match"] is False, (
            f"[{mode['name']}] expected reset to change weights: {body}"
        )
        assert body["mismatches"]


@pytest.mark.parametrize("wc_server", _DISTRIBUTED_MODES, indirect=True)
class TestWeightCheckerTP2DP2EP:
    """Verify a real checkpoint across TP=2, DP=2, and EP."""

    def test_reset_reload_and_compare_real_weights(self, wc_server):
        mode, url = wc_server

        initial = weight_checker(url, "checksum")
        assert initial.status_code == 200, initial.text
        initial_body = initial.json()
        assert initial_body["baseline_created"] is True
        assert len(initial_body["engines"]) == mode["dp"], (
            f"[{mode['name']}] expected checksums from {mode['dp']} engines, "
            f"got {len(initial_body['engines'])}"
        )

        reset = weight_checker(url, "reset")
        assert reset.status_code == 200, reset.text

        reloaded = collective_rpc(url, "reload_weights")
        assert reloaded.status_code == 200, reloaded.text

        current = weight_checker(url, "checksum")
        assert current.status_code == 200, current.text
        assert current.json()["baseline_created"] is False
        assert current.json()["checksums"] == initial_body["checksums"], (
            f"[{mode['name']}] reloaded checkpoint differs from initial weights"
        )

        comparison = weight_checker(url, "compare")
        assert comparison.status_code == 200, comparison.text
        assert comparison.json() == {"match": True, "mismatches": []}
