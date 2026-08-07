# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the POST /weight_checker development endpoint."""

import os

import pytest
import requests

from .conftest import (
    gen,
    health,
    is_sleeping,
    ok,
    poll_until,
    server,
    sleep,
    wake,
    weight_checker,
)


def _mode(tp: int, ep: bool):
    key = f"tp{tp}ep" if ep else f"tp{tp}"
    return pytest.param({"tp": tp, "ep": ep, "name": key}, id=key)


_MODE_TP1 = _mode(1, False)
_MODE_TP2 = _mode(2, False)
_MODE_TP2EP = _mode(2, True)


def _requested_modes(available: list) -> list:
    want = os.environ.get("VLLM_TEST_MODES", "")
    if not want:
        return available
    requested = set(want.split(","))
    return [mode for mode in available if mode.id in requested]


_API_MODES = _requested_modes([_MODE_TP1])
_TP_MODES = _requested_modes([_MODE_TP2])
_EP_MODES = _requested_modes([_MODE_TP2EP])
_PORT_BY_MODE = {"tp1": 8830, "tp2": 9830, "tp2ep": 10830}


def _mode_args(mode: dict) -> list[str]:
    args: list[str] = []
    if mode["tp"] > 1:
        args += ["--tensor-parallel-size", str(mode["tp"])]
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
        dummy_weights=True,
        extra_args=_mode_args(mode),
    ) as url:
        yield mode, url


@pytest.fixture
def awake_wc_server(wc_server):
    """Ensure a sleep-related test leaves the shared server fully awake."""
    yield wc_server

    _, url = wc_server
    assert wake(url) == 200
    assert poll_until(lambda: not is_sleeping(url), timeout=30)


@pytest.mark.parametrize("wc_server", _API_MODES, indirect=True)
class TestWeightCheckerAPI:
    """API and state semantics that only need a single-engine server."""

    def test_compare_without_snapshot_returns_400(self, wc_server):
        mode, url = wc_server
        response = weight_checker(url, "compare")
        assert response.status_code == 400, (
            f"[{mode['name']}] expected 400, got "
            f"{response.status_code}: {response.text}"
        )
        assert health(url) == 200

    @pytest.mark.parametrize("payload", [{}, {"action": "frobnicate"}])
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

    def test_checksum_does_not_affect_snapshot(self, wc_server):
        mode, url = wc_server
        assert weight_checker(url, "snapshot").status_code == 200
        assert weight_checker(url, "checksum").status_code == 200

        response = weight_checker(url, "compare")
        assert response.status_code == 200, response.text
        assert response.json() == {"match": True, "mismatches": []}, (
            f"[{mode['name']}] checksum changed the stored snapshot"
        )

    def test_snapshot_compare_is_one_shot(self, wc_server):
        mode, url = wc_server
        assert weight_checker(url, "snapshot").status_code == 200

        first = weight_checker(url, "compare")
        assert first.status_code == 200, first.text
        assert first.json() == {"match": True, "mismatches": []}

        second = weight_checker(url, "compare")
        assert second.status_code == 400, (
            f"[{mode['name']}] expected 400 for a consumed snapshot, got "
            f"{second.status_code}: {second.text}"
        )

    def test_sleep_wake_does_not_change_weights(self, awake_wc_server):
        mode, url = awake_wc_server
        assert weight_checker(url, "snapshot").status_code == 200
        assert sleep(url) == 200
        assert wake(url) == 200
        assert poll_until(lambda: not is_sleeping(url), timeout=30)

        response = weight_checker(url, "compare")
        assert response.status_code == 200, response.text
        assert response.json()["match"] is True, (
            f"[{mode['name']}] sleep/wake changed weights: {response.json()}"
        )

    def test_generate_does_not_change_weights(self, wc_server):
        mode, url = wc_server
        assert weight_checker(url, "snapshot").status_code == 200
        assert ok(gen(url))

        response = weight_checker(url, "compare")
        assert response.status_code == 200, response.text
        assert response.json()["match"] is True, (
            f"[{mode['name']}] generation changed weights: {response.json()}"
        )

    def test_reset_changes_weights(self, wc_server):
        """Run last because reset destructively randomizes model weights."""
        mode, url = wc_server
        assert weight_checker(url, "snapshot").status_code == 200

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


@pytest.mark.parametrize("wc_server", _TP_MODES, indirect=True)
class TestWeightCheckerTP:
    """Tensor-parallel checksum aggregation coverage."""

    def test_checksum_and_compare_work_with_tp(self, wc_server):
        mode, url = wc_server
        response = weight_checker(url, "checksum")
        assert response.status_code == 200, response.text
        checksums = response.json()["checksums"]

        assert weight_checker(url, "snapshot").status_code == 200
        comparison = weight_checker(url, "compare")
        assert comparison.status_code == 200, comparison.text
        assert comparison.json() == {"match": True, "mismatches": []}


@pytest.mark.parametrize("wc_server", _EP_MODES, indirect=True)
class TestWeightCheckerEP:
    """Expert-parallel checksum coverage."""

    def test_checksum_covers_local_experts(self, wc_server):
        mode, url = wc_server
        response = weight_checker(url, "checksum")
        assert response.status_code == 200, response.text
        checksums = response.json()["checksums"]
        assert any("expert" in name for name in checksums), (
            "expected expert-gated MoE weights, got keys like "
            f"{sorted(checksums)[:5]}"
        )
