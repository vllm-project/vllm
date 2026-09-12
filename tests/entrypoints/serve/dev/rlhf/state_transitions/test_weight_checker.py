# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP checksum baselines, corruption detection, and checkpoint restoration.

The API smoke uses one real small model so reset can always be undone. Rank
aggregation additionally runs with TP2/DP2/EP on four GPUs. Pure baseline and
merge errors are covered by entrypoints/unit_tests/test_weight_checker.py;
transport correctness remains in tests/distributed/test_weight_transfer.py.
"""

import os

import pytest
import regex as re
import requests

from tests.entrypoints.serve.dev.rlhf.conftest import (
    collective_rpc,
    gen,
    health,
    ok,
    reusable_server,
    weight_checker,
)

# Parent cleanup runs once after the shared server has fully shut down.
pytestmark = pytest.mark.skip_global_cleanup


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
    return pytest.param(
        value, id=key, marks=[pytest.mark.distributed] if tp * dp > 1 else []
    )


_MODE_TP1 = _mode(1, real_weights=True)
_MODE_TP2DP2EP = _mode(2, dp=2, ep=True, real_weights=True)


def _requested_modes(available: list) -> list:
    want = os.environ.get("VLLM_TEST_MODES", "")
    if not want:
        return available
    requested = set(want.split(","))
    return [mode for mode in available if mode.id in requested]


_API_MODES = _requested_modes([_MODE_TP1])
_DISTRIBUTED_MODES = _requested_modes([_MODE_TP2DP2EP])


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
def wc_server(request, num_gpus_available):
    """Start one server for each test class and parallel mode."""
    mode = request.param
    if num_gpus_available < mode["tp"] * mode["dp"]:
        pytest.skip("Insufficient GPUs for the requested checker topology")
    model = (
        os.environ.get("VLLM_TEST_MOE_MODEL", "TitanML/tiny-mixtral")
        if mode["ep"]
        else os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B")
    )
    with reusable_server(
        model=model,
        timeout=900,
        dummy_weights=not mode["real_weights"],
        extra_args=_mode_args(mode)
        + (["--hf-overrides", '{"sliding_window": null}'] if mode["ep"] else []),
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
        "payload", [{}, {"action": "snapshot"}, {"action": "frobnicate"}, [], None]
    )
    def test_invalid_action_returns_400(self, wc_server, payload):
        mode, url = wc_server
        response = requests.post(f"{url}/weight_checker", json=payload, timeout=10)
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
        assert first.json()["baseline_created"] is True
        assert checksums
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", digest) for digest in checksums.values()
        )

        second = weight_checker(url, "checksum")
        assert second.status_code == 200, second.text
        assert second.json()["baseline_created"] is False
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

    def test_reset_changes_weights_and_reload_restores_them(self, wc_server):
        """A failed assertion still restores weights before the next test."""
        _, url = wc_server
        original = weight_checker(url, "checksum")
        original.raise_for_status()
        try:
            reset = weight_checker(url, "reset")
            reset.raise_for_status()
            comparison = weight_checker(url, "compare")
            comparison.raise_for_status()
            assert comparison.json()["match"] is False
            assert comparison.json()["mismatches"]
        finally:
            collective_rpc(url, "reload_weights").raise_for_status()
        restored = weight_checker(url, "checksum")
        restored.raise_for_status()
        assert restored.json()["checksums"] == original.json()["checksums"]
        comparison = weight_checker(url, "compare")
        comparison.raise_for_status()
        assert comparison.json() == {"match": True, "mismatches": []}
        assert ok(gen(url))


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

        ranks = set()
        for key in initial_body["checksums"]:
            match = re.fullmatch(r"dp(\d+):pp0:pcp0:tp(\d+):ep\d+:.+", key)
            assert match is not None, f"Unqualified checksum key: {key}"
            ranks.add(tuple(map(int, match.groups())))
        assert ranks == {
            (dp, tp) for dp in range(mode["dp"]) for tp in range(mode["tp"])
        }
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
