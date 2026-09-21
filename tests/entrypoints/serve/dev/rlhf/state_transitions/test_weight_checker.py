# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP checksum baselines, corruption detection, and checkpoint restoration.

The API smoke uses one real small model so reset can always be undone; rank
aggregation additionally runs with TP2/DP2/EP on four GPUs.
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
    pause,
    reusable_server,
    weight_checker,
)

pytestmark = pytest.mark.skip_global_cleanup


def _mode(tp: int, dp: int = 1, ep: bool = False, real_weights: bool = False):
    """Build a parametrized parallel-mode spec for the server fixture."""
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
    """Keep only the modes named in VLLM_TEST_MODES, or all of them if unset."""
    want = os.environ.get("VLLM_TEST_MODES", "")
    if not want:
        return available
    requested = set(want.split(","))
    return [mode for mode in available if mode.id in requested]


_API_MODES = _requested_modes([_MODE_TP1])
_DISTRIBUTED_MODES = _requested_modes([_MODE_TP2DP2EP])


def _mode_args(mode: dict) -> list[str]:
    """Translate a mode spec into vLLM server CLI flags."""
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
    """API semantics that only need a single-engine server."""

    def test_compare_requires_a_baseline(self, wc_server):
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

    def test_paused_server_is_conflict(self, wc_server):
        """A paused engine cannot hash or rewrite its weights.

        ``compare`` is sent without a baseline, so 409 must win over the 400
        that a missing baseline would otherwise produce.
        """
        mode, url = wc_server
        assert pause(url) == 200
        try:
            for action in ("checksum", "reset", "compare"):
                response = weight_checker(url, action)
                assert response.status_code == 409, (
                    f"[{mode['name']}] expected 409 for {action} while paused, "
                    f"got {response.status_code}: {response.text}"
                )
            assert health(url) == 200
        finally:
            # Leave the shared class-scoped server unpaused for later tests.
            assert requests.post(f"{url}/resume", timeout=10).status_code == 200

    def test_checksum_is_stable_and_stateless(self, wc_server):
        mode, url = wc_server
        first = weight_checker(url, "checksum")
        assert first.status_code == 200, first.text
        assert set(first.json()) == {"checksums"}, first.text
        checksums = first.json()["checksums"]
        assert checksums
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", digest) for digest in checksums.values()
        )

        second = weight_checker(url, "checksum")
        assert second.status_code == 200, second.text
        assert checksums == second.json()["checksums"], (
            f"[{mode['name']}] checksum changed while weights were unchanged"
        )

        # Statelessness means a comparison can be repeated, and any API
        # process can serve it.
        for _ in range(2):
            comparison = weight_checker(url, "compare", checksums)
            assert comparison.status_code == 200, comparison.text
            assert comparison.json() == {"match": True, "mismatches": []}

    def test_compare_reports_changed_and_missing_tensors(self, wc_server):
        mode, url = wc_server
        baseline = weight_checker(url, "checksum")
        baseline.raise_for_status()
        baseline_checksums = baseline.json()["checksums"]

        keys = list(baseline_checksums)
        changed_key, missing_key = keys[0], keys[1]
        altered = dict(baseline_checksums)
        altered[changed_key] = "0" * 64
        del altered[missing_key]

        comparison = weight_checker(url, "compare", altered)
        assert comparison.status_code == 200, comparison.text
        assert comparison.json() == {
            "match": False,
            "mismatches": sorted([changed_key, missing_key]),
        }, mode["name"]

    def test_reset_changes_weights_and_reload_restores_them(self, wc_server):
        """A failed assertion still restores weights before the next test."""
        _, url = wc_server
        original = weight_checker(url, "checksum")
        original.raise_for_status()
        baseline = original.json()["checksums"]
        try:
            reset = weight_checker(url, "reset")
            reset.raise_for_status()
            comparison = weight_checker(url, "compare", baseline)
            comparison.raise_for_status()
            assert comparison.json()["match"] is False
            assert comparison.json()["mismatches"]
        finally:
            collective_rpc(url, "reload_weights").raise_for_status()
        restored = weight_checker(url, "checksum")
        restored.raise_for_status()
        assert restored.json()["checksums"] == baseline
        comparison = weight_checker(url, "compare", baseline)
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
        initial_checksums = initial.json()["checksums"]

        ranks = set()
        for key in initial_checksums:
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
        assert current.json()["checksums"] == initial_checksums, (
            f"[{mode['name']}] reloaded checkpoint differs from initial weights"
        )

        comparison = weight_checker(url, "compare", initial_checksums)
        assert comparison.status_code == 200, comparison.text
        assert comparison.json() == {"match": True, "mismatches": []}
