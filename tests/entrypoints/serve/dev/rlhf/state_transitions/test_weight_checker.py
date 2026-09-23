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
    reusable_server,
    weight_checker,
)
from vllm.utils.weight_checksum_utils import split_checksum_key

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


def _rank_prefixes(checksums: dict[str, str]) -> list[str]:
    """Return the rank prefixes present in a checksum response.

    Derived from the keys rather than hardcoded, so it tracks whatever
    topology the fixture actually started.
    """
    return sorted({split_checksum_key(key)[0] for key in checksums})


def _matches(comparison: requests.Response, checksums: dict[str, str]) -> bool:
    """Whether a comparison reports a match over exactly the given ranks."""
    return comparison.json() == {
        "match": True,
        "mismatches": [],
        "ranks": _rank_prefixes(checksums),
    }


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
            assert _matches(comparison, checksums), comparison.text

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
            "ranks": _rank_prefixes(baseline_checksums),
        }, mode["name"]

    def test_compare_holds_the_callers_other_reports_to_the_baseline(
        self, wc_server
    ):
        """Extra reports turn compare into a replica check over the wire."""
        mode, url = wc_server
        baseline = weight_checker(url, "checksum")
        baseline.raise_for_status()
        checksums = baseline.json()["checksums"]

        comparison = weight_checker(url, "compare", checksums, [checksums])
        assert comparison.status_code == 200, comparison.text
        assert _matches(comparison, checksums), comparison.text

        # A report that disagrees must be reported, not ignored.
        diverged = dict(checksums)
        diverged_key = next(iter(diverged))
        diverged[diverged_key] = "0" * 64
        comparison = weight_checker(url, "compare", checksums, [diverged])
        assert comparison.status_code == 200, comparison.text
        assert comparison.json() == {
            "match": False,
            "mismatches": [diverged_key],
            "ranks": _rank_prefixes(checksums),
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
        assert _matches(comparison, baseline), comparison.text
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
        assert _matches(comparison, initial_checksums), comparison.text
