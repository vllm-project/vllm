# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import regex as re
import requests

from tests.entrypoints.serve.dev.rlhf.conftest import gen, health, ok, server
from tests.utils import multi_gpu_test


def weight_checker(url: str, action: str, baseline=None) -> requests.Response:
    body = {"action": action}
    if baseline is not None:
        body["baseline"] = baseline
    return requests.post(f"{url}/weight_checker", json=body, timeout=300)


def checksums(url: str) -> dict[str, str]:
    response = weight_checker(url, "checksum")
    response.raise_for_status()
    return response.json()["checksums"]


def reset_reload_and_compare(url: str) -> None:
    baseline = checksums(url)
    try:
        weight_checker(url, "reset").raise_for_status()
        assert weight_checker(url, "compare", baseline).json()["match"] is False
    finally:
        requests.post(
            f"{url}/collective_rpc", json={"method": "reload_weights"}, timeout=300
        ).raise_for_status()
    assert weight_checker(url, "compare", baseline).json() == {
        "match": True,
        "mismatches": [],
    }


@pytest.fixture(scope="module")
def server_url():
    with server() as url:
        yield url


@pytest.mark.parametrize(
    "payload",
    [{}, {"action": "frobnicate"}, {"action": "compare"}, [], None],
)
def test_invalid_request_returns_400(server_url, payload):
    response = requests.post(f"{server_url}/weight_checker", json=payload, timeout=10)
    assert response.status_code == 400, response.text
    assert health(server_url) == 200


def test_checksum_is_stable_and_compare_reports_differences(server_url):
    baseline = checksums(server_url)
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in baseline.values())
    assert checksums(server_url) == baseline

    changed, missing, *_ = baseline
    altered = {**baseline, changed: "0" * 64}
    del altered[missing]
    assert weight_checker(server_url, "compare", altered).json() == {
        "match": False,
        "mismatches": sorted([changed, missing]),
    }


def test_reset_changes_weights_and_reload_restores_them(server_url):
    reset_reload_and_compare(server_url)
    assert ok(gen(server_url))


@multi_gpu_test(num_gpus=4)
def test_keys_cover_every_tp_and_dp_rank():
    args = ["--tensor-parallel-size", "2", "--data-parallel-size", "2"]
    with server(extra_args=args, port=8771, timeout=600) as url:
        ranks = {
            re.fullmatch(r"dp(\d):pp0:pcp0:tp(\d):ep0:.+", key).groups()
            for key in checksums(url)
        }
        assert ranks == {(dp, tp) for dp in "01" for tp in "01"}
        reset_reload_and_compare(url)
