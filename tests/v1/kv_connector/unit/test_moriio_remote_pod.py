# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    fill_remote_pod_from_config,
    fold_local_rank,
    pod_index,
)

_EXTRA = {"remote_hosts": ["10.0.0.1", "10.0.0.2"], "remote_dp_size_local": 8}


def test_fill_from_serve_config_when_router_omits_hosts():
    params = {"remote_dp_rank": 12}
    fill_remote_pod_from_config(params, _EXTRA, is_producer=False)
    assert params["remote_hosts"] == ["10.0.0.1", "10.0.0.2"]
    assert params["remote_dp_size_local"] == 8
    # The engine holding the request owns the notify; the pod-0-only
    # default would drop it for child-pod ranks.
    assert params["is_request_leader"] is True


def test_producer_fill_does_not_claim_leadership():
    params = {"remote_dp_rank": 12}
    fill_remote_pod_from_config(params, _EXTRA, is_producer=True)
    assert params["remote_hosts"] == ["10.0.0.1", "10.0.0.2"]
    assert "is_request_leader" not in params


@pytest.mark.parametrize(
    ("rank", "host", "port_offset"),
    [(0, "10.0.0.1", 0), (7, "10.0.0.1", 7), (8, "10.0.0.2", 0), (12, "10.0.0.2", 4)],
)
def test_rank_maps_to_its_own_pod(rank, host, port_offset):
    params = {"remote_dp_rank": rank}
    fill_remote_pod_from_config(params, _EXTRA, is_producer=True)
    dp_local = params["remote_dp_size_local"]
    assert params["remote_hosts"][pod_index(rank, dp_local)] == host
    assert fold_local_rank(rank, dp_local) == port_offset


def test_router_values_win():
    params = {"remote_hosts": ["10.9.9.9"], "remote_dp_rank": 1}
    fill_remote_pod_from_config(params, _EXTRA, is_producer=False)
    assert params["remote_hosts"] == ["10.9.9.9"]
    assert "remote_dp_size_local" not in params
    assert "is_request_leader" not in params


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"remote_hosts": ["10.0.0.1"]},
        {"remote_dp_size_local": 8},
        {"remote_hosts": [""], "remote_dp_size_local": 8},
        {"remote_hosts": ["10.0.0.1"], "remote_dp_size_local": 0},
        {"remote_hosts": ["10.0.0.1"], "remote_dp_size_local": "eight"},
    ],
)
def test_incomplete_config_is_ignored(extra):
    params = {"remote_dp_rank": 12}
    fill_remote_pod_from_config(params, extra, is_producer=False)
    assert params == {"remote_dp_rank": 12}


def test_comma_separated_hosts():
    params = {}
    fill_remote_pod_from_config(
        params,
        {"remote_hosts": "10.0.0.1, 10.0.0.2", "remote_dp_size_local": "8"},
        is_producer=True,
    )
    assert params["remote_hosts"] == ["10.0.0.1", "10.0.0.2"]
    assert params["remote_dp_size_local"] == 8


def test_second_call_is_a_no_op():
    params = {"remote_dp_rank": 12}
    fill_remote_pod_from_config(params, _EXTRA, is_producer=False)
    first = dict(params)
    fill_remote_pod_from_config(params, _EXTRA, is_producer=False)
    assert params == first
