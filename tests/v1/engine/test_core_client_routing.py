# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.engine import core_client as core_client_module
from vllm.v1.engine.core_client import DPLBAsyncMPClient


def _make_dplb_client() -> DPLBAsyncMPClient:
    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = 1
    client.reqs_in_flight = {}
    client.core_engines = [b"\x00\x00", b"\x01\x00", b"\x02\x00", b"\x03\x00"]
    client.lb_engines = [[3, 0], [0, 0], [0, 0], [0, 0]]
    client.eng_start_index = 0
    client.engine_ranks_managed = [0, 1, 2, 3]
    client.active_data_parallel_size = 2
    return client


def test_dplb_routes_explicit_sleeping_rank_to_active_engine():
    client = _make_dplb_client()
    request = SimpleNamespace(
        request_id="sleeping-rank-request",
        data_parallel_rank=3,
        pooling_params=None,
    )

    chosen_engine = client.get_core_engine_for_request(request)

    assert chosen_engine == client.core_engines[1]
    assert client.reqs_in_flight[request.request_id] == chosen_engine
    assert client.lb_engines == [[3, 0], [1, 0], [0, 0], [0, 0]]


def test_dplb_counts_explicit_active_rank_once():
    client = _make_dplb_client()
    request = SimpleNamespace(
        request_id="active-rank-request",
        data_parallel_rank=1,
        pooling_params=None,
    )

    chosen_engine = client.get_core_engine_for_request(request)

    assert chosen_engine == client.core_engines[1]
    assert client.lb_engines == [[3, 0], [1, 0], [0, 0], [0, 0]]


def test_dplb_routes_late_interaction_sleeping_rank_to_active_engine(monkeypatch):
    client = _make_dplb_client()
    request = SimpleNamespace(
        request_id="late-interaction-request",
        data_parallel_rank=None,
        pooling_params=object(),
    )
    monkeypatch.setattr(
        core_client_module,
        "get_late_interaction_engine_index",
        lambda pooling_params, dp_size: 3,
    )

    chosen_engine = client.get_core_engine_for_request(request)

    assert chosen_engine == client.core_engines[1]
    assert client.lb_engines == [[3, 0], [1, 0], [0, 0], [0, 0]]
