# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from types import SimpleNamespace

import pytest

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOConnectorMetadata,
    MoRIIOConstants,
    MoRIIOMode,
    get_moriio_node_hosts,
    get_moriio_trusted_remote_hosts,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
    _pick_remote_rank_host,
)


@pytest.mark.parametrize(
    ("node_hosts", "expected"),
    [
        ("prefill-a, prefill-b, ", ["prefill-a", "prefill-b"]),
        (["prefill-a", " prefill-b ", ""], ["prefill-a", "prefill-b"]),
        (None, ["local-host"]),
    ],
)
def test_moriio_node_hosts_come_from_explicit_config_or_local_fallback(
    node_hosts, expected
):
    config = KVTransferConfig(
        kv_connector_extra_config={}
        if node_hosts is None
        else {"node_hosts": node_hosts}
    )

    assert get_moriio_node_hosts(config, "local-host") == expected


def test_moriio_trusted_remote_hosts_use_explicit_config_or_node_hosts():
    config = KVTransferConfig(
        kv_connector_extra_config={
            "trusted_remote_hosts": "peer-a, peer-b",
            "node_hosts": ["local-a"],
        }
    )
    assert get_moriio_trusted_remote_hosts(config, ["local-a"]) == frozenset(
        {"peer-a", "peer-b"}
    )

    config = KVTransferConfig(kv_connector_extra_config={})
    assert get_moriio_trusted_remote_hosts(config, ["local-a"]) == frozenset(
        {"local-a"}
    )


@pytest.mark.parametrize(
    ("tp_rank", "expected"),
    [
        (0, "prefill-a"),
        (7, "prefill-a"),
        (8, "prefill-b"),
        (15, "prefill-b"),
    ],
)
def test_remote_rank_host_uses_tp_rank_for_multi_host_tp(tp_rank, expected):
    assert (
        _pick_remote_rank_host(
            "fallback",
            ["prefill-a", "prefill-b"],
            tp_size=16,
            tp_rank=tp_rank,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("remote_dp_rank", "expected"),
    [
        (0, "prefill-a"),
        (7, "prefill-a"),
        (8, "prefill-b"),
        (15, "prefill-b"),
    ],
)
def test_remote_rank_host_uses_dp_rank_when_remote_dp_spans_hosts(
    remote_dp_rank, expected
):
    assert (
        _pick_remote_rank_host(
            "fallback",
            ["prefill-a", "prefill-b"],
            tp_size=1,
            tp_rank=0,
            remote_dp_size=16,
            remote_dp_rank=remote_dp_rank,
        )
        == expected
    )


def test_connector_metadata_uses_remote_hosts_for_plain_request_id():
    metadata = MoRIIOConnectorMetadata(trusted_remote_hosts={"prefill-a", "prefill-b"})

    metadata.add_new_req(
        request_id="plain-request-id",
        local_block_ids=[1],
        kv_transfer_params={
            "transfer_id": "tx-1",
            "remote_block_ids": [2],
            "remote_engine_id": "prefill-engine",
            "remote_hosts": ["prefill-a", "prefill-b"],
            "remote_tp_size": 16,
            "remote_dp_size": 16,
            "remote_dp_rank": 8,
        },
    )

    req_meta = metadata.reqs_to_recv["plain-request-id"]
    assert req_meta.remote_host == "prefill-a"
    assert req_meta.remote_handshake_port == int(MoRIIOConstants.DEFAULT_HANDSHAKE_PORT)
    assert req_meta.remote_notify_port == int(MoRIIOConstants.DEFAULT_NOTIFY_PORT)
    assert req_meta.remote_hosts == ["prefill-a", "prefill-b"]
    assert req_meta.tp_size == 16
    assert req_meta.remote_dp_size == 16
    assert req_meta.remote_dp_rank == 8


def test_connector_metadata_rejects_untrusted_remote_hosts():
    metadata = MoRIIOConnectorMetadata(trusted_remote_hosts={"prefill-a"})

    with pytest.raises(ValueError, match="untrusted remote_hosts"):
        metadata.add_new_req(
            request_id="plain-request-id",
            local_block_ids=[1],
            kv_transfer_params={
                "transfer_id": "tx-1",
                "remote_block_ids": [2],
                "remote_engine_id": "prefill-engine",
                "remote_hosts": ["prefill-a", "metadata.example"],
                "remote_tp_size": 16,
            },
        )


def test_scheduler_rejects_untrusted_remote_hosts_before_notify(monkeypatch):
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.mode = MoRIIOMode.WRITE
    scheduler.tp_size = 2
    scheduler.dp_rank = 0
    scheduler.transfer_id_to_request_id = {}
    scheduler.request_id_to_transfer_id = {}
    scheduler.trusted_remote_hosts = frozenset({"prefill-a"})
    sent = []
    monkeypatch.setattr(
        scheduler,
        "send_notify_block",
        lambda **kwargs: sent.append(kwargs),
    )
    request = SimpleNamespace(
        request_id=(
            "___prefill_addr_host:prefill-a,handshake:6301,notify:61005"
            "___decode_addr_host:decode-a,handshake:6301,notify:61005"
            "_0123456789abcdef0123456789abcdef"
        ),
        kv_transfer_params={
            "transfer_id": "tx-1",
            "do_remote_prefill": True,
            "remote_hosts": ["prefill-a", "notify.example"],
        },
    )

    with pytest.raises(ValueError, match="untrusted remote_hosts"):
        scheduler.update_state_after_alloc(
            request,
            SimpleNamespace(get_block_ids=lambda: ([1, 2],)),
            0,
        )

    assert sent == []


@pytest.mark.asyncio
async def test_toy_proxy_forwards_node_hosts_to_prefill_and_decode(monkeypatch):
    from examples.disaggregated.disaggregated_serving import (
        moriio_toy_proxy_server as proxy,
    )

    captured_prefill = {}
    captured_decode = {}
    prefill_hosts = ["prefill-a", "prefill-b"]
    decode_hosts = ["decode-a", "decode-b"]

    class FakeRequest:
        async def get_json(self):
            return {"model": "test-model", "prompt": "hello", "max_tokens": 2}

    async def fake_send_request_to_prefill(
        endpoint, req_data, request_id, selected_prefill_dp_rank
    ):
        captured_prefill.update(
            {
                "endpoint": endpoint,
                "request_id": request_id,
                "selected_prefill_dp_rank": selected_prefill_dp_rank,
                "req_data": copy.deepcopy(req_data),
            }
        )
        return {
            "kv_transfer_params": {
                "remote_engine_id": "prefill-engine",
                "remote_block_ids": [1, 2],
                "transfer_id": req_data["kv_transfer_params"]["transfer_id"],
                "remote_hosts": ["unused-read-mode-host"],
            }
        }

    async def fake_start_decode_request(endpoint, req_data, request_id):
        captured_decode.update(
            {
                "endpoint": endpoint,
                "request_id": request_id,
                "req_data": copy.deepcopy(req_data),
            }
        )
        return object(), object()

    def fake_stream_decode_response(session, response, request_id):
        async def stream():
            yield b"ok"

        return stream()

    async def fake_make_response(response):
        return response

    monkeypatch.setattr(
        proxy,
        "prefill_instances",
        [
            {
                "request_address": "http://prefill.example/v1",
                "zmq_address": "host:prefill-a,handshake:6301,notify:61005",
                "dp_size": 1,
                "tp_size": 16,
                "node_hosts": prefill_hosts,
            }
        ],
    )
    monkeypatch.setattr(
        proxy,
        "decode_instances",
        [
            {
                "request_address": "http://decode.example/v1",
                "zmq_address": "host:decode-a,handshake:6301,notify:61005",
                "dp_size": 1,
                "tp_size": 16,
                "node_hosts": decode_hosts,
            }
        ],
    )
    monkeypatch.setattr(proxy, "TRANSFER_TYPE", "READ")
    monkeypatch.setattr(proxy, "request_nums", 0)
    monkeypatch.setattr(proxy, "send_request_to_prefill", fake_send_request_to_prefill)
    monkeypatch.setattr(proxy, "start_decode_request", fake_start_decode_request)
    monkeypatch.setattr(proxy, "stream_decode_response", fake_stream_decode_response)
    monkeypatch.setattr(proxy, "make_response", fake_make_response)

    await proxy.handle_request("/chat/completions", FakeRequest())

    assert captured_prefill["endpoint"] == (
        "http://prefill.example/v1/chat/completions"
    )
    assert (
        captured_prefill["req_data"]["kv_transfer_params"]["remote_hosts"]
        == decode_hosts
    )
    assert captured_decode["endpoint"] == "http://decode.example/v1/chat/completions"
    assert (
        captured_decode["req_data"]["kv_transfer_params"]["remote_hosts"]
        == prefill_hosts
    )
    assert captured_decode["req_data"]["kv_transfer_params"]["tp_size"] == 16
