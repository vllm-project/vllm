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
    get_moriio_request_id_trusted_hosts,
    get_moriio_trusted_remote_hosts,
    validate_moriio_remote_host,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
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


def test_moriio_trusted_remote_hosts_explicit_or_empty():
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
    # Opt-in: unconfigured -> empty allowlist (no enforcement), NOT the local
    # node_hosts, so the default multi-node cross-host READ flow is not rejected.
    assert get_moriio_trusted_remote_hosts(config, ["local-a"]) == frozenset()


def test_connector_metadata_accepts_remote_hosts_when_trusted_unconfigured():
    # Regression: with no trusted_remote_hosts configured the plural remote_hosts
    # path must be a no-op (accept) so the default multi-node cross-host READ
    # works. The earlier fallback to local node_hosts rejected every cross-host
    # peer forwarded by the proxy.
    metadata = MoRIIOConnectorMetadata(trusted_remote_hosts=frozenset())

    metadata.add_new_req(
        request_id="plain-request-id",
        local_block_ids=[1],
        kv_transfer_params={
            "transfer_id": "tx-1",
            "remote_block_ids": [2],
            "remote_engine_id": "prefill-engine",
            "remote_hosts": ["prefill-a", "prefill-b"],
            "remote_tp_size": 16,
        },
    )

    req_meta = metadata.reqs_to_recv["plain-request-id"]
    assert req_meta.remote_hosts == ["prefill-a", "prefill-b"]


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
    scheduler._request_id_trusted_hosts = frozenset()
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
        return object(), SimpleNamespace(headers={})

    def fake_stream_decode_response(session, response, request_id):
        async def stream():
            yield b"ok"

        return stream()

    async def fake_make_response(response):
        return SimpleNamespace(headers={})

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


def _multi_host_recv_meta():
    """Decode-side ReqMeta for a 2-node prefill peer (TP=16, DP=16)."""
    metadata = MoRIIOConnectorMetadata(trusted_remote_hosts={"prefill-a", "prefill-b"})
    metadata.add_new_req(
        request_id="glue-multihost",
        local_block_ids=[1],
        kv_transfer_params={
            "transfer_id": "tx-1",
            "remote_block_ids": [2],
            "remote_engine_id": "prefill-engine",
            "remote_hosts": ["prefill-a", "prefill-b"],
            "remote_tp_size": 16,
            "remote_dp_size": 16,
        },
    )
    return metadata.reqs_to_recv["glue-multihost"]


@pytest.mark.parametrize(
    ("tp_rank", "expected"),
    [
        (0, "prefill-a"),
        (8, "prefill-b"),
        (15, "prefill-b"),
    ],
)
def test_worker_pick_remote_host_routes_by_tp_rank(tp_rank, expected):
    # Proves _pick_remote_host feeds self.tp_rank + meta.remote_hosts into the
    # pure picker (TP branch), so each decode worker targets its own peer node.
    meta = _multi_host_recv_meta()
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = tp_rank
    assert worker._pick_remote_host(meta) == expected


@pytest.mark.parametrize(
    ("dp_rank", "expected"),
    [
        (0, "prefill-a"),
        (8, "prefill-b"),
        (15, "prefill-b"),
    ],
)
def test_worker_pick_host_for_dp_rank_routes_by_dp_rank(dp_rank, expected):
    # Proves _pick_host_for_dp_rank forwards meta.remote_dp_size + the dp_rank
    # arg (DP branch). tp_rank is pinned to 0 so only dp_rank can move the host.
    meta = _multi_host_recv_meta()
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = 0
    assert worker._pick_host_for_dp_rank(meta, dp_rank) == expected


@pytest.mark.parametrize("tp_rank", [0, 7])
def test_worker_pick_remote_host_single_host_passthrough(tp_rank):
    # Proves the TP<=8 / single-host path is untouched: no host list means
    # _pick_remote_host returns meta.remote_host unchanged for any tp_rank.
    metadata = MoRIIOConnectorMetadata()
    metadata.add_new_req(
        request_id="glue-singlehost",
        local_block_ids=[1],
        kv_transfer_params={
            "transfer_id": "tx-1",
            "remote_block_ids": [2],
            "remote_engine_id": "prefill-engine",
            "remote_host": "prefill-solo",
            "remote_handshake_port": 6301,
            "remote_notify_port": 61005,
            "remote_tp_size": 8,
        },
    )
    meta = metadata.reqs_to_recv["glue-singlehost"]
    assert meta.remote_hosts is None
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = tp_rank
    assert worker._pick_remote_host(meta) == "prefill-solo"


def test_validate_moriio_remote_host():
    trusted = frozenset({"prefill-a"})
    key = "kv_transfer_params['remote_host']"

    # (a) value=None: singular path is optional, so a missing host is a no-op.
    validate_moriio_remote_host(None, trusted, key)
    # (b) no trust list configured: default single-host flow stays unenforced.
    validate_moriio_remote_host("anything.example", (), key)
    # (c) host present in the trust list is accepted.
    validate_moriio_remote_host("prefill-a", trusted, key)
    # (d) host absent from a non-empty trust list is rejected.
    with pytest.raises(ValueError, match="untrusted host"):
        validate_moriio_remote_host("evil.example", trusted, key)


def test_connector_metadata_rejects_untrusted_remote_host_singular():
    metadata = MoRIIOConnectorMetadata(trusted_remote_hosts={"prefill-a"})

    # Ports are supplied so the request_id parse branch is skipped: without the
    # singular-host guard the untrusted host would be accepted outright.
    with pytest.raises(ValueError, match="untrusted host"):
        metadata.add_new_req(
            request_id="plain-request-id",
            local_block_ids=[1],
            kv_transfer_params={
                "transfer_id": "tx-1",
                "remote_block_ids": [2],
                "remote_engine_id": "prefill-engine",
                "remote_host": "evil.example",
                "remote_handshake_port": 6301,
                "remote_notify_port": 61005,
                "remote_tp_size": 16,
            },
        )


def test_connector_metadata_accepts_trusted_remote_host_singular():
    metadata = MoRIIOConnectorMetadata(trusted_remote_hosts={"prefill-a"})

    metadata.add_new_req(
        request_id="plain-request-id",
        local_block_ids=[1],
        kv_transfer_params={
            "transfer_id": "tx-1",
            "remote_block_ids": [2],
            "remote_engine_id": "prefill-engine",
            "remote_host": "prefill-a",
            "remote_handshake_port": 6301,
            "remote_notify_port": 61005,
            "remote_tp_size": 16,
        },
    )

    # A trusted singular host survives validation and is registered as the peer.
    assert metadata.reqs_to_recv["plain-request-id"].remote_host == "prefill-a"


def test_scheduler_release_write_rejects_untrusted_remote_host(monkeypatch):
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.trusted_remote_hosts = frozenset({"prefill-a"})
    scheduler.tp_size = 1
    sent = []
    monkeypatch.setattr(
        scheduler,
        "_send_transfer_release",
        lambda *args, **kwargs: sent.append((args, kwargs)),
    )

    # remote_notify_port present -> request_id parse is skipped; the guard must
    # fail closed before any release dial reaches the untrusted host.
    with pytest.raises(ValueError, match="untrusted host"):
        scheduler._release_write_prefill_blocks(
            "req",
            {
                "transfer_id": "tx",
                "remote_host": "evil.example",
                "remote_notify_port": 6100,
            },
        )

    assert sent == []


def test_scheduler_release_write_sends_for_trusted_remote_host(monkeypatch):
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.trusted_remote_hosts = frozenset({"prefill-a"})
    scheduler._request_id_trusted_hosts = frozenset()
    scheduler.tp_size = 1
    sent = []
    monkeypatch.setattr(
        scheduler,
        "_send_transfer_release",
        lambda transfer_id, host, port: sent.append((transfer_id, host, port)),
    )

    scheduler._release_write_prefill_blocks(
        "req",
        {"transfer_id": "tx", "remote_host": "prefill-a", "remote_notify_port": 6100},
    )

    # A trusted host reaches the release dial with the resolved transfer target.
    assert sent == [("tx", "prefill-a", 6100)]


def test_scheduler_release_write_none_host_uses_request_id(monkeypatch):
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.trusted_remote_hosts = frozenset({"prefill-a"})
    scheduler._request_id_trusted_hosts = frozenset()
    scheduler.tp_size = 1
    sent = []
    monkeypatch.setattr(
        scheduler,
        "_send_transfer_release",
        lambda transfer_id, host, port: sent.append((transfer_id, host, port)),
    )

    request_id = (
        "___prefill_addr_host:prefill-a,handshake:6301,notify:6100"
        "___decode_addr_host:decode-a,handshake:6301,notify:6100"
        "_0123456789abcdef0123456789abcdef"
    )

    # remote_host omitted: the guard returns early (no false reject) and the
    # notify address is recovered from the request_id, then dialed.
    scheduler._release_write_prefill_blocks(request_id, {"transfer_id": "tx"})

    assert sent == [("tx", "prefill-a", 6100)]


def test_scheduler_update_state_rejects_untrusted_remote_host_singular(monkeypatch):
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
            "remote_host": "evil.example",
            "remote_notify_port": 61005,
        },
    )

    # remote_notify_port present -> parse skipped; the singular host must be
    # rejected before any notify is emitted to the untrusted peer.
    with pytest.raises(ValueError, match="untrusted host"):
        scheduler.update_state_after_alloc(
            request,
            SimpleNamespace(get_block_ids=lambda: ([1, 2],)),
            0,
        )

    assert sent == []


def _request_id_with_prefill_host(host):
    """Router request_id whose embedded *prefill* peer host is ``host``.

    Consumer-side resolution (``is_producer=False``) matches ``_PREFILL_ZMQ_RE``
    and parses the prefill zmq_address, so ``host`` becomes the
    request_id-derived ``remote_host`` (notify port 6100).
    """
    return (
        f"___prefill_addr_host:{host},handshake:6301,notify:6100"
        "___decode_addr_host:decode-a,handshake:6301,notify:6100"
        "_0123456789abcdef0123456789abcdef"
    )


def test_request_id_trusted_hosts_empty_when_unconfigured():
    # Opt-in: without an explicit trusted_remote_hosts the allowlist is empty,
    # which makes the request_id-derived validate a no-op (default flow intact).
    config = KVTransferConfig(kv_connector_extra_config={})
    assert get_moriio_request_id_trusted_hosts(config, ["local-1"]) == frozenset()


def test_request_id_trusted_hosts_union_node_hosts_when_configured():
    # Configured: explicit peers UNION this instance's own node_hosts (the local
    # host is trivially safe to accept from a request_id).
    config = KVTransferConfig(
        kv_connector_extra_config={"trusted_remote_hosts": "prefill-a"}
    )
    assert get_moriio_request_id_trusted_hosts(config, ["local-1"]) == frozenset(
        {"prefill-a", "local-1"}
    )


def test_add_new_req_rejects_untrusted_request_id_host():
    # No direct remote_host -> host is resolved from the client-controllable
    # request_id; with a configured allowlist an untrusted host must be rejected.
    metadata = MoRIIOConnectorMetadata(request_id_trusted_hosts={"prefill-a"})

    with pytest.raises(ValueError, match="untrusted host"):
        metadata.add_new_req(
            request_id=_request_id_with_prefill_host("evil.example"),
            local_block_ids=[1],
            kv_transfer_params={"transfer_id": "tx-1"},
        )

    assert metadata.reqs_to_recv == {}


def test_add_new_req_accepts_trusted_request_id_host():
    metadata = MoRIIOConnectorMetadata(request_id_trusted_hosts={"prefill-a"})
    request_id = _request_id_with_prefill_host("prefill-a")

    metadata.add_new_req(
        request_id=request_id,
        local_block_ids=[1],
        kv_transfer_params={"transfer_id": "tx-1", "remote_block_ids": [2]},
    )

    # A request_id host inside the allowlist survives and is registered.
    assert metadata.reqs_to_recv[request_id].remote_host == "prefill-a"


def test_add_new_req_skips_request_id_validation_when_unconfigured():
    # No-regression: unconfigured allowlist (empty) -> the request_id-derived
    # host is NOT validated, so an arbitrary embedded host flows through exactly
    # as it did before the opt-in guard existed.
    metadata = MoRIIOConnectorMetadata()
    request_id = _request_id_with_prefill_host("evil.example")

    metadata.add_new_req(
        request_id=request_id,
        local_block_ids=[1],
        kv_transfer_params={"transfer_id": "tx-1"},
    )

    assert metadata.reqs_to_recv[request_id].remote_host == "evil.example"


def test_scheduler_release_rejects_untrusted_request_id_host(monkeypatch):
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.trusted_remote_hosts = frozenset({"prefill-a"})
    scheduler._request_id_trusted_hosts = frozenset({"prefill-a"})
    scheduler.tp_size = 1
    sent = []
    monkeypatch.setattr(
        scheduler,
        "_send_transfer_release",
        lambda *args, **kwargs: sent.append((args, kwargs)),
    )

    # remote_host omitted -> host recovered from the request_id; an untrusted
    # host must fail closed before any release dial leaves the box.
    with pytest.raises(ValueError, match="untrusted host"):
        scheduler._release_write_prefill_blocks(
            _request_id_with_prefill_host("evil.example"),
            {"transfer_id": "tx"},
        )

    assert sent == []


def test_scheduler_release_accepts_trusted_request_id_host(monkeypatch):
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.trusted_remote_hosts = frozenset({"prefill-a"})
    scheduler._request_id_trusted_hosts = frozenset({"prefill-a"})
    scheduler.tp_size = 1
    sent = []
    monkeypatch.setattr(
        scheduler,
        "_send_transfer_release",
        lambda transfer_id, host, port: sent.append((transfer_id, host, port)),
    )

    scheduler._release_write_prefill_blocks(
        _request_id_with_prefill_host("prefill-a"),
        {"transfer_id": "tx"},
    )

    # A request_id host inside the allowlist reaches the release dial.
    assert sent == [("tx", "prefill-a", 6100)]


def test_scheduler_release_skips_request_id_validation_when_unconfigured(monkeypatch):
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    # Effective direct-host trust list is populated (its unconfigured default is
    # node_hosts), but the request_id allowlist is empty. The request_id host
    # must be checked against the EMPTY allowlist, not the effective one -- else
    # the opt-in default flow would false-reject legitimate peers.
    scheduler.trusted_remote_hosts = frozenset({"prefill-a"})
    scheduler._request_id_trusted_hosts = frozenset()
    scheduler.tp_size = 1
    sent = []
    monkeypatch.setattr(
        scheduler,
        "_send_transfer_release",
        lambda transfer_id, host, port: sent.append((transfer_id, host, port)),
    )

    scheduler._release_write_prefill_blocks(
        _request_id_with_prefill_host("evil.example"),
        {"transfer_id": "tx"},
    )

    # Unconfigured request_id trust -> arbitrary embedded host flows through.
    assert sent == [("tx", "evil.example", 6100)]


def test_scheduler_update_state_rejects_untrusted_request_id_host(monkeypatch):
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.mode = MoRIIOMode.WRITE
    scheduler.tp_size = 2
    scheduler.dp_rank = 0
    scheduler.transfer_id_to_request_id = {}
    scheduler.request_id_to_transfer_id = {}
    scheduler.trusted_remote_hosts = frozenset({"prefill-a"})
    scheduler._request_id_trusted_hosts = frozenset({"prefill-a"})
    sent = []
    monkeypatch.setattr(
        scheduler,
        "send_notify_block",
        lambda **kwargs: sent.append(kwargs),
    )
    request = SimpleNamespace(
        request_id=_request_id_with_prefill_host("evil.example"),
        kv_transfer_params={"transfer_id": "tx-1", "do_remote_prefill": True},
    )

    # remote_host omitted -> resolved from request_id; the untrusted host must be
    # rejected before any notify reaches the peer.
    with pytest.raises(ValueError, match="untrusted host"):
        scheduler.update_state_after_alloc(
            request,
            SimpleNamespace(get_block_ids=lambda: ([1, 2],)),
            0,
        )

    assert sent == []


def test_scheduler_update_state_skips_request_id_validation_when_unconfigured(
    monkeypatch,
):
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.mode = MoRIIOMode.WRITE
    scheduler.tp_size = 1
    scheduler.dp_rank = 0
    scheduler.transfer_id_to_request_id = {}
    scheduler.request_id_to_transfer_id = {}
    scheduler.trusted_remote_hosts = frozenset({"prefill-a"})
    scheduler._request_id_trusted_hosts = frozenset()
    sent = []
    monkeypatch.setattr(
        scheduler,
        "send_notify_block",
        lambda **kwargs: sent.append(kwargs),
    )
    request = SimpleNamespace(
        request_id=_request_id_with_prefill_host("evil.example"),
        kv_transfer_params={"transfer_id": "tx-1", "do_remote_prefill": True},
    )

    # No-regression: unconfigured request_id trust -> arbitrary embedded host is
    # accepted and the notify path proceeds to the resolved peer.
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda: ([1, 2],)),
        0,
    )

    assert [kw["host"] for kw in sent] == ["evil.example"]
