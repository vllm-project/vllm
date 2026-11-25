# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from unittest.mock import Mock

import pytest
from fastapi import Request

from vllm.config.kv_events import KVEventsConfig
from vllm.entrypoints.serve.instrumentator.basic import kv_event_sources
from vllm.v1.engine import PeerLookupInfo

PEER_LOOKUP = PeerLookupInfo(
    enabled=True,
    engine_id="engine-a",
    side_channel_host="10.0.0.2",
    side_channel_port=5600,
    tp_size=1,
    dcp_size=1,
    pp_size=1,
    connector="NixlConnector",
    connector_version="17",
    layout_fingerprint="abc",
)


def make_request(
    sources: dict[int, KVEventsConfig], peer_lookups: dict[int, PeerLookupInfo]
) -> Mock:
    request = Mock(spec=Request)
    client = Mock()
    client.get_kv_event_sources.return_value = sources
    client.get_peer_lookup_infos.return_value = peer_lookups
    request.app.state.engine_client = client
    return request


@pytest.mark.asyncio
async def test_kv_event_sources_without_peer_lookup_is_unchanged():
    config = KVEventsConfig(
        enable_kv_cache_events=True, publisher="zmq", endpoint="tcp://10.0.0.2:5557"
    )
    result = await kv_event_sources(make_request({0: config}, {}))
    assert result == {0: dataclasses.asdict(config)}
    assert "peer_lookup" not in result[0]


@pytest.mark.asyncio
async def test_kv_event_sources_adds_peer_lookup_only_where_present():
    config = KVEventsConfig(
        enable_kv_cache_events=True, publisher="zmq", endpoint="tcp://10.0.0.2:5557"
    )
    result = await kv_event_sources(
        make_request({0: config, 1: config}, {1: PEER_LOOKUP})
    )
    assert "peer_lookup" not in result[0]
    assert result[1] == {
        **dataclasses.asdict(config),
        "peer_lookup": dataclasses.asdict(PEER_LOOKUP),
    }
    assert set(result[1]["peer_lookup"]) == {
        "enabled",
        "engine_id",
        "side_channel_host",
        "side_channel_port",
        "tp_size",
        "dcp_size",
        "pp_size",
        "connector",
        "connector_version",
        "layout_fingerprint",
    }


@pytest.mark.asyncio
async def test_kv_event_sources_peer_lookup_without_kv_events():
    result = await kv_event_sources(make_request({}, {0: PEER_LOOKUP}))
    assert result == {
        0: {
            **dataclasses.asdict(KVEventsConfig()),
            "peer_lookup": dataclasses.asdict(PEER_LOOKUP),
        }
    }
    assert result[0]["enable_kv_cache_events"] is False


@pytest.mark.asyncio
async def test_kv_event_sources_render_only_server():
    request = Mock(spec=Request)
    request.app.state.engine_client = None
    assert await kv_event_sources(request) == {}
