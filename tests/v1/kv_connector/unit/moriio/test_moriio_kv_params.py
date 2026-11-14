# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject malformed request metadata before MoRIIO schedules a remote load."""

from copy import deepcopy

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import make_kv_cache_group
from vllm import SamplingParams
from vllm.config import CacheConfig, DeviceConfig, KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_common
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOConnectorMetadata,
    get_role,
    set_role,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnector,
)
from vllm.exceptions import VLLMValidationError
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
)
from vllm.v1.request import Request

PEER_ADDRESS = "host:127.0.0.1,handshake:6301,notify:61005"
ROUTER_REQUEST_ID = (
    f"___prefill_addr_{PEER_ADDRESS}___decode_addr_{PEER_ADDRESS}_{'0' * 32}"
)


@pytest.fixture
def connector():
    original_role = get_role()
    config = VllmConfig(
        device_config=DeviceConfig("cpu"),
        cache_config=CacheConfig(block_size=32),
        kv_transfer_config=KVTransferConfig(
            kv_connector="MoRIIOConnector",
            kv_role="kv_consumer",
            kv_connector_extra_config={
                "read_mode": False,
                "host_ip": "127.0.0.1",
                "handshake_port": 6301,
                "notify_port": 61005,
            },
        ),
    )
    spec = AttentionSpec(
        block_size=32, num_kv_heads=4, head_size=64, dtype=torch.float16
    )
    cache = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[make_kv_cache_group(["layer0"], spec)],
    )
    instance = MoRIIOConnector(config, KVConnectorRole.SCHEDULER, cache)
    try:
        yield instance
    finally:
        instance.shutdown()
        set_role(original_role)


def make_request(params, request_id="request"):
    return Request(
        request_id=request_id,
        prompt_token_ids=[1] * 32,
        sampling_params=SamplingParams(
            max_tokens=1, extra_args={"kv_transfer_params": params}
        ),
        pooling_params=None,
    )


@pytest.mark.parametrize(
    "field", ["remote_dp_size", "remote_dp_size_local", "remote_dp_rank"]
)
@pytest.mark.parametrize("value", ["x", [1], float("inf")])
def test_invalid_dp_integer_is_rejected_before_remote_load(connector, field, value):
    request = make_request({"do_remote_prefill": True, field: value}, ROUTER_REQUEST_ID)
    original_params = deepcopy(request.kv_transfer_params)

    with pytest.raises(VLLMValidationError, match=field) as error:
        connector.validate_request(request)

    assert error.value.parameter == "kv_transfer_params"
    assert request.kv_transfer_params == original_params
    assert connector.connector_scheduler.request_id_to_transfer_id == {}


@pytest.mark.parametrize("params", ["x", [1], 1])
def test_non_dictionary_metadata_is_rejected(connector, params):
    with pytest.raises(VLLMValidationError, match="dictionary"):
        connector.validate_request(make_request(params))


@pytest.mark.parametrize(
    "peer_params",
    [
        {},
        {"remote_host": "127.0.0.1"},
        {"remote_notify_port": 61005},
        {"remote_host": "", "remote_notify_port": 61005},
        {"remote_host": ["127.0.0.1"], "remote_notify_port": 61005},
        {"remote_host": "127.0.0.1", "remote_notify_port": "x"},
        {"remote_host": "127.0.0.1", "remote_notify_port": float("inf")},
        {"remote_host": "127.0.0.1", "remote_notify_port": 0},
        {"remote_host": "127.0.0.1", "remote_notify_port": 65536},
    ],
)
def test_missing_or_malformed_write_peer_is_rejected(connector, peer_params):
    request = make_request({"do_remote_prefill": True, **peer_params})
    with pytest.raises(VLLMValidationError, match="remote_(host|notify_port)"):
        connector.validate_request(request)
    assert connector.connector_scheduler.request_id_to_transfer_id == {}


@pytest.mark.parametrize(
    "dp_params",
    [
        {},
        {
            "remote_dp_size": None,
            "remote_dp_size_local": None,
            "remote_dp_rank": None,
        },
        {"remote_dp_size": "2", "remote_dp_size_local": None, "remote_dp_rank": "1"},
    ],
)
@pytest.mark.parametrize("request_id", ["sidecar-request", ROUTER_REQUEST_ID])
def test_valid_router_and_sidecar_metadata_preserves_remote_load(
    connector, dp_params, request_id
):
    params = {
        "do_remote_prefill": True,
        "transfer_id": "transfer",
        "remote_engine_id": "prefill",
        "remote_block_ids": [[1]],
        "remote_host": "127.0.0.1",
        "remote_handshake_port": 6301,
        "remote_notify_port": "61005",
        **dp_params,
    }
    request = make_request(params, request_id)
    original_params = deepcopy(params)
    connector.validate_request(request)

    assert request.kv_transfer_params == original_params
    assert connector.get_num_new_matched_tokens(request, 0) == (32, True)
    metadata = MoRIIOConnectorMetadata()
    metadata.add_new_req(request_id, [[2]], params)
    registered = metadata.reqs_to_recv[request_id]
    assert registered.remote_dp_size == int(dp_params.get("remote_dp_size") or 1)
    assert registered.remote_dp_size_local == registered.remote_dp_size
    assert registered.remote_dp_rank == int(dp_params.get("remote_dp_rank") or 0)
    assert registered.remote_block_ids == [[1]]


def test_router_address_takes_precedence_over_sidecar_fields(connector):
    params = {
        "do_remote_prefill": True,
        "remote_host": None,
        "remote_notify_port": "ignored",
    }
    connector.validate_request(make_request(params, ROUTER_REQUEST_ID))
    assert moriio_common.get_moriio_write_notify_address(ROUTER_REQUEST_ID, params) == (
        "127.0.0.1",
        61005,
    )


def test_write_cleanup_preserves_explicit_address_precedence(connector):
    params = {
        "do_remote_prefill": True,
        "remote_host": "127.0.0.2",
        "remote_notify_port": 61006,
    }
    connector.validate_request(make_request(params, ROUTER_REQUEST_ID))
    assert moriio_common.get_moriio_write_notify_address(ROUTER_REQUEST_ID, params) == (
        "127.0.0.1",
        61005,
    )
    assert moriio_common.get_moriio_write_notify_address(
        ROUTER_REQUEST_ID, params, prefer_explicit=True
    ) == ("127.0.0.2", 61006)


def test_valid_router_address_does_not_hide_invalid_release_address(connector):
    request = make_request(
        {
            "do_remote_prefill": True,
            "remote_host": "127.0.0.1",
            "remote_notify_port": "x",
        },
        ROUTER_REQUEST_ID,
    )
    with pytest.raises(VLLMValidationError, match="remote_notify_port"):
        connector.validate_request(request)


def test_malformed_router_address_is_not_hidden_by_sidecar_fallback(connector):
    request_id = ROUTER_REQUEST_ID.replace("notify:61005", "notify:x")
    request = make_request(
        {
            "do_remote_prefill": True,
            "remote_host": "127.0.0.1",
            "remote_notify_port": 61005,
        },
        request_id,
    )
    with pytest.raises(VLLMValidationError, match="Malformed zmq_address"):
        connector.validate_request(request)


@pytest.mark.parametrize("params", [None, {}])
def test_local_request_does_not_need_peer_metadata(connector, params):
    request = make_request(params)
    connector.validate_request(request)
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
