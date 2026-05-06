# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import KVTransferConfig
from vllm.v1.engine.kv_consumer_validation import validate_kv_consumer_request


def _make_request(kv_transfer_params=None):
    return SimpleNamespace(
        request_id="request-0",
        pooling_params=None,
        kv_transfer_params=kv_transfer_params,
    )


def _remote_prefill_params():
    return {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_engine_id": "prefill-engine",
        "remote_request_id": "prefill-request",
        "remote_host": "127.0.0.1",
        "remote_port": 14579,
        "remote_block_ids": [0, 1, 2],
    }


def _nixl_config(kv_role: str = "kv_consumer"):
    return KVTransferConfig(kv_connector="NixlConnector", kv_role=kv_role)


def _make_engine_core_request(kv_transfer_params=None):
    return SimpleNamespace(
        request_id="request-0",
        pooling_params=None,
        sampling_params=SimpleNamespace(
            extra_args={"kv_transfer_params": kv_transfer_params}
            if kv_transfer_params is not None
            else None
        ),
    )


def test_kv_consumer_rejects_missing_kv_transfer_params():
    request = _make_request()

    with pytest.raises(ValueError, match="requires non-empty kv_transfer_params"):
        validate_kv_consumer_request(request, _nixl_config())


@pytest.mark.parametrize(
    "kv_transfer_params",
    [
        {},
        {"do_remote_prefill": False},
        {"do_remote_decode": True, "do_remote_prefill": False},
    ],
)
def test_kv_consumer_rejects_non_remote_prefill_params(kv_transfer_params):
    request = _make_request(kv_transfer_params)

    with pytest.raises(ValueError, match="refusing local prefill"):
        validate_kv_consumer_request(request, _nixl_config())


@pytest.mark.parametrize(
    "missing_param",
    ["remote_engine_id", "remote_request_id", "remote_host", "remote_port"],
)
def test_nixl_kv_consumer_rejects_missing_remote_metadata(missing_param):
    request = _make_request(_remote_prefill_params())
    request.kv_transfer_params.pop(missing_param)

    with pytest.raises(ValueError, match=missing_param):
        validate_kv_consumer_request(request, _nixl_config())


@pytest.mark.parametrize("remote_block_ids", [None])
def test_nixl_kv_consumer_rejects_missing_remote_block_ids(remote_block_ids):
    request = _make_request(_remote_prefill_params())
    request.kv_transfer_params["remote_block_ids"] = remote_block_ids

    with pytest.raises(ValueError, match="remote_block_ids"):
        validate_kv_consumer_request(request, _nixl_config())


def test_nixl_kv_consumer_accepts_valid_remote_prefill_params():
    request = _make_request(_remote_prefill_params())

    validate_kv_consumer_request(request, _nixl_config())


def test_nixl_kv_consumer_accepts_empty_remote_block_ids_for_full_hit():
    request = _make_request(_remote_prefill_params())
    request.kv_transfer_params["remote_block_ids"] = []

    validate_kv_consumer_request(request, _nixl_config())


def test_kv_both_preserves_existing_decode_only_behavior():
    request = _make_request()

    validate_kv_consumer_request(request, _nixl_config(kv_role="kv_both"))


def test_kv_consumer_rejects_engine_core_request_without_remote_prefill():
    request = _make_engine_core_request()

    with pytest.raises(ValueError, match="requires non-empty kv_transfer_params"):
        validate_kv_consumer_request(request, _nixl_config())


def test_kv_consumer_accepts_engine_core_request_with_remote_prefill():
    request = _make_engine_core_request(_remote_prefill_params())

    validate_kv_consumer_request(request, _nixl_config())
