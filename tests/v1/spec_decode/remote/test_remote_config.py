# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RemoteDraftConfig field validation."""

import pytest
from pydantic import ValidationError

from vllm.config.speculative import RemoteDraftConfig


def test_defaults_are_production_safe():
    config = RemoteDraftConfig(endpoint="unix:///run/vllm/drafter.sock")
    assert config.transport == "auto"
    assert config.failure_policy == "target_only"
    assert config.request_timeout_ms > 0
    assert config.startup_timeout_ms > 0


def test_inline_transport_is_explicitly_supported():
    config = RemoteDraftConfig(
        endpoint="unix:///run/vllm/drafter.sock", transport="inline"
    )
    assert config.transport == "inline"


def test_endpoint_is_required():
    with pytest.raises(ValidationError):
        RemoteDraftConfig()


@pytest.mark.parametrize("endpoint", ["", "   "])
def test_empty_endpoint_rejected(endpoint):
    with pytest.raises(ValidationError):
        RemoteDraftConfig(endpoint=endpoint)


@pytest.mark.parametrize(
    "endpoint", ["http://127.0.0.1:5555", "tcp://127.0.0.1", "drafter.sock"]
)
def test_malformed_endpoint_rejected(endpoint):
    with pytest.raises(ValidationError, match="endpoint"):
        RemoteDraftConfig(endpoint=endpoint)


@pytest.mark.parametrize("transport", ["pinned_host", "zmq", "carrier_pigeon"])
def test_unimplemented_transport_rejected(transport):
    with pytest.raises(ValidationError):
        RemoteDraftConfig(endpoint="unix:///x.sock", transport=transport)


def test_nonpositive_timeout_rejected():
    with pytest.raises(ValidationError):
        RemoteDraftConfig(endpoint="unix:///x.sock", request_timeout_ms=0)


def test_cuda_ipc_requires_unix_socket():
    assert (
        RemoteDraftConfig(endpoint="unix:///x.sock", transport="cuda_ipc").transport
        == "cuda_ipc"
    )
    with pytest.raises(ValidationError, match="UNIX socket"):
        RemoteDraftConfig(endpoint="tcp://127.0.0.1:9999", transport="cuda_ipc")
