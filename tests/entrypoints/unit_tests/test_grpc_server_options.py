# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gRPC Core silently ignores unknown channel-arg keys, so a misspelt key
cannot be caught at runtime; these tests pin the exact strings. The options
live in a module with no optional-extra imports so they run in CI without
``smg-grpc-servicer``.
"""

import pytest

from vllm.entrypoints.grpc_options import grpc_server_options


@pytest.fixture
def options() -> dict[str, int | bool]:
    opts = grpc_server_options()
    assert len(opts) == len(dict(opts)), "duplicate channel-arg keys"
    return dict(opts)


def test_min_ping_interval_uses_grpc_core_key(options):
    assert options["grpc.http2.min_ping_interval_without_data_ms"] == 10000
    assert "grpc.http2.min_recv_ping_interval_without_data_ms" not in options


def test_ping_strikes_disabled(options):
    assert options["grpc.http2.max_ping_strikes"] == 0


def test_keepalive_permitted_without_calls(options):
    assert options["grpc.keepalive_permit_without_calls"] is True
