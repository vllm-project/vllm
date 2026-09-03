# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guards the ping-tolerance channel args of the vLLM gRPC server.

A non-streaming ``Generate`` sends no DATA frames while decoding, so the
server's tolerance for client PINGs is the only thing keeping a long request
alive. gRPC Core silently ignores unknown channel-arg keys, so a misspelt key
cannot be caught at runtime; these tests pin the exact strings from
``include/grpc/impl/channel_arg_names.h``.
"""

import pytest

grpc_server = pytest.importorskip("vllm.entrypoints.grpc_server")


@pytest.fixture
def options() -> dict[str, int | bool]:
    opts = grpc_server.grpc_server_options()
    assert len(opts) == len(dict(opts)), "duplicate channel-arg keys"
    return dict(opts)


def test_min_ping_interval_uses_grpc_core_key(options):
    assert options["grpc.http2.min_ping_interval_without_data_ms"] == 10000
    assert "grpc.http2.min_recv_ping_interval_without_data_ms" not in options


def test_ping_strikes_disabled(options):
    assert options["grpc.http2.max_ping_strikes"] == 0


def test_keepalive_permitted_without_calls(options):
    assert options["grpc.keepalive_permit_without_calls"] is True
