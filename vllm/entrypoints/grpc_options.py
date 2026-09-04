# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Channel arguments for the vLLM gRPC server.

Kept free of optional-extra imports so the values can be unit-tested without
``smg-grpc-servicer`` installed.
"""


def grpc_server_options() -> list[tuple[str, int | bool]]:
    """Return the channel arguments passed to ``grpc.aio.server``.

    A non-streaming ``Generate`` sends nothing while the engine decodes, so
    every client keepalive or BDP-probe PING received in that window is judged
    against ``min_ping_interval_without_data_ms``. A PING arriving sooner than
    that floor is a strike. Strikes reset only when the server writes a frame,
    which never happens mid-decode, and once they exceed ``max_ping_strikes``
    gRPC Core sends ``GOAWAY(ENHANCE_YOUR_CALM, "too_many_pings")`` and fails
    every in-flight RPC on the connection.

    Fixing the interval key (the old spelling was silently ignored, leaving
    the 300s default in force) makes a 30s client keepalive a good PING.
    ``max_ping_strikes = 0`` disables the check outright and is the setting
    that actually protects us: it also covers sub-floor BDP probes and, with
    ``keepalive_permit_without_calls``, keepalives on an idle connection,
    where gRPC Core applies a fixed two-hour floor regardless of the option.
    With strikes disabled the interval value is inert and documents intent.
    """
    return [
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1),
        # GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS; the string
        # value has no "recv".
        ("grpc.http2.min_ping_interval_without_data_ms", 10000),
        ("grpc.http2.max_ping_strikes", 0),
        ("grpc.keepalive_permit_without_calls", True),
    ]
