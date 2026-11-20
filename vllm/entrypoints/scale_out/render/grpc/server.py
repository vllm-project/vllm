# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gRPC listener for `vllm launch render --grpc-port`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    import grpc

    from vllm.entrypoints.scale_out.render.serving import ServingRender

logger = init_logger(__name__)

RENDER_SERVICE_NAME = "vllm.Render"


def _import_grpc():
    try:
        import grpc
        from grpc_health.v1 import health, health_pb2, health_pb2_grpc

        from vllm.entrypoints.scale_out.render.grpc import render_pb2_grpc
        from vllm.entrypoints.scale_out.render.grpc.servicer import RenderServicer
    except ImportError as e:
        raise ImportError(
            "--grpc-port needs grpcio, grpcio-health-checking and the "
            "generated vllm.Render stubs. Install grpcio and "
            "grpcio-health-checking, and run "
            "`python -m vllm.entrypoints.scale_out.render.grpc.compile_protos` "
            f"in a source checkout ({e})"
        ) from e
    return grpc, health, health_pb2, health_pb2_grpc, render_pb2_grpc, RenderServicer


async def start_render_grpc_server(
    serving_render: ServingRender, host: str, port: int
) -> grpc.aio.Server:
    """Start `vllm.Render` and `grpc.health.v1` on `host:port`.

    Shares `serving_render` with the HTTP server running in the same event loop.
    """
    grpc, health, health_pb2, health_pb2_grpc, render_pb2_grpc, RenderServicer = (
        _import_grpc()
    )

    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ]
    )
    render_pb2_grpc.add_RenderServicer_to_server(RenderServicer(serving_render), server)

    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    address = f"{host}:{port}"
    if server.add_insecure_port(address) == 0:
        raise OSError(f"failed to bind the gRPC render server to {address}")
    await server.start()

    serving = health_pb2.HealthCheckResponse.SERVING
    await health_servicer.set("", serving)
    await health_servicer.set(RENDER_SERVICE_NAME, serving)
    logger.info("Serving gRPC %s on %s", RENDER_SERVICE_NAME, address)
    return server
