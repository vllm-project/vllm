# SPDX-License-Identifier: Apache-2.0
"""mp-server-side coordinator registration helpers.

These functions are the integration point an MP server uses to join the
coordinator. They send plain HTTP requests with a caller-provided
``httpx.AsyncClient`` (the MP server's generic client) and use the shared
:mod:`schemas` models -- there is no dedicated client object, mirroring how the
coordinator just calls mp endpoints directly.

The MP server's FastAPI lifespan launches :func:`keep_registered` as a task and
cancels it on shutdown. It is best-effort: registration/heartbeat failures are
logged and retried, and it never takes the MP server down.
"""

# Standard
import asyncio
import contextlib

# Third Party
from pydantic import ValidationError
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.schemas import RegisterRequest, RegisterResponse
from lmcache.v1.rpc_utils import get_ip

logger = init_logger(__name__)

_DEFAULT_HEARTBEAT_INTERVAL = 5.0


async def register(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    http_port: int,
    advertise_ip: str,
    instance_id: str = "",
    p2p_advertised_url: str = "",
    mq_port: int = 0,
) -> str:
    """Register an MP server with the coordinator and return its id.

    Args:
        client: The HTTP client to send with.
        base_url: Coordinator base URL (e.g. ``http://host:9300``).
        http_port: This MP server's HTTP port to advertise.
        advertise_ip: IP the coordinator should reach this server at.
        instance_id: Desired id; empty lets the coordinator assign one.
        p2p_advertised_url: Transfer-channel URL to advertise for P2P. Empty
            when P2P is disabled.
        mq_port: Port of this server's ZMQ message-queue server for P2P lookup
            RPCs. 0 when P2P is disabled.

    Returns:
        The registered instance id (coordinator-assigned if ``instance_id`` was
        empty).

    Raises:
        httpx.HTTPError: If the request fails or returns a non-2xx status.
    """
    body = RegisterRequest(
        instance_id=instance_id,
        ip=advertise_ip,
        http_port=http_port,
        p2p_advertised_url=p2p_advertised_url,
        mq_port=mq_port,
    )
    response = await client.post(f"{base_url}/instances", json=body.model_dump())
    response.raise_for_status()
    return RegisterResponse.model_validate(response.json()).instance_id


async def keep_registered(
    client: httpx.AsyncClient,
    coordinator_url: str,
    *,
    http_port: int,
    instance_id: str = "",
    advertise_ip: str = "",
    heartbeat_interval: float = _DEFAULT_HEARTBEAT_INTERVAL,
    p2p_advertised_url: str = "",
    mq_port: int = 0,
) -> None:
    """Register, heartbeat on a timer, and deregister on cancellation.

    Run as an asyncio task on the MP server's event loop and cancelled on
    shutdown. Resilient: transient failures (network blips, 5xx, malformed or
    version-skewed responses) are logged and retried on the next tick while the
    current identity is kept, so a down coordinator never stops the MP server
    and never spawns a duplicate registration. The identity is rebuilt only on
    an explicit ``404`` (the coordinator forgot us) or after a failed initial
    registration.

    Args:
        client: The HTTP client to send with.
        coordinator_url: Coordinator base URL.
        http_port: This MP server's HTTP port to advertise.
        instance_id: Desired id; empty lets the coordinator assign one.
        advertise_ip: IP the coordinator should reach this server at; defaults to
            the machine's outbound IP.
        heartbeat_interval: Seconds between heartbeats.
        p2p_advertised_url: Transfer-channel URL to advertise for P2P. Empty
            when P2P is disabled.
        mq_port: Port of this server's ZMQ message-queue server for P2P lookup
            RPCs. 0 when P2P is disabled.
    """
    base_url = coordinator_url.rstrip("/")
    ip = advertise_ip or get_ip()
    assigned_id: str | None = None
    try:
        while True:
            try:
                if assigned_id is None:
                    assigned_id = await register(
                        client,
                        base_url,
                        http_port=http_port,
                        advertise_ip=ip,
                        instance_id=instance_id,
                        p2p_advertised_url=p2p_advertised_url,
                        mq_port=mq_port,
                    )
                    logger.info("Registered with coordinator as %s", assigned_id)
                else:
                    response = await client.put(
                        f"{base_url}/instances/{assigned_id}/heartbeat"
                    )
                    if response.status_code == 404:
                        # Coordinator forgot us: drop id, re-register next tick.
                        logger.info(
                            "Coordinator no longer knows %s; re-registering",
                            assigned_id,
                        )
                        assigned_id = None
                    else:
                        response.raise_for_status()
            except (httpx.HTTPError, ValueError, ValidationError) as e:
                # Transient: keep the current id so the next tick retries and
                # shutdown can still deregister. ValueError/ValidationError cover
                # bad JSON or schema skew; CancelledError still propagates.
                logger.warning("Coordinator registration/heartbeat failed: %s", e)
            await asyncio.sleep(heartbeat_interval)
    finally:
        if assigned_id is not None:
            with contextlib.suppress(httpx.HTTPError, asyncio.CancelledError):
                await client.delete(f"{base_url}/instances/{assigned_id}")
