# SPDX-License-Identifier: Apache-2.0
"""REST resource for fleet membership: the ``/instances`` collection.

mp servers register themselves, heartbeat, and deregister here; operators can
list the fleet. Endpoints operate directly on the shared ``InstanceRegistry``
reached via the :class:`CoordinatorContext` -- membership is thin enough to need
no service layer.
"""

# Standard
from typing import Any
import time
import uuid

# Third Party
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.registry import MPInstance
from lmcache.v1.mp_coordinator.schemas import (
    HeartbeatResponse,
    RegisterRequest,
    RegisterResponse,
)

logger = init_logger(__name__)

router = APIRouter()


@router.post("/instances")
async def register_instance(
    body: RegisterRequest, request: Request
) -> RegisterResponse:
    """Register (or re-register) an mp server.

    The body is validated by :class:`RegisterRequest`; FastAPI returns 422 on a
    malformed body. An empty ``instance_id`` is replaced with a generated one,
    returned in the response so the caller learns its assigned id.

    Returns:
        A :class:`RegisterResponse` carrying the (possibly generated) id.
    """
    instance_id = body.instance_id or f"mp-{uuid.uuid4().hex}"
    # Wall-clock registration_time for display; monotonic last_heartbeat_time for
    # NTP-safe stale detection (see registry.stale). register() does the
    # exists-check and write under one lock, so the re_registered flag is correct
    # even under concurrent registrations of the same id.
    re_registered = get_context(request).registry.register(
        MPInstance(
            instance_id=instance_id,
            ip=body.ip,
            http_port=body.http_port,
            registration_time=time.time(),
            last_heartbeat_time=time.monotonic(),
            metadata=dict(body.metadata),
            p2p_advertised_url=body.p2p_advertised_url,
            mq_port=body.mq_port,
        )
    )
    logger.info("Registered instance %s at %s:%s", instance_id, body.ip, body.http_port)
    return RegisterResponse(instance_id=instance_id, re_registered=re_registered)


@router.put("/instances/{instance_id}/heartbeat")
async def heartbeat(instance_id: str, request: Request) -> Any:
    """Record a heartbeat for an instance.

    Returns:
        ``{"instance_id": str}`` with 200 if known, or a 404 JSON error if the
        instance is unknown (the caller should re-register via ``POST
        /instances``).
    """
    if get_context(request).registry.update_heartbeat(instance_id, time.monotonic()):
        return HeartbeatResponse(instance_id=instance_id)
    return JSONResponse(
        status_code=404,
        content={"error": f"unknown instance {instance_id}; re-register"},
    )


@router.delete("/instances/{instance_id}")
async def deregister_instance(instance_id: str, request: Request) -> Response:
    """Deregister an mp server.

    Idempotent: returns 204 whether or not the instance was registered.

    Returns:
        An empty 204 response.
    """
    if get_context(request).registry.deregister(instance_id) is not None:
        logger.info("Deregistered instance %s", instance_id)
    else:
        logger.info("Instance %s not registered, skipping deregistration", instance_id)
    return Response(status_code=204)


@router.get("/instances")
async def list_instances(request: Request) -> Any:
    """List all registered mp servers.

    Returns:
        ``{"instances": [ {instance_id, ip, http_port, ...}, ... ]}``.
    """
    instances = [
        {
            "instance_id": instance.instance_id,
            "ip": instance.ip,
            "http_port": instance.http_port,
            "registration_time": instance.registration_time,
            "metadata": instance.metadata,
            "p2p_advertised_url": instance.p2p_advertised_url,
            "mq_port": instance.mq_port,
        }
        for instance in get_context(request).registry.all_instances()
    ]
    return {"instances": instances}
