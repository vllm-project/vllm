# SPDX-License-Identifier: Apache-2.0
"""Basic-info endpoints: liveness, health, status, and version.

These are the small, dependency-light routes used to inspect that the
server is up and what it is running:

- ``GET /`` — static liveness payload (does not touch the engine).
- ``GET /healthcheck`` — Kubernetes liveness/readiness probe.
- ``GET /status`` — detailed internal state of all MP components.
- ``GET /version``, ``/lmc_version``, ``/commit_id`` — version descriptors,
  re-exposed from the shared ``internal_api_server.vllm`` version router.
"""

# Standard
from typing import Any

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.internal_api_server.vllm.version_api import router as _version_router

router = APIRouter()

# Re-expose the shared version routes (/version, /lmc_version, /commit_id)
# as part of the basic-info group.
router.include_router(_version_router)


@router.get("/")
async def root() -> dict[str, str]:
    """
    Basic liveness check endpoint.
    Returns:
        dict: A dictionary containing the status and service name.
    """
    return {"status": "ok", "service": "LMCache HTTP API"}


@router.get("/healthcheck")
async def healthcheck(request: Request) -> Any:
    """
    Health check endpoint for k8s liveness/readiness probes.

    Checks:
        - HTTP server is alive (implicit: if you get a response)
        - Cache engine is alive
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": "engine not initialized",
            },
        )

    return {"status": "healthy"}


@router.get("/status")
async def status(request: Request) -> Any:
    """
    Detailed status endpoint for inspecting internal state
    of all MP components (L1 cache, L2 adapters, controllers,
    sessions).
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "engine not initialized"},
        )
    return engine.report_status()
