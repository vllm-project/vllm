# SPDX-License-Identifier: Apache-2.0
# Third Party
from fastapi import APIRouter
from prometheus_client import REGISTRY, generate_latest
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# First Party
from lmcache.observability import reset_observability_metrics

router = APIRouter()


@router.get("/metrics")
async def get_metrics(request: Request):
    """
    Provide Prometheus metrics data
    """
    metrics_data = generate_latest(REGISTRY)
    return PlainTextResponse(content=metrics_data, media_type="text/plain")


@router.post("/metrics/reset")
async def reset_metrics():
    """
    Reset Prometheus metrics to their initial state.
    """
    reset_observability_metrics()
    return PlainTextResponse(content="ok", media_type="text/plain")
