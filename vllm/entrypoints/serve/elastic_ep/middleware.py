# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Awaitable

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# Process-local states used to pause serving during scaling and after this rank
# has been removed from the external-LB topology. A retired API process remains
# available for health and scaling-status queries until its orchestrator exits it.
_scaling_elastic_ep = False
_elastic_ep_rank_retired = False
_SCALING_OBSERVABILITY_PATHS = frozenset({"/health", "/is_scaling_elastic_ep"})


def get_scaling_elastic_ep():
    return _scaling_elastic_ep


def set_scaling_elastic_ep(value):
    global _scaling_elastic_ep
    _scaling_elastic_ep = value


def get_elastic_ep_rank_retired():
    return _elastic_ep_rank_retired


def set_elastic_ep_rank_retired(value):
    global _elastic_ep_rank_retired
    _elastic_ep_rank_retired = value


class ScalingMiddleware:
    """
    Middleware that pauses serving while the model is scaling or after the
    local external-LB rank has been retired.

    Health and scaling-status requests remain available so an external
    orchestrator can observe the operation and remove a retired process.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] != "http":
            return self.app(scope, receive, send)

        # Keep observability available while serving is paused or this rank is
        # waiting for the external orchestrator to remove its process.
        rank_retired = get_elastic_ep_rank_retired()
        serving_paused = get_scaling_elastic_ep() or rank_retired
        if serving_paused and scope["path"] not in _SCALING_OBSERVABILITY_PATHS:
            error = (
                "This data-parallel rank has been retired and is awaiting shutdown."
                if rank_retired
                else "The model is currently scaling. Please try again later."
            )
            response = JSONResponse(
                content={"error": error},
                status_code=503,
            )
            return response(scope, receive, send)

        return self.app(scope, receive, send)
