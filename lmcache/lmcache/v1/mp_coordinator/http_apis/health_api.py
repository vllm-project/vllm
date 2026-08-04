# SPDX-License-Identifier: Apache-2.0
"""Liveness endpoint for the mp coordinator (k8s probe)."""

# Standard
from typing import Any

# Third Party
from fastapi import APIRouter

router = APIRouter()


@router.get("/healthz")
async def healthz() -> Any:
    """Report coordinator liveness.

    Returns:
        ``{"status": "healthy"}``.
    """
    return {"status": "healthy"}
