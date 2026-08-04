# SPDX-License-Identifier: Apache-2.0
"""
API endpoint for monitoring periodic threads.
"""

# Standard
from typing import Optional

# Third Party
from fastapi import APIRouter, Query
from starlette.requests import Request
from starlette.responses import JSONResponse

# First Party
from lmcache.v1.periodic_thread import PeriodicThreadRegistry, ThreadLevel

router = APIRouter()


@router.get("/periodic-threads")
async def get_periodic_threads(
    request: Request,
    level: Optional[str] = Query(
        None,
        description="Filter by thread level (critical, high, medium, low)",
    ),
    running_only: bool = Query(
        False,
        description="Only show running threads",
    ),
    active_only: bool = Query(
        False,
        description="Only show active threads",
    ),
):
    """
    Get information about registered periodic threads.

    Returns a summary of all periodic threads including:
    - Total, running, and active counts by level
    - Individual thread status with last run time and summary
    """
    registry = PeriodicThreadRegistry.get_instance()

    # Get all threads
    if level:
        try:
            thread_level = ThreadLevel(level.lower())
            threads = registry.get_by_level(thread_level)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Invalid level: {level}. "
                    f"Valid values: critical, high, medium, low"
                },
            )
    else:
        threads = registry.get_all()

    # Apply filters
    if running_only:
        threads = [t for t in threads if t.is_running]
    if active_only:
        threads = [t for t in threads if t.is_active]

    # Build response
    thread_statuses = [t.get_status() for t in threads]

    # Get summary
    summary = registry.get_summary()

    return JSONResponse(
        content={
            "summary": {
                "total_count": summary["total_count"],
                "running_count": summary["running_count"],
                "active_count": summary["active_count"],
                "by_level": summary["by_level"],
            },
            "threads": thread_statuses,
        }
    )


@router.get("/periodic-threads/{thread_name}")
async def get_periodic_thread(
    request: Request,
    thread_name: str,
):
    """
    Get detailed information about a specific periodic thread.
    """
    registry = PeriodicThreadRegistry.get_instance()
    thread = registry.get(thread_name)

    if thread is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Thread not found: {thread_name}"},
        )

    return JSONResponse(content=thread.get_status())


@router.get("/periodic-threads-health")
async def get_periodic_threads_health(request: Request):
    """
    Quick health check for periodic threads.

    Returns:
    - healthy: True if all critical/high level threads are active
    - unhealthy_threads: List of inactive critical/high threads
    """
    registry = PeriodicThreadRegistry.get_instance()

    unhealthy_threads = []

    # Check critical and high level threads
    for level in [ThreadLevel.CRITICAL, ThreadLevel.HIGH]:
        for thread in registry.get_by_level(level):
            if thread.is_running and not thread.is_active:
                unhealthy_threads.append(
                    {
                        "name": thread.name,
                        "level": thread.level.value,
                        "last_run_ago": thread.get_status().get("last_run_ago"),
                        "interval": thread.interval,
                    }
                )

    return JSONResponse(
        content={
            "healthy": len(unhealthy_threads) == 0,
            "unhealthy_count": len(unhealthy_threads),
            "unhealthy_threads": unhealthy_threads,
        }
    )
