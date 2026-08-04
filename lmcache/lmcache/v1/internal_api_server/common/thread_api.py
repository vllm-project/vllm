# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import sys
import threading
import traceback

# Third Party
from fastapi import APIRouter, Query
from starlette.requests import Request
from starlette.responses import PlainTextResponse

router = APIRouter()


@router.get("/threads")
async def get_threads(
    request: Request,
    name: Optional[str] = Query(
        None, description="Filter by thread name (fuzzy match)"
    ),
    thread_id: Optional[int] = Query(None, description="Filter by thread ID"),
):
    """Return information about active threads with optional filtering"""
    threads = threading.enumerate()

    filtered_threads = []
    for t in threads:
        # Apply filters
        if name and name.lower() not in t.name.lower():
            continue
        if thread_id and t.ident != thread_id:
            continue
        filtered_threads.append(t)

    thread_info = []

    for t in filtered_threads:
        # Basic thread info with creation time
        info = f"Thread: {t}\n"

        # Get stack trace if available
        try:
            stack_frames = (
                sys._current_frames().get(t.ident) if t.ident is not None else None
            )
            if stack_frames:
                stack_trace = traceback.format_stack(stack_frames)
                info += "Stack trace:\n" + "".join(stack_trace)
            else:
                info += "No stack trace available\n"
        except AttributeError:
            info += "Stack trace unavailable\n"

        thread_info.append(info)

    # Add summary section
    summary = "\n\n=== Thread Summary ===\n"
    summary += f"Total threads: {len(filtered_threads)}\n"

    return PlainTextResponse(
        content="\n\n".join(thread_info) + summary, media_type="text/plain"
    )
