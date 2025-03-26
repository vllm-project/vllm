# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import Counter, Gauge
from starlette.background import BackgroundTask, BackgroundTasks

http_error_counter = Counter(
    "http_error_count",
    "Total error count across requests with non 2xx response code",
)

server_load_gauge = Gauge(
    "server_load",
    "Total number of requests currently being processed by the server",
    multiprocess_mode="sum")


async def listen_for_disconnect(request: Request) -> None:
    """Returns if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    """Decorator that allows a route handler to be cancelled by client
    disconnections.

    This does _not_ use request.is_disconnected, which does not work with
    middleware. Instead this follows the pattern from
    starlette.StreamingResponse, which simultaneously awaits on two tasks- one
    to wait for an http disconnect message, and the other to do the work that we
    want done. When the first task finishes, the other is cancelled.

    A core assumption of this method is that the body of the request has already
    been read. This is a safe assumption to make for fastapi handlers that have
    already parsed the body of the request into a pydantic model for us.
    This decorator is unsafe to use elsewhere, as it will consume and throw away
    all incoming messages for the request while it looks for a disconnect
    message.

    In the case where a `StreamingResponse` is returned by the handler, this
    wrapper will stop listening for disconnects and instead the response object
    will start listening for disconnects.
    """

    # Functools.wraps is required for this wrapper to appear to fastapi as a
    # normal route handler, with the correct request type hinting.
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):

        # The request is either the second positional arg or `raw_request`
        request = args[1] if len(args) > 1 else kwargs["raw_request"]

        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))

        done, pending = await asyncio.wait([handler_task, cancellation_task],
                                           return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()

        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


def decrement_server_load(request: Request):
    request.app.state.server_load_metrics -= 1


def http_middleware(func):

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        raw_request = get_raw_request(args, kwargs)

        if raw_request is None:
            raise ValueError(
                "raw_request required when http middleware is enabled")

        if not raw_request.app.state.enable_http_middleware:
            return await func(*args, **kwargs)

        increment_server_load(raw_request)
        try:
            response = await func(*args, **kwargs)
        except Exception:
            decrement_server_load(raw_request)
            http_error_counter.inc()
            raise

        handle_response_background(response, raw_request)
        check_and_increment_error_counter(response)

        return response

    return wrapper


def get_raw_request(args, kwargs):
    return kwargs.get("raw_request", args[1] if len(args) > 1 else None)


def increment_server_load(raw_request):
    raw_request.app.state.server_load_metrics += 1


def handle_response_background(response, raw_request):
    if isinstance(response, (JSONResponse, StreamingResponse)):
        if response.background is None:
            response.background = BackgroundTask(decrement_server_load,
                                                 raw_request)
        elif isinstance(response.background, BackgroundTasks):
            response.background.add_task(decrement_server_load, raw_request)
        elif isinstance(response.background, BackgroundTask):
            tasks = BackgroundTasks()
            tasks.add_task(response.background.func, *response.background.args,
                           **response.background.kwargs)
            tasks.add_task(decrement_server_load, raw_request)
            response.background = tasks
    else:
        decrement_server_load(raw_request)


def check_and_increment_error_counter(response):
    status_code = (response.status_code if hasattr(response, "status_code")
                   else response.code if hasattr(response, "code") else None)
    if status_code and not (200 <= status_code < 300):
        http_error_counter.inc()
