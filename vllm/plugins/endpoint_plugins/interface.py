# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract for `vllm.endpoint_plugins` entry points.

An endpoint plugin adds HTTP routes to the OpenAI compatible API server.
It registers routes and optionally per app state used by those routes, and
may serve a listener of its own (see `EndpointPlugin`). It must not open new
paths into the engine by reaching the engine the same way an in-tree serving
handler does via `EngineClient` (e.g. `engine_client.collective_rpc(...)`).

If a plugin also needs engine side behavior (a new worker side RPC method,
a custom stat, etc.) pair this entry point with one registered under
`vllm.general_plugins` (see `vllm/plugins/__init__.py`). The
`general_plugins` entry installs the engine side method and the
`endpoint_plugins` entry exposes it over HTTP. The two are registered and
loaded independently where neither implies the other.

Plugins are opt-in. See `load_endpoint_plugins` in `vllm/plugins/__init__.py`
for the loading/gating rules and `docs/usage/security.md` for the security
posture of exposing plugin defined routes.

The CPU only render server (see `build_and_serve_renderer` in
`vllm/entrypoints/launchers/render`) has no `EngineClient`. A plugin
eligible for the `render` task (`required_tasks` is `None` or includes
`"render"`) still gets `attach_router` called but `init_state` receives
`engine_client=None`. Plugins that cannot function without an engine should
either exclude `"render"` from `required_tasks` or check for `None` in
`init_state`/their route handlers and degrade gracefully.
"""

import asyncio
import contextlib
import time
from argparse import Namespace
from collections.abc import Sequence
from copy import copy
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.datastructures import State
from starlette.routing import BaseRoute

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.tasks import SupportedTask

logger = init_logger(__name__)


@runtime_checkable
class EndpointPlugin(Protocol):
    """Protocol implemented by `vllm.endpoint_plugins` entry point factories.

    An entry point registered under the `vllm.endpoint_plugins` group must
    resolve to a zero argument callable (a class or factory function) that
    returns an object satisfying this protocol.

    A plugin may also define these optional lifecycle methods. They are looked
    up with `getattr` and are not protocol members:

    - `async start(state)`: called in load order during lifespan startup. If
      it raises, the plugin cleans up its own partial resources; plugins that
      already started are stopped in reverse order and startup is aborted.
    - `ready(state) -> bool`: a cheap synchronous readiness predicate read by
      `/health`. `False` or an exception returns 503.
    - `async drain(state, timeout)`: called concurrently on shutdown, before
      the engine shuts down, with the remaining `--shutdown-timeout` budget.
      Core sets `state.draining` before any drain runs; the plugin's own
      admission paths must check it, and its drain closes its listener.
      Overruns are cancelled; drain must honor cancellation.
    - `async stop(state)`: called exactly once per started plugin, in reverse
      order, after HTTP stops. It must return promptly and not wait for
      in-flight work (that is `drain`'s job); core does not time-bound it.

    `state.request_shutdown()` starts the same graceful shutdown as SIGTERM.
    A plugin that serves a non-HTTP listener (e.g. gRPC) must set the class
    attribute `owns_non_http_listener = True`; such plugins require
    `--api-server-count 1`. That check does not cover data parallel external
    load balancing, where each rank runs its own API server: the listener
    must then use a distinct port per rank (e.g. base + data parallel rank).
    """

    name: str
    """Unique plugin name used in logs and for `VLLM_PLUGINS` allowlisting."""

    required_tasks: "tuple[SupportedTask, ...] | None"
    """Tasks the server must support for this plugin to be loaded.

    The plugin is loaded only if this set intersects the server's
    `supported_tasks`. `None` means the plugin has no task requirement and
    is always eligible (subject to the `VLLM_PLUGINS` allowlist).
    """

    def attach_router(self, app: FastAPI) -> None:
        """Register this plugin's routes on `app`.

        Called once during `build_app()` after all core routers have been
        attached. HTTP routes registered here replace earlier routes with the
        same path template and method. Other methods remain available.
        """
        ...

    async def init_state(
        self, engine_client: "EngineClient | None", state: State, args: Namespace
    ) -> None:
        """Initialize per app state consumed by this plugin's routes.

        Called once during `init_app_state()` after core state has been
        initialized. Use `engine_client` (e.g. `collective_rpc`) to reach
        the engine. Do not open new engine access paths.

        `engine_client` is `None` on the CPU only render server which has
        no engine. This only happens for plugins eligible for the `render`
        task (`required_tasks` is `None` or includes `"render"`). Handle
        `None` explicitly (e.g. skip engine dependent setup, or have route
        handlers return an error) if the plugin is loadable for `render` but
        cannot function without an engine.
        """
        ...


def attach_endpoint_plugins(
    app: FastAPI, supported_tasks: tuple["SupportedTask", ...]
) -> None:
    """Phase A of endpoint plugin wiring: discover, gate and attach routes.

    Hooks run after all core routers have been attached. Plugin HTTP routes
    replace earlier operations with the same path template and method in both
    routing and OpenAPI. No-ops when no plugins are discovered/allowlisted.
    """
    from vllm.plugins import load_endpoint_plugins

    endpoint_plugins = load_endpoint_plugins(supported_tasks)
    existing_routes = {id(route): route for route in app.routes}
    for plugin in endpoint_plugins:
        plugin.attach_router(app)

    routes: list[BaseRoute] = []
    for route in app.routes:
        if id(route) in existing_routes or not isinstance(route, APIRoute):
            routes.append(route)
            continue

        insert_at = len(routes)
        retained: list[BaseRoute] = []
        for previous in routes:
            if (
                isinstance(previous, APIRoute)
                and previous.path == route.path
                and previous.methods & route.methods
            ):
                logger.info(
                    "Endpoint plugin overrides %s %s",
                    ", ".join(sorted(previous.methods & route.methods)),
                    route.path,
                )
                # Keep replacements ahead of mounts, preserving other methods.
                insert_at = min(insert_at, len(retained))
                methods = previous.methods - route.methods
                if not methods:
                    continue
                previous = copy(previous)
                previous.methods = methods
            retained.append(previous)
        retained.insert(insert_at, route)
        routes = retained

    app.router.routes[:] = routes
    if endpoint_plugins:
        app.openapi_schema = None
    app.state.endpoint_plugins = endpoint_plugins


async def init_endpoint_plugins_state(
    engine_client: EngineClient | None, state: State, args: Namespace
) -> None:
    """Phase B of endpoint plugin wiring: initialize per app plugin state.

    `state.endpoint_plugins` is set by `_attach_endpoint_plugins` (Phase A)
    in `build_app`. Some `init_app_state` callers (e.g. `run_batch.py`)
    build their own bare `State` without going through `build_app`. As a result
    `endpoint_plugins` may be absent and are treated that the same as "none attached".

    `engine_client` is `None` for the CPU only render server which has no
    engine (see `init_render_app_state`). Plugins must handle a `None`
    `engine_client` themselves (see `EndpointPlugin.init_state`).

    Also sets `state.draining = False`, so plugins can read it directly.
    """
    state.draining = False
    plugins = getattr(state, "endpoint_plugins", [])
    listeners = [p.name for p in plugins if getattr(p, "owns_non_http_listener", False)]
    api_server_count = getattr(args, "api_server_count", None)
    if listeners and api_server_count not in (None, 1):
        raise ValueError(
            f"Endpoint plugins {listeners} serve a non-HTTP listener, which "
            f"requires --api-server-count 1 (got {api_server_count})."
        )
    for plugin in plugins:
        await plugin.init_state(engine_client, state, args)


async def start_endpoint_plugins(state: State, plugins: Sequence[Any]) -> None:
    """Start `plugins` in order, stopping the started ones if one fails."""
    state.started_endpoint_plugins = started = []
    for plugin in plugins:
        if (start := getattr(plugin, "start", None)) is not None:
            try:
                await start(state)
            except BaseException:
                logger.error("Endpoint plugin %s failed to start", plugin.name)
                await stop_endpoint_plugins(state)
                raise
        started.append(plugin)


def endpoint_plugins_ready(state: State) -> bool:
    """Return whether every endpoint plugin reports ready."""
    for plugin in getattr(state, "endpoint_plugins", []):
        if (ready := getattr(plugin, "ready", None)) is None:
            continue
        try:
            if not ready(state):
                return False
        except Exception:
            logger.warning(
                "Endpoint plugin %s readiness failed", plugin.name, exc_info=True
            )
            return False
    return True


async def _drain(plugin: Any, state: State, timeout: float) -> None:
    try:
        await plugin.drain(state, timeout)
    except asyncio.CancelledError:
        logger.warning("Endpoint plugin %s drain was cancelled", plugin.name)
        raise
    except Exception:
        logger.exception("Endpoint plugin %s failed to drain", plugin.name)


async def drain_endpoint_plugins(state: State, deadline: float) -> None:
    """Set `state.draining`, then drain started plugins concurrently.

    `state.draining` is set before anything is awaited. Drains still running
    at the deadline, or when the caller is cancelled, are cancelled and
    awaited, so none outlives this call. A drain that swallows
    `CancelledError` therefore delays shutdown.

    Args:
        state: The app state.
        deadline: A `time.monotonic()` deadline. If it has already passed, no
            drain is called.

    """
    state.draining = True
    plugins = [
        plugin
        for plugin in getattr(state, "started_endpoint_plugins", [])
        if getattr(plugin, "drain", None) is not None
    ]
    timeout = deadline - time.monotonic()
    if timeout <= 0 or not plugins:
        return
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(timeout), asyncio.TaskGroup() as tg:
            for plugin in plugins:
                tg.create_task(_drain(plugin, state, timeout))


async def stop_endpoint_plugins(state: State) -> None:
    """Stop started plugins in reverse order, each exactly once.

    A graceful shutdown still in progress (`state.shutdown_task`, e.g. when
    the watchdog stops HTTP or a later `start` fails while plugins drain) is
    cancelled and awaited first, so no drain overlaps a `stop`. In the normal
    path it has already finished and nothing is awaited. Cancellation of
    the caller is deferred until every plugin has stopped.
    """
    shutdown = getattr(state, "shutdown_task", None)
    if shutdown is not None and shutdown.cancel():
        while not shutdown.done():  # defer our own cancellation, as below
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait([shutdown])
    started = getattr(state, "started_endpoint_plugins", [])
    while started:
        plugin = started.pop()
        if (stop := getattr(plugin, "stop", None)) is None:
            continue
        try:
            await stop(state)
        except (Exception, asyncio.CancelledError):
            logger.exception("Endpoint plugin %s failed to stop", plugin.name)
    if (task := asyncio.current_task()) is not None and task.cancelling():
        raise asyncio.CancelledError
