# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract for `vllm.endpoint_plugins` entry points.

An endpoint plugin adds HTTP routes to the OpenAI compatible API server.
Its scope is HTTP surface only. It registers routes and optionally
per app state used by those routes. It must not open new paths into the
engine by reaching the engine the same way an in-tree serving handler does
via `EngineClient` (e.g. `engine_client.collective_rpc(...)`).

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
import time
from argparse import Namespace
from copy import copy
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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


@runtime_checkable
class SupportsStart(Protocol):
    """Optional endpoint plugin startup capability."""

    async def start(self, state: State) -> None:
        """Acquire resources after state initialization, before HTTP accepts.

        If startup fails or is cancelled, clean up partially acquired resources
        before raising. The server only stops plugins whose startup completed.
        """
        ...


@runtime_checkable
class SupportsStop(Protocol):
    """Optional endpoint plugin cleanup capability."""

    async def stop(self, state: State) -> None:
        """Release resources before app state is deleted.

        The engine may already be stopped. Cleanup is cooperative and has no
        deadline; it must settle any background work owned by the plugin.
        """
        ...


@runtime_checkable
class SupportsReady(Protocol):
    """Optional synchronous endpoint plugin readiness capability."""

    def ready(self, state: State) -> bool:
        """Return whether this process can accept new plugin work owners."""
        ...


@runtime_checkable
class SupportsDrain(Protocol):
    """Optional endpoint plugin session drain capability."""

    async def drain(self, state: State, timeout: float) -> None:
        """Settle accepted work before engine shutdown, within `timeout` seconds.

        The server sets `state.draining` before calling any drain hook. Plugin
        admission paths must reject new work owners, including new sessions on
        existing connections, while allowing accepted owners to submit final
        engine work. On completion, error or cancellation, settle or abort and
        await all owned work so it cannot submit later engine requests. The
        server does not track plugin sessions or provide an abort hook.

        Hooks run concurrently under a shared deadline. Cancellation cleanup
        remains cooperative and unbounded; `stop` runs after it settles.
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
    """
    state.draining = False
    for plugin in getattr(state, "endpoint_plugins", []):
        await plugin.init_state(engine_client, state, args)


async def start_endpoint_plugins(state: State) -> None:
    """Start plugins in load order, recording those eligible for cleanup."""
    started: list[EndpointPlugin] = []
    state.started_endpoint_plugins = started
    for plugin in getattr(state, "endpoint_plugins", []):
        if isinstance(plugin, SupportsStart):
            await plugin.start(state)
        started.append(plugin)


def endpoint_plugins_ready(state: State) -> bool:
    """Check every loaded readiness provider; a missing predicate is ready."""
    plugins: list[EndpointPlugin] = getattr(state, "endpoint_plugins", [])
    for plugin in plugins:
        if isinstance(plugin, SupportsReady):
            try:
                if not plugin.ready(state):
                    return False
            except Exception:
                logger.exception("Endpoint plugin %s readiness failed", plugin.name)
                return False
    return True


async def drain_endpoint_plugins(state: State, deadline: float) -> None:
    """Close plugin admission and drain concurrently against a monotonic deadline."""
    state.draining = True
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return

    async def drain(plugin: SupportsDrain, name: str) -> None:
        try:
            await plugin.drain(state, max(0.0, deadline - time.monotonic()))
        except Exception:
            logger.exception("Endpoint plugin %s failed to drain", name)

    try:
        async with asyncio.timeout(remaining), asyncio.TaskGroup() as group:
            plugins: list[EndpointPlugin] = getattr(
                state, "started_endpoint_plugins", []
            )
            for plugin in plugins:
                if isinstance(plugin, SupportsDrain):
                    group.create_task(drain(plugin, plugin.name))
    except TimeoutError:
        logger.warning("Endpoint plugin drain deadline expired")


async def stop_endpoint_plugins(state: State) -> None:
    """Stop eligible plugins once in reverse order, deferring cancellation."""
    shutdown_task = getattr(state, "shutdown_task", None)
    if shutdown_task is not None and not shutdown_task.done():
        # Do not interrupt cancellation cleanup already in progress.
        if not shutdown_task.cancelling():
            shutdown_task.cancel()
        while not shutdown_task.done():
            try:
                await asyncio.shield(shutdown_task)
            except asyncio.CancelledError:
                # Repeated lifespan cancellations must not release session owners.
                pass
            except Exception:
                logger.exception("Endpoint plugin shutdown failed")
    started: list[EndpointPlugin] = getattr(state, "started_endpoint_plugins", [])
    while started:
        plugin = started.pop()
        if isinstance(plugin, SupportsStop):
            try:
                await plugin.stop(state)
            except (Exception, asyncio.CancelledError):
                logger.exception("Endpoint plugin %s failed to stop", plugin.name)
    if (task := asyncio.current_task()) is not None and task.cancelling():
        raise asyncio.CancelledError
