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
import inspect
import time
from argparse import Namespace
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.datastructures import State
from starlette.routing import BaseRoute

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launchers.utils.constants import (
    ENDPOINT_PLUGIN_STOP_TIMEOUT_DEFAULT,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.tasks import SupportedTask

logger = init_logger(__name__)

# A `ready` call longer than asyncio's default `slow_callback_duration` is logged.
_SLOW_READY_MS = 100


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
        `start` and `stop` run in the same task, so a resource bound to the
        task that entered it, such as an anyio cancel scope, can be held from
        `start` until `stop`.
        """
        ...


@runtime_checkable
class SupportsStop(Protocol):
    """Optional endpoint plugin cleanup capability."""

    async def stop(self, state: State) -> None:
        """Release resources before app state is deleted.

        The engine may already be stopped. The hook must settle any background
        work owned by the plugin. All `stop` hooks share one time limit,
        `--endpoint-plugin-stop-timeout` (default 5 seconds). When it expires
        the server cancels the running hook, stops waiting for it and does not
        call the remaining hooks. Cleanup remains cooperative: a hook that
        ignores cancellation keeps running alongside later cleanup, and one
        that blocks the event loop cannot be bounded.
        """
        ...


@runtime_checkable
class SupportsReady(Protocol):
    """Optional synchronous endpoint plugin readiness capability."""

    def ready(self, state: State) -> bool:
        """Return whether this process can accept new plugin work owners.

        Called on the event loop while a `/ready` probe is handled, so it must
        return a `bool` quickly, without blocking or I/O.
        """
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
        remains cooperative: a hook that ignores it holds shutdown before
        engine shutdown. `stop` runs after it settles.
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


def validate_endpoint_plugin_hooks(plugin: EndpointPlugin) -> None:
    for name in ("start", "stop", "drain"):
        hook = getattr(plugin, name, None)
        if hook is not None and not inspect.iscoroutinefunction(hook):
            raise TypeError(
                f"Endpoint plugin {plugin.name}: `{name}` must be an `async def` "
                f"coroutine function; any other `{name}` is not supported. "
                "Rename the attribute if it is not a lifecycle hook."
            )
    ready = getattr(plugin, "ready", None)
    if ready is None:
        return
    if inspect.iscoroutinefunction(ready):
        problem = "an async `ready` is not supported"
    elif not callable(ready):
        problem = "`ready` is not callable"
    else:
        return
    raise TypeError(
        f"Endpoint plugin {plugin.name}: {problem}; define `ready` with `def`, "
        "or rename the attribute."
    )


def cancel_endpoint_plugin_startup(state: State) -> None:
    """Cancel acquisition once, without interrupting rollback or listener binding."""
    state.endpoint_plugin_startup_cancelled = True
    lifecycle = getattr(state, "endpoint_plugin_starting", None)
    if lifecycle is not None:
        lifecycle.cancel_start()


def _stop_timeout(state: State) -> int:
    # `build_app` is public, so its `args` may lack the field.
    return getattr(
        getattr(state, "args", None),
        "endpoint_plugin_stop_timeout",
        ENDPOINT_PLUGIN_STOP_TIMEOUT_DEFAULT,
    )


class _PluginLifecycle:
    """Runs one plugin's `start` and `stop` in one task.

    Some resources are bound to the task that entered them: anyio cancel
    scopes and task groups (an MCP `ClientSession` uses them) raise
    `RuntimeError` when exited from another task, `asyncio.timeout` and
    `asyncio.TaskGroup` cancel the task that entered them, and a context
    variable set in `start` is not visible to another task. A plugin may hold
    such a resource from `start` until `stop`, so the task runs `start`, waits
    for a stop request, then runs `stop`. Running the two hooks in one task
    also means `stop` cannot overlap `start`.
    """

    def __init__(self, plugin: EndpointPlugin, state: State) -> None:
        self.plugin = plugin
        self.state = state
        # Set once `start` has returned, raised or been cancelled.
        self._start_settled = asyncio.Event()
        self._start_error: BaseException | None = None
        self._start_cancelled = False
        self._stop_requested = asyncio.Event()
        self._skip_stop = False
        self._stop_called = False
        self.task = asyncio.create_task(self._run())
        # The task can be cancelled before its first step, when `_run` cannot
        # record the outcome itself.
        self.task.add_done_callback(self._settle_start_on_task_end)

    def _settle_start_on_task_end(self, task: asyncio.Task) -> None:
        if not task.cancelled():
            # `_run` records or logs every exception it raises; retrieve it so
            # the event loop does not report it again.
            task.exception()
        if not self._start_settled.is_set():
            self._start_cancelled = True
            self._start_settled.set()

    def cancel_start(self) -> None:
        """Cancel a running `start` once, so later calls cannot interrupt the
        cleanup it does before propagating the cancellation."""
        if not self._start_settled.is_set() and not self.task.cancelling():
            self.task.cancel()

    def request_stop(self) -> None:
        self._stop_requested.set()

    def end_without_stop(self) -> None:
        """End the waiting task without calling `stop`."""
        self._skip_stop = True
        self._stop_requested.set()

    async def wait_started(self) -> bool:
        """Wait until `start` returns, raises or is cancelled.

        Cancelling the caller cancels a running `start`; the wait continues
        until `start` ends, so repeated cancellation cannot leave it running.

        Returns:
            True if the caller was cancelled after `start` returned. The plugin
            is then started and the caller must stop it.

        Raises:
            The exception raised by `start`, or `asyncio.CancelledError` if
            `start` or the caller was cancelled.

        """
        caller_cancelled = False
        while not self._start_settled.is_set():
            try:
                await self._start_settled.wait()
            except asyncio.CancelledError:
                caller_cancelled = True
                self.cancel_start()
        if isinstance(self._start_error, (KeyboardInterrupt, SystemExit)):
            # The plugin task re-raised it, so it has already left the event
            # loop; asyncio is now cancelling the remaining tasks.
            raise asyncio.CancelledError
        if self._start_error is not None:
            raise self._start_error
        if self._start_cancelled:
            raise asyncio.CancelledError
        return caller_cancelled

    async def _run(self) -> None:
        plugin = self.plugin
        try:
            if isinstance(plugin, SupportsStart):
                await plugin.start(self.state)
        except asyncio.CancelledError:
            self._start_cancelled = True
            self._start_settled.set()
            raise
        except (KeyboardInterrupt, SystemExit) as e:
            # Re-raise so these keep leaving the event loop as asyncio does
            # for any task.
            self._start_error = e
            self._start_settled.set()
            raise
        except BaseException as e:
            # The caller re-raises the exception, so this task need not.
            self._start_error = e
            self._start_settled.set()
            return
        self._start_settled.set()
        try:
            await self._stop_requested.wait()
        except asyncio.CancelledError:
            if self._stop_requested.is_set():
                # Shutdown requested `stop` and the cancellation arrived before
                # this task woke up. Run `stop` as requested and absorb the
                # cancellation: the caller bounds `stop` with the time limit,
                # cancelling it again at the deadline, and `stop_endpoint_plugins`
                # re-raises the lifespan cancellation after cleanup.
                if not self._skip_stop:
                    await self._stop()
                return
            # Something other than shutdown cancelled the task: event loop
            # teardown, or a failed child of an anyio task group the plugin
            # opened in `start`. Release the plugin's resources before ending.
            if isinstance(plugin, SupportsStop):
                logger.warning(
                    "Endpoint plugin %s task was cancelled before shutdown; "
                    "calling its stop hook",
                    plugin.name,
                )
                try:
                    async with asyncio.timeout(_stop_timeout(self.state)):
                        await self._stop()
                except Exception:
                    logger.exception("Endpoint plugin %s failed to stop", plugin.name)
            raise
        if not self._skip_stop:
            await self._stop()

    async def _stop(self) -> None:
        if self._stop_called or not isinstance(self.plugin, SupportsStop):
            return
        self._stop_called = True
        await self.plugin.stop(self.state)


async def start_endpoint_plugins(state: State) -> None:
    """Start plugins in load order, recording those eligible for cleanup.

    Each plugin with a `start` or `stop` hook gets a task that runs both hooks
    (see `_PluginLifecycle`).
    """
    started: list[EndpointPlugin] = []
    # `lifecycles[i]` runs the hooks of `started[i]`; None for a plugin
    # without `start` or `stop`.
    lifecycles: list[_PluginLifecycle | None] = []
    state.started_endpoint_plugins = started
    state.endpoint_plugin_lifecycles = lifecycles
    for plugin in getattr(state, "endpoint_plugins", []):
        if getattr(state, "endpoint_plugin_startup_cancelled", False):
            raise asyncio.CancelledError
        # A separate name keeps `plugin` typed as `EndpointPlugin`.
        has_hooks = isinstance(plugin, (SupportsStart, SupportsStop))
        if not has_hooks:
            started.append(plugin)
            lifecycles.append(None)
            continue
        lifecycle = _PluginLifecycle(plugin, state)
        state.endpoint_plugin_starting = lifecycle
        try:
            caller_cancelled = await lifecycle.wait_started()
        finally:
            state.endpoint_plugin_starting = None
        started.append(plugin)
        lifecycles.append(lifecycle)
        if caller_cancelled:
            raise asyncio.CancelledError


def endpoint_plugins_ready(state: State) -> bool:
    """Check every loaded readiness provider; a missing predicate is ready."""
    plugins: list[EndpointPlugin] = getattr(state, "endpoint_plugins", [])
    for plugin in plugins:
        if isinstance(plugin, SupportsReady) and not _endpoint_plugin_ready(
            plugin, plugin.name, state
        ):
            return False
    return True


def _endpoint_plugin_ready(plugin: SupportsReady, name: str, state: State) -> bool:
    start = time.perf_counter()
    try:
        result = plugin.ready(state)
    except Exception:
        logger.exception("Endpoint plugin %s readiness failed", name)
        return False
    if time.perf_counter() - start > _SLOW_READY_MS / 1000:
        logger.warning_once(
            "Endpoint plugin %s: a `ready` call took over %d ms; `ready` runs "
            "on the event loop for every /ready probe and must return quickly",
            name,
            _SLOW_READY_MS,
            scope="process",
        )
    if isinstance(result, bool):
        return result
    if inspect.iscoroutine(result):
        result.close()
    logger.error_once(
        "Endpoint plugin %s: `ready` must return a bool; treating the plugin "
        "as not ready",
        name,
        scope="process",
    )
    return False


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

    plugins: list[EndpointPlugin] = getattr(state, "started_endpoint_plugins", [])
    tasks = {
        asyncio.create_task(drain(plugin, plugin.name)): plugin.name
        for plugin in plugins
        if isinstance(plugin, SupportsDrain)
    }
    if not tasks:
        return
    cancelled = False
    try:
        await asyncio.wait(tasks, timeout=remaining)
    except asyncio.CancelledError:
        cancelled = True
    pending = [task for task in tasks if not task.done()]
    if pending and not cancelled:
        logger.warning(
            "Endpoint plugin drain deadline expired; cancelling drains still "
            "running: %s",
            ", ".join(tasks[task] for task in pending),
        )
    for task in pending:
        task.cancel()
    # Like a TaskGroup, defer repeated cancellation until the drains settle.
    while pending:
        try:
            await asyncio.wait(pending)
        except asyncio.CancelledError:
            cancelled = True
        pending = [task for task in pending if not task.done()]
    if cancelled:
        raise asyncio.CancelledError


# Plugin tasks whose `stop` outlived the time limit, referenced until they end.
_abandoned_stop_tasks: set[asyncio.Task] = set()


def _discard_abandoned_stop_task(name: str, task: asyncio.Task) -> None:
    _abandoned_stop_tasks.discard(task)
    # Retrieve the exception so the event loop does not report it.
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.debug(
            "Endpoint plugin %s failed to stop after the time limit",
            name,
            exc_info=error,
        )


async def _stop_endpoint_plugin(lifecycle: _PluginLifecycle, deadline: float) -> bool:
    """Request one plugin's `stop` and wait until its task ends or the deadline
    expires.

    Returns:
        False if the task was still running at the deadline and was abandoned.

    """
    name = lifecycle.plugin.name
    task = lifecycle.task
    loop = asyncio.get_running_loop()
    lifecycle.request_stop()
    while not task.done() and (remaining := deadline - loop.time()) > 0:
        try:
            # Not `asyncio.wait_for`, which awaits the cancelled task.
            await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            # The lifespan task was cancelled: pass it on to the hook and give
            # it the rest of the time limit to finish.
            task.cancel()
    if not task.done():
        task.cancel()
        _abandoned_stop_tasks.add(task)
        task.add_done_callback(partial(_discard_abandoned_stop_task, name))
        return False
    try:
        task.result()
    except (Exception, asyncio.CancelledError):
        logger.exception("Endpoint plugin %s failed to stop", name)
    return True


async def stop_endpoint_plugins(state: State) -> None:
    """Wait for the shutdown sequence, then stop eligible plugins once in
    reverse order within one time limit, deferring cancellation."""
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
    lifecycles: list[_PluginLifecycle | None] = getattr(
        state, "endpoint_plugin_lifecycles", []
    )
    timeout = _stop_timeout(state)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    abandoned = None
    while started:
        plugin = started[-1]
        lifecycle = lifecycles[-1]
        if (
            lifecycle is None
            or lifecycle.task.done()
            or not isinstance(plugin, SupportsStop)
        ):
            # A task that already ended has called `stop` if it was going to
            # (see `_PluginLifecycle._run`); a start-only plugin's task just
            # ends.
            started.pop()
            lifecycles.pop()
            if lifecycle is not None:
                lifecycle.end_without_stop()
            continue
        if deadline <= loop.time():
            break
        started.pop()
        lifecycles.pop()
        if not await _stop_endpoint_plugin(lifecycle, deadline):
            abandoned = plugin.name
            break
    skipped = [p.name for p in reversed(started) if isinstance(p, SupportsStop)]
    for remaining in lifecycles:
        if remaining is not None:
            remaining.end_without_stop()
    lifecycles.clear()
    started.clear()
    if abandoned is not None or skipped:
        logger.warning(
            "Endpoint plugin stop hooks exceeded the %d second limit "
            "(--endpoint-plugin-stop-timeout); abandoned: %s; skipped: %s",
            timeout,
            abandoned or "none",
            ", ".join(skipped) or "none",
        )
    if (task := asyncio.current_task()) is not None and task.cancelling():
        raise asyncio.CancelledError
