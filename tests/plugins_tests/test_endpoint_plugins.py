# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the `vllm.endpoint_plugins` framework (RFC #46565).

Uses the worked in repo example plugin (`vllm_add_dummy_endpoint_plugin`,
installed via `tests/plugins/vllm_add_dummy_endpoint_plugin`) exercising both
`EndpointPlugin` hooks against a fake `EngineClient`, unit tests for the
`load_endpoint_plugins` gating matrix, an e2e test that drives a real HTTP
request through the plugin's route, and lifecycle tests that run the optional
`start`/`stop` capabilities through `serve_http` and the server lifespan.
"""

import asyncio
import contextlib
import os
import signal
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import APIRouter, FastAPI
from starlette.datastructures import State
from vllm_add_dummy_endpoint_plugin import DummyAdminEndpointPlugin

from vllm.config import VllmConfig
from vllm.entrypoints.common.cli_args import make_arg_parser
from vllm.entrypoints.launchers.app import build_app
from vllm.entrypoints.launchers.launcher import (
    create_server_socket,
    serve_http,
    terminate_if_errored,
)
from vllm.plugins import load_endpoint_plugins
from vllm.plugins.endpoint_plugins.interface import (
    EndpointPlugin,
    SupportsStart,
    SupportsStop,
    attach_endpoint_plugins,
    init_endpoint_plugins_state,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

_TIMEOUT_S = 10


class _RaisingEndpointPlugin:
    """Factory that raises to exercise the "instantiation fails" path."""

    name = "raising_endpoint_plugin"
    required_tasks = None

    def __init__(self):
        raise RuntimeError("boom")


class _ShadowEndpointPlugin:
    name = "shadow_endpoint_plugin"
    required_tasks = None

    def __init__(self, path: str, include_in_schema: bool):
        self.path = path
        self.include_in_schema = include_in_schema

    def attach_router(self, app: FastAPI) -> None:
        assert str(app.url_path_for("show_version")) == "/version"
        app.openapi()

        @app.get(self.path, status_code=209, include_in_schema=self.include_in_schema)
        async def plugin_version(plugin_arg: str) -> dict[str, str]:
            return {"served_by": plugin_arg}

    async def init_state(self, engine_client, state, args) -> None:
        pass


class _RouterEndpointPlugin(DummyAdminEndpointPlugin):
    def __init__(self, router: APIRouter):
        super().__init__()
        self.router = router

    def attach_router(self, app: FastAPI) -> None:
        app.include_router(self.router)


class _FakeEngineClient:
    """Minimal stand in exercising `collective_rpc`. Not a real engine."""

    def __init__(self, rpc_result: Any = None):
        self.rpc_result = rpc_result
        self.rpc_calls: list[tuple[str, tuple, dict]] = []

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.rpc_calls.append((method, args, kwargs or {}))
        return self.rpc_result


def _build_args() -> Namespace:
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers()
    serve_parser = subparsers.add_parser("serve")
    make_arg_parser(serve_parser)
    return serve_parser.parse_args([])


def _fake_loader(factories: dict[str, Any]):
    def _load_plugins_by_group(group: str) -> dict[str, Any]:
        assert group == "vllm.endpoint_plugins"
        return factories

    return _load_plugins_by_group


def test_dummy_plugin_satisfies_protocol():
    assert isinstance(DummyAdminEndpointPlugin(), EndpointPlugin)


def test_no_plugins_loaded_when_allowlist_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)

    assert load_endpoint_plugins(("generate",)) == []


def test_no_plugins_loaded_when_allowlist_is_empty_string(
    monkeypatch: pytest.MonkeyPatch,
):
    """`VLLM_PLUGINS=""` parses to `[""]`, not `None` (see `vllm.envs`), so it
    must be treated as a (non strict) allowlist matching no plugin name, not
    as "unset"."""
    monkeypatch.setenv("VLLM_PLUGINS", "")

    assert load_endpoint_plugins(("generate",)) == []


def test_plugin_loaded_when_allowlisted_and_task_matches(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_PLUGINS", "dummy_admin_endpoint_plugin")

    plugins = load_endpoint_plugins(("generate",))

    assert len(plugins) == 1
    assert isinstance(plugins[0], DummyAdminEndpointPlugin)


def test_plugin_skipped_when_required_tasks_miss(monkeypatch: pytest.MonkeyPatch):
    class _GenerateOnlyPlugin(DummyAdminEndpointPlugin):
        required_tasks = ("generate",)

    monkeypatch.setenv("VLLM_PLUGINS", "dummy_admin_endpoint_plugin")
    monkeypatch.setattr(
        "vllm.plugins.load_plugins_by_group",
        _fake_loader({"dummy_admin_endpoint_plugin": _GenerateOnlyPlugin}),
    )

    assert load_endpoint_plugins(("embed",)) == []
    assert len(load_endpoint_plugins(("generate",))) == 1


def test_plugin_loaded_when_required_tasks_is_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_PLUGINS", "dummy_admin_endpoint_plugin")

    assert len(load_endpoint_plugins(supported_tasks=None)) == 1


def test_plugin_skipped_when_required_tasks_set_but_supported_tasks_none(
    monkeypatch: pytest.MonkeyPatch,
):
    class _GenerateOnlyPlugin(DummyAdminEndpointPlugin):
        required_tasks = ("generate",)

    monkeypatch.setenv("VLLM_PLUGINS", "dummy_admin_endpoint_plugin")
    monkeypatch.setattr(
        "vllm.plugins.load_plugins_by_group",
        _fake_loader({"dummy_admin_endpoint_plugin": _GenerateOnlyPlugin}),
    )

    assert load_endpoint_plugins(supported_tasks=None) == []


def test_factory_raising_is_logged_and_skipped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "VLLM_PLUGINS", "raising_endpoint_plugin,dummy_admin_endpoint_plugin"
    )
    monkeypatch.setattr(
        "vllm.plugins.load_plugins_by_group",
        _fake_loader(
            {
                "raising_endpoint_plugin": _RaisingEndpointPlugin,
                "dummy_admin_endpoint_plugin": DummyAdminEndpointPlugin,
            }
        ),
    )

    plugins = load_endpoint_plugins(("generate",))

    assert len(plugins) == 1
    assert isinstance(plugins[0], DummyAdminEndpointPlugin)


def test_attach_is_noop_when_nothing_discovered(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)

    app = FastAPI()
    attach_endpoint_plugins(app, ("generate",))

    assert app.state.endpoint_plugins == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/version", "/metrics", "/health", "/ready"])
@pytest.mark.parametrize("include_in_schema", [True, False])
async def test_endpoint_plugin_override_preserves_hook_and_schema(
    monkeypatch: pytest.MonkeyPatch, path: str, include_in_schema: bool
):
    """Hooks see core routes; overrides win over both HTTP routes and mounts."""
    args = _build_args()
    monkeypatch.setenv("VLLM_PLUGINS", "shadow_endpoint_plugin")
    monkeypatch.setattr(
        "vllm.plugins.load_plugins_by_group",
        _fake_loader(
            {
                "shadow_endpoint_plugin": lambda: _ShadowEndpointPlugin(
                    path, include_in_schema
                )
            }
        ),
    )
    app = build_app(args, supported_tasks=())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path, params={"plugin_arg": "plugin"})
        schema = (await client.get("/openapi.json")).json()

    assert response.status_code == 209
    assert response.json() == {"served_by": "plugin"}
    if include_in_schema:
        operation = schema["paths"][path]["get"]
        assert "209" in operation["responses"]
        assert "200" not in operation["responses"]
        assert operation["parameters"][0]["name"] == "plugin_arg"
        assert operation["parameters"][0]["required"] is True
        response_schema = operation["responses"]["209"]["content"]["application/json"][
            "schema"
        ]
        assert response_schema["additionalProperties"] == {"type": "string"}
    else:
        assert path not in schema["paths"]


@pytest.mark.asyncio
async def test_endpoint_plugin_override_preserves_other_methods(
    monkeypatch: pytest.MonkeyPatch,
):
    """Overriding GET preserves POST's validation and the shared core route."""
    core = APIRouter()

    @core.api_route("/shared", methods=["GET", "POST"], status_code=201)
    async def original(value: int) -> dict[str, int]:
        return {"value": value}

    plugin = APIRouter()

    @plugin.get("/shared", status_code=202)
    async def replacement():
        return {"served_by": "plugin"}

    app = FastAPI()
    app.include_router(core)
    original_route = app.routes[-1]
    monkeypatch.setattr(
        "vllm.plugins.load_endpoint_plugins",
        lambda supported_tasks: [_RouterEndpointPlugin(plugin)],
    )
    attach_endpoint_plugins(app, ())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/shared")
        post_response = await client.post("/shared", params={"value": 7})
        invalid_response = await client.post("/shared", params={"value": "invalid"})

    assert response.status_code == 202
    assert response.json() == {"served_by": "plugin"}
    assert post_response.status_code == 201
    assert post_response.json() == {"value": 7}
    assert invalid_response.status_code == 422
    assert original_route.methods == {"GET", "POST"}
    assert core.routes[0].methods == {"GET", "POST"}
    operations = app.openapi()["paths"]["/shared"]
    assert "202" in operations["get"]["responses"]
    assert "201" in operations["post"]["responses"]


@pytest.mark.asyncio
async def test_later_endpoint_plugin_overrides_same_operation(
    monkeypatch: pytest.MonkeyPatch,
):
    """The last registered override wins while unrelated plugin routes survive."""
    first = APIRouter()

    @first.get("/shared", status_code=201)
    @first.get("/plugins/first")
    async def first_handler():
        return {"served_by": "first"}

    second = APIRouter()

    @second.get("/shared", status_code=202)
    async def second_handler():
        return {"served_by": "second"}

    monkeypatch.setattr(
        "vllm.plugins.load_endpoint_plugins",
        lambda supported_tasks: [
            _RouterEndpointPlugin(first),
            _RouterEndpointPlugin(second),
        ],
    )
    app = FastAPI()
    attach_endpoint_plugins(app, ())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/shared")
        unrelated = await client.get("/plugins/first")
        unsupported = await client.delete("/shared")

    assert response.status_code == 202
    assert response.json() == {"served_by": "second"}
    assert unrelated.status_code == 200
    assert unrelated.json() == {"served_by": "first"}
    assert unsupported.status_code == 405
    responses = app.openapi()["paths"]["/shared"]["get"]["responses"]
    assert "202" in responses
    assert "201" not in responses


@pytest.mark.asyncio
async def test_init_state_is_noop_without_phase_a(monkeypatch: pytest.MonkeyPatch):
    """`init_app_state` callers that never ran `build_app` (e.g.
    `run_batch.py`, which builds a bare `State()`) must not crash just
    because `state.endpoint_plugins` was never set."""
    monkeypatch.setenv("VLLM_PLUGINS", "dummy_admin_endpoint_plugin")

    state = State()
    await init_endpoint_plugins_state(_FakeEngineClient(), state, _build_args())

    assert not hasattr(state, "dummy_engine_client")


@pytest.mark.asyncio
async def test_render_server_attaches_endpoint_plugins_with_no_engine_client(
    monkeypatch: pytest.MonkeyPatch,
):
    """The CPU only render server has no `EngineClient` but a plugin eligible
    for the `render` task (`required_tasks` is `None` or includes `"render"`)
    still gets its routes attached at Phase A. Phase B passes `None` for
    `engine_client` and it's up to the plugin to handle that."""
    monkeypatch.setenv("VLLM_PLUGINS", "dummy_admin_endpoint_plugin")

    args = _build_args()
    app = build_app(args, ("render",))

    assert len(app.state.endpoint_plugins) == 1
    assert any(
        getattr(route, "path", None) == "/v1/admin/scheduler_config"
        for route in app.routes
    )

    await init_endpoint_plugins_state(None, app.state, args)

    assert app.state.dummy_engine_client is None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/admin/scheduler_config")

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_endpoint_plugin_end_to_end(monkeypatch: pytest.MonkeyPatch):
    """Phase A (attach) + Phase B (init) wired through `build_app` then
    exercised with a real HTTP request against the worked example plugin."""
    monkeypatch.setenv("VLLM_PLUGINS", "dummy_admin_endpoint_plugin")

    args = _build_args()
    app = build_app(args, supported_tasks=())

    assert len(app.state.endpoint_plugins) == 1
    assert any(
        getattr(route, "path", None) == "/v1/admin/scheduler_config"
        for route in app.routes
    )

    fake_engine_client = _FakeEngineClient(rpc_result=["cfg-a", "cfg-b"])
    await init_endpoint_plugins_state(fake_engine_client, app.state, args)

    assert app.state.dummy_engine_client is fake_engine_client

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/admin/scheduler_config")

    assert response.status_code == 200
    assert response.json() == {"scheduler_config": ["cfg-a", "cfg-b"]}
    assert fake_engine_client.rpc_calls == [("get_scheduler_config", (), {})]


class _LegacyEndpointPlugin:
    """A plugin written before the optional `start`/`stop` capabilities."""

    name = "legacy_endpoint_plugin"
    required_tasks = None

    def __init__(self, events: list[str] | None = None, name: str | None = None):
        self.events = [] if events is None else events
        if name is not None:
            self.name = name

    def attach_router(self, app: FastAPI) -> None:
        pass

    async def init_state(self, engine_client, state, args) -> None:
        pass


class _StartOnlyPlugin(_LegacyEndpointPlugin):
    async def start(self, state: State) -> None:
        self.events.append(f"start:{self.name}")


class _StopOnlyPlugin(_LegacyEndpointPlugin):
    async def stop(self, state: State) -> None:
        self.events.append(f"stop:{self.name}")


def _dummy(plugin_name: str, events: list[str]) -> DummyAdminEndpointPlugin:
    plugin = DummyAdminEndpointPlugin(events)
    plugin.name = plugin_name
    return plugin


class _ServingEngineClient(_FakeEngineClient):
    """External engine service double for `serve_http`: healthy until core
    calls `shutdown` (from its executor thread), which it records in `events`
    and signals through `stopped`."""

    def __init__(self, events: list[str], shutdown_timeout: int = 30):
        super().__init__()
        self.events = events
        self.vllm_config = VllmConfig(shutdown_timeout=shutdown_timeout)
        self.errored = False
        self.is_running = True
        self.should_keep_api_server_alive = False
        self.loop = asyncio.get_running_loop()
        self.stopped = asyncio.Event()

    async def check_health(self) -> None:
        pass

    def shutdown(self, timeout: float | None = None) -> None:
        self.shutdown_allowance = timeout
        self.events.append("engine_shutdown")
        self.loop.call_soon_threadsafe(self.stopped.set)


class _NativeServing:
    """Stands in for `state.openai_serving_transcription`, which the lifespan
    shuts down by duck typing on a synchronous `shutdown()`."""

    def __init__(self, events: list[str]):
        self.events = events

    def shutdown(self) -> None:
        self.events.append("native_shutdown")


async def _build_plugin_app(
    monkeypatch: pytest.MonkeyPatch,
    plugins: list,
    engine_client: _ServingEngineClient | None,
    native: _NativeServing | None = None,
) -> FastAPI:
    monkeypatch.setattr(
        "vllm.plugins.load_endpoint_plugins", lambda supported_tasks: plugins
    )
    args = _build_args()
    app = build_app(args, supported_tasks=())
    app.state.engine_client = engine_client
    app.state.log_stats = False
    if native is not None:
        app.state.openai_serving_transcription = native
    await init_endpoint_plugins_state(engine_client, app.state, args)
    return app


async def _wait_closed(serve_task: asyncio.Task) -> None:
    """Bound the test's wait for serving and cleanup to finish."""
    async with asyncio.timeout(_TIMEOUT_S):
        await serve_task


@asynccontextmanager
async def _serving(
    monkeypatch: pytest.MonkeyPatch,
    plugins: list,
    engine_client: _ServingEngineClient | None,
    native: _NativeServing | None = None,
) -> AsyncIterator[tuple[State, asyncio.Task, str]]:
    """Serve a `build_app` app through `serve_http` on a free port.

    Yields the app state (which outlives the lifespan's `del app.state`), the
    `serve_http` task and the base URL. The server is torn down on exit even
    if the test fails.
    """
    app = await _build_plugin_app(monkeypatch, plugins, engine_client, native)
    state = app.state
    sock = create_server_socket(("127.0.0.1", 0), reuse_port=False)
    port = sock.getsockname()[1]

    async def serve_and_shutdown() -> None:
        await (
            await serve_http(
                app, sock, host="127.0.0.1", port=port, log_level="warning"
            )
        )

    serve_task = asyncio.create_task(serve_and_shutdown())
    try:
        yield state, serve_task, f"http://127.0.0.1:{port}"
    finally:
        if not serve_task.done():
            if (server := getattr(state, "server", None)) is not None:
                server.should_exit = True
            else:
                serve_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await _wait_closed(serve_task)
        sock.close()


async def _wait_started(state: State, serve_task: asyncio.Task) -> None:
    async with asyncio.timeout(_TIMEOUT_S):
        while not getattr(getattr(state, "server", None), "started", False):
            assert not serve_task.done(), serve_task.result()
            await asyncio.sleep(0.01)


async def _health(base_url: str) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        return await client.get(f"{base_url}/health")


def test_start_and_stop_capabilities_are_independently_optional():
    """Legacy plugins satisfy `EndpointPlugin` alone; a plugin may implement
    either capability without the other."""
    legacy = _LegacyEndpointPlugin()
    assert isinstance(legacy, EndpointPlugin)
    assert not isinstance(legacy, SupportsStart)
    assert not isinstance(legacy, SupportsStop)

    assert isinstance(_StartOnlyPlugin(), SupportsStart)
    assert not isinstance(_StartOnlyPlugin(), SupportsStop)
    assert not isinstance(_StopOnlyPlugin(), SupportsStart)
    assert isinstance(_StopOnlyPlugin(), SupportsStop)

    dummy = DummyAdminEndpointPlugin()
    assert isinstance(dummy, EndpointPlugin)
    assert isinstance(dummy, SupportsStart)
    assert isinstance(dummy, SupportsStop)


@pytest.mark.asyncio
async def test_legacy_plugin_serves_and_shuts_down_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    """A plugin without `start`/`stop` serves, keeps `/health` as is and sees
    the engine-first shutdown it always had."""
    events: list[str] = []
    engine = _ServingEngineClient(events)
    async with _serving(monkeypatch, [_LegacyEndpointPlugin()], engine) as (
        state,
        serve_task,
        url,
    ):
        await _wait_started(state, serve_task)
        response = await _health(url)
        assert (response.status_code, response.content) == (200, b"")

        state.request_shutdown()
        await _wait_closed(serve_task)

    assert events == ["engine_shutdown"]


@pytest.mark.asyncio
async def test_start_and_stop_order_around_engine_shutdown(
    monkeypatch: pytest.MonkeyPatch,
):
    """`start` runs in load order before HTTP accepts and can already reach
    `state.request_shutdown`; after the engine stops, every plugin with
    `stop` stops in reverse load order, including stop-only ones; repeated
    `request_shutdown` calls are idempotent."""
    events: list[str] = []

    class _Probing(DummyAdminEndpointPlugin):
        name = "first"

        async def start(self, state: State) -> None:
            assert callable(state.request_shutdown)
            self.events.append(f"start:{self.name}:accepting={state.server.started}")

    plugins = [
        _Probing(events),
        _StartOnlyPlugin(events, "second"),
        _StopOnlyPlugin(events, "third"),
    ]
    engine = _ServingEngineClient(events)
    async with _serving(monkeypatch, plugins, engine) as (state, serve_task, url):
        await _wait_started(state, serve_task)
        assert (await _health(url)).status_code == 200

        state.request_shutdown()
        state.request_shutdown()
        await _wait_closed(serve_task)

    assert events == [
        "start:first:accepting=False",
        "start:second",
        "engine_shutdown",
        "stop:third",
        "stop:first",
    ]


@pytest.mark.asyncio
async def test_sigterm_shares_the_request_shutdown_event(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []
    engine = _ServingEngineClient(events)
    async with _serving(monkeypatch, [_dummy("plugin", events)], engine) as (
        state,
        serve_task,
        _,
    ):
        await _wait_started(state, serve_task)

        os.kill(os.getpid(), signal.SIGTERM)
        async with asyncio.timeout(_TIMEOUT_S):
            await engine.stopped.wait()
        state.request_shutdown()
        await _wait_closed(serve_task)

    assert events == ["start:plugin", "engine_shutdown", "stop:plugin"]


@pytest.mark.asyncio
async def test_request_shutdown_during_start_stops_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
):
    """A `start` that requests shutdown (and keeps running until the engine
    has stopped) still completes; the server never serves, and `stop` runs
    once."""
    events: list[str] = []
    engine = _ServingEngineClient(events)

    class _Aborting(DummyAdminEndpointPlugin):
        name = "aborting"

        async def start(self, state: State) -> None:
            await super().start(state)
            state.request_shutdown()
            async with asyncio.timeout(_TIMEOUT_S):
                await engine.stopped.wait()

    async with _serving(monkeypatch, [_Aborting(events)], engine) as (
        _,
        serve_task,
        url,
    ):
        await _wait_closed(serve_task)
        with pytest.raises(httpx.ConnectError):
            await _health(url)

    assert events == ["start:aborting", "engine_shutdown", "stop:aborting"]


@pytest.mark.asyncio
async def test_shutdown_during_socket_binding_leaves_no_listener(
    monkeypatch: pytest.MonkeyPatch,
):
    """Pause real socket binding before Uvicorn records the listener, so a
    shutdown request cannot cancel binding and orphan the open socket."""
    events: list[str] = []
    engine = _ServingEngineClient(events)
    binding = asyncio.Event()
    release = asyncio.Event()
    shutdown_waiting = asyncio.Event()
    listeners: list[asyncio.Server] = []
    loop = asyncio.get_running_loop()
    create_server = loop.create_server

    async def paused_create_server(*args, **kwargs):
        listener = await create_server(*args, **kwargs)
        listeners.append(listener)
        binding.set()
        await release.wait()
        return listener

    monkeypatch.setattr(loop, "create_server", paused_create_server)
    async with _serving(monkeypatch, [_dummy("plugin", events)], engine) as (
        state,
        serve_task,
        url,
    ):
        try:
            async with asyncio.timeout(_TIMEOUT_S):
                await binding.wait()
                wait_for_startup = state.server.startup_complete.wait

                async def observed_wait() -> bool:
                    shutdown_waiting.set()
                    return await wait_for_startup()

                monkeypatch.setattr(
                    state.server.startup_complete, "wait", observed_wait
                )
                state.request_shutdown()
                await shutdown_waiting.wait()
            release.set()
            await _wait_closed(serve_task)
            with pytest.raises(httpx.ConnectError):
                await _health(url)
        finally:
            release.set()
            for listener in listeners:
                listener.close()
                await listener.wait_closed()
    assert events == ["start:plugin", "engine_shutdown", "stop:plugin"]


@pytest.mark.asyncio
async def test_watchdog_exit_stops_plugins_without_engine_shutdown(
    monkeypatch: pytest.MonkeyPatch,
):
    """When the watchdog exits the server after engine death, `stop` still
    runs exactly once."""
    events: list[str] = []
    engine = _ServingEngineClient(events)
    async with _serving(monkeypatch, [_dummy("plugin", events)], engine) as (
        state,
        serve_task,
        _,
    ):
        await _wait_started(state, serve_task)

        engine.errored = True
        engine.is_running = False
        terminate_if_errored(state.server, engine)
        await _wait_closed(serve_task)

    assert events == ["start:plugin", "stop:plugin"]


@pytest.mark.asyncio
async def test_stop_failures_do_not_skip_peers_or_native_cleanup(
    monkeypatch: pytest.MonkeyPatch,
):
    """A `stop` that raises, including `CancelledError`, is logged; the other
    plugins still stop in reverse order and the native transcription
    handler is still shut down afterwards."""
    events: list[str] = []

    class _Erroring(DummyAdminEndpointPlugin):
        name = "erroring"

        async def stop(self, state: State) -> None:
            await super().stop(state)
            raise RuntimeError("stop failed")

    class _Cancelling(DummyAdminEndpointPlugin):
        name = "cancelling"

        async def stop(self, state: State) -> None:
            await super().stop(state)
            raise asyncio.CancelledError

    plugins = [_dummy("first", events), _Erroring(events), _Cancelling(events)]
    engine = _ServingEngineClient(events)
    native = _NativeServing(events)
    async with _serving(monkeypatch, plugins, engine, native) as (
        state,
        serve_task,
        _,
    ):
        await _wait_started(state, serve_task)

        state.request_shutdown()
        await _wait_closed(serve_task)

    assert events == [
        "start:first",
        "start:erroring",
        "start:cancelling",
        "engine_shutdown",
        "stop:cancelling",
        "stop:erroring",
        "stop:first",
        "native_shutdown",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
async def test_failed_start_rolls_back_in_reverse_once(
    monkeypatch: pytest.MonkeyPatch, error: type[BaseException]
):
    """A failing `start` aborts lifespan startup: the plugins ahead of it,
    including a stop-only one, stop in reverse order exactly once, later
    plugins never start and native cleanup still runs."""
    events: list[str] = []

    class _Failing(DummyAdminEndpointPlugin):
        name = "failing"

        async def start(self, state: State) -> None:
            await super().start(state)
            raise error("bind failed")

    plugins = [
        _dummy("first", events),
        _StopOnlyPlugin(events, "second"),
        _Failing(events),
        _dummy("never", events),
    ]
    native = _NativeServing(events)
    app = await _build_plugin_app(monkeypatch, plugins, None, native)

    with pytest.raises(error):
        async with app.router.lifespan_context(app):
            pytest.fail("startup must not complete")

    assert events == [
        "start:first",
        "start:failing",
        "stop:second",
        "stop:first",
        "native_shutdown",
    ]


@pytest.mark.asyncio
async def test_cancelled_lifespan_shutdown_still_stops_remaining_plugins(
    monkeypatch: pytest.MonkeyPatch,
):
    """Cancelling the lifespan while a `stop` is pending is deferred until the
    remaining plugins and native cleanup have run, then re-raised."""
    events: list[str] = []
    entered = asyncio.Event()

    class _Blocking(DummyAdminEndpointPlugin):
        name = "blocking"

        async def stop(self, state: State) -> None:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.events.append(f"stop:{self.name}:cancelled")
                raise

    plugins = [_dummy("first", events), _Blocking(events), _dummy("last", events)]
    native = _NativeServing(events)
    app = await _build_plugin_app(monkeypatch, plugins, None, native)

    async def run_lifespan() -> None:
        async with app.router.lifespan_context(app):
            pass

    task = asyncio.create_task(run_lifespan())
    async with asyncio.timeout(_TIMEOUT_S):
        await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(_TIMEOUT_S):
            await task

    assert events == [
        "start:first",
        "start:blocking",
        "start:last",
        "stop:last",
        "stop:blocking:cancelled",
        "stop:first",
        "native_shutdown",
    ]


class _DrainPlugin(_LegacyEndpointPlugin):
    def __init__(self, events: list[str], name: str = "drain"):
        super().__init__(events, name)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cleaned = asyncio.Event()

    async def drain(self, state: State, timeout: float) -> None:
        self.events.append(f"drain:{self.name}")
        self.entered.set()
        try:
            await self.release.wait()
        finally:
            self.events.append(f"settled:{self.name}")
            self.cleaned.set()

    async def stop(self, state: State) -> None:
        self.events.append(f"stop:{self.name}")


@pytest.mark.asyncio
@pytest.mark.parametrize("render", [False, True])
@pytest.mark.parametrize("predicate", [True, False, RuntimeError("not ready"), None])
async def test_ready_checks_loaded_plugins_without_changing_health(
    monkeypatch, render, predicate
):
    class _Ready(_LegacyEndpointPlugin):
        def ready(self, state: State) -> bool:
            if isinstance(predicate, Exception):
                raise predicate
            return predicate

    plugins = [_LegacyEndpointPlugin() if predicate is None else _Ready()]
    engine = None if render else _ServingEngineClient([])
    app = await _build_plugin_app(monkeypatch, plugins, engine)
    # No lifespan/start: readiness applies to every loaded provider.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        expected = 200 if predicate is True or predicate is None else 503
        assert (await client.get("/ready")).status_code == expected
        assert (await client.get("/health")).status_code == 200
        app.state.draining = True
        assert (await client.get("/ready")).status_code == 503
        assert (await client.get("/health")).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [None, "dead", "ordinary"])
async def test_ready_mirrors_engine_health_and_sagemaker_ping(monkeypatch, error):
    from vllm.entrypoints.serve.instrumentator.health import health
    from vllm.entrypoints.serve.sagemaker.api_router import health as ping_health
    from vllm.v1.engine.exceptions import EngineDeadError

    assert ping_health is health

    class _Engine(_ServingEngineClient):
        async def check_health(self) -> None:
            if error == "dead":
                raise EngineDeadError()
            if error == "ordinary":
                raise RuntimeError("health failed")

    app = await _build_plugin_app(monkeypatch, [], _Engine([]))
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        expected = {None: 200, "dead": 503, "ordinary": 500}[error]
        assert (await client.get("/health")).status_code == expected
        assert (await client.get("/ready")).status_code == expected


@pytest.mark.asyncio
async def test_ready_checks_draining_before_and_after_engine_await(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    class _Engine(_ServingEngineClient):
        async def check_health(self) -> None:
            entered.set()
            await release.wait()

    app = await _build_plugin_app(monkeypatch, [], _Engine([]))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        request = asyncio.create_task(client.get("/ready"))
        try:
            async with asyncio.timeout(_TIMEOUT_S):
                await entered.wait()
            app.state.draining = True
            release.set()
            assert (await request).status_code == 503
            entered.clear()
            assert (await client.get("/ready")).status_code == 503
            assert not entered.is_set()
        finally:
            release.set()
            await request


@pytest.mark.asyncio
async def test_accepted_owner_finishes_before_engine_shutdown(monkeypatch):
    events: list[str] = []
    engine = _ServingEngineClient(events)
    plugin = DummyAdminEndpointPlugin(events)
    async with _serving(monkeypatch, [plugin], engine) as (state, task, url):
        await _wait_started(state, task)
        async with httpx.AsyncClient(base_url=url) as client:
            assert (await client.post("/plugins/dummy/session")).status_code == 200
            state.request_shutdown()
            async with asyncio.timeout(_TIMEOUT_S):
                await plugin.drain_entered.wait()
            assert state.draining
            # Reuse the existing HTTP connection to request a new owner.
            assert (await client.post("/plugins/dummy/session")).status_code == 503
            assert (await client.get("/ready")).status_code == 503
            assert (await client.get("/health")).status_code == 200
            assert (await client.post("/plugins/dummy/finish")).status_code == 200
        await _wait_closed(task)
    assert engine.rpc_calls == [("finish_session", (), {})]
    assert events == [
        "start:dummy_admin_endpoint_plugin",
        "draining",
        "owner_final_submit",
        "owner_done",
        "engine_shutdown",
        "stop:dummy_admin_endpoint_plugin",
    ]
    assert engine.shutdown_allowance is not None
    assert 0 < engine.shutdown_allowance < 30


@pytest.mark.asyncio
async def test_drain_timeout_isolates_errors_and_settles_peers(monkeypatch):
    events: list[str] = []
    hanging = _DrainPlugin(events, "hanging")
    healthy = _DrainPlugin(events, "healthy")

    class _Error(_DrainPlugin):
        async def drain(self, state: State, timeout: float) -> None:
            await hanging.entered.wait()
            try:
                raise RuntimeError("drain failed")
            finally:
                events.append("abort:error")
                healthy.release.set()

    engine = _ServingEngineClient(events, shutdown_timeout=1)
    plugins = [hanging, healthy, _Error(events, "error")]
    async with _serving(monkeypatch, plugins, engine) as (state, task, _):
        await _wait_started(state, task)
        state.request_shutdown()
        await _wait_closed(task)
    assert "abort:error" in events
    assert healthy.cleaned.is_set() and hanging.cleaned.is_set()
    assert events.index("settled:healthy") < events.index("settled:hanging")
    assert events.index("settled:hanging") < events.index("engine_shutdown")
    assert events.index("engine_shutdown") < events.index("stop:error")
    assert engine.shutdown_allowance == 0


@pytest.mark.asyncio
async def test_zero_timeout_sets_draining_skips_drains_and_stops(monkeypatch):
    events: list[str] = []
    plugin = _DrainPlugin(events)
    engine = _ServingEngineClient(events, shutdown_timeout=0)
    async with _serving(monkeypatch, [plugin], engine) as (state, task, _):
        await _wait_started(state, task)
        state.request_shutdown()
        await _wait_closed(task)
    assert state.draining
    assert not plugin.entered.is_set()
    assert events == ["engine_shutdown", "stop:drain"]
    assert engine.shutdown_allowance == 0


@pytest.mark.asyncio
async def test_watchdog_exit_during_drain_settles_before_stop(monkeypatch):
    events: list[str] = []
    plugin = _DrainPlugin(events)
    engine = _ServingEngineClient(events)
    async with _serving(monkeypatch, [plugin], engine) as (state, task, _):
        await _wait_started(state, task)
        state.request_shutdown()
        async with asyncio.timeout(_TIMEOUT_S):
            await plugin.entered.wait()
        engine.errored, engine.is_running = True, False
        terminate_if_errored(state.server, engine)
        await _wait_closed(task)
    assert events == ["drain:drain", "settled:drain", "stop:drain"]


@pytest.mark.asyncio
@pytest.mark.parametrize("startup_failure", [False, True])
@pytest.mark.parametrize("already_cancelling", [False, True])
async def test_stop_interlock_defers_repeated_cancellation_until_owners_settle(
    monkeypatch, startup_failure, already_cancelling
):
    events: list[str] = []
    cleanup_entered, release_cleanup = asyncio.Event(), asyncio.Event()
    plugin = _DrainPlugin(events)

    async def shutdown() -> None:
        plugin.entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_entered.set()
            await release_cleanup.wait()
            events.append("owner_settled")

    class _Failing(_StartOnlyPlugin):
        async def start(self, state: State) -> None:
            await plugin.entered.wait()
            raise RuntimeError("startup failed")

    plugins = [plugin, _Failing()] if startup_failure else [plugin]
    app = await _build_plugin_app(monkeypatch, plugins, None, _NativeServing(events))
    shutdown_task = asyncio.create_task(shutdown())
    app.state.shutdown_task = shutdown_task
    if already_cancelling:
        async with asyncio.timeout(_TIMEOUT_S):
            await plugin.entered.wait()
            shutdown_task.cancel()
            await cleanup_entered.wait()

    async def lifespan() -> None:
        async with app.router.lifespan_context(app):
            await plugin.entered.wait()

    task = asyncio.create_task(lifespan())
    # Enter lifespan cleanup before cancelling it, even when shutdown was
    # already unwinding before lifespan started.
    await asyncio.sleep(0)
    try:
        async with asyncio.timeout(_TIMEOUT_S):
            await cleanup_entered.wait()
        task.cancel()
        # Force two independent cancellation deliveries, not two queued counts.
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert events == []
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await _wait_closed(task)
        assert events == ["owner_settled", "stop:drain", "native_shutdown"]
        assert shutdown_task.done()
    finally:
        release_cleanup.set()
        shutdown_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await shutdown_task
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await task


def test_readiness_and_drain_capabilities_are_independent() -> None:
    from vllm.plugins.endpoint_plugins.interface import SupportsDrain, SupportsReady

    class _Ready(_LegacyEndpointPlugin):
        def ready(self, state: State) -> bool:
            return True

    # These assignments are also checked statically by mypy.
    ready: SupportsReady = _Ready()
    drain: SupportsDrain = _DrainPlugin([])
    assert isinstance(ready, SupportsReady)
    assert not isinstance(ready, SupportsDrain)
    assert isinstance(drain, SupportsDrain)
    assert not isinstance(drain, SupportsReady)
    assert isinstance(_LegacyEndpointPlugin(), EndpointPlugin)
    assert not isinstance(_LegacyEndpointPlugin(), SupportsReady)
    assert not isinstance(_LegacyEndpointPlugin(), SupportsDrain)


@pytest.mark.asyncio
async def test_real_render_server_drains_accepted_owner():
    from tests.utils import RemoteLaunchRenderServer

    with RemoteLaunchRenderServer(
        "vllm-project/tiny-random-LlamaForCausalLM",
        ["--shutdown-timeout", "10"],
        env_dict={"VLLM_PLUGINS": "dummy_admin_endpoint_plugin"},
    ) as server:
        async with httpx.AsyncClient(base_url=server.url_for("")) as client:
            assert (await client.post("/plugins/dummy/session")).status_code == 200
            server.proc.send_signal(signal.SIGTERM)
            async with asyncio.timeout(_TIMEOUT_S):
                while (await client.get("/ready")).status_code == 200:
                    await asyncio.sleep(0.01)
            assert (await client.get("/ready")).status_code == 503
            assert (await client.get("/health")).status_code == 200
            assert (await client.post("/plugins/dummy/session")).status_code == 503
            response = await client.post("/plugins/dummy/finish")
            assert response.status_code == 200
            assert response.json() == {"finished_while_draining": True}
        assert await asyncio.to_thread(server.proc.wait, timeout=_TIMEOUT_S) == 0
