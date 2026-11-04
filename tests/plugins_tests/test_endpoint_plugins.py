# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the `vllm.endpoint_plugins` framework (RFC #46565).

Uses the worked in repo example plugin (`vllm_add_dummy_endpoint_plugin`,
installed via `tests/plugins/vllm_add_dummy_endpoint_plugin`) exercising both
`EndpointPlugin` hooks against a fake `EngineClient`, unit tests for the
`load_endpoint_plugins` gating matrix and an e2e test that drives a real HTTP
request through the plugin's route.
"""

import asyncio
import os
import signal
import time
from argparse import Namespace
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
    attach_endpoint_plugins,
    drain_endpoint_plugins,
    init_endpoint_plugins_state,
    stop_endpoint_plugins,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


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
@pytest.mark.parametrize("path", ["/version", "/metrics"])
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
    from starlette.datastructures import State

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
    """A plugin written before the optional lifecycle methods existed."""

    name = "legacy_endpoint_plugin"
    required_tasks = None

    def attach_router(self, app: FastAPI) -> None:
        pass

    async def init_state(self, engine_client, state, args) -> None:
        pass


class _ServingEngineClient(_FakeEngineClient):
    """Fake engine for `serve_http`; records `shutdown` into `events`."""

    def __init__(self, events: list[str], shutdown_timeout: int):
        super().__init__()
        self.events = events
        self.vllm_config = VllmConfig(shutdown_timeout=shutdown_timeout)
        self.errored = False
        self.is_running = True
        self.should_keep_api_server_alive = False
        self.shutdown_timeouts: list[float] = []

    async def check_health(self) -> None:
        pass

    def shutdown(self, timeout: float | None = None) -> None:
        self.events.append("engine_shutdown")
        self.shutdown_timeouts.append(timeout)


async def _build_plugin_app(
    monkeypatch: pytest.MonkeyPatch,
    plugins: list,
    engine_client: _ServingEngineClient | None,
    supported_tasks: tuple = (),
) -> FastAPI:
    monkeypatch.setattr(
        "vllm.plugins.load_endpoint_plugins", lambda supported_tasks: plugins
    )
    args = _build_args()
    app = build_app(args, supported_tasks)
    app.state.engine_client = engine_client
    app.state.log_stats = False
    await init_endpoint_plugins_state(engine_client, app.state, args)
    return app


async def _serve(
    monkeypatch: pytest.MonkeyPatch,
    plugins: list,
    engine_client: _ServingEngineClient | None,
    supported_tasks: tuple = (),
):
    """Start `serve_http` on a free port and wait until it serves.

    Returns the app, the `serve_http` task and the base URL.
    """
    app = await _build_plugin_app(monkeypatch, plugins, engine_client, supported_tasks)
    sock = create_server_socket(("127.0.0.1", 0), reuse_port=False)
    port = sock.getsockname()[1]
    serve_task = asyncio.create_task(
        serve_http(app, sock, host="127.0.0.1", port=port, log_level="warning")
    )
    async with asyncio.timeout(10):
        while not getattr(getattr(app.state, "server", None), "started", False):
            await asyncio.sleep(0.01)
    return app, serve_task, f"http://127.0.0.1:{port}"


async def _health(base_url: str) -> int:
    async with httpx.AsyncClient() as client:
        return (await client.get(f"{base_url}/health")).status_code


def test_legacy_plugin_satisfies_protocol():
    assert isinstance(_LegacyEndpointPlugin(), EndpointPlugin)


@pytest.mark.asyncio
async def test_lifecycle_order_on_request_shutdown(monkeypatch: pytest.MonkeyPatch):
    """`state.draining` is set before any drain runs, drains run concurrently
    before the engine stops, and stops run in reverse start order after HTTP
    stops."""
    events: list[str] = []
    entered = {"first": asyncio.Event(), "second": asyncio.Event()}
    release_drain = asyncio.Event()

    class _Draining(DummyAdminEndpointPlugin):
        async def drain(self, state, timeout):
            events.append(f"drain:{self.name}:draining={state.draining}")
            entered[self.name].set()
            await release_drain.wait()

    class _First(_Draining):
        name = "first"

    class _Second(_Draining):
        name = "second"

    engine = _ServingEngineClient(events, shutdown_timeout=30)
    app, serve_task, url = await _serve(
        monkeypatch, [_First(events), _Second(events)], engine
    )
    assert await _health(url) == 200

    app.state.request_shutdown()
    async with asyncio.timeout(5):  # sequential drains never enter the second
        await asyncio.gather(*(event.wait() for event in entered.values()))
    assert await _health(url) == 503
    release_drain.set()
    shutdown = await serve_task
    await shutdown

    assert events == [
        "start:first",
        "start:second",
        "drain:first:draining=True",
        "drain:second:draining=True",
        "engine_shutdown",
        "stop:second",
        "stop:first",
    ]
    assert 0 < engine.shutdown_timeouts[0] <= 30


@pytest.mark.asyncio
async def test_sigterm_drains_and_stops(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    engine = _ServingEngineClient(events, shutdown_timeout=30)
    app, serve_task, _ = await _serve(
        monkeypatch, [DummyAdminEndpointPlugin(events)], engine
    )

    os.kill(os.getpid(), signal.SIGTERM)
    await (await serve_task)

    name = DummyAdminEndpointPlugin.name
    assert events == [
        f"start:{name}",
        f"drain:{name}",
        "engine_shutdown",
        f"stop:{name}",
    ]


@pytest.mark.asyncio
async def test_failed_start_stops_started_plugins_once(
    monkeypatch: pytest.MonkeyPatch,
):
    """A failing `start` aborts lifespan startup (uvicorn then exits); only
    plugins that started are stopped, in reverse order and exactly once;
    later plugins never start."""
    events: list[str] = []

    def plugin(plugin_name: str, fail: bool = False):
        class _Plugin(DummyAdminEndpointPlugin):
            name = plugin_name

            async def start(self, state):
                await super().start(state)
                if fail:
                    raise RuntimeError("bind failed")

        return _Plugin(events)

    plugins = [plugin("a"), plugin("b"), plugin("c", fail=True), plugin("d")]
    app = await _build_plugin_app(monkeypatch, plugins, engine_client=None)
    with pytest.raises(RuntimeError, match="bind failed"):
        async with app.router.lifespan_context(app):
            pass

    assert events == ["start:a", "start:b", "start:c", "stop:b", "stop:a"]


@pytest.mark.asyncio
async def test_drain_overrun_and_error_are_isolated(monkeypatch: pytest.MonkeyPatch):
    """A hanging drain is cancelled at the deadline and an erroring drain is
    logged; neither stops the other drains, the engine shutdown or `stop`.
    The engine gets only the time left."""
    events: list[str] = []

    class _Hanging(DummyAdminEndpointPlugin):
        name = "hanging"

        async def drain(self, state, timeout):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                events.append("drain:hanging:cancelled")
                raise

    class _Erroring(DummyAdminEndpointPlugin):
        name = "erroring"

        async def drain(self, state, timeout):
            raise RuntimeError("drain failed")

    engine = _ServingEngineClient(events, shutdown_timeout=1)
    plugins = [_Hanging(events), _Erroring(events), DummyAdminEndpointPlugin(events)]
    app, serve_task, _ = await _serve(monkeypatch, plugins, engine)

    app.state.request_shutdown()
    await (await serve_task)

    name = DummyAdminEndpointPlugin.name
    assert f"drain:{name}" in events
    assert events.index("drain:hanging:cancelled") < events.index("engine_shutdown")
    assert [e for e in events if not e.startswith("drain:")][-4:] == [
        "engine_shutdown",
        f"stop:{name}",
        "stop:erroring",
        "stop:hanging",
    ]
    assert 0 <= engine.shutdown_timeouts[0] < 0.5


@pytest.mark.asyncio
async def test_shutdown_timeout_zero_skips_drain(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    engine = _ServingEngineClient(events, shutdown_timeout=0)
    app, serve_task, _ = await _serve(
        monkeypatch, [DummyAdminEndpointPlugin(events)], engine
    )
    state = app.state

    state.request_shutdown()
    shutdown = await serve_task
    assert state.draining is True
    await shutdown

    name = DummyAdminEndpointPlugin.name
    assert events == [f"start:{name}", "engine_shutdown", f"stop:{name}"]
    assert engine.shutdown_timeouts == [0]


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", [True, False], ids=["engine", "render"])
async def test_health_reflects_plugin_readiness(
    monkeypatch: pytest.MonkeyPatch, engine: bool
):
    """`/health` returns 503 while a plugin is not ready or its readiness
    check raises, with an engine and on the engine-less render server."""
    plugin = DummyAdminEndpointPlugin()
    engine_client = _ServingEngineClient([], shutdown_timeout=0) if engine else None
    tasks = ("generate",) if engine else ("render",)
    app, serve_task, url = await _serve(monkeypatch, [plugin], engine_client, tasks)

    assert await _health(url) == 200
    plugin.is_ready = False
    assert await _health(url) == 503

    def raise_error(state):
        raise RuntimeError("listener lost")

    plugin.ready = raise_error
    assert await _health(url) == 503

    app.state.request_shutdown()
    await (await serve_task)


@pytest.mark.asyncio
async def test_watchdog_exit_stops_without_drain(monkeypatch: pytest.MonkeyPatch):
    """When the watchdog stops the server after engine death, `stop` still
    runs exactly once; `drain` and the engine shutdown do not."""
    events: list[str] = []
    engine = _ServingEngineClient(events, shutdown_timeout=30)
    app, serve_task, _ = await _serve(
        monkeypatch, [DummyAdminEndpointPlugin(events)], engine
    )

    engine.errored = True
    engine.is_running = False
    terminate_if_errored(app.state.server, engine)
    await (await serve_task)

    name = DummyAdminEndpointPlugin.name
    assert events == [f"start:{name}", f"stop:{name}"]


class _HangingDrain(DummyAdminEndpointPlugin):
    def __init__(self, events: list[str]):
        super().__init__(events)
        self.entered = asyncio.Event()

    async def drain(self, state, timeout):
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.events.append("drain:cancelled")
            raise


@pytest.mark.asyncio
async def test_cancelled_shutdown_cancels_drains():
    """Cancelling the shutdown cancels running drains rather than leaving them
    behind to race `stop`."""
    state = State()
    state.started_endpoint_plugins = [plugin := _HangingDrain([])]
    task = asyncio.create_task(drain_endpoint_plugins(state, time.monotonic() + 30))
    await asyncio.wait_for(plugin.entered.wait(), 10)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert plugin.events == ["drain:cancelled"]


@pytest.mark.asyncio
async def test_watchdog_exit_during_drain_cancels_drain_before_stop(
    monkeypatch: pytest.MonkeyPatch,
):
    """If the engine dies while plugins drain, the drain is cancelled and
    finished before `stop` runs."""
    events: list[str] = []
    engine = _ServingEngineClient(events, shutdown_timeout=30)
    plugin = _HangingDrain(events)
    app, serve_task, _ = await _serve(monkeypatch, [plugin], engine)

    app.state.request_shutdown()
    await asyncio.wait_for(plugin.entered.wait(), 10)
    engine.errored = True
    engine.is_running = False
    terminate_if_errored(app.state.server, engine)
    await (await serve_task)

    name = DummyAdminEndpointPlugin.name
    assert events == [f"start:{name}", "drain:cancelled", f"stop:{name}"]


@pytest.mark.asyncio
async def test_stop_cancels_running_drain_first():
    """`stop` never overlaps a drain, e.g. when a later plugin's `start`
    fails while a shutdown requested during startup is draining."""
    state = State()
    state.started_endpoint_plugins = [plugin := _HangingDrain([])]
    state.shutdown_task = asyncio.create_task(
        drain_endpoint_plugins(state, time.monotonic() + 30)
    )
    await asyncio.wait_for(plugin.entered.wait(), 10)

    await stop_endpoint_plugins(state)

    assert state.shutdown_task.cancelled()
    assert plugin.events == ["drain:cancelled", f"stop:{plugin.name}"]


@pytest.mark.asyncio
async def test_cancelled_stop_still_stops_plugins():
    """Cancelling `stop` while it waits for a drain to unwind defers the
    cancellation until every plugin has stopped."""

    class _SlowUnwind(_HangingDrain):
        unwinding = asyncio.Event()
        release = asyncio.Event()

        async def drain(self, state, timeout):
            try:
                await super().drain(state, timeout)
            finally:
                self.unwinding.set()
                await self.release.wait()

    state = State()
    state.started_endpoint_plugins = [plugin := _SlowUnwind([])]
    state.shutdown_task = asyncio.create_task(
        drain_endpoint_plugins(state, time.monotonic() + 30)
    )
    await asyncio.wait_for(plugin.entered.wait(), 10)
    stop = asyncio.create_task(stop_endpoint_plugins(state))
    await asyncio.wait_for(plugin.unwinding.wait(), 10)

    stop.cancel()
    await asyncio.sleep(0)
    plugin.release.set()
    with pytest.raises(asyncio.CancelledError):
        await stop
    assert plugin.events == ["drain:cancelled", f"stop:{plugin.name}"]


@pytest.mark.asyncio
async def test_non_http_listener_requires_single_api_server(
    monkeypatch: pytest.MonkeyPatch,
):
    class _ListenerPlugin(DummyAdminEndpointPlugin):
        owns_non_http_listener = True

    monkeypatch.setattr(
        "vllm.plugins.load_endpoint_plugins",
        lambda supported_tasks: [_ListenerPlugin()],
    )
    args = _build_args()
    app = build_app(args, supported_tasks=())

    args.api_server_count = 2
    with pytest.raises(ValueError, match="--api-server-count 1"):
        await init_endpoint_plugins_state(None, app.state, args)

    args.api_server_count = None
    await init_endpoint_plugins_state(None, app.state, args)


@pytest.mark.asyncio
async def test_legacy_plugin_lifecycle_unchanged(monkeypatch: pytest.MonkeyPatch):
    """A plugin without lifecycle methods serves, reports healthy and shuts
    down as before."""
    events: list[str] = []
    engine = _ServingEngineClient(events, shutdown_timeout=30)
    app, serve_task, url = await _serve(monkeypatch, [_LegacyEndpointPlugin()], engine)

    assert await _health(url) == 200
    app.state.request_shutdown()
    await (await serve_task)

    assert events == ["engine_shutdown"]
