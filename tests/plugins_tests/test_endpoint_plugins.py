# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the `vllm.endpoint_plugins` framework (RFC #46565).

Uses the worked in repo example plugin (`vllm_add_dummy_endpoint_plugin`,
installed via `tests/plugins/vllm_add_dummy_endpoint_plugin`) exercising both
`EndpointPlugin` hooks against a fake `EngineClient`, unit tests for the
`load_endpoint_plugins` gating matrix, an e2e test that drives a real HTTP
request through the plugin's route, and lifecycle tests that run the optional
`start`/`stop` capabilities through `serve_http` and the server lifespan. Live
engine tests require an accepted owner to generate real tokens after drain begins.
"""

import asyncio
import contextlib
import contextvars
import gc
import inspect
import logging
import os
import signal
import subprocess
import sys
import time
import warnings
from argparse import Namespace
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx
import pytest
import uvicorn
from fastapi import APIRouter, FastAPI, Request
from packaging.version import Version
from starlette.datastructures import State
from vllm_add_dummy_endpoint_plugin import DummyAdminEndpointPlugin

from tests.utils import VLLM_PATH, RemoteLaunchRenderServer
from vllm.config import VllmConfig
from vllm.entrypoints.common.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.launchers.app import build_app
from vllm.entrypoints.launchers.launcher import (
    create_server_socket,
    serve_http,
    terminate_if_errored,
)
from vllm.logger import configure_logging_from_args
from vllm.plugins import load_endpoint_plugins
from vllm.plugins.endpoint_plugins.interface import (
    EndpointPlugin,
    _PluginLifecycle,
    attach_endpoint_plugins,
    cancel_endpoint_plugin_startup,
    init_endpoint_plugins_state,
    start_endpoint_plugins,
    stop_endpoint_plugins,
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


def _build_args(*cli_args: str) -> Namespace:
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers()
    serve_parser = subparsers.add_parser("serve")
    make_arg_parser(serve_parser)
    return serve_parser.parse_args(list(cli_args))


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
@pytest.mark.parametrize("path", ["/version", "/metrics", "/ready"])
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


class _StopOnlyPlugin(_LegacyEndpointPlugin):
    async def stop(self, state: State) -> None:
        self.events.append(f"stop:{self.name}")


def _dummy(plugin_name: str, events: list[str]) -> DummyAdminEndpointPlugin:
    plugin = DummyAdminEndpointPlugin(events)
    plugin.name = plugin_name
    return plugin


class _ServingEngineClient(_FakeEngineClient):
    """External engine service double for `serve_http`: healthy until core
    calls `shutdown`, which it records in `events`."""

    def __init__(self, events: list[str], shutdown_timeout: float = 30):
        super().__init__()
        self.events = events
        self.vllm_config = VllmConfig(shutdown_timeout=shutdown_timeout)
        self.errored = False
        self.is_running = True
        self.should_keep_api_server_alive = False

    async def check_health(self) -> None:
        pass

    def shutdown(self, timeout: float | None = None) -> None:
        self.shutdown_allowance = timeout
        self.events.append("engine_shutdown")


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
    stop_timeout: int = 5,
    cli_args: tuple[str, ...] = (),
) -> FastAPI:
    monkeypatch.setattr(
        "vllm.plugins.load_endpoint_plugins", lambda supported_tasks: plugins
    )
    args = _build_args(f"--endpoint-plugin-stop-timeout={stop_timeout}", *cli_args)
    app = build_app(args, supported_tasks=())
    app.state.engine_client = engine_client
    app.state.log_stats = False
    if native is not None:
        app.state.openai_serving_transcription = native
    await init_endpoint_plugins_state(engine_client, app.state, args)
    return app


class _AsyncCallable:
    async def __call__(self, state: State) -> None:
        pass


async def _async_ready(self, state: State) -> bool:
    return True


def _async_hook_required(hook: str) -> str:
    return (
        f"`{hook}` must be an `async def` coroutine function; any other `{hook}` "
        "is not supported. Rename the attribute if it is not a lifecycle hook."
    )


_SYNC_READY_REQUIRED = "; define `ready` with `def`, or rename the attribute."


@pytest.mark.parametrize(
    "hook, value, message",
    [
        *(
            pytest.param(hook, value, _async_hook_required(hook), id=f"{hook}-{kind}")
            for hook in ("start", "stop", "drain")
            for kind, value in (
                ("sync_function", lambda self, state: None),
                ("async_callable_object", _AsyncCallable()),
            )
        ),
        pytest.param(
            "ready",
            _async_ready,
            "an async `ready` is not supported" + _SYNC_READY_REQUIRED,
            id="ready-async_function",
        ),
        pytest.param(
            "ready",
            True,
            "`ready` is not callable" + _SYNC_READY_REQUIRED,
            id="ready-not_callable",
        ),
    ],
)
def test_invalid_hook_is_rejected_at_load(
    monkeypatch: pytest.MonkeyPatch, hook: str, value: Any, message: str
):
    """A hook that the server cannot call as intended would fail only when
    called, or would not be recognized as a hook; reject it by name."""

    class _SyncHook(_LegacyEndpointPlugin):
        name = "sync_hook_plugin"

    setattr(_SyncHook, hook, value)
    monkeypatch.setenv("VLLM_PLUGINS", _SyncHook.name)
    monkeypatch.setattr(
        "vllm.plugins.load_plugins_by_group",
        _fake_loader({_SyncHook.name: _SyncHook}),
    )
    with pytest.raises(TypeError) as error:
        load_endpoint_plugins(())
    assert str(error.value) == f"Endpoint plugin {_SyncHook.name}: {message}"


@pytest.mark.parametrize("timeout", ["0", "-1"])
def test_non_positive_stop_timeout_is_refused(timeout: str):
    """The server refuses a stop time limit that is not positive before it
    loads the model."""
    args = _build_args(f"--endpoint-plugin-stop-timeout={timeout}")
    with pytest.raises(
        ValueError, match="--endpoint-plugin-stop-timeout must be positive"
    ):
        validate_parsed_serve_args(args)


@pytest.mark.parametrize(
    "cli_args",
    [
        pytest.param(("--endpoint-plugin-drain-timeout=-1",), id="negative"),
        pytest.param(
            ("--shutdown-timeout=2", "--endpoint-plugin-drain-timeout=3"),
            id="above_shutdown_timeout",
        ),
    ],
)
def test_drain_timeout_outside_shutdown_timeout_is_refused(cli_args: tuple[str, ...]):
    """The server refuses a drain time limit that is negative or exceeds
    `--shutdown-timeout` before it loads the model."""
    args = _build_args(*cli_args)
    with pytest.raises(
        ValueError,
        match="--endpoint-plugin-drain-timeout must be between 0 and "
        "--shutdown-timeout",
    ):
        validate_parsed_serve_args(args)


@pytest.mark.asyncio
async def test_plugins_stop_in_reverse_with_errors_isolated(
    monkeypatch: pytest.MonkeyPatch,
):
    """Plugins start in load order before serving; every eligible plugin,
    including a stop-only one, stops once in reverse order even when a peer's
    `stop` raises or cannot be called; native transcription cleanup runs
    last."""
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

    class _BadSignature(DummyAdminEndpointPlugin):
        name = "bad_signature"

        async def stop(self) -> None:  # type: ignore[override]
            raise AssertionError("a stop without `state` cannot be called")

    plugins = [
        _dummy("first", events),
        _StopOnlyPlugin(events, "second"),
        _Erroring(events),
        _Cancelling(events),
        _BadSignature(events),
    ]
    app = await _build_plugin_app(monkeypatch, plugins, None, _NativeServing(events))

    async with app.router.lifespan_context(app):
        events.append("serving")

    assert events == [
        "start:first",
        "start:erroring",
        "start:cancelling",
        "start:bad_signature",
        "serving",
        "stop:cancelling",
        "stop:erroring",
        "stop:second",
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


class _HangingStopPlugin(DummyAdminEndpointPlugin):
    """`stop` never returns on its own. With `release` it also ignores
    cancellation until the test sets it."""

    def __init__(
        self,
        events: list[str],
        release: asyncio.Event | None = None,
        name: str = "hanging",
    ):
        super().__init__(events)
        self.name = name
        self.stopping = asyncio.Event()
        self.release = release

    async def stop(self, state: State) -> None:
        await super().stop(state)
        self.stopping.set()
        if self.release is None:
            await asyncio.Event().wait()
            return
        while not self.release.is_set():
            with contextlib.suppress(asyncio.CancelledError):
                await self.release.wait()


@pytest.mark.asyncio
async def test_hung_stop_is_bounded_during_startup_rollback(
    monkeypatch: pytest.MonkeyPatch,
):
    """The stop time limit also covers rollback: once a preceding plugin's
    `stop` hangs past it, earlier plugins are not stopped, native cleanup
    runs and the failed `start` still surfaces its own error."""
    events: list[str] = []

    class _Failing(DummyAdminEndpointPlugin):
        name = "failing"

        async def start(self, state: State) -> None:
            await super().start(state)
            raise RuntimeError("bind failed")

    plugins = [_dummy("first", events), _HangingStopPlugin(events), _Failing(events)]
    app = await _build_plugin_app(
        monkeypatch, plugins, None, _NativeServing(events), stop_timeout=1
    )

    with pytest.raises(RuntimeError, match="bind failed"):
        async with asyncio.timeout(_TIMEOUT_S), app.router.lifespan_context(app):
            pytest.fail("startup must not complete")

    assert events == [
        "start:first",
        "start:hanging",
        "start:failing",
        "stop:hanging",
        "native_shutdown",
    ]


@pytest.mark.asyncio
async def test_lifespan_cancellation_during_stop_is_bounded_and_reraised(
    monkeypatch: pytest.MonkeyPatch,
):
    """Cancelling the lifespan task while a `stop` runs cancels that hook. When
    the hook ignores it, the wait still ends at the stop time limit, native
    cleanup runs, and the cancellation is re-raised."""
    events: list[str] = []
    release = asyncio.Event()
    hanging = _HangingStopPlugin(events, release)
    plugins = [_dummy("first", events), hanging]
    app = await _build_plugin_app(
        monkeypatch, plugins, None, _NativeServing(events), stop_timeout=1
    )

    async def run_lifespan() -> None:
        async with app.router.lifespan_context(app):
            events.append("serving")

    task = asyncio.create_task(run_lifespan())
    try:
        async with asyncio.timeout(_TIMEOUT_S):
            await hanging.stopping.wait()
        task.cancel()
        # Awaiting `task` here would wait for the hook if the deadline failed.
        done, _ = await asyncio.wait({task}, timeout=_TIMEOUT_S)
        assert done, "lifespan cleanup waited for the hung stop"
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events == [
        "start:first",
        "start:hanging",
        "serving",
        "stop:hanging",
        "native_shutdown",
    ]


@pytest.fixture
def plugin_logs(caplog_vllm: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    # Building the app can import a dependency that raises the root logger's
    # level, which would hide these warnings from the first test that runs.
    caplog_vllm.set_level(logging.INFO, logger="vllm.plugins")
    return caplog_vllm


@pytest.mark.asyncio
async def test_stop_hooks_share_one_time_limit(
    monkeypatch: pytest.MonkeyPatch, plugin_logs: pytest.LogCaptureFixture
):
    """The stop time limit is total, not per hook: once one hung `stop` uses
    it up, core abandons that hook, does not call the remaining `stop` hooks
    and names both plugins in one warning."""
    events: list[str] = []
    plugins = [
        _HangingStopPlugin(events, name="first"),
        _HangingStopPlugin(events, name="second"),
    ]
    app = await _build_plugin_app(
        monkeypatch, plugins, None, _NativeServing(events), stop_timeout=1
    )

    async with asyncio.timeout(_TIMEOUT_S), app.router.lifespan_context(app):
        events.append("serving")

    assert events == [
        "start:first",
        "start:second",
        "serving",
        "stop:second",
        "native_shutdown",
    ]
    warnings = [r.getMessage() for r in plugin_logs.records]
    assert [w for w in warnings if "stop hooks" in w] == [
        (
            "Endpoint plugin stop hooks exceeded the 1 second limit "
            "(--endpoint-plugin-stop-timeout); abandoned: second; skipped: first"
        )
    ]
    assert await _pending_plugin_tasks() == []


_plugin_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_plugin_context", default="unset"
)


class _TaskScopedPlugin(DummyAdminEndpointPlugin):
    """Holds task-scoped resources from `start` to `stop`: an anyio cancel
    scope, which an MCP `ClientSession` opens, and a context variable."""

    def __init__(self, events: list[str], name: str):
        super().__init__(events)
        self.name = name
        self.tasks: list[asyncio.Task | None] = []

    async def start(self, state: State) -> None:
        await super().start(state)
        self.tasks.append(asyncio.current_task())
        self.scope = anyio.CancelScope()
        self.scope.__enter__()
        _plugin_context.set(self.name)

    async def stop(self, state: State) -> None:
        self.tasks.append(asyncio.current_task())
        self.events.append(f"context:{_plugin_context.get()}")
        self.scope.__exit__(None, None, None)
        await super().stop(state)


@pytest.mark.asyncio
async def test_start_and_stop_run_in_one_task_per_plugin(
    monkeypatch: pytest.MonkeyPatch,
):
    """Each plugin's `start` and `stop` run in the same task, and each plugin
    has its own task, so a plugin can exit in `stop` an anyio cancel scope it
    entered in `start` and read context variables it set there."""
    events: list[str] = []
    plugins = [_TaskScopedPlugin(events, "first"), _TaskScopedPlugin(events, "second")]
    app = await _build_plugin_app(monkeypatch, plugins, None, _NativeServing(events))

    async with asyncio.timeout(_TIMEOUT_S), app.router.lifespan_context(app):
        events.append("serving")

    assert events == [
        "start:first",
        "start:second",
        "serving",
        "context:second",
        "stop:second",
        "context:first",
        "stop:first",
        "native_shutdown",
    ]
    assert await _pending_plugin_tasks() == []
    first, second = plugins
    assert first.tasks[0] is first.tasks[1]
    assert second.tasks[0] is second.tasks[1]
    assert first.tasks[0] is not second.tasks[0]


class _HangingStartPlugin(DummyAdminEndpointPlugin):
    """`start` waits until cancelled, then records the cancellation."""

    name = "hanging_start"

    def __init__(self, events: list[str]):
        super().__init__(events)
        self.starting = asyncio.Event()

    async def start(self, state: State) -> None:
        await super().start(state)
        self.starting.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.events.append("start_cancelled:hanging_start")
            raise


@pytest.mark.asyncio
async def test_cancelled_startup_stops_only_started_plugins(
    monkeypatch: pytest.MonkeyPatch,
):
    """Cancelling plugin startup cancels the `start` that is running. That
    plugin is not stopped, later plugins never start, and the plugins ahead
    of it stop in reverse order."""
    events: list[str] = []
    hanging = _HangingStartPlugin(events)
    plugins = [_dummy("first", events), hanging, _dummy("never", events)]
    app = await _build_plugin_app(monkeypatch, plugins, None, _NativeServing(events))

    async def run_lifespan() -> None:
        async with app.router.lifespan_context(app):
            pytest.fail("startup must not complete")

    task = asyncio.create_task(run_lifespan())
    async with asyncio.timeout(_TIMEOUT_S):
        await hanging.starting.wait()
        cancel_endpoint_plugin_startup(app.state)
        with pytest.raises(asyncio.CancelledError):
            await task

    assert events == [
        "start:first",
        "start:hanging_start",
        "start_cancelled:hanging_start",
        "stop:first",
        "native_shutdown",
    ]


@pytest.mark.asyncio
async def test_plugin_task_cancelled_before_shutdown_stops_once():
    """If something other than shutdown cancels a started plugin's task, for
    example event loop teardown, the task calls `stop` before it ends, and a
    later shutdown does not call `stop` again."""
    events: list[str] = []
    plugin = _TaskScopedPlugin(events, "first")
    state = State()
    state.endpoint_plugins = [plugin]
    state.args = _build_args()

    async with asyncio.timeout(_TIMEOUT_S):
        await start_endpoint_plugins(state)
        (plugin_task,) = (t for t in asyncio.all_tasks() if t in plugin.tasks)
        plugin_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await plugin_task
        await stop_endpoint_plugins(state)

    assert events == ["start:first", "context:first", "stop:first"]
    assert plugin.tasks[0] is plugin.tasks[1]


async def _pending_plugin_tasks() -> list[asyncio.Task]:
    """Plugin lifecycle tasks still running, after letting ended ones settle."""
    for _ in range(3):
        await asyncio.sleep(0)
    return [
        t
        for t in asyncio.all_tasks()
        if not t.done()
        and "_PluginLifecycle" in getattr(t.get_coro(), "__qualname__", "")
    ]


class _CleanupOnCancelStartPlugin(DummyAdminEndpointPlugin):
    """`start` waits until cancelled, then waits for `release` before ending,
    raising `error` if set."""

    name = "slow_cancel"

    def __init__(self, events: list[str], error: type[Exception] | None = None):
        super().__init__(events)
        self.starting = asyncio.Event()
        self.cleaning = asyncio.Event()
        self.release = asyncio.Event()
        self.error = error

    async def start(self, state: State) -> None:
        await super().start(state)
        self.starting.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.events.append("start_cancelled:slow_cancel")
            self.cleaning.set()
            while not self.release.is_set():
                with contextlib.suppress(asyncio.CancelledError):
                    await self.release.wait()
            if self.error is not None:
                raise self.error("cleanup failed") from None
            raise


@pytest.mark.asyncio
@pytest.mark.parametrize("cancels", [1, 2])
@pytest.mark.parametrize("error", [None, RuntimeError])
async def test_lifespan_cancellation_during_start(
    monkeypatch: pytest.MonkeyPatch,
    plugin_logs: pytest.LogCaptureFixture,
    cancels: int,
    error: type[Exception] | None,
):
    """Cancelling the lifespan task, once or again while the running `start`
    cleans up, cancels that `start` and waits for it to end. The plugin is not
    stopped and the plugins ahead of it stop in reverse order. The lifespan
    re-raises the cancellation after cleanup; an exception raised by `start`
    while cleaning up is its context."""
    events: list[str] = []
    slow = _CleanupOnCancelStartPlugin(events, error)
    plugins = [_dummy("first", events), slow, _dummy("never", events)]
    app = await _build_plugin_app(monkeypatch, plugins, None, _NativeServing(events))

    async def run_lifespan() -> None:
        async with app.router.lifespan_context(app):
            pytest.fail("startup must not complete")

    task = asyncio.create_task(run_lifespan())
    async with asyncio.timeout(_TIMEOUT_S):
        await slow.starting.wait()
        task.cancel()
        await slow.cleaning.wait()
        for _ in range(cancels - 1):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        slow.release.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
    if error is not None:
        assert isinstance(exc_info.value.__context__, error)

    assert events == [
        "start:first",
        "start:slow_cancel",
        "start_cancelled:slow_cancel",
        "stop:first",
        "native_shutdown",
    ]
    assert not [r for r in plugin_logs.records if "failed to stop" in r.getMessage()]
    assert await _pending_plugin_tasks() == []


@pytest.mark.asyncio
async def test_start_cancelled_before_plugin_task_runs():
    """Cancelling startup after a plugin's task is created, before the task
    first runs, ends the wait for `start` instead of hanging it."""
    events: list[str] = []
    lifecycle = _PluginLifecycle(_dummy("first", events), State())
    lifecycle.cancel_start()
    async with asyncio.timeout(_TIMEOUT_S):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.wait_started()
    assert events == []


@pytest.mark.asyncio
async def test_skipped_plugin_task_cancelled_does_not_stop():
    """A plugin skipped at the stop time limit is never stopped, even if its
    task is cancelled before it wakes up."""
    events: list[str] = []
    lifecycle = _PluginLifecycle(_dummy("first", events), State())
    async with asyncio.timeout(_TIMEOUT_S):
        assert not await lifecycle.wait_started()
        lifecycle.end_without_stop()
        lifecycle.task.cancel()
        await lifecycle.task
    assert events == ["start:first"]


@pytest.mark.asyncio
async def test_repeated_startup_cancellation_does_not_interrupt_cleanup():
    """Shutdown can call `cancel_endpoint_plugin_startup` more than once, for
    example a signal followed by a watchdog exit. Only the first call cancels
    the running `start`, so the cleanup it does before propagating the
    cancellation runs to completion."""
    events: list[str] = []
    release = asyncio.Event()

    class _Cleanup(DummyAdminEndpointPlugin):
        name = "cleanup"
        starting = asyncio.Event()
        cleaning = asyncio.Event()

        async def start(self, state: State) -> None:
            self.starting.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cleaning.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    events.append("cleanup_interrupted")
                    raise
                events.append("cleanup_done")
                raise

    plugin = _Cleanup(events)
    state = State()
    state.endpoint_plugins = [plugin]
    task = asyncio.create_task(start_endpoint_plugins(state))
    async with asyncio.timeout(_TIMEOUT_S):
        await plugin.starting.wait()
        cancel_endpoint_plugin_startup(state)
        await plugin.cleaning.wait()
        cancel_endpoint_plugin_startup(state)
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert events == ["cleanup_done"]


_EXITING_START_CHILD = """
import asyncio, sys
from starlette.datastructures import State
from vllm_add_dummy_endpoint_plugin import DummyAdminEndpointPlugin
from vllm.plugins.endpoint_plugins.interface import (
    start_endpoint_plugins, stop_endpoint_plugins)

class Exiting(DummyAdminEndpointPlugin):
    name = "exiting"
    async def start(self, state):
        raise {error}

class Printed(list):
    def append(self, event):
        super().append(event)
        print("EVENT=" + event, flush=True)

first = DummyAdminEndpointPlugin(Printed())
first.name = "first"
state = State()
state.endpoint_plugins = [first, Exiting()]

async def lifespan():
    try:
        await start_endpoint_plugins(state)
    finally:
        await stop_endpoint_plugins(state)

asyncio.run(lifespan())
"""


@pytest.mark.parametrize(
    ("error", "returncode"),
    [("SystemExit(42)", 42), ("KeyboardInterrupt", -signal.SIGINT)],
)
def test_process_exit_from_start(error: str, returncode: int):
    """`SystemExit` and `KeyboardInterrupt` from `start` leave the event loop,
    as they do from any asyncio task. The plugins ahead stop once during event
    loop teardown, and nothing reports an unretrieved task exception."""
    result = subprocess.run(
        [sys.executable, "-c", _EXITING_START_CHILD.format(error=error)],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    )
    output = result.stdout + result.stderr
    assert result.returncode == returncode, output
    events = [
        line.removeprefix("EVENT=")
        for line in result.stdout.splitlines()
        if line.startswith("EVENT=")
    ]
    assert events == ["start:first", "stop:first"], output
    for text in ("never retrieved", "failed to stop", "cancelled before shutdown"):
        assert text not in output, output


@pytest.mark.asyncio
async def test_failed_task_group_child_stops_plugin_once(
    plugin_logs: pytest.LogCaptureFixture,
):
    """When a child of an anyio task group opened in `start` fails, the task
    group cancels the plugin's task. The task logs a warning and calls `stop`,
    and shutdown does not call `stop` again."""
    events: list[str] = []
    child_failed = asyncio.Event()

    class _TaskGroupPlugin(DummyAdminEndpointPlugin):
        name = "task_group"

        async def start(self, state: State) -> None:
            await super().start(state)
            self.stack = contextlib.AsyncExitStack()
            group = await self.stack.enter_async_context(anyio.create_task_group())

            async def child() -> None:
                await asyncio.sleep(0)
                child_failed.set()
                raise RuntimeError("transport closed")

            group.start_soon(child)

        async def stop(self, state: State) -> None:
            await super().stop(state)
            with contextlib.suppress(Exception):
                await self.stack.aclose()

    state = State()
    state.endpoint_plugins = [_TaskGroupPlugin(events)]
    state.args = _build_args()

    async with asyncio.timeout(_TIMEOUT_S):
        await start_endpoint_plugins(state)
        await child_failed.wait()
        await _pending_plugin_tasks()
        await stop_endpoint_plugins(state)

    assert events == ["start:task_group", "stop:task_group"]
    assert [
        r.getMessage()
        for r in plugin_logs.records
        if "cancelled before" in r.getMessage()
    ] == [
        (
            "Endpoint plugin task_group task was cancelled before shutdown; "
            "calling its stop hook"
        )
    ]
    assert await _pending_plugin_tasks() == []


class _PrintedEvents(list):
    """Events a child process reports as they happen, so the parent sees
    them even when the child is terminated by a signal."""

    def append(self, event: str) -> None:
        super().append(event)
        print(f"EVENT={event}", flush=True)


_STOP_CASES = ("held_stop", "hung_stop", "stop_bad_signature")


async def _serve_in_child(
    case: str,
    trigger: str = "SIGTERM",
    stop_signals: tuple[str, ...] = (),
) -> None:
    """Run real `serve_http` under uvloop and shut it down as `case` says.

    Plugin "B" blocks in `start` until cancelled, or raises from it in
    `start_raises`. In `_STOP_CASES` its `start` returns, `trigger` requests
    shutdown once the server is serving, and its `stop` is held: `held_stop`
    sends itself `stop_signals`, returning once each is handled; `hung_stop`
    waits until cancelled. In `stop_bad_signature` its `stop` does not accept
    `state`, so calling it raises.
    """
    # As the launchers do, so that the parent can read this process's warnings.
    configure_logging_from_args(_build_args())
    events = _PrintedEvents()
    engine = _ServingEngineClient(events)
    blocked = asyncio.Event()

    class _Blocking(DummyAdminEndpointPlugin):
        name = "B"

        def attach_router(self, app: FastAPI) -> None:
            @app.post("/plugins/B/request_shutdown")
            async def request_shutdown(raw_request: Request):
                raw_request.app.state.request_shutdown()

        async def start(self, state: State) -> None:
            await super().start(state)
            assert not state.server.started
            if case in _STOP_CASES:
                return
            blocked.set()
            if case == "start_raises":
                raise RuntimeError("bind failed")
            if case == "request_shutdown":
                state.request_shutdown()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                events.append("partial_cleanup:B")
                raise

        async def stop(self, state: State) -> None:
            await super().stop(state)
            if case == "hung_stop":
                await asyncio.Event().wait()
            # A signal has been handled once the SIGUSR1 sent after it has.
            handled = asyncio.Event()
            asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, handled.set)
            for name in stop_signals:
                os.kill(os.getpid(), signal.Signals[name])
                os.kill(os.getpid(), signal.SIGUSR1)
                await handled.wait()
                handled.clear()

    class _BadSignature(_Blocking):
        async def stop(self) -> None:  # type: ignore[override]
            raise AssertionError("a stop without `state` cannot be called")

    blocking = _BadSignature if case == "stop_bad_signature" else _Blocking

    # The real launcher's engine context outlives `serve_http`, including when
    # Uvicorn raises SystemExit on startup failure.
    @asynccontextmanager
    async def engine_context():
        try:
            yield
        finally:
            events.append("engine_context_exit")

    async with engine_context(), contextlib.AsyncExitStack() as stack:
        monkeypatch = stack.enter_context(pytest.MonkeyPatch.context())
        plugins = [_dummy("A", events), blocking(events), _dummy("C", events)]
        # Only a `stop` meant to outlive the limit gets a short one.
        stop_timeout = 1 if case == "hung_stop" else 6 * _TIMEOUT_S
        app = await _build_plugin_app(
            monkeypatch, plugins, engine, _NativeServing(events), stop_timeout
        )
        state = app.state
        sock = create_server_socket(("127.0.0.1", 0), reuse_port=False)
        port = sock.getsockname()[1]
        server_task = asyncio.create_task(
            serve_http(
                app,
                sock,
                host="127.0.0.1",
                port=port,
                log_level="warning",
            )
        )

        async with asyncio.timeout(_TIMEOUT_S):
            if case in _STOP_CASES:
                while not getattr(getattr(state, "server", None), "started", False):
                    await asyncio.sleep(0.01)
            else:
                await blocked.wait()
        if case == "watchdog":
            engine.errored, engine.is_running = True, False
            terminate_if_errored(state.server, engine)
        elif case in _STOP_CASES and trigger == "request_shutdown":
            async with httpx.AsyncClient() as client:
                url = f"http://127.0.0.1:{port}/plugins/B/request_shutdown"
                assert (await client.post(url)).status_code == 200
        elif case in _STOP_CASES:
            os.kill(os.getpid(), signal.Signals[trigger])
        elif case not in ("request_shutdown", "start_raises"):
            os.kill(os.getpid(), signal.SIGTERM)

        async with asyncio.timeout(_TIMEOUT_S):
            try:
                await (await server_task)
            finally:
                sock.close()
        with pytest.raises(httpx.ConnectError):
            async with httpx.AsyncClient() as client:
                await client.get(f"http://127.0.0.1:{port}/health")


_ROLLED_BACK = ["partial_cleanup:B", "stop:A", "native_shutdown"]
_EXITED = ["engine_context_exit"]
_STOPPING = ["start:C", "engine_shutdown", "stop:C", "stop:B"]
_STOPPED = [*_STOPPING, "stop:A", "native_shutdown", *_EXITED]
# Uvicorn >= 0.50 exits with status 3 when lifespan startup fails; older
# versions return from `serve`.
_STARTUP_FAILED = 3 if Version(uvicorn.__version__) >= Version("0.50") else 0


def _held_stop(trigger: str, *stop_signals: str) -> dict:
    return {"trigger": trigger, "stop_signals": stop_signals}


@pytest.mark.parametrize(
    "case, kwargs, returncode, expected",
    [
        (
            "sigterm",
            {},
            _STARTUP_FAILED,
            ["engine_shutdown", *_ROLLED_BACK, *_EXITED],
        ),
        (
            "request_shutdown",
            {},
            _STARTUP_FAILED,
            ["engine_shutdown", *_ROLLED_BACK, *_EXITED],
        ),
        ("watchdog", {}, _STARTUP_FAILED, [*_ROLLED_BACK, *_EXITED]),
        (
            "start_raises",
            {},
            _STARTUP_FAILED,
            ["stop:A", "native_shutdown", *_EXITED],
        ),
        ("held_stop", _held_stop("SIGINT", "SIGTERM"), 0, _STOPPED),
        ("held_stop", _held_stop("request_shutdown", "SIGTERM"), 0, _STOPPED),
        ("hung_stop", {}, 0, [*_STOPPING, "native_shutdown", *_EXITED]),
        (
            "stop_bad_signature",
            {},
            0,
            [event for event in _STOPPED if event != "stop:B"],
        ),
    ],
)
def test_shutdown_during_plugin_lifecycle(
    case: str, kwargs: dict, returncode: int, expected: list[str]
):
    """Shutdown during a blocked `start` cancels it and rolls back the
    started plugins before the listener closes, and the process exits as
    Uvicorn does after a failed startup. Signals that arrive while `stop`
    hooks run do not skip the rest of cleanup. A `stop` that outlives its
    time limit is cancelled and does not hold up the rest of shutdown, and
    one that cannot be called does not hold up the other `stop` hooks."""
    code = (
        "import runpy, uvloop; "
        f"scope = runpy.run_path({__file__!r}); "
        f"uvloop.run(scope['_serve_in_child']({case!r}, **{kwargs!r}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=6 * _TIMEOUT_S,
        env={
            **os.environ,
            # The child runs this file, which imports `tests.utils`.
            "PYTHONPATH": str(VLLM_PATH.resolve())
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        },
    )
    events = [
        line.removeprefix("EVENT=")
        for line in result.stdout.splitlines()
        if line.startswith("EVENT=")
    ]
    output = result.stdout + result.stderr
    assert result.returncode == returncode, output
    assert events == ["start:A", "start:B", *expected], output
    abandoned = "abandoned: B; skipped: A" in output
    assert abandoned == (case == "hung_stop"), output
    # Every plugin task ends and every task exception is retrieved.
    assert "Task was destroyed but it is pending" not in output, output
    assert "exception was never retrieved" not in output, output


class _StartOnlyPlugin(_LegacyEndpointPlugin):
    async def start(self, state: State) -> None:
        self.events.append(f"start:{self.name}")


@asynccontextmanager
async def _serving(
    monkeypatch: pytest.MonkeyPatch,
    plugins: list,
    engine: _ServingEngineClient,
    *cli_args: str,
):
    """Run real `serve_http` on a loopback socket; yield (state, task, url)."""
    app = await _build_plugin_app(
        monkeypatch, plugins, engine, _NativeServing([]), cli_args=cli_args
    )
    sock = create_server_socket(("127.0.0.1", 0), reuse_port=False)
    port = sock.getsockname()[1]
    task = asyncio.create_task(
        serve_http(app, sock, host="127.0.0.1", port=port, log_level="warning")
    )
    try:
        yield app.state, task, f"http://127.0.0.1:{port}"
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(BaseException):
            await task
        sock.close()


async def _wait_started(state: State, task: asyncio.Task) -> None:
    async with asyncio.timeout(_TIMEOUT_S):
        while not getattr(getattr(state, "server", None), "started", False):
            assert not task.done(), "server exited before starting"
            await asyncio.sleep(0.01)


async def _wait_closed(task: asyncio.Task) -> None:
    async with asyncio.timeout(_TIMEOUT_S):
        shutdown = await task
        if inspect.isawaitable(shutdown):
            await shutdown


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


class _ReadyPlugin(_LegacyEndpointPlugin):
    def __init__(self, result: object, name: str | None = None):
        super().__init__(name=name)
        self.result = result

    def ready(self, state: State) -> Any:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plugin, engine_health, draining, ready_status",
    [
        pytest.param(None, "ok", False, 200, id="no-ready-capability"),
        pytest.param(False, "ok", False, 503, id="plugin-not-ready"),
        pytest.param(RuntimeError("x"), "ok", False, 503, id="plugin-raises"),
        pytest.param(True, "ok", True, 503, id="draining"),
        pytest.param(True, "dead", False, 503, id="engine-dead"),
        pytest.param(True, "error", False, 500, id="engine-error"),
        pytest.param(True, "error", True, 503, id="draining-before-engine-error"),
        pytest.param(True, "drains", False, 503, id="draining-during-engine-check"),
        pytest.param(
            True, "drains-then-error", False, 503, id="draining-during-engine-error"
        ),
        pytest.param(False, None, False, 503, id="render-not-ready"),
    ],
)
async def test_ready_reports_plugin_readiness_and_draining(
    monkeypatch, plugin, engine_health, draining, ready_status
):
    from vllm.v1.engine.exceptions import EngineDeadError

    class _Engine(_ServingEngineClient):
        async def check_health(self) -> None:
            if engine_health == "dead":
                raise EngineDeadError()
            if engine_health == "error":
                raise RuntimeError("health failed")
            if engine_health in ("drains", "drains-then-error"):
                app.state.draining = True
            if engine_health == "drains-then-error":
                raise RuntimeError("health failed")

    engine = _Engine([]) if engine_health else None
    plugins = [_LegacyEndpointPlugin() if plugin is None else _ReadyPlugin(plugin)]
    app = await _build_plugin_app(monkeypatch, plugins, engine)
    # No lifespan/start: readiness applies to every loaded provider.
    app.state.draining = draining
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/ready")).status_code == ready_status


async def _probe_ready(monkeypatch: pytest.MonkeyPatch, plugin: Any) -> list[int]:
    """Probe `/ready` twice on a render server app with only `plugin`."""
    app = await _build_plugin_app(monkeypatch, [plugin], None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return [(await client.get("/ready")).status_code for _ in range(2)]


def _messages(caplog: pytest.LogCaptureFixture, plugin_name: str) -> list[str]:
    messages = (record.getMessage() for record in caplog.records)
    return [message for message in messages if plugin_name in message]


@pytest.mark.asyncio
async def test_ready_returning_a_coroutine_is_not_ready_and_closed(monkeypatch):
    """A synchronous wrapper that returns a coroutine passes the loader check;
    the coroutine counts as not ready and is closed, never left unawaited."""
    coroutines: list[Any] = []

    class _Wrapper(_LegacyEndpointPlugin):
        name = "coroutine_ready_plugin"

        def ready(self, state: State) -> Any:
            coroutines.append(_async_ready(self, state))
            return coroutines[-1]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert await _probe_ready(monkeypatch, _Wrapper()) == [503, 503]
        states = [inspect.getcoroutinestate(coroutine) for coroutine in coroutines]
        coroutines.clear()
        gc.collect()
    assert states == [inspect.CORO_CLOSED] * 2
    assert not [w for w in caught if "never awaited" in str(w.message)]


@pytest.mark.asyncio
async def test_non_bool_ready_result_is_not_ready_and_logged_once(
    monkeypatch, plugin_logs
):
    name = "truthy_ready_plugin"
    assert await _probe_ready(monkeypatch, _ReadyPlugin(1, name)) == [503, 503]
    assert _messages(plugin_logs, name) == [
        (
            f"Endpoint plugin {name}: `ready` must return a bool; "
            "treating the plugin as not ready"
        )
    ]


@pytest.mark.asyncio
async def test_slow_ready_is_logged_once(monkeypatch, plugin_logs):
    """`ready` runs on the event loop for every probe, so one slow call is
    reported, but only once."""

    class _SlowReady(_LegacyEndpointPlugin):
        name = "slow_ready_plugin"

        def ready(self, state: State) -> bool:
            time.sleep(0.15)
            return True

    assert await _probe_ready(monkeypatch, _SlowReady()) == [200, 200]
    assert _messages(plugin_logs, _SlowReady.name) == [
        (
            f"Endpoint plugin {_SlowReady.name}: a `ready` call took over 100 ms; "
            "`ready` runs on the event loop for every /ready probe and must "
            "return quickly"
        )
    ]


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
    # Drains finished early, and engine shutdown gets at least half of 30 s.
    assert 15 <= engine.shutdown_allowance < 30


@pytest.mark.asyncio
async def test_drain_timeout_isolates_errors_and_settles_peers(
    monkeypatch, plugin_logs
):
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
    assert engine.shutdown_allowance is not None
    assert 0 < engine.shutdown_allowance <= 0.5
    assert _messages(plugin_logs, "deadline") == [
        (
            "Endpoint plugin drain deadline expired; cancelling drains still "
            "running: hanging"
        )
    ]


@pytest.mark.asyncio
async def test_drain_past_half_the_shutdown_timeout_leaves_engine_the_rest(
    monkeypatch,
):
    """Plugin drains get the first half of `--shutdown-timeout`: a drain still
    running then is cancelled, and engine shutdown gets the remainder."""
    events: list[str] = []
    plugin = _DrainPlugin(events)
    engine = _ServingEngineClient(events, shutdown_timeout=1)
    async with _serving(monkeypatch, [plugin], engine) as (state, task, _):
        await _wait_started(state, task)
        state.request_shutdown()
        await _wait_closed(task)
    assert not plugin.release.is_set()
    assert events == ["drain:drain", "settled:drain", "engine_shutdown", "stop:drain"]
    assert engine.shutdown_allowance is not None
    assert 0 < engine.shutdown_allowance <= 0.5


@pytest.mark.asyncio
async def test_drain_past_the_drain_timeout_leaves_engine_the_rest(monkeypatch):
    """`--endpoint-plugin-drain-timeout` replaces the half split: a drain still
    running after 1 of 3 seconds is cancelled, and engine shutdown gets the
    remaining 2 seconds rather than 1.5."""
    events: list[str] = []
    plugin = _DrainPlugin(events)
    engine = _ServingEngineClient(events, shutdown_timeout=3)
    drain_timeout = "--endpoint-plugin-drain-timeout=1"
    async with _serving(monkeypatch, [plugin], engine, drain_timeout) as (
        state,
        task,
        _,
    ):
        await _wait_started(state, task)
        state.request_shutdown()
        await _wait_closed(task)
    assert not plugin.release.is_set()
    assert events == ["drain:drain", "settled:drain", "engine_shutdown", "stop:drain"]
    assert engine.shutdown_allowance is not None
    assert 1.5 < engine.shutdown_allowance <= 2


@pytest.mark.asyncio
async def test_drain_deadline_warning_is_logged_before_drains_settle(
    monkeypatch, plugin_logs
):
    """The warning names the drains still running when the deadline expires,
    before they are cancelled, so a drain that ignores cancellation is named
    while it holds shutdown."""
    events: list[str] = []
    release = asyncio.Event()

    class _IgnoresCancellation(_DrainPlugin):
        async def drain(self, state: State, timeout: float) -> None:
            while not release.is_set():
                with contextlib.suppress(asyncio.CancelledError):
                    await release.wait()
            events.append("settled:ignores")

    warning = (
        "Endpoint plugin drain deadline expired; cancelling drains still "
        "running: ignores"
    )
    engine = _ServingEngineClient(events, shutdown_timeout=1)
    plugin = _IgnoresCancellation(events, "ignores")
    try:
        async with _serving(monkeypatch, [plugin], engine) as (state, task, _):
            await _wait_started(state, task)
            state.request_shutdown()
            async with asyncio.timeout(_TIMEOUT_S):
                while _messages(plugin_logs, "deadline") != [warning]:
                    await asyncio.sleep(0.01)
            assert events == []
            release.set()
            await _wait_closed(task)
    finally:
        release.set()
    assert events == ["settled:ignores", "engine_shutdown", "stop:ignores"]


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
async def test_repeated_shutdown_cancellation_waits_for_drain_cleanup(monkeypatch):
    events: list[str] = []
    cleanup_entered, release_cleanup = asyncio.Event(), asyncio.Event()

    class _SlowCleanup(_DrainPlugin):
        async def drain(self, state: State, timeout: float) -> None:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_entered.set()
                await release_cleanup.wait()
                events.append("settled:drain")

    plugin = _SlowCleanup(events)
    engine = _ServingEngineClient(events)
    async with _serving(monkeypatch, [plugin], engine) as (state, task, _):
        await _wait_started(state, task)
        state.request_shutdown()
        async with asyncio.timeout(_TIMEOUT_S):
            await plugin.entered.wait()
            state.shutdown_task.cancel()
            await cleanup_entered.wait()
        state.shutdown_task.cancel()
        await asyncio.sleep(0)
        assert not state.shutdown_task.done()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await state.shutdown_task
        engine.errored, engine.is_running = True, False
        terminate_if_errored(state.server, engine)
        await _wait_closed(task)
    assert events == ["settled:drain", "stop:drain"]


@pytest.mark.asyncio
@pytest.mark.parametrize("startup_failure", [False, True])
@pytest.mark.parametrize("already_cancelling", [False, True])
async def test_stop_interlock_defers_repeated_cancellation_until_owners_settle(
    monkeypatch, startup_failure, already_cancelling
):
    events: list[str] = []
    cleanup_entered, release_cleanup = asyncio.Event(), asyncio.Event()
    startup_reached = asyncio.Event()
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
            startup_reached.set()
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
            startup_reached.set()
            await plugin.entered.wait()

    task = asyncio.create_task(lifespan())
    try:
        async with asyncio.timeout(_TIMEOUT_S):
            # Startup must reach the predecessor before cancellation can test
            # whether its cleanup waits for shutdown's session owners.
            await startup_reached.wait()
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


@pytest.mark.asyncio
async def test_real_render_server_drains_accepted_owner():
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


_LIVE_MODEL = "vllm-project/tiny-random-LlamaForCausalLM"
_LIVE_ARGS = [
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "256",
    "--enforce-eager",
    "--gpu-memory-utilization",
    "0.05",
    "--max-num-seqs",
    "4",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown_source", ["sigterm", "callback"])
async def test_live_engine_drains_accepted_owner(shutdown_source: str):
    """An accepted owner can submit real generation after host drain begins."""
    from tests.entrypoints.launchers.test_shutdown import (
        _assert_children_cleaned_up,
        _get_child_pids,
    )
    from tests.utils import RemoteOpenAIServer

    with RemoteOpenAIServer(
        _LIVE_MODEL,
        [*_LIVE_ARGS, "--shutdown-timeout", "30"],
        env_dict={"VLLM_PLUGINS": "dummy_admin_endpoint_plugin"},
        max_wait_seconds=480,
    ) as server:
        child_pids = _get_child_pids(server.proc.pid)
        async with httpx.AsyncClient(
            base_url=server.url_for(""),
            timeout=30,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        ) as client:
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/ready")).status_code == 200
            assert (await client.post("/plugins/dummy/generate")).status_code == 200

            accepted = await client.post(
                "/plugins/dummy/session", params={"real_generation": True}
            )
            assert accepted.status_code == 200
            assert accepted.json() == {"accepted": True}
            connection = accepted.extensions["network_stream"]
            if shutdown_source == "sigterm":
                server.proc.send_signal(signal.SIGTERM)
            else:
                shutdown = await client.post("/plugins/dummy/shutdown")
                assert shutdown.status_code == 200
                assert shutdown.json() == {"shutdown_requested": True}

            async with asyncio.timeout(_TIMEOUT_S):
                while (await client.get("/ready")).status_code == 200:
                    await asyncio.sleep(0.01)
            assert (await client.get("/ready")).status_code == 503
            assert (await client.get("/health")).status_code == 200
            rejected = await client.post(
                "/plugins/dummy/session", params={"real_generation": True}
            )
            assert rejected.status_code == 503
            assert rejected.extensions["network_stream"] is connection
            assert (await client.post("/plugins/dummy/generate")).status_code == 503

            response = await client.post("/plugins/dummy/finish")
            assert response.status_code == 200
            result = response.json()
            assert result["submitted_while_draining"] is True
            assert result["finished_while_draining"] is True
            assert result["finished"] is True
            assert result["finish_reason"] == "length"
            assert len(result["token_ids"]) == 16

        assert (
            await asyncio.to_thread(
                server.proc.wait, timeout=server._get_process_termination_timeout()
            )
            == 0
        )
        await _assert_children_cleaned_up(child_pids)
