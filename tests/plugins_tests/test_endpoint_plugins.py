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
import logging
import os
import signal
import subprocess
import sys
from argparse import Namespace
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import APIRouter, FastAPI, Request
from packaging.version import Version
from starlette.datastructures import State
from vllm_add_dummy_endpoint_plugin import DummyAdminEndpointPlugin

from tests.utils import VLLM_PATH
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

    def __init__(self, events: list[str]):
        super().__init__()
        self.events = events
        self.vllm_config = VllmConfig(shutdown_timeout=30)
        self.errored = False
        self.is_running = True
        self.should_keep_api_server_alive = False

    async def check_health(self) -> None:
        pass

    def shutdown(self, timeout: float | None = None) -> None:
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
) -> FastAPI:
    monkeypatch.setattr(
        "vllm.plugins.load_endpoint_plugins", lambda supported_tasks: plugins
    )
    args = _build_args(f"--endpoint-plugin-stop-timeout={stop_timeout}")
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


@pytest.mark.parametrize("hook", ["start", "stop"])
@pytest.mark.parametrize(
    "value",
    [lambda self, state: None, _AsyncCallable()],
    ids=["sync_function", "async_callable_object"],
)
def test_sync_lifecycle_hook_is_rejected_at_load(
    monkeypatch: pytest.MonkeyPatch, hook: str, value: Any
):
    """A hook that is not an `async def` coroutine function would fail only
    when awaited, or would not be recognized as a hook; reject it by name."""

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
    assert str(error.value) == (
        f"Endpoint plugin {_SyncHook.name}: `{hook}` must be an `async def` "
        f"coroutine function; any other `{hook}` is not supported. Rename the "
        "attribute if it is not a lifecycle hook."
    )


@pytest.mark.parametrize("timeout", ["0", "-1"])
def test_non_positive_stop_timeout_is_refused(timeout: str):
    """The server refuses a stop time limit that is not positive before it
    loads the model."""
    args = _build_args(f"--endpoint-plugin-stop-timeout={timeout}")
    with pytest.raises(
        ValueError, match="--endpoint-plugin-stop-timeout must be positive"
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
