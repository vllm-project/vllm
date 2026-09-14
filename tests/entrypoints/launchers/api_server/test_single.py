# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ExitStack lifecycle tests for the single-process API server launcher."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
from collections import Counter
from typing import Any
from unittest import mock

import pytest

from vllm.entrypoints.launchers.api_server import single


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        model="m",
        uds=None,
        host="127.0.0.1",
        port=8000,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_ca_certs=None,
        ssl_cert_reqs=None,
        ssl_ciphers=None,
        uvicorn_log_level="info",
        disable_uvicorn_access_log=False,
        h11_max_incomplete_event_size=None,
        h11_max_header_count=None,
    )


async def drain(ticks: int = 10) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.fixture
def env(monkeypatch) -> Any:
    """Patch every external boundary of run_server; return an event recorder."""
    events: list[str] = []
    rec = events.append

    class FakeSocket:
        def close(self) -> None:
            rec("socket.close")

    sock = FakeSocket()
    monkeypatch.setattr(
        single,
        "setup_listen_address",
        lambda a, *, reuse_port: (rec("setup") or "http://x", sock),
    )
    monkeypatch.setattr(
        single,
        "cleanup_listen_socket",
        lambda s, uds=None: (rec("cleanup_socket"), s.close()),
    )

    engine = mock.MagicMock()
    engine.model_config = mock.MagicMock()
    engine.get_supported_tasks = mock.AsyncMock(return_value=[])
    engine.shutdown = mock.MagicMock(side_effect=lambda t: rec("engine.shutdown"))

    eng_args = mock.MagicMock()
    eng_args.create_engine_config.return_value = mock.MagicMock(shutdown_timeout=42)
    monkeypatch.setattr(
        single.AsyncEngineArgs, "from_cli_args", classmethod(lambda cls, a: eng_args)
    )

    async def build_engine(*a, **kw):
        rec("build_engine")
        return engine

    monkeypatch.setattr(single, "build_async_engine_client", build_engine)

    app = mock.MagicMock()
    app.state = mock.MagicMock()

    async def init_app(*a, **kw):
        rec("init_app")
        return app

    monkeypatch.setattr(single, "init_app", init_app)

    started = asyncio.Event()
    release = asyncio.Event()

    class Server:
        def __init__(self, cfg):
            self.should_exit = False

        async def serve(self, sockets=None):
            started.set()
            await release.wait()

    monkeypatch.setattr(single, "init_uvicorn", lambda *a, **kw: mock.MagicMock())
    monkeypatch.setattr(single, "NoSignalServer", Server)

    async def watchdog(*a):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            rec("watchdog.cancel")
            raise

    monkeypatch.setattr(single, "watchdog_loop", watchdog)

    def start_ssl(args, cfg, stack):
        stack.callback(rec, "ssl.cleanup")

    monkeypatch.setattr(single, "start_ssl_refresher_if_needed", start_ssl)

    for name in (
        "set_ulimit",
        "decorate_logs",
        "log_version_and_model",
        "log_non_default_args",
        "validate_api_server_args",
        "init_parser_plugin",
    ):
        monkeypatch.setattr(single, name, lambda *a, **kw: None)
    monkeypatch.setattr(single, "setup_interrupt_handler", lambda: None)

    # Mirror the real function's ExitStack registrations without touching
    # the concrete loop class (patching the abstract base has no effect).
    def install_signals(loop, server, stack):
        for sig in (signal.SIGINT, signal.SIGTERM):
            stack.callback(rec, f"remove({signal.Signals(sig).name})")
        stack.callback(rec, "shutdown.cancel")

    monkeypatch.setattr(single, "_install_signal_handlers", install_signals)

    return mock.MagicMock(
        events=events,
        engine=engine,
        app=app,
        sock=sock,
        started=started,
        release=release,
    )


# ---------------------------------------------------------------------------
# ExitStack — LIFO cleanup and error paths
# ---------------------------------------------------------------------------


class TestExitStack:
    @pytest.mark.asyncio
    async def test_lifo_cleanup_on_clean_exit(self, env) -> None:
        """Registered resources unwind in LIFO order on a clean exit."""
        with contextlib.ExitStack() as stack:
            task = asyncio.create_task(single.run_server(make_args(), stack))
            await env.started.wait()
            env.release.set()
            await task
        await drain()

        e = env.events
        # All resources were released.
        assert "socket.close" in e and "engine.shutdown" in e
        assert "ssl.cleanup" in e and "watchdog.cancel" in e
        assert "remove(SIGINT)" in e and "remove(SIGTERM)" in e

        # LIFO order among synchronous callbacks.
        assert e.index("ssl.cleanup") < e.index("remove(SIGTERM)")
        assert e.index("remove(SIGTERM)") < e.index("remove(SIGINT)")
        assert e.index("remove(SIGINT)") < e.index("engine.shutdown")
        assert e.index("engine.shutdown") < e.index("socket.close")

    @pytest.mark.asyncio
    async def test_socket_released_when_engine_build_fails(
        self, env, monkeypatch
    ) -> None:
        """A failure after bind must still release the socket."""

        async def boom(*a, **kw):
            raise RuntimeError("engine failed")

        monkeypatch.setattr(single, "build_async_engine_client", boom)

        with pytest.raises(RuntimeError, match="engine failed"):  # noqa: SIM117
            with contextlib.ExitStack() as stack:
                await single.run_server(make_args(), stack)
        await drain()

        assert "socket.close" in env.events
        # engine.shutdown was never registered -> must not run.
        assert "engine.shutdown" not in env.events

    @pytest.mark.asyncio
    async def test_cleanup_when_server_raises(self, env, monkeypatch) -> None:
        """A failure in server.serve still unwinds every resource."""

        class Failing:
            def __init__(self, cfg):
                self.should_exit = False

            async def serve(self, sockets=None):
                raise RuntimeError("serve failed")

        monkeypatch.setattr(single, "NoSignalServer", Failing)

        with pytest.raises(RuntimeError, match="serve failed"):  # noqa: SIM117
            with contextlib.ExitStack() as stack:
                await single.run_server(make_args(), stack)
        await drain()

        assert "socket.close" in env.events
        assert "engine.shutdown" in env.events
        assert "ssl.cleanup" in env.events


# ---------------------------------------------------------------------------
# Watchdog cancellation
# ---------------------------------------------------------------------------


class TestWatchdogCancellation:
    @pytest.mark.asyncio
    async def test_cancelled_at_most_once(self, env) -> None:
        """Guard against the duplicate ``watchdog_task.cancel()`` regression.

        ``asyncio.Task.cancel`` is immutable, so count via a task factory.
        """
        calls: list[int] = []

        class SpyTask(asyncio.Task):
            def cancel(self, *a, **kw):
                calls.append(id(self))
                return super().cancel(*a, **kw)

        loop = asyncio.get_running_loop()
        prev = loop.get_task_factory()
        loop.set_task_factory(lambda lp, coro: SpyTask(coro, loop=lp))
        try:
            with contextlib.ExitStack() as stack:
                task = asyncio.create_task(single.run_server(make_args(), stack))
                await env.started.wait()
                env.release.set()
                await task
            await drain()
        finally:
            loop.set_task_factory(prev)

        assert all(n <= 1 for n in Counter(calls).values()), Counter(calls)
