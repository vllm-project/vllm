# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket
import threading
import time
from contextlib import nullcontext
from multiprocessing import connection
from threading import Event
from types import SimpleNamespace

import msgspec
import pytest
import torch
import zmq

import vllm.platforms as platforms
from vllm.utils import torch_utils
from vllm.v1.engine import core as core_module
from vllm.v1.engine import utils as engine_utils
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.utils import (
    CoreEngine,
    CoreEngineLaunch,
    CoreEngineProcManager,
    EngineZmqAddresses,
    wait_for_engine_startup,
)
from vllm.v1.executor import UniProcExecutor, multiproc_executor

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    (
        "executor_class",
        "local_engine_count",
        "initial_threads",
        "user_threads",
        "fail_start",
        "expected",
    ),
    [
        (UniProcExecutor, 1, 192, None, False, (12, "12", "1")),
        (UniProcExecutor, 2, 192, None, False, (6, "6", "1")),
        (UniProcExecutor, 2, 192, None, True, (6, "6", "1")),
        (UniProcExecutor, 2, 5, None, False, (5, "5", "1")),
        (UniProcExecutor, 2, 1, None, False, (1, "1", "1")),
        (UniProcExecutor, 2, 5, "5", False, (5, "5", None)),
        (
            multiproc_executor.MultiprocExecutor,
            2,
            192,
            None,
            False,
            (192, None, None),
        ),
    ],
)
def test_engine_core_startup_threads_are_scoped_to_launch(
    monkeypatch: pytest.MonkeyPatch,
    executor_class,
    local_engine_count: int,
    initial_threads: int,
    user_threads: str | None,
    fail_start: bool,
    expected: tuple[int, str | None, str | None],
):
    """UniProc workers inherit their CPU share and restore the parent."""
    marker = torch_utils.OMP_NUM_THREADS_SET_BY_VLLM
    monkeypatch.delenv(marker, raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    if user_threads is not None:
        monkeypatch.setenv("OMP_NUM_THREADS", user_threads)
    monkeypatch.setattr(torch_utils, "available_cpu_count", lambda: 12)
    platform = SimpleNamespace(
        is_cpu=lambda: False, is_cuda_alike=lambda: True, is_xpu=lambda: False
    )
    monkeypatch.setattr(engine_utils, "current_platform", platform)
    threads = SimpleNamespace(count=initial_threads)
    setter_calls = []

    def set_threads(count):
        setter_calls.append(count)
        threads.count = count

    monkeypatch.setattr(torch, "get_num_threads", lambda: threads.count)
    monkeypatch.setattr(torch, "set_num_threads", set_threads)
    start_states = []

    def thread_state():
        return (
            threads.count,
            os.environ.get("OMP_NUM_THREADS"),
            os.environ.get(marker),
        )

    def start_process():
        start_states.append(thread_state())
        if fail_start:
            raise RuntimeError("start failed")

    def get_context():
        assert thread_state() == (initial_threads, user_threads, None)
        return SimpleNamespace(
            Process=lambda **kwargs: SimpleNamespace(
                name=kwargs["name"],
                exitcode=1 if fail_start else None,
                start=start_process,
            )
        )

    monkeypatch.setattr(engine_utils, "get_mp_context", get_context)
    monkeypatch.setattr(engine_utils, "shutdown", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        engine_utils.numa_utils,
        "configure_subprocess",
        lambda *args, **kwargs: nullcontext(),
    )
    with (
        pytest.raises(RuntimeError, match="start failed")
        if fail_start
        else nullcontext()
    ):
        manager = CoreEngineProcManager(
            local_engine_count=local_engine_count,
            start_index=0,
            local_start_index=0,
            vllm_config=SimpleNamespace(
                shutdown_timeout=0,
                parallel_config=SimpleNamespace(
                    data_parallel_size=local_engine_count,
                    assigned_physical_gpu_ids=None,
                    use_ray=False,
                ),
            ),
            local_client=True,
            handshake_address="unused",
            executor_class=executor_class,
            log_stats=False,
        )
        manager._finalizer.detach()
    assert start_states == [expected] * (1 if fail_start else local_engine_count)
    assert thread_state() == (initial_threads, user_threads, None)
    assert setter_calls == (
        [expected[0], initial_threads]
        if executor_class is UniProcExecutor and expected[0] < initial_threads
        else []
    )


@pytest.mark.parametrize(
    ("is_rocm", "request_timeout", "manager_timeout", "process_timeout"),
    [
        (True, 0, 0, 15.0),
        (True, 0, 7, 7),
        (True, 0, None, None),
        (False, 0, 0, 0),
        (True, 7, 0, 0),
    ],
)
def test_engine_core_process_shutdown_timeout(
    monkeypatch: pytest.MonkeyPatch,
    is_rocm: bool,
    request_timeout: float | None,
    manager_timeout: float | None,
    process_timeout: float | None,
):
    manager = object.__new__(CoreEngineProcManager)
    manager._request_shutdown_timeout = request_timeout
    manager.manager_stopped = Event()
    manager.processes = [object()]
    detach_results = iter((object(), None))
    manager._finalizer = SimpleNamespace(detach=lambda: next(detach_results))

    shutdown_calls = []
    monkeypatch.setattr(
        engine_utils,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: is_rocm),
    )
    monkeypatch.setattr(
        engine_utils,
        "shutdown",
        lambda processes, timeout: shutdown_calls.append((processes, timeout)),
    )

    manager.shutdown(timeout=manager_timeout)
    manager.shutdown(timeout=manager_timeout)

    assert manager.manager_stopped.is_set()
    assert shutdown_calls == [(manager.processes, process_timeout)]


@pytest.mark.parametrize(
    (
        "is_rocm",
        "shutdown_state",
        "has_work",
        "shutdown_timeout",
        "exit_code",
        "expected_calls",
    ),
    [
        (
            True,
            EngineShutdownState.SHUTTING_DOWN,
            False,
            0,
            None,
            ["shutdown", "freeze"],
        ),
        (False, EngineShutdownState.SHUTTING_DOWN, False, 0, None, ["shutdown"]),
        (True, EngineShutdownState.RUNNING, False, 0, None, ["shutdown"]),
        (True, EngineShutdownState.SHUTTING_DOWN, True, 0, None, ["shutdown"]),
        (True, EngineShutdownState.SHUTTING_DOWN, False, 7, None, ["shutdown"]),
        (True, EngineShutdownState.SHUTTING_DOWN, False, 0, 1, ["shutdown"]),
    ],
)
def test_freeze_gc_after_clean_rocm_engine_core_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    is_rocm: bool,
    shutdown_state: EngineShutdownState,
    has_work: bool,
    shutdown_timeout: int,
    exit_code: int | None,
    expected_calls: list[str],
):
    calls: list[str] = []
    vllm_config = SimpleNamespace(shutdown_timeout=shutdown_timeout)
    proc = SimpleNamespace(
        shutdown_state=EngineShutdownState.RUNNING,
        has_work=lambda: has_work,
        vllm_config=vllm_config,
    )

    def run_busy_loop():
        proc.shutdown_state = shutdown_state
        raise SystemExit(exit_code)

    proc.run_busy_loop = run_busy_loop
    proc.shutdown = lambda: calls.append("shutdown")
    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        numa_bind=False,
        reconfigure_for_independent_dp_rank=lambda: None,
    )
    vllm_config.parallel_config = parallel_config

    for name in (
        "maybe_register_config_serialize_by_value",
        "set_process_title",
        "maybe_init_worker_tracer",
        "decorate_logs",
    ):
        monkeypatch.setattr(core_module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(core_module, "EngineCoreProc", lambda *args, **kwargs: proc)
    monkeypatch.setattr(
        core_module,
        "SignalCallback",
        lambda callback: SimpleNamespace(trigger=lambda: None, stop=lambda: None),
    )
    monkeypatch.setattr(core_module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_rocm=lambda: is_rocm)
    )
    monkeypatch.setattr(core_module.gc, "freeze", lambda: calls.append("freeze"))

    with pytest.raises(SystemExit):
        EngineCoreProc.run_engine_core(vllm_config=vllm_config)

    assert calls == expected_calls


class _FinishedProcess:
    name = "RustFrontend"

    def __init__(self, sentinel):
        self.sentinel = sentinel

    @property
    def exitcode(self):
        return 1


def test_wait_for_engine_startup_reports_watched_process_exit():
    ctx = zmq.Context()
    handshake_socket = ctx.socket(zmq.ROUTER)
    recv, send = connection.Pipe(duplex=False)
    send.close()

    parallel_config = SimpleNamespace(
        data_parallel_size_local=1,
        data_parallel_hybrid_lb=False,
        data_parallel_external_lb=False,
    )

    try:
        launch = CoreEngineLaunch(
            engine_manager=None,
            coordinator=None,
            addresses=EngineZmqAddresses(inputs=[], outputs=[]),
            tensor_queue=None,
        )
        launch.watched_frontend_processes = [_FinishedProcess(recv)]
        with pytest.raises(RuntimeError) as exc_info:
            wait_for_engine_startup(
                handshake_socket,
                [CoreEngine()],
                parallel_config,
                coordinated_dp=False,
                cache_config=None,
                launch=launch,
            )
    finally:
        recv.close()
        handshake_socket.close(linger=0)
        ctx.term()

    assert "Frontend process failed during engine core initialization" in str(
        exc_info.value
    )
    assert "Failed frontend proc(s): {'RustFrontend': 1}" in str(exc_info.value)


# Messages no engine sends, as seen from port scanners plus the shapes the
# handshake parse has to reject.
_STRAY_PAYLOADS = [
    b"\x03",  # a msgpack int
    msgspec.msgpack.encode({"probe": 1}),  # a map, but no handshake fields
    msgspec.msgpack.encode({"status": "HELLO", "local": "yes"}),  # nearly one
    msgspec.msgpack.encode("probe"),
    b"\xa3\xe0\x80\x80",  # a str holding invalid UTF-8: UnicodeDecodeError
    b"\x81\x91\x01\x02",  # a map keyed by an array, which decodes to a tuple key
    b"\x81\x80\x01",  # a map keyed by a map: TypeError before msgspec 0.22
    b"GET / HTTP/1.1\r\n\r\n",
    b"",  # an empty frame
    b"\x91" * 20000 + b"\x01",  # arrays nested past the limit: RecursionError
    b"\x81\x01" * 20000 + b"\x01",  # maps nested the same way
]
# The request in issue #38677: a metrics scraper pointed at the RPC port. With
# no ZMTP greeting, libzmq reads it as an unversioned peer and its first bytes
# as that peer's identity.
_HTTP_SCRAPE = (
    b"GET /metrics HTTP/1.1\r\nHost: 172.26.43.196:13345\r\n"
    b"User-Agent: vm_promscrape\r\nAccept: text/plain;version=0.0.4;q=1,*/*;q=0.1"
    b"\r\nAccept-Encoding: gzip\r\nX-Prometheus-Scrape-Timeout-Seconds: 10\r\n\r\n"
)


class _Startup:
    """``wait_for_engine_startup`` on a ROUTER bound to loopback TCP, as the
    rank 0 front-end binds it in multi-node DP, run on a thread."""

    def __init__(self, core_engines, local_count):
        self.ctx = zmq.Context()
        self.ctx.setsockopt(zmq.LINGER, 0)
        self.socket = self.ctx.socket(zmq.ROUTER)
        port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.address = f"tcp://127.0.0.1:{port}"
        self.port = port
        parallel_config = SimpleNamespace(
            data_parallel_size_local=local_count,
            data_parallel_hybrid_lb=True,
            data_parallel_external_lb=False,
        )
        launch = CoreEngineLaunch(
            engine_manager=None,
            coordinator=None,
            addresses=EngineZmqAddresses(inputs=[], outputs=[]),
            tensor_queue=None,
        )
        self.error: BaseException | None = None
        # Peers stay open until close(): a socket collected early would drop
        # what it has queued before the ROUTER reads it.
        self.peers: list[zmq.Socket] = []
        self.thread = threading.Thread(
            target=self._run,
            args=(core_engines, parallel_config, launch),
            daemon=True,
        )
        self.thread.start()

    def _run(self, core_engines, parallel_config, launch):
        try:
            wait_for_engine_startup(
                self.socket,
                core_engines,
                parallel_config,
                coordinated_dp=False,
                cache_config=None,
                launch=launch,
            )
        except BaseException as e:
            self.error = e

    def finished(self, timeout: float = 15.0) -> bool:
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def connect(self, kind, identity: bytes | None = None, option=None) -> zmq.Socket:
        sock = self.ctx.socket(kind)
        if identity is not None:
            sock.setsockopt(zmq.IDENTITY, identity)
        if option is not None:
            sock.setsockopt(*option)
        sock.connect(self.address)
        self.peers.append(sock)
        return sock

    def send_strays(self) -> None:
        """Everything a ROUTER accepts from a peer that is not an engine."""
        dealer = self.connect(zmq.DEALER)
        for payload in _STRAY_PAYLOADS:
            dealer.send(payload)
        dealer.send_multipart([b"\x03", b"\x03"])
        # An engine-sized identity that no engine of this front-end has.
        self.connect(zmq.DEALER, identity=CoreEngine(7).identity).send(b"\x03")
        for payload in (b"\x03", msgspec.msgpack.encode({"probe": 1})):
            req = self.connect(zmq.REQ)  # an empty delimiter frame first
            req.send(payload)
        router = self.connect(zmq.ROUTER, option=(zmq.CONNECT_ROUTING_ID, b"fe"))
        time.sleep(0.2)  # a ROUTER can only address a peer once connected
        router.send_multipart([b"fe", b"\x03"])
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as raw:
            raw.sendall(_HTTP_SCRAPE)
            time.sleep(0.2)
        time.sleep(0.5)  # let the front-end take all of it off the socket

    def close(self) -> None:
        self.ctx.destroy(linger=0)
        self.thread.join(timeout=5)


def _engine(startup: _Startup, index: int, local: bool, junk_first: bool = False):
    """An engine's side of the handshake, up to and including HELLO."""
    sock = startup.connect(zmq.DEALER, identity=CoreEngine(index).identity)
    if junk_first:
        # Junk under an engine's own identity is dropped too.
        sock.send(b"\x03")
    EngineCoreProc.startup_handshake(sock, local_client=local, headless=False)
    return sock


def _ready(sock: zmq.Socket, local: bool) -> None:
    sock.send(
        msgspec.msgpack.encode({"status": "READY", "local": local, "headless": False})
    )


def test_wait_for_engine_startup_drops_messages_no_engine_sent(monkeypatch):
    """Stray traffic on the handshake socket is dropped and startup completes."""
    monkeypatch.setattr(core_module, "HANDSHAKE_TIMEOUT_MINS", 0.2)
    startup = _Startup([CoreEngine(0, local=True), CoreEngine(1, local=False)], 1)
    try:
        startup.send_strays()
        assert startup.error is None, f"startup failed: {startup.error!r}"
        assert startup.thread.is_alive(), "startup ended before any engine"

        local = _engine(startup, 0, local=True, junk_first=True)
        remote = _engine(startup, 1, local=False)
        startup.send_strays()
        _ready(local, local=True)
        _ready(remote, local=False)

        assert startup.finished(), "startup did not complete"
        assert startup.error is None, f"startup failed: {startup.error!r}"
    finally:
        startup.close()


def test_wait_for_engine_startup_still_fails_on_an_unexpected_rank():
    """A well-formed HELLO from a rank this front-end does not own still fails
    startup instead of being dropped, which would leave it waiting forever."""
    startup = _Startup([CoreEngine(0, local=True)], 1)
    try:
        startup.send_strays()
        stranger = startup.connect(zmq.DEALER, identity=CoreEngine(5).identity)
        stranger.send(
            msgspec.msgpack.encode(
                {"status": "HELLO", "local": False, "headless": False}
            )
        )
        assert startup.finished(), "startup did not fail"
        assert isinstance(startup.error, RuntimeError)
        assert str(startup.error).endswith("unexpected data parallel rank: 5")
    finally:
        startup.close()
