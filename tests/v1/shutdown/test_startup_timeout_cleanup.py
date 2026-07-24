# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GH-32116: wait_for_engine_startup() startup deadline.

Two failure modes are addressed by the fix:

Problem A – launcher blocks forever on startup timeout
    API server workers time out waiting for the engine ready signal
    (VLLM_ENGINE_READY_TIMEOUT_S), die, but the launcher's
    wait_for_engine_startup() had no matching deadline so it kept blocking
    until warmup completed (up to 19 min in the reported case), holding all
    GPU memory during that window.

Problem B – SIGINT / exception during startup bypasses explicit cleanup
    Any BaseException propagated through the launch_core_engines() generator
    (Ctrl-C SystemExit, RuntimeError from engine dying mid-handshake)
    bypassed the explicit cleanup call to local_engine_manager.shutdown(),
    falling back to weakref.finalize() with a hardcoded 5 s grace and no
    logging.

These tests exercise wait_for_engine_startup() in isolation (no GPU, no real
model).  Integration-level verification (full launch_core_engines() with a
real subprocess and GPU memory check) is left to manual testing with the
VLLM_ENGINE_READY_TIMEOUT_S environment variable.
"""

import multiprocessing as mp
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import zmq

from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.v1.engine.utils import CoreEngine, EngineZmqAddresses, wait_for_engine_startup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sleep_forever() -> None:
    """Subprocess target: sleep until killed.

    Stands in for an engine core that is still warming up and has not yet
    sent its HELLO message on the handshake socket.
    """
    time.sleep(3600)


def _make_parallel_config(*, local: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        data_parallel_size_local=local,
        data_parallel_hybrid_lb=False,
        data_parallel_external_lb=False,
    )


def _make_proc_manager(proc: mp.process.BaseProcess) -> MagicMock:
    """Return a CoreEngineProcManager mock backed by a real process.

    The real process contributes a working sentinel FD so the ZMQ poller can
    register it for liveness detection just as production code does.
    """
    pm = MagicMock()
    pm.sentinels.return_value = [proc.sentinel]
    pm.finished_procs.return_value = {}
    return pm


@pytest.fixture()
def silent_engine_proc():
    """A real subprocess that sleeps forever and never sends HELLO."""
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_sleep_forever, daemon=False)
    proc.start()
    yield proc
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)


# ---------------------------------------------------------------------------
# Problem A: wait_for_engine_startup must respect VLLM_ENGINE_READY_TIMEOUT_S
# ---------------------------------------------------------------------------


def test_wait_for_engine_startup_raises_timeout_on_silent_engine(
    monkeypatch: pytest.MonkeyPatch,
    silent_engine_proc: "mp.Process",
) -> None:
    """wait_for_engine_startup() must raise TimeoutError promptly when
    VLLM_ENGINE_READY_TIMEOUT_S elapses with no HELLO from the engine.

    Before the fix the function had no overall deadline and would block until
    the engine finally sent HELLO (potentially minutes), so this test would
    hang forever on unpatched code.
    """
    timeout_s = 2
    monkeypatch.setattr(
        "vllm.v1.engine.utils.envs.VLLM_ENGINE_READY_TIMEOUT_S",
        timeout_s,
    )

    ipc_addr = get_open_zmq_ipc_path()
    addresses = EngineZmqAddresses(inputs=["unused"], outputs=["unused"])
    parallel_config = _make_parallel_config(local=1)
    proc_manager = _make_proc_manager(silent_engine_proc)
    engine = CoreEngine(index=0, local=True)

    start = time.monotonic()
    with (
        zmq_socket_ctx(ipc_addr, zmq.ROUTER, bind=True) as sock,
        pytest.raises(TimeoutError),
    ):
        wait_for_engine_startup(
            sock,
            addresses,
            [engine],
            parallel_config,
            coordinated_dp=False,
            cache_config=MagicMock(),
            proc_manager=proc_manager,
            coord_process=None,
        )
    elapsed = time.monotonic() - start

    # Must fire close to the deadline:
    #   lower bound: >= 90% of the budget (not spuriously early)
    #   upper bound: <  3× the budget (not stuck for a whole extra poll cycle)
    assert elapsed >= timeout_s * 0.9, (
        f"TimeoutError fired too early: {elapsed:.2f}s "
        f"(expected >= {timeout_s * 0.9:.2f}s)"
    )
    assert elapsed < timeout_s * 3, (
        f"wait_for_engine_startup blocked for {elapsed:.2f}s; "
        f"expected < {timeout_s * 3:.2f}s — deadline logic may be missing"
    )


def test_wait_for_engine_startup_timeout_message_is_informative(
    monkeypatch: pytest.MonkeyPatch,
    silent_engine_proc: "mp.Process",
) -> None:
    """The TimeoutError message must mention the env var so users know how
    to extend the timeout for large models."""
    monkeypatch.setattr(
        "vllm.v1.engine.utils.envs.VLLM_ENGINE_READY_TIMEOUT_S",
        1,
    )
    ipc_addr = get_open_zmq_ipc_path()
    addresses = EngineZmqAddresses(inputs=["unused"], outputs=["unused"])
    engine = CoreEngine(index=0, local=True)

    with (
        zmq_socket_ctx(ipc_addr, zmq.ROUTER, bind=True) as sock,
        pytest.raises(TimeoutError, match="VLLM_ENGINE_READY_TIMEOUT_S"),
    ):
        wait_for_engine_startup(
            sock,
            addresses,
            [engine],
            _make_parallel_config(local=1),
            coordinated_dp=False,
            cache_config=MagicMock(),
            proc_manager=_make_proc_manager(silent_engine_proc),
            coord_process=None,
        )


# ---------------------------------------------------------------------------
# Happy path regression: fix must not break normal startup
# ---------------------------------------------------------------------------


def _fake_engine_handshake(ipc_addr: str, identity: bytes) -> None:
    """Simulate the two-step HELLO/READY handshake an engine performs on the
    launcher's ROUTER socket.  Runs in a subprocess."""
    import msgspec
    import zmq as _zmq

    ctx = _zmq.Context.instance()
    sock = ctx.socket(_zmq.DEALER)
    sock.identity = identity
    try:
        sock.connect(ipc_addr)
        sock.send(
            msgspec.msgpack.encode(
                {"status": "HELLO", "local": True, "headless": False}
            )
        )
        sock.recv()  # wait for init reply from launcher
        sock.send(
            msgspec.msgpack.encode(
                {"status": "READY", "local": True, "headless": False}
            )
        )
        time.sleep(1)  # keep alive until launcher drains the event
    finally:
        sock.close(linger=0)
        ctx.term()


def test_wait_for_engine_startup_succeeds_on_hello_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wait_for_engine_startup() must return normally when the engine
    completes the HELLO/READY handshake within the timeout budget."""
    monkeypatch.setattr(
        "vllm.v1.engine.utils.envs.VLLM_ENGINE_READY_TIMEOUT_S",
        10,
    )

    ipc_addr = get_open_zmq_ipc_path()
    addresses = EngineZmqAddresses(inputs=["unused"], outputs=["unused"])
    engine = CoreEngine(index=0, local=True)
    identity = engine.identity

    ctx_mp = mp.get_context("fork")
    engine_proc = ctx_mp.Process(
        target=_fake_engine_handshake,
        args=(ipc_addr, identity),
        daemon=True,
    )
    engine_proc.start()
    try:
        proc_manager = _make_proc_manager(engine_proc)
        with zmq_socket_ctx(ipc_addr, zmq.ROUTER, bind=True) as sock:
            # Must complete without raising.
            wait_for_engine_startup(
                sock,
                addresses,
                [engine],
                _make_parallel_config(local=1),
                coordinated_dp=False,
                cache_config=MagicMock(),
                proc_manager=proc_manager,
                coord_process=None,
            )
    finally:
        engine_proc.kill()
        engine_proc.join(timeout=5)
