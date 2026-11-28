# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The zygote must start processes that behave like spawned ones, and fall
back to spawn whenever it cannot."""

import atexit
import contextlib
import ctypes
import multiprocessing as mp
import multiprocessing.spawn
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection
from pathlib import Path

import psutil
import pytest

from vllm.utils import zygote
from vllm.utils.system_utils import kill_process_tree

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the zygote is Linux-only"
)


def _report(conn) -> None:
    conn.send(
        {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "parent": mp.parent_process().pid,  # type: ignore[union-attr]
            "name": mp.current_process().name,
            "env": os.environ.get("ZYGOTE_TEST"),
            "library_path": os.environ.get("LD_LIBRARY_PATH"),
            "hash_randomization": sys.flags.hash_randomization,
        }
    )
    conn.close()


def _exit(code: int) -> None:
    sys.exit(code)


def _sleep() -> None:
    time.sleep(60)


def _report_num_threads(conn) -> None:
    import torch

    conn.send(torch.get_num_threads())


def _touch_at_exit(path: str) -> None:
    atexit.register(lambda: open(path, "w").close())


def _report_grandchild(conn) -> None:
    ctx = zygote.get_context()
    r, w = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_report, args=(w,))
    proc.start()
    w.close()
    conn.send(r.recv())
    proc.join()


def _start_sleeper(conn) -> None:
    proc = zygote.get_context().Process(target=_sleep)
    proc.start()
    conn.send(proc.pid)
    proc.join()


def _stubborn_worker(conn, requester):
    received_term = False

    def on_term(*_):
        nonlocal received_term
        received_term = True

    signal.signal(signal.SIGTERM, on_term)
    conn.send((os.environ[zygote.ADDRESS_ENV], os.getppid(), requester, os.getpid()))
    while not received_term:
        time.sleep(0.01)
    conn.send(not psutil.pid_exists(requester))
    _sleep()


def _start_stubborn_worker(conn):
    worker = zygote.get_context().Process(
        target=_stubborn_worker, args=(conn, os.getpid())
    )
    worker.start()
    worker.join()


def _create_orphan_tree(fd, preload_dir):
    sys.path.insert(0, preload_dir)
    zygote.start(preload=["short_orphan_grace"])
    root = zygote.get_context().Process(
        target=_start_stubborn_worker, args=(Connection(fd),)
    )
    root.start()
    root.join()


def _run(ctx, target, *args, **kwargs) -> dict:
    r, w = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(w, *args), **kwargs)
    proc.start()
    w.close()
    info = r.recv()
    proc.join()
    assert proc.exitcode == 0
    return info


@pytest.fixture(scope="module")
def ctx():
    if not zygote._SUPPORTED:
        pytest.skip("the zygote is Linux-only")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "zygote")
        patch.setenv("PYTHONHASHSEED", "42")
        patch.setenv("OMP_NUM_THREADS", "2")
        patch.setenv(zygote.ADDRESS_ENV, "")
        patch.setattr(zygote, "_zygote", None)
        patch.setattr(zygote, "_alive_w", None)
        zygote.start(preload=["torch"])
        try:
            yield zygote.get_context()
        finally:
            assert zygote._alive_w is not None and zygote._zygote is not None
            os.close(zygote._alive_w)
            zygote._zygote.wait(timeout=10)


def test_child_runs_like_a_spawned_process(ctx, monkeypatch):
    monkeypatch.setenv("ZYGOTE_TEST", "set after the zygote started")
    info = _run(ctx, _report, name="Reporter")
    assert info["env"] == "set after the zygote started"
    assert info["parent"] == os.getpid()
    assert info["name"] == "Reporter"
    # Forked by the zygote, not by us.
    assert info["ppid"] not in (os.getpid(), info["pid"])


def test_changed_library_path_is_forwarded_to_zygote_child(ctx, monkeypatch):
    path = "/test/requester/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", path)
    info = _run(ctx, _report)
    assert zygote._zygote is not None and info["ppid"] == zygote._zygote.pid
    assert info["library_path"] == path


def test_exit_code_and_terminate(ctx):
    from vllm.v1.engine.utils import _exited_cleanly

    proc = ctx.Process(target=_exit, args=(3,))
    proc.start()
    assert zygote.connection.wait([proc.sentinel], timeout=10)
    assert proc.forked_by_zygote and not _exited_cleanly(proc)
    proc.join(10)
    assert proc.exitcode == 3

    proc = ctx.Process(target=_sleep, daemon=True)
    proc.start()
    proc.terminate()
    proc.join(10)
    assert proc.exitcode == -signal.SIGTERM


def test_child_sizes_torch_threads_from_its_environment(ctx, monkeypatch):
    # The zygote imported torch under its own environment.
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert _run(ctx, _report_num_threads) == 3


def test_exit_handlers_run_as_in_spawned_process(ctx, tmp_path):
    marker = tmp_path / "exited"
    proc = ctx.Process(target=_touch_at_exit, args=(str(marker),))
    proc.start()
    proc.join()
    assert proc.exitcode == 0 and marker.exists()


def test_grandchild_is_forked_by_the_same_zygote(ctx):
    child = _run(ctx, _report)
    grandchild = _run(ctx, _report_grandchild)
    assert grandchild["ppid"] == child["ppid"]


@linux_only
def test_creator_sigkill_reaps_orphan_tree(tmp_path):
    (tmp_path / "short_orphan_grace.py").write_text(
        "from vllm.utils import zygote\nzygote._ORPHAN_GRACE_S = 1.0\n"
    )
    r, w = mp.Pipe(duplex=False)
    cmd = (
        "from tests.utils_.test_zygote import _create_orphan_tree; "
        f"_create_orphan_tree({w.fileno()}, {str(tmp_path)!r})"
    )
    with (
        r,
        w,
        subprocess.Popen(
            [sys.executable, "-c", cmd],
            pass_fds=(w.fileno(),),
            start_new_session=True,
            stderr=subprocess.PIPE,
        ) as creator,
    ):
        try:
            w.close()
            assert r.poll(60), "grandchild did not start"
            address, server, root, worker = r.recv()
            assert psutil.Process(server).ppid() == creator.pid
            assert psutil.Process(root).ppid() == server
            assert Path(address).is_socket()
            creator.kill()
            # The zygote and resource tracker both hold stderr until exit.
            creator.communicate(timeout=20)
            assert creator.returncode == -signal.SIGKILL
            assert r.poll() and r.recv(), (
                "grandchild must receive SIGTERM after its requester is reaped"
            )
            assert not psutil.pid_exists(root) and not psutil.pid_exists(worker)
            assert not Path(address).parent.exists()
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(creator.pid, signal.SIGKILL)


@pytest.mark.parametrize("reap_first", [False, True])
def test_kill_process_tree_reaches_processes_forked_for_the_tree(ctx, reap_first):
    # The sleeper descends from proc, but its parent is the zygote.
    r, w = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_start_sleeper, args=(w,))
    proc.start()
    w.close()
    sleeper = psutil.Process(r.recv())
    try:
        if reap_first:
            proc.kill()
            proc.join(10)
            assert proc.exitcode == -signal.SIGKILL
        kill_process_tree(proc.pid)
        proc.join(10)
        assert proc.exitcode == -signal.SIGKILL
        sleeper.wait(timeout=10)
    finally:
        if sleeper.is_running():
            sleeper.kill()


def test_changed_exec_time_variable_falls_back_to_spawn(ctx, monkeypatch):
    # Hash randomization is fixed when an interpreter starts: a forked child
    # could not honor the new value.
    with monkeypatch.context() as patch:
        patch.setenv("PYTHONHASHSEED", "0")
        info = _run(ctx, _report)
        assert info["ppid"] == os.getpid()
        assert info["hash_randomization"] == 0

    assert zygote._zygote is not None
    assert _run(ctx, _report)["ppid"] == zygote._zygote.pid


def test_other_interpreter_falls_back_to_spawn(ctx, monkeypatch):
    # The same interpreter under another path, as another virtualenv would be.
    executable = os.path.join(os.path.dirname(sys.executable), ".", "python3")
    assert os.path.exists(executable)
    monkeypatch.setattr(multiprocessing.spawn, "_python_exe", os.fsencode(executable))
    assert _run(ctx, _report)["ppid"] == os.getpid()


@pytest.mark.parametrize("failure", ["missing", "timeout", "hazard"])
def test_failed_zygote_address_is_bypassed_on_subsequent_forks(
    ctx, monkeypatch, tmp_path, failure
):
    live_address = os.environ[zygote.ADDRESS_ENV]
    address = tmp_path / "unavailable.sock"
    monkeypatch.setenv(zygote.ADDRESS_ENV, str(address))

    def reject_fork(listener):
        conn, _ = listener.accept()
        with conn:
            _, fds = zygote._recv_msg(conn, zygote._MAX_FDS)
            try:
                zygote._send_msg(conn, {"error": "preload hazard", "unavailable": True})
            finally:
                for fd in fds:
                    os.close(fd)

    with socket.socket(socket.AF_UNIX) as listener, ThreadPoolExecutor(1) as pool:
        if failure != "missing":
            listener.bind(str(address))
            listener.listen()
            listener.settimeout(5)
        reply = pool.submit(reject_fork, listener) if failure == "hazard" else None
        with monkeypatch.context() as attempt:
            if failure == "timeout":
                # Leave the fork request in the listening socket's backlog.
                attempt.setattr(zygote, "_TIMEOUT_S", 0.1)
            assert _run(ctx, _report)["ppid"] == os.getpid()
        if reply is not None:
            reply.result(timeout=5)

    address.unlink(missing_ok=True)
    address.symlink_to(live_address)
    assert _run(ctx, _report)["ppid"] == os.getpid()
    monkeypatch.setenv(zygote.ADDRESS_ENV, live_address)
    assert zygote._zygote is not None
    assert _run(ctx, _report)["ppid"] == zygote._zygote.pid


@linux_only
def test_start_failure_is_reported(monkeypatch, tmp_path):
    # Unix socket paths are limited to about 100 bytes.
    long_dir = tmp_path / ("d" * 100)
    long_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(long_dir))
    monkeypatch.setattr(zygote, "_zygote", None)
    address = os.environ.get(zygote.ADDRESS_ENV)
    with pytest.raises(zygote.ZygoteUnavailableError):
        zygote.start()
    assert os.environ.get(zygote.ADDRESS_ENV) == address


@pytest.mark.parametrize(
    ("args", "starts"),
    [
        (["serve", "model"], True),
        (["run-batch"], True),
        (["serve", "model", "-h"], False),
        (["run-batch", "--help"], False),
        (["serve", "--help=all"], False),
        (["bench"], False),
        ([], False),
    ],
)
def test_opted_in_cli_falls_back_only_for_engine_launches(monkeypatch, args, starts):
    def unavailable():
        raise zygote.ZygoteUnavailableError("socket unavailable")

    monkeypatch.setattr(zygote, "start", unavailable)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "zygote")
    zygote.maybe_start_for_cli(["vllm", *args])
    expected = "spawn" if starts else "zygote"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == expected


def test_child_env():
    start = {"A": "1", "LD_PRELOAD": "x", "CHANGED": "old"}
    current = {**start, "CHANGED": "new", "ADDED": "1"}
    depends_on = ["LD_PRELOAD"]

    env = zygote._child_env(start, current, {**start, "OWN": "1"}, depends_on)
    # The requester's variables, plus what preloading changed.
    assert env == {**current, "OWN": "1"}
    # A requester that already made the same changes is served too.
    assert zygote._child_env(start, current, current, depends_on) == current

    # Both interpreters can edit their library path independently during imports.
    assert zygote._child_env(
        {"LD_LIBRARY_PATH": "base"},
        {"LD_LIBRARY_PATH": "server:base"},
        {"LD_LIBRARY_PATH": "requester:base"},
        zygote._EXEC_TIME_ENV,
    ) == {"LD_LIBRARY_PATH": "requester:base"}

    changed = {**start, "LD_PRELOAD": "y"}
    assert "LD_PRELOAD" in zygote._child_env(start, current, changed, depends_on)
    changed = {**start, "CHANGED": "other"}
    assert "CHANGED" in zygote._child_env(start, current, changed, depends_on)


def test_fork_hazard_names_threads():
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait, name="preload-helper")
    thread.start()
    try:
        hazard = zygote._fork_hazard()
        assert hazard is not None and "preload-helper" in hazard
    finally:
        stop.set()
        thread.join()


@pytest.mark.parametrize("exitcode", [0, 7, -signal.SIGKILL])
def test_concurrent_exitcode_and_join_read_one_status(monkeypatch, exitcode):
    from vllm.v1.engine.utils import _exited_cleanly

    # Stand in for the OS parent, which reports the status once and closes
    # its end of the pipe. Both observers must see the same status.
    status_r, status_w = os.pipe()
    zygote.write_signed(status_w, exitcode)
    os.close(status_w)

    def launch(popen, proc):
        popen.pid = os.getpid()
        popen.sentinel = status_r

    monkeypatch.setattr(zygote._ZygotePopen, "_launch", launch)
    monkeypatch.setattr(zygote, "_SUPPORTED", True)
    monkeypatch.setenv(zygote.ADDRESS_ENV, "test")

    # Exercise the Linux waitid branch even on systems without waitid.
    def waitid(*args):
        raise ChildProcessError

    for name in ("P_PID", "WEXITED", "WNOWAIT"):
        monkeypatch.setattr(os, name, 0, raising=False)
    monkeypatch.setattr(os, "waitid", waitid, raising=False)

    proc = zygote.get_context().Process()
    proc.start()
    ready = threading.Barrier(2)
    wait = zygote.connection.wait

    def simultaneous_wait(fds, timeout):
        result = wait(fds, timeout)
        ready.wait(timeout=5)
        return result

    monkeypatch.setattr(zygote.connection, "wait", simultaneous_wait)
    try:
        with ThreadPoolExecutor(2) as pool:
            status = pool.submit(lambda: proc.exitcode)
            proc.join()
            assert status.result() == exitcode
        assert proc.exitcode == exitcode
        assert _exited_cleanly(proc) == (exitcode == 0)
    finally:
        os.close(status_r)


@pytest.mark.parametrize("result", [3, 0, 4])
def test_fork_hazard_checks_loaded_cuda_driver(monkeypatch, result):
    class Driver:
        def cuDeviceGetCount(self, count):
            return result

    def open_loaded_driver(name, mode):
        assert name == "libcuda.so.1" and mode == os.RTLD_NOLOAD
        return Driver()

    monkeypatch.setattr(ctypes, "CDLL", open_loaded_driver)
    monkeypatch.setattr(threading, "enumerate", lambda: [threading.main_thread()])
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert (zygote._fork_hazard() is None) == (result == 3)


def test_default_preload_includes_current_gpu_runner(monkeypatch):
    from importlib.util import find_spec

    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: True)
    modules = zygote._default_preload()
    assert modules is not None and "vllm.v1.worker.gpu.model_runner" in modules
    assert all(find_spec(name) is not None for name in modules)


def _report_affinity(conn):
    assert hasattr(os, "sched_getaffinity")
    conn.send((os.getppid(), sorted(os.sched_getaffinity(0))))


@pytest.mark.parametrize("fallback", [False, True])
def test_numa_binding_reaches_child_on_fork_and_spawn(ctx, monkeypatch, fallback):
    from vllm.utils.numa_utils import NumaBinding, zygote_numa_binding

    assert hasattr(os, "sched_getaffinity")
    original = os.sched_getaffinity(0)
    cpus = [min(original)]
    if fallback:
        monkeypatch.setenv(zygote.ADDRESS_ENV, "unreachable")
    token = zygote_numa_binding.set(
        NumaBinding(cpus=cpus, memory_policy=None, nodes=[])
    )
    try:
        ppid, affinity = _run(ctx, _report_affinity)
    finally:
        zygote_numa_binding.reset(token)
    assert (ppid == os.getpid()) == fallback
    assert affinity == cpus
    assert os.sched_getaffinity(0) == original


@linux_only
def test_serve_with_zygote():
    from vllm.platforms import current_platform

    if not current_platform.is_cuda_alike():
        pytest.skip("default zygote preloads require CUDA or ROCm")
    from tests.utils import RemoteOpenAIServer

    model = "Qwen/Qwen3-0.6B"
    with RemoteOpenAIServer(
        model,
        [
            "--distributed-executor-backend",
            "mp",
            "--enforce-eager",
            "--max-model-len",
            "128",
        ],
        env_dict={"VLLM_WORKER_MULTIPROC_METHOD": "zygote"},
    ) as server:
        api = psutil.Process(server.proc.pid)
        (server_zygote,) = [p for p in api.children() if p.name().endswith("::Zygote")]
        # TP1 with the mp executor launches one EngineCore and a separate worker.
        roles = {p.name().split("::")[-1] for p in server_zygote.children()}
        assert {"EngineCore", "Worker"} <= roles
        result = server.get_client().completions.create(
            model=model, prompt="Hello", max_tokens=1, temperature=0
        )
        assert result.usage.completion_tokens == 1
