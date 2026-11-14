# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import io
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import pytest

from vllm.utils import system_utils
from vllm.utils.system_utils import (
    _maybe_force_spawn,
    suppress_stdout,
    unique_filepath,
)


def test_unique_filepath():
    temp_dir = tempfile.mkdtemp()
    path_fn = lambda i: Path(temp_dir) / f"file_{i}.txt"
    paths = set()
    for _ in range(10):
        path = unique_filepath(path_fn)
        path.write_text("test")
        paths.add(path)
    assert len(paths) == 10
    assert len(list(Path(temp_dir).glob("*.txt"))) == 10


@pytest.mark.parametrize("start_method", [None, "fork"])
@pytest.mark.parametrize(
    "numa_args, force_spawn",
    [
        (["--numa-bind"], True),
        (["--numa-bind-nodes", "0"], True),
        (["--numa-bind-nodes=0"], True),
        (["--numa-bind-cpus", "0-3"], True),
        (["--numa-bind-cpus=0-3"], True),
        (["--numa_bind"], True),
        (["--numa_bind_nodes", "0"], True),
        (["--numa_bind_cpus=0-3"], True),
        ([], False),
        (["--no-numa-bind"], False),
    ],
)
def test_numa_bind_cli_start_method(monkeypatch, start_method, numa_args, force_spawn):
    if start_method is None:
        monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    else:
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", start_method)
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", *numa_args])
    for check in (
        "is_in_ray_actor",
        "cuda_is_initialized",
        "xpu_is_initialized",
        "in_wsl",
    ):
        monkeypatch.setattr(system_utils, check, lambda: False)

    _maybe_force_spawn()
    expected_method = "spawn" if force_spawn else start_method
    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == expected_method


def test_suppress_stdout_when_sys_stdout_has_no_fd(capfd):
    with contextlib.redirect_stdout(io.StringIO()), suppress_stdout():
        os.write(1, b"c library output\n")
    assert "c library output" not in capfd.readouterr().out


def test_suppress_stdout_keeps_stderr_when_sys_stdout_is_stderr(capfd, monkeypatch):
    monkeypatch.setattr(sys, "stdout", sys.stderr)
    with suppress_stdout():
        os.write(1, b"c library output\n")
        os.write(sys.stderr.fileno(), b"error message\n")
    out, err = capfd.readouterr()
    assert "c library output" not in out
    assert "error message" in err


def test_kill_process_tree_skips_child_that_exits_mid_walk(monkeypatch):
    sleep = [sys.executable, "-c", "import time; time.sleep(60)"]
    target, *children = procs = [subprocess.Popen(sleep) for _ in range(4)]
    try:
        # List the children while they are alive, then let one exit and be
        # reaped before the walk reaches it, so psutil's name() raises.
        listed = [psutil.Process(p.pid) for p in children]
        children[1].terminate()
        children[1].wait(timeout=10)
        monkeypatch.setattr(
            psutil.Process, "children", lambda self, recursive=False: listed
        )

        system_utils.kill_process_tree(target.pid)

        # The exited child died from our SIGTERM; the rest were SIGKILLed.
        assert children[1].returncode == -signal.SIGTERM
        for p in (children[0], children[2], target):
            assert p.wait(timeout=10) == -signal.SIGKILL
    finally:
        for p in procs:
            p.kill()
            p.wait()


def test_forked_child_can_use_openmp_after_parent_did():
    """A forked child deadlocks in libgomp if its parent already ran a parallel
    region, unless get_mp_context() releases the parent's OpenMP workers."""
    script = """
import torch
from vllm.utils.system_utils import get_mp_context

torch.set_num_threads(4)
x = torch.randn(1024, 1024)
torch.einsum("in,jn->ij", x, x)
proc = get_mp_context().Process(target=torch.zeros, args=(64 * 1024 * 1024,))
proc.start()
proc.join(60)
hung = proc.is_alive()
proc.kill()
raise SystemExit(1 if hung else proc.exitcode)
"""
    env = os.environ | {"VLLM_WORKER_MULTIPROC_METHOD": "fork"}
    result = subprocess.run([sys.executable, "-c", script], env=env, timeout=120)
    assert result.returncode == 0


_PARENT_DEATH_PARENT = """
import os
import sys
import threading
import time

from vllm.utils.system_utils import get_mp_context, set_parent_death_signal


def child(pid_file, arm_delay):
    time.sleep(arm_delay)
    set_parent_death_signal()
    with open(pid_file + ".tmp", "w") as f:
        f.write(str(os.getpid()))
    os.replace(pid_file + ".tmp", pid_file)
    time.sleep(600)


def start_and_wait_armed(proc, pid_file):
    proc.start()
    while not os.path.exists(pid_file):
        time.sleep(0.05)


if __name__ == "__main__":
    mode, pid_file = sys.argv[1], sys.argv[2]
    early = mode == "exit_before_arming"
    proc = get_mp_context().Process(
        target=child, args=(pid_file + ".child" if early else pid_file, 2.0 * early)
    )
    if mode == "thread":
        starter = threading.Thread(target=start_and_wait_armed, args=(proc, pid_file))
        starter.start()
        starter.join()
    elif early:
        proc.start()
        with open(pid_file, "w") as f:
            f.write(str(proc.pid))
        os._exit(0)
    else:
        start_and_wait_armed(proc, pid_file)
    time.sleep(600)
"""


def _child_pid(pid_file: Path, timeout: float = 60) -> int:
    deadline = time.monotonic() + timeout
    while not pid_file.exists():
        assert time.monotonic() < deadline, "child never armed"
        time.sleep(0.05)
    return int(pid_file.read_text())


def _gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(sys.platform != "linux", reason="prctl is Linux-only")
@pytest.mark.parametrize("mode", ["kill_parent", "thread", "exit_before_arming"])
def test_set_parent_death_signal(tmp_path, mode):
    script = tmp_path / "parent.py"
    script.write_text(_PARENT_DEATH_PARENT)
    pid_file = tmp_path / "child.pid"
    parent = subprocess.Popen(
        [sys.executable, str(script), mode, str(pid_file)],
        env={**os.environ, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
    )
    child_pid = None
    try:
        if mode == "exit_before_arming":
            parent.wait(timeout=60)
            child_pid = _child_pid(pid_file)
            assert _gone(child_pid, timeout=30)
            return
        child_pid = _child_pid(pid_file)
        if mode == "thread":
            assert not _gone(child_pid, timeout=2)
            return
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=30)
        assert _gone(child_pid, timeout=30)
    finally:
        with contextlib.suppress(ProcessLookupError):
            parent.kill()
        if child_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)
