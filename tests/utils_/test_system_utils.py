# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import io
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import psutil

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
    for i in range(10):
        path = unique_filepath(path_fn)
        path.write_text("test")
        paths.add(path)
    assert len(paths) == 10
    assert len(list(Path(temp_dir).glob("*.txt"))) == 10


def test_numa_bind_forces_spawn(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "--numa-bind"])
    _maybe_force_spawn()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


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


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def test_kill_process_tree_skips_child_that_exits_mid_walk(monkeypatch):
    target, *children = procs = [_sleeper() for _ in range(4)]
    try:
        # List the children while they are alive, then let one exit and be
        # reaped before the walk reaches it, so psutil's name() raises.
        listed = [psutil.Process(p.pid) for p in children]
        children[1].kill()
        children[1].wait()
        monkeypatch.setattr(
            psutil.Process, "children", lambda self, recursive=False: listed
        )

        system_utils.kill_process_tree(target.pid)

        for p in (children[0], children[2], target):
            assert p.wait(timeout=10) == -signal.SIGKILL
    finally:
        for p in procs:
            p.kill()
            p.wait()


def test_kill_process_tree_returns_when_target_exits_before_children(monkeypatch):
    target = _sleeper()
    list_children = psutil.Process.children

    def exit_then_list(self, recursive=False):
        # The target exits and is reaped after the lookup, so psutil raises
        # NoSuchProcess from inside children().
        target.kill()
        target.wait()
        return list_children(self, recursive=recursive)

    monkeypatch.setattr(psutil.Process, "children", exit_then_list)
    try:
        system_utils.kill_process_tree(target.pid)
    finally:
        target.kill()
        target.wait()
