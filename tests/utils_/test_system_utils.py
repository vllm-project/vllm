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
from types import SimpleNamespace

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


@pytest.mark.parametrize("start_method", [None, "fork", "zygote"])
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
    expected_method = (
        "spawn" if force_spawn and start_method != "zygote" else start_method
    )
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


@pytest.mark.parametrize("exit_during_walk", [False, True])
def test_kill_process_tree_reaps_worker_of_exited_requester(
    monkeypatch, exit_during_walk
):
    from vllm.utils import zygote

    with (
        subprocess.Popen([sys.executable, "-c", "pass"]) as parent,
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        ) as worker,
    ):
        try:
            if exit_during_walk:
                children = psutil.Process.children

                def walk(proc, recursive):
                    if proc.pid == parent.pid:
                        parent.wait(timeout=5)
                        raise psutil.NoSuchProcess(proc.pid)
                    return children(proc, recursive=recursive)

                monkeypatch.setattr(psutil.Process, "children", walk)
            else:
                parent.wait(timeout=5)
            monkeypatch.setattr(
                zygote,
                "descendants",
                lambda pids: [worker.pid] if parent.pid in pids else [],
            )
            system_utils.kill_process_tree(parent.pid)
            assert worker.wait(timeout=5) == -signal.SIGKILL
        finally:
            worker.kill()


@pytest.mark.parametrize("wsl", [False, True])
def test_zygote_uses_clean_server_state_and_respects_wsl(monkeypatch, wsl):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "zygote")
    monkeypatch.setattr(system_utils, "is_in_ray_actor", lambda: False)
    monkeypatch.setattr(system_utils, "in_wsl", lambda: wsl)
    for check in ("cuda_is_initialized", "xpu_is_initialized"):
        monkeypatch.setattr(system_utils, check, lambda: pytest.fail("queried parent"))
    expected = "spawn" if wsl else "zygote"
    assert system_utils.get_mp_context().get_start_method() == expected


def test_zygote_respects_ray_actor_spawn(monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "zygote")
    monkeypatch.setattr(system_utils, "is_in_ray_actor", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            get_runtime_context=lambda: SimpleNamespace(gcs_address="cluster-address"),
        ),
    )
    monkeypatch.setenv("RAY_ADDRESS", "")
    assert system_utils.get_mp_context().get_start_method() == "spawn"
    assert os.environ["RAY_ADDRESS"] == "cluster-address"
