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
    """A forked vLLM worker can use OpenMP after its parent ran a parallel region."""
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


def test_process_forked_after_openmp_can_launch_forked_workers():
    """A process forked after its parent used OpenMP must be able to use OpenMP
    and to launch forked workers, as an engine started inside it does."""
    script = """
import os
import signal
import torch
from vllm.utils.system_utils import get_mp_context

def use_openmp():
    torch.set_num_threads(4)
    x = torch.randn(1024, 1024)
    torch.einsum("in,jn->ij", x, x)

use_openmp()
pid = os.fork()
if pid == 0:
    signal.alarm(60)
    use_openmp()
    proc = get_mp_context().Process(target=use_openmp)
    proc.start()
    signal.alarm(0)
    proc.join(60)
    if proc.is_alive():
        proc.kill()
        proc.join()
        os._exit(1)
    os._exit(proc.exitcode)
raise SystemExit(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
"""
    env = os.environ | {"VLLM_WORKER_MULTIPROC_METHOD": "fork"}
    result = subprocess.run([sys.executable, "-c", script], env=env, timeout=180)
    assert result.returncode == 0


def test_spawn_mode_skips_the_libgomp_release_on_fork():
    """A child that imports vLLM after its parent used OpenMP inherits a pool
    without its threads, and joining them hangs. Spawn mode must not try."""
    script = """
import os
import signal
import torch

torch.set_num_threads(4)
x = torch.randn(1024, 1024)
torch.einsum("in,jn->ij", x, x)
pid = os.fork()
if pid == 0:
    signal.alarm(60)
    import vllm.utils.system_utils  # noqa: F401

    grandchild = os.fork()
    if grandchild == 0:
        os._exit(0)
    os._exit(os.waitstatus_to_exitcode(os.waitpid(grandchild, 0)[1]))
raise SystemExit(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
"""
    env = os.environ | {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    result = subprocess.run([sys.executable, "-c", script], env=env, timeout=180)
    assert result.returncode == 0


def test_llm_starts_in_process_forked_after_openmp():
    """`LLM` must start and generate in a process forked after its parent used
    OpenMP: its engine is forked from that process, which has no worker threads."""
    script = """
import os
import signal
import torch
from vllm import LLM, SamplingParams

torch.set_num_threads(4)
x = torch.randn(1024, 1024)
torch.einsum("in,jn->ij", x, x)
pid = os.fork()
if pid == 0:
    signal.alarm(900)
    llm = LLM(
        model="hmellor/tiny-random-LlamaForCausalLM",
        enforce_eager=True,
        gpu_memory_utilization=0.2,
        max_model_len=64,
    )
    params = SamplingParams(max_tokens=2, ignore_eos=True)
    outputs = llm.generate("Hello", params, use_tqdm=False)
    assert len(outputs[0].outputs[0].token_ids) == 2
    llm.llm_engine.engine_core.shutdown()
    os._exit(0)
raise SystemExit(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
"""
    env = os.environ | {"VLLM_WORKER_MULTIPROC_METHOD": "fork"}
    result = subprocess.run([sys.executable, "-c", script], env=env, timeout=1000)
    assert result.returncode == 0
