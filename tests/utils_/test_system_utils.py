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


def test_frontend_imports_do_not_call_cuda_is_available():
    """Importing vLLM's public frontend APIs must not call
    torch.cuda.is_available(), torch.cuda._lazy_init(), or import
    torch.utils.cpp_extension so default 'fork' remains safe."""
    script = """
import sys
import torch.cuda

calls = []
orig_is_available = torch.cuda.is_available
orig_lazy_init = torch.cuda._lazy_init

def tracked_is_available():
    calls.append("is_available")
    return orig_is_available()

def tracked_lazy_init():
    calls.append("_lazy_init")
    return orig_lazy_init()

torch.cuda.is_available = tracked_is_available
torch.cuda._lazy_init = tracked_lazy_init

from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine  # noqa: F401

assert not calls, f"Unexpected CUDA calls during frontend import: {calls}"
assert "torch.utils.cpp_extension" not in sys.modules
assert "xgrammar" not in sys.modules
assert "llguidance" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_fla_check_shared_mem_avoids_triton_driver_on_nvidia():
    """Vendored FLA's check_shared_mem() must query CUDA properties via
    cuda_get_device_properties on NVIDIA rather than triton.runtime.driver.active,
    which calls torch.cuda.is_available() and poisons fork()."""
    script = """
import torch.cuda
import vllm.third_party.flash_linear_attention.ops.utils as fla_utils

def fail_is_available():
    raise AssertionError("torch.cuda.is_available() should not be called")

get_all_calls = []

def fail_get_all():
    get_all_calls.append("called")
    raise AssertionError("get_all_max_shared_mem() should not be called on NVIDIA")

queries = []
def fake_cuda_get_device_properties(device, names, init_cuda=False):
    queries.append((device, tuple(names)))
    return (166912,)

torch.cuda.is_available = fail_is_available
fla_utils.is_amd = False
fla_utils.is_nvidia = True
fla_utils.get_all_max_shared_mem = fail_get_all
fla_utils.cuda_get_device_properties = fake_cuda_get_device_properties
fla_utils.check_shared_mem.cache_clear()

result_ok = fla_utils.check_shared_mem("ampere")
assert not get_all_calls, "get_all_max_shared_mem() was called on NVIDIA"
assert result_ok is True
assert queries == [(0, ("shared_memory_per_block_optin",))]
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_triton_utils_importing_avoids_cuda_driver_is_active():
    """triton_utils.importing must determine active nvidia/amd drivers via
    current_platform without calling driver.is_active()."""
    pytest.importorskip("triton.backends")
    script = """
import importlib
import types
import triton.backends
from vllm.platforms import current_platform, vllm_version_matches_substr

def fail_is_active():
    raise AssertionError("driver.is_active() should not be called for nvidia/amd")

fake_driver = types.SimpleNamespace(is_active=fail_is_active)
triton.backends.backends = {
    "cpu": types.SimpleNamespace(driver=types.SimpleNamespace(is_active=lambda: True)),
    "nvidia": types.SimpleNamespace(driver=fake_driver),
    "amd": types.SimpleNamespace(driver=fake_driver),
}

import vllm.triton_utils.importing as importing
importlib.reload(importing)
expected = (
    current_platform.is_cuda()
    or current_platform.is_rocm()
    or vllm_version_matches_substr("cpu")
)
assert importing.HAS_TRITON == expected
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
