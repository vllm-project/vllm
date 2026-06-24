# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from pathlib import Path

from vllm.utils.system_utils import _maybe_force_spawn, unique_filepath


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

def test_forkserver_opt_in_survives_cuda_init(monkeypatch):
    """Regression test: when the user explicitly opts into
    VLLM_WORKER_MULTIPROC_METHOD=forkserver, _maybe_force_spawn must NOT
    silently rewrite it to "spawn" — even when CUDA is initialized in the
    parent. The forkserver helper process forks workers from a clean snapshot
    taken before CUDA init, so the CUDA-init hazard that motivates forcing
    spawn does not apply."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "forkserver")
    monkeypatch.setattr("sys.argv", ["vllm", "serve"])
    # Simulate CUDA initialized in the parent process — without the
    # forkserver early-return, _maybe_force_spawn would rewrite to "spawn".
    monkeypatch.setattr(
        "vllm.utils.system_utils.cuda_is_initialized", lambda: True
    )
    _maybe_force_spawn()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "forkserver"


def test_forkserver_opt_in_survives_numa_bind(monkeypatch):
    """Same contract under --numa-bind: forkserver opt-in is preserved."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "forkserver")
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "--numa-bind"])
    _maybe_force_spawn()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "forkserver"


def test_spawn_opt_in_still_short_circuits(monkeypatch):
    """Sanity: existing "spawn" early-return still works (regression guard
    against accidentally clobbering the original branch)."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "--numa-bind"])
    _maybe_force_spawn()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_unset_with_cuda_init_still_forces_spawn(monkeypatch):
    """When the user has NOT opted in, CUDA-init still forces spawn."""
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr("sys.argv", ["vllm", "serve"])
    monkeypatch.setattr(
        "vllm.utils.system_utils.cuda_is_initialized", lambda: True
    )
    _maybe_force_spawn()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"

