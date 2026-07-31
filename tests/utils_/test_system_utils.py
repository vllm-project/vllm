# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from pathlib import Path

import pytest

from vllm.platforms import current_platform
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


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_bad_fork_forces_spawn():
    """A child forked from a CUDA-initialized parent must not fork again.

    ``torch.cuda.is_initialized()`` reports False in such a child even though
    CUDA is unusable there, so the check must key off the bad-fork state.
    """
    import torch

    torch.zeros(1, device="cuda")

    pid = os.fork()
    if pid == 0:
        os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
        try:
            _maybe_force_spawn()
            forced = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"
        except Exception:
            forced = False
        os._exit(0 if forced else 1)

    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
