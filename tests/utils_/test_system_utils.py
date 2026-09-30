# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

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


def test_suppress_stdout_without_fileno():
    # `sys.stdout` may have no underlying file descriptor (e.g. inside
    # `contextlib.redirect_stdout(io.StringIO())` or a Jupyter kernel).
    # suppress_stdout() must not crash in that case.
    with contextlib.redirect_stdout(io.StringIO()):
        with suppress_stdout():
            pass


def test_suppress_stdout_does_not_suppress_stderr():
    # When `sys.stdout` is bound to stderr (fd 2), suppress_stdout() must still
    # target the real stdout (fd 1) and leave stderr untouched.
    real_stdout = sys.stdout
    try:
        sys.stdout = sys.stderr
        with suppress_stdout():
            os.write(2, b"CRITICAL ERROR MESSAGE\n")
    finally:
        sys.stdout = real_stdout
