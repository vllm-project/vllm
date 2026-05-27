# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from pathlib import Path

import pytest

import __main__
from vllm.utils.system_utils import (
    _check_spawn_stdin_entrypoint,
    _maybe_force_spawn,
    get_mp_context,
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


def test_spawn_rejects_stdin_main(monkeypatch):
    monkeypatch.setattr(__main__, "__file__", "<stdin>", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        _check_spawn_stdin_entrypoint("spawn")

    message = str(exc_info.value)
    assert "standard input" in message
    assert "VLLM_ENABLE_V1_MULTIPROCESSING=0" in message


def test_get_mp_context_rejects_stdin_with_spawn(monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setattr(__main__, "__file__", "<stdin>", raising=False)

    with pytest.raises(RuntimeError, match="standard input"):
        get_mp_context()


def test_spawn_allows_python_c_without_main_file(monkeypatch):
    monkeypatch.delattr(__main__, "__file__", raising=False)

    _check_spawn_stdin_entrypoint("spawn")


def test_spawn_allows_importable_main_file(monkeypatch, tmp_path):
    main_file = tmp_path / "main.py"
    main_file.write_text("pass")
    monkeypatch.setattr(__main__, "__file__", str(main_file), raising=False)

    _check_spawn_stdin_entrypoint("spawn")


def test_fork_allows_stdin_main(monkeypatch):
    monkeypatch.setattr(__main__, "__file__", "<stdin>", raising=False)

    _check_spawn_stdin_entrypoint("fork")
