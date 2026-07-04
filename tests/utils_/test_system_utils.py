# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from pathlib import Path

import pytest

from vllm.utils.system_utils import (
    _matching_library_path,
    _maybe_force_spawn,
    find_loaded_library,
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


def _maps_line(path: str) -> str:
    return f"7f0000000000-7f0000010000 r-xp 00000000 00:1f 12345 {path}\n"


@pytest.mark.parametrize(
    "lib_name,path,expected",
    [
        # Real runtime variants must match.
        ("libcudart", "/usr/local/cuda/lib64/libcudart.so", True),
        ("libcudart", "/usr/local/cuda/lib64/libcudart.so.12", True),
        ("libcudart", "/opt/lib/libcudart-d0da41ae.so.11.0", True),
        ("libamdhip64", "/opt/rocm/lib/libamdhip64.so.7", True),
        # Python extension modules (`.` separated) must match.
        ("cumem_allocator", "/x/cumem_allocator.cpython-312-x86_64.so", True),
        # tilelang's stub must NOT be mistaken for the CUDA runtime (#47548).
        ("libcudart", "/x/tilelang/lib/libcudart_stub.so", False),
        # Unrelated library that merely contains the name as a substring.
        ("libcudart", "/x/libcudarter.so", False),
    ],
)
def test_matching_library_path(lib_name, path, expected):
    result = _matching_library_path(lib_name, _maps_line(path))
    assert (result == path) is expected
    if not expected:
        assert result is None


def test_find_loaded_library_skips_lookalike(monkeypatch, tmp_path):
    # tilelang's stub is loaded before the real runtime; the lookalike must be
    # skipped and the real library returned (regression for #47548).
    real = "/usr/local/cuda/lib64/libcudart.so.12"
    maps = tmp_path / "maps"
    maps.write_text(
        _maps_line("/x/tilelang/lib/libcudart_stub.so") + _maps_line(real)
    )

    import builtins

    real_open = builtins.open
    monkeypatch.setattr(
        builtins,
        "open",
        lambda f, *a, **k: real_open(maps, *a, **k)
        if f == "/proc/self/maps"
        else real_open(f, *a, **k),
    )
    assert find_loaded_library("libcudart") == real


def test_numa_bind_forces_spawn(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "--numa-bind"])
    _maybe_force_spawn()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
