# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open

import pytest

from vllm.utils import system_utils
from vllm.utils.system_utils import (
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


def test_numa_bind_forces_spawn(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "--numa-bind"])
    _maybe_force_spawn()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


# ---------------------------------------------------------------------------
# find_loaded_library
# ---------------------------------------------------------------------------
#
# The function walks /proc/self/maps for a substring match. Stub libraries
# (``lib<name>_stub.so``) are skipped so the iteration can find a real
# loaded library further down the maps. The tests below mock ``open`` so
# they run on any platform without needing the real /proc/self/maps.


def _maps(*lines: str) -> str:
    """Build a synthetic /proc/self/maps blob from `address path` pairs.

    Real /proc/self/maps lines look like::

        7f1234500000-7f1234600000 r-xp 00000000 00:01 1234   /usr/lib/libfoo.so

    The function only cares about the first space-separated address column
    and the path, so we use a simplified form ``"<addr> <path>"``.
    """
    return "\n".join(lines) + "\n"


def _patch_proc_maps(monkeypatch, body: str) -> None:
    """Patch ``open`` inside ``vllm.utils.system_utils`` to return ``body``."""
    m = mock_open(read_data=body)
    monkeypatch.setattr(system_utils, "open", m, raising=False)


def test_find_loaded_library_returns_first_real_match(monkeypatch):
    body = _maps(
        "7f00 r-xp 00000000 00:01 1 /usr/local/cuda/lib64/libcudart.so.13.0.96",
    )
    _patch_proc_maps(monkeypatch, body)
    assert (
        find_loaded_library("libcudart") == "/usr/local/cuda/lib64/libcudart.so.13.0.96"
    )


def test_find_loaded_library_returns_none_when_unloaded(monkeypatch):
    body = _maps("7f00 r-xp 0 0:0 0 /usr/lib/libsomething_else.so")
    _patch_proc_maps(monkeypatch, body)
    assert find_loaded_library("libcudart") is None


def test_find_loaded_library_skips_stub_and_continues(monkeypatch):
    """Regression: when a stub library appears before the real one in the
    maps, the function must skip the stub and keep iterating to find the
    real library — not return the stub's path.

    Concrete case: tilelang on aarch64 ships ``libcudart_stub.so`` with
    zero CUDA runtime symbols. PyTorch later loads the real libcudart.
    Returning the stub here would cause ``AttributeError`` at the first
    callsite that does ``getattr(lib, <symbol>)``.
    """
    body = _maps(
        "7f00 r-xp 0 0:0 0 /opt/tilelang/lib/libcudart_stub.so",
        "7f10 r-xp 0 0:0 0 /usr/local/cuda/lib64/libcudart.so.13",
    )
    _patch_proc_maps(monkeypatch, body)
    assert find_loaded_library("libcudart") == "/usr/local/cuda/lib64/libcudart.so.13"


def test_find_loaded_library_returns_none_when_only_stub_loaded(monkeypatch):
    """When the only libcudart-named library loaded is a stub, the function
    returns None so callers can fall through to env-var fallback (e.g.
    ``VLLM_CUDART_SO_PATH``) instead of returning a broken handle."""
    body = _maps(
        "7f00 r-xp 0 0:0 0 /opt/tilelang/lib/libcudart_stub.so",
    )
    _patch_proc_maps(monkeypatch, body)
    assert find_loaded_library("libcudart") is None


def test_find_loaded_library_filter_applies_to_arbitrary_lib_name(monkeypatch):
    """The stub filter is by filename pattern (``_stub`` substring), not
    libcudart-specific. ROCm's ``libamdhip64`` and any other library
    consumed via this helper benefit equally."""
    body = _maps(
        "7f00 r-xp 0 0:0 0 /opt/somewhere/libamdhip64_stub.so",
        "7f10 r-xp 0 0:0 0 /opt/rocm/lib/libamdhip64.so.7",
    )
    _patch_proc_maps(monkeypatch, body)
    assert find_loaded_library("libamdhip64") == "/opt/rocm/lib/libamdhip64.so.7"


def test_find_loaded_library_dedupes_repeated_mapping_lines(monkeypatch):
    """Real /proc/self/maps repeats a single library across multiple
    mapping lines (one per page-protection segment). The function should
    not re-process the same path; a later fallback after the same path
    appears twice should still find the real library."""
    body = _maps(
        "7f00 r-xp 0 0:0 0 /opt/tilelang/lib/libcudart_stub.so",
        "7f08 rw-p 0 0:0 0 /opt/tilelang/lib/libcudart_stub.so",
        "7f10 r-xp 0 0:0 0 /usr/local/cuda/lib64/libcudart.so.13",
        "7f18 rw-p 0 0:0 0 /usr/local/cuda/lib64/libcudart.so.13",
    )
    _patch_proc_maps(monkeypatch, body)
    assert find_loaded_library("libcudart") == "/usr/local/cuda/lib64/libcudart.so.13"


def test_find_loaded_library_assert_unrelated_filename(monkeypatch):
    """Defensive: if a substring match returns something that doesn't
    actually start with the requested lib_name (e.g. a memory map that
    happens to mention the substring inside another path component),
    the assert should still fire so the caller is alerted to the
    surprise rather than silently using a wrong file."""
    body = _maps(
        "7f00 r-xp 0 0:0 0 /opt/junk/notlibcudart.so",
    )
    _patch_proc_maps(monkeypatch, body)
    with pytest.raises(AssertionError, match="Unexpected filename"):
        find_loaded_library("libcudart")
