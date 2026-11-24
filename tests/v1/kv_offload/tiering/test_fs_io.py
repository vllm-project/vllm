# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-file access tracking through both filesystem I/O implementations."""

import errno
import os
import time

import pytest

from vllm.logger import _print_warning_once
from vllm.v1.kv_offload.tiering.fs import io


@pytest.fixture(params=[False, True], ids=["python", "native"])
def backend(request, monkeypatch):
    if request.param and not io._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io, "_HAS_FSIO_C", request.param)


@pytest.mark.parametrize("operation", ["load", "reuse"])
def test_tracking_refreshes_atime_without_rewriting(tmp_path, backend, operation):
    path = tmp_path / "block.bin"
    path.write_bytes(b"cache")
    # Future atime prevents automatic relatime updates; explicit tracking must
    # replace it with now. Old mtime must remain exactly unchanged.
    future = time.time_ns() + 10**15
    os.utime(path, ns=(future, 1_000_000_000))
    before = path.stat()
    start = time.time_ns()
    buffer = bytearray(b"other")
    if operation == "load":
        io.batch_load_block([str(path)], memoryview(buffer), [0], 5, False, True)
        assert buffer == b"cache"
    else:
        assert (
            io.batch_store_block([str(path)], memoryview(buffer), [0], 5, False, True)
            == 0
        )
    after = path.stat()
    assert start <= after.st_atime_ns <= time.time_ns()
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ino == before.st_ino
    assert path.read_bytes() == b"cache"


def test_disabled_reuse_does_not_refresh(tmp_path, backend):
    path = tmp_path / "block.bin"
    path.write_bytes(b"cache")
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    before = path.stat()
    assert io.batch_store_block([str(path)], memoryview(b"other"), [0], 5, False) == 0
    after = path.stat()
    assert after.st_atime_ns == before.st_atime_ns
    assert after.st_mtime_ns == before.st_mtime_ns


def test_partial_load_tracks_successful_prefix(tmp_path, backend):
    path = tmp_path / "block.bin"
    path.write_bytes(b"cache")
    os.utime(path, ns=(time.time_ns() + 10**15, 1_000_000_000))
    start = time.time_ns()
    buffer = bytearray(10)
    with pytest.raises(OSError) as error:
        io.batch_load_block(
            [str(path), str(tmp_path / "missing.bin")],
            memoryview(buffer),
            [0, 5],
            5,
            False,
            True,
        )
    assert error.value.num_succeeded == 1
    assert buffer[:5] == b"cache"
    assert start <= path.stat().st_atime_ns <= time.time_ns()
    assert path.stat().st_mtime_ns == 1_000_000_000


def test_new_store_preserves_data_and_count(tmp_path, backend):
    path = tmp_path / "new.bin"
    assert (
        io.batch_store_block([str(path)], memoryview(b"cache"), [0], 5, False, True)
        == 1
    )
    assert path.read_bytes() == b"cache"


@pytest.mark.parametrize("operation", ["load", "reuse"])
def test_metadata_failure_does_not_fail_transfer(
    tmp_path, monkeypatch, caplog_vllm, operation
):
    path = tmp_path / "block.bin"
    path.write_bytes(b"cache")
    monkeypatch.setattr(io, "_HAS_FSIO_C", False)
    _print_warning_once.cache_clear()

    def denied(*args, **kwargs):
        raise PermissionError(errno.EPERM, "metadata update denied")

    monkeypatch.setattr(os, "utime", denied)
    buffer = bytearray(b"other")
    for _ in range(3):
        if operation == "load":
            io.batch_load_block([str(path)], memoryview(buffer), [0], 5, False, True)
            assert buffer == b"cache"
        else:
            assert (
                io.batch_store_block(
                    [str(path)], memoryview(buffer), [0], 5, False, True
                )
                == 0
            )
    assert path.read_bytes() == b"cache"
    warnings = [
        r for r in caplog_vllm.records if "refresh cache access time" in r.message
    ]
    assert len(warnings) == 1
    _print_warning_once.cache_clear()
