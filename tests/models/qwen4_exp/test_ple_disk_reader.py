# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched O_DIRECT span reads (fs_io_C.batch_read_spans) used by PLE disk
offload: every span must match the file's bytes, whatever its alignment."""

import os

import numpy as np
import pytest

fs_io_C = pytest.importorskip("vllm.fs_io_C")


def _open_direct(path):
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError:
        pytest.skip("filesystem does not support O_DIRECT")


@pytest.fixture(scope="module")
def backend():
    return fs_io_C.read_spans_init(4, 8)


@pytest.fixture
def files(tmp_path):
    rng = np.random.default_rng(0)
    out = []
    # Sizes that end mid-block, so the last span reads a partial block.
    for i, size in enumerate((3 * 2**20 + 123, 2**20 + 7)):
        data = rng.integers(0, 256, size, dtype=np.uint8)
        path = tmp_path / f"shard{i}.bin"
        data.tofile(path)
        out.append((_open_direct(path), data))
    yield out
    for fd, _ in out:
        os.close(fd)


def _read(fds, offsets, length):
    fds = np.asarray(fds, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    out = np.empty((len(offsets), length), dtype=np.uint8)
    fs_io_C.batch_read_spans(fds, offsets, length, out)
    return out


@pytest.mark.parametrize("length", [1, 160, 264, 4096, 5000])
@pytest.mark.parametrize("n", [1, 37, 20000])
def test_spans_match_file_bytes(backend, files, length, n):
    rng = np.random.default_rng(n * length)
    which = rng.integers(0, len(files), n)
    sizes = np.array([len(data) for _, data in files])
    offsets = rng.integers(0, sizes[which] - length + 1)
    # Duplicates, a span at offset 0 and one that ends exactly at EOF.
    which[: min(n, 3)], offsets[: min(n, 3)] = which[0], offsets[0]
    if n > 3:
        which[3], offsets[3] = 0, 0
        which[4], offsets[4] = 1, sizes[1] - length
    fds = [files[w][0] for w in which]
    out = _read(fds, offsets, length)
    for i in range(n):
        data = files[which[i]][1]
        np.testing.assert_array_equal(out[i], data[offsets[i] : offsets[i] + length])


def test_empty_batch(backend, files):
    assert _read([], [], 160).shape == (0, 160)


def test_read_past_eof_raises(backend, files):
    fd, data = files[0]
    with pytest.raises(OSError):
        _read([fd], [len(data) - 10], 160)


@pytest.mark.parametrize(
    "fds, offsets, length, out_rows",
    [
        ([0], [0], 0, 1),  # zero length
        ([0], [0], 2**17, 1),  # longer than 64 KiB
        ([0, 0], [0], 160, 2),  # length mismatch
        ([0], [0], 160, 0),  # out too small
        ([0], [-1], 160, 1),  # negative offset
    ],
)
def test_bad_arguments_raise(backend, files, fds, offsets, length, out_rows):
    fds = [files[0][0] if f == 0 else f for f in fds]
    out = np.empty((out_rows, max(length, 1)), dtype=np.uint8)
    with pytest.raises(ValueError):
        fs_io_C.batch_read_spans(
            np.asarray(fds, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64),
            length,
            out,
        )


def test_backend_is_reported(backend):
    assert backend in ("io_uring", "pread")
    assert fs_io_C.read_spans_init() == backend
