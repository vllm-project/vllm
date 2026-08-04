# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
import errno
import os
import platform

# Third Party
import pytest

# First Party
from tests.v1.storage_backend.raw_block_test_utils import (
    RAW_BLOCK_CI_BLOCK_ALIGN,
    RAW_BLOCK_CI_CAPACITY_BYTES,
    make_raw_block_file,
)

lmcache_rust_raw_block_io = pytest.importorskip("lmcache_rust_raw_block_io")
RawBlockDevice = lmcache_rust_raw_block_io.RawBlockDevice


def _is_skip_safe_io_error(exc: BaseException) -> bool:
    if getattr(exc, "errno", None) in {errno.EINVAL, errno.ENOSYS, errno.EPERM}:
        return True
    msg = str(exc).lower()
    return any(
        text in msg
        for text in (
            "function not implemented",
            "invalid argument",
            "io_uring init failed",
            "not supported",
            "operation not permitted",
            "unsupported",
        )
    )


def test_raw_block_device_posix_roundtrip_on_tmp_file(tmp_path):
    path = make_raw_block_file(tmp_path)
    dev = RawBlockDevice(
        str(path),
        writable=True,
        use_odirect=False,
        alignment=RAW_BLOCK_CI_BLOCK_ALIGN,
        io_engine="posix",
        iouring_queue_depth=8,
    )

    try:
        assert dev.size_bytes() == RAW_BLOCK_CI_CAPACITY_BYTES

        payload1 = bytearray(b"raw-block-posix-ci-payload")
        payload2 = bytearray(bytes(range(64)))
        out1 = bytearray(len(payload1))
        out2 = bytearray(len(payload2))

        dev.pwrite_from_buffer(4096, payload1, len(payload1), len(payload1))
        dev.pwrite_from_buffer(8192, payload2, len(payload2), len(payload2))

        dev.pread_into(4096, out1, len(out1), len(out1))
        dev.pread_into(8192, out2, len(out2), len(out2))

        assert out1 == payload1
        assert out2 == payload2
    finally:
        dev.close()


def test_raw_block_device_read_past_capacity_raises(tmp_path):
    path = make_raw_block_file(tmp_path)
    dev = RawBlockDevice(
        str(path),
        writable=True,
        use_odirect=False,
        alignment=RAW_BLOCK_CI_BLOCK_ALIGN,
        io_engine="posix",
        iouring_queue_depth=8,
    )

    try:
        out = bytearray(1)
        with pytest.raises(RuntimeError, match="unexpected EOF"):
            dev.pread_into(RAW_BLOCK_CI_CAPACITY_BYTES, out, len(out), len(out))
    finally:
        dev.close()


@pytest.mark.skipif(platform.system() != "Linux", reason="io_uring is Linux only")
def test_raw_block_device_iouring_best_effort_roundtrip(tmp_path):
    path = make_raw_block_file(tmp_path)
    dev = None
    try:
        dev = RawBlockDevice(
            str(path),
            writable=True,
            use_odirect=False,
            alignment=RAW_BLOCK_CI_BLOCK_ALIGN,
            io_engine="io_uring",
            iouring_queue_depth=8,
        )

        payload = bytearray(b"raw-block-iouring-ci-payload")
        out = bytearray(len(payload))

        batch_id = dev.batched_write([4096], [payload], [len(payload)])
        dev.wait_iouring(batch_id)
        batch_id = dev.batched_read([4096], [out], [len(out)])
        dev.wait_iouring(batch_id)

        assert out == payload
    except Exception as e:
        if _is_skip_safe_io_error(e):
            pytest.skip(f"io_uring is unavailable on this runner: {e}")
        raise
    finally:
        if dev is not None:
            dev.close()


@pytest.mark.skipif(
    os.getenv("LMCACHE_RUN_ODIRECT_SMOKE") != "1",
    reason="O_DIRECT smoke is opt-in and not part of default PR CI",
)
def test_raw_block_device_odirect_optional_smoke(tmp_path):
    path = make_raw_block_file(tmp_path)
    dev = None
    try:
        dev = RawBlockDevice(
            str(path),
            writable=True,
            use_odirect=True,
            alignment=RAW_BLOCK_CI_BLOCK_ALIGN,
            io_engine="posix",
            iouring_queue_depth=8,
        )

        payload = bytearray([17]) * RAW_BLOCK_CI_BLOCK_ALIGN
        out = bytearray(len(payload))
        dev.pwrite_from_buffer(4096, payload, len(payload), len(payload))
        dev.pread_into(4096, out, len(out), len(out))
        assert out == payload
    except Exception as e:
        if _is_skip_safe_io_error(e):
            pytest.skip(f"O_DIRECT is unavailable on this runner: {e}")
        raise
    finally:
        if dev is not None:
            dev.close()
