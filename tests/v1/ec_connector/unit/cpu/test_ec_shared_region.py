# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECSharedRegion (the mmap substrate)."""

import contextlib
import ctypes
import errno
import mmap
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
    _wait_for_file_size,
)


def _make_region(num_blocks: int = 8, block_size_bytes: int = 64) -> ECSharedRegion:
    return ECSharedRegion(
        engine_id=str(uuid.uuid4()),
        num_blocks=num_blocks,
        block_size_bytes=block_size_bytes,
    )


@pytest.fixture
def region() -> ECSharedRegion:
    r = _make_region()
    yield r
    r.cleanup()


@contextlib.contextmanager
def _region(**kwargs):
    """Context manager: create one region, clean up on exit."""
    r = _make_region(**kwargs)
    try:
        yield r
    finally:
        r.cleanup()


def _page_residency(mmap_obj: mmap.mmap, length: int) -> list[bool]:
    """Return Linux page-residency bits for a writable mmap."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        mincore = libc.mincore
    except AttributeError:
        pytest.skip("mincore is unavailable")

    page_count = (length + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    mincore.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    mincore.restype = ctypes.c_int
    vector = (ctypes.c_ubyte * page_count)()
    address = ctypes.addressof(ctypes.c_ubyte.from_buffer(mmap_obj))
    result = mincore(ctypes.c_void_p(address), length, vector)
    if result != 0:
        raise OSError(ctypes.get_errno())
    return [bool(value & 1) for value in vector]


# ── mmap sharing between two instances ───────────────────────────────────────


def test_second_instance_opens_existing_file_and_shares_memory():
    instance_id = str(uuid.uuid4())
    r1 = ECSharedRegion(
        engine_id=instance_id,
        num_blocks=4,
        block_size_bytes=64,
    )
    try:
        r2 = ECSharedRegion(
            engine_id=instance_id,
            num_blocks=4,
            block_size_bytes=64,
        )
        try:
            assert r1._is_creator
            assert not r2._is_creator
            # Both map the same physical pages; writes via r1 are visible via r2.
            r1.blocks[0, :4] = torch.tensor([10, 20, 30, 40], dtype=torch.int8)
            assert r2.blocks[0, :4].tolist() == [10, 20, 30, 40]
        finally:
            r2.cleanup()
    finally:
        r1.cleanup()


def test_only_creator_unlinks_file_on_cleanup():
    """Critical contract: if the non-creator unlinks, the creator's mmap path
    becomes a dangling backing file and a third opener would create a new one
    out from under the creator."""
    instance_id = str(uuid.uuid4())
    r1 = ECSharedRegion(
        engine_id=instance_id,
        num_blocks=4,
        block_size_bytes=64,
    )
    path = r1._mmap_path
    r2 = ECSharedRegion(
        engine_id=instance_id,
        num_blocks=4,
        block_size_bytes=64,
    )

    # Non-creator goes away first — file must still be on disk for r1.
    r2.cleanup()
    assert os.path.exists(path), "non-creator cleanup must not unlink the file"

    # Creator goes away — file is removed.
    r1.cleanup()
    assert not os.path.exists(path), "creator cleanup must unlink the file"


# ── _wait_for_file_size ──────────────────────────────────────────────────────


def test_wait_for_file_size_returns_when_already_big_enough(tmp_path):
    """The fast path: file already at expected size — return immediately."""
    p = tmp_path / "f.bin"
    p.write_bytes(b"\x00" * 128)
    fd = os.open(str(p), os.O_RDONLY)
    try:
        _wait_for_file_size(fd, expected_size=128, timeout=1.0)  # must not raise
    finally:
        os.close(fd)


def test_wait_for_file_size_times_out_when_file_stays_empty(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"")
    fd = os.open(str(p), os.O_RDONLY)
    try:
        with pytest.raises(TimeoutError):
            _wait_for_file_size(fd, expected_size=4096, timeout=0.05)
    finally:
        os.close(fd)


# ── MADV_POPULATE_WRITE pre-faulting / fallback ──────────────────────────────


def test_madvise_success_selects_madvise_population(monkeypatch):
    """A successful probe must keep using MADV_POPULATE_WRITE — the fallback
    helper must never be invoked on a kernel that accepts the advice.

    Spies on both helpers so we can verify call counts: 1 probe call over a
    single page + 1 populate call over the whole flat region, fallback 0 times.
    """
    from vllm.distributed.ec_transfer.ec_connector.cpu import ec_shared_region as esr

    madvise_calls: list[tuple[int, int, int]] = []
    fallback_calls: list[tuple[int, int]] = []

    def _spy_madvise(mm, off, ln):
        madvise_calls.append((off, ln, id(mm)))

    def _spy_fallback(mm, off, ln):
        fallback_calls.append((off, ln))

    monkeypatch.setattr(esr, "_madvise_populate_write", _spy_madvise)
    monkeypatch.setattr(esr, "_fallback_populate_write", _spy_fallback)

    num_blocks, block_size_bytes = 4, mmap.PAGESIZE
    with _region(num_blocks=num_blocks, block_size_bytes=block_size_bytes):
        total = num_blocks * block_size_bytes
        mmap_id = madvise_calls[0][2]
        assert madvise_calls == [
            (0, mmap.PAGESIZE, mmap_id),  # probe
            (0, total, mmap_id),  # flat pre-fault of the whole region
        ]
        assert fallback_calls == [], (
            "native-path constructor must not invoke the fallback helper"
        )


def test_madvise_einval_selects_fallback_for_whole_region(monkeypatch):
    """An EINVAL probe must select the fallback for the whole flat region.

    This is the regression under test: on Linux < 5.14 the constant is absent
    from the `mmap` module, so advice=23 reaches a kernel that rejects it and
    every ECSharedRegion construction used to abort with EINVAL.
    """
    from vllm.distributed.ec_transfer.ec_connector.cpu import ec_shared_region as esr

    fallback_calls: list[tuple[int, int]] = []
    real_fallback = esr._fallback_populate_write

    def _raise_einval(mm, off, ln):
        raise OSError(errno.EINVAL, "simulated unsupported kernel")

    def _spy_fallback(mm, off, ln):
        fallback_calls.append((off, ln))
        return real_fallback(mm, off, ln)

    monkeypatch.setattr(esr, "_madvise_populate_write", _raise_einval)
    monkeypatch.setattr(esr, "_fallback_populate_write", _spy_fallback)

    num_blocks, block_size_bytes = 4, mmap.PAGESIZE
    with _region(num_blocks=num_blocks, block_size_bytes=block_size_bytes) as r:
        assert fallback_calls == [(0, num_blocks * block_size_bytes)]
        # Region is still fully usable after taking the fallback path.
        r.blocks[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
        assert r.blocks[0, :4].tolist() == [1, 2, 3, 4]


def test_non_creator_does_not_probe_or_populate(monkeypatch):
    """Only the creator pre-faults. A second opener must not probe the advice
    nor re-touch pages the creator already populated."""
    from vllm.distributed.ec_transfer.ec_connector.cpu import ec_shared_region as esr

    instance_id = str(uuid.uuid4())
    r1 = ECSharedRegion(
        engine_id=instance_id, num_blocks=4, block_size_bytes=mmap.PAGESIZE
    )
    try:
        calls: list[str] = []
        monkeypatch.setattr(
            esr,
            "_madvise_populate_write",
            lambda mm, off, ln: calls.append("madvise"),
        )
        monkeypatch.setattr(
            esr,
            "_fallback_populate_write",
            lambda mm, off, ln: calls.append("fallback"),
        )
        r2 = ECSharedRegion(
            engine_id=instance_id, num_blocks=4, block_size_bytes=mmap.PAGESIZE
        )
        try:
            assert not r2._is_creator
            assert calls == []
        finally:
            r2.cleanup()
    finally:
        r1.cleanup()


def test_fallback_populate_write_preserves_bytes_and_faults_pages():
    """Fallback preserves existing bytes and touches every target page.

    `|= 0` rather than `= 0` matters here: a peer worker may already have
    written EC data into the shared mmap before we pre-fault it.
    """
    from vllm.distributed.ec_transfer.ec_connector.cpu import ec_shared_region as esr

    size = 3 * mmap.PAGESIZE
    if not hasattr(mmap, "MADV_DONTNEED"):
        pytest.skip("MADV_DONTNEED is unavailable")

    mmap_obj = mmap.mmap(
        -1,
        size,
        flags=mmap.MAP_SHARED,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    try:
        mmap_obj.madvise(mmap.MADV_DONTNEED, 0, size)
        if any(_page_residency(mmap_obj, size)):
            pytest.skip("kernel did not discard anonymous mmap pages")

        mmap_obj[0] = 0xAB
        assert _page_residency(mmap_obj, size) == [True, False, False]

        esr._fallback_populate_write(mmap_obj, 0, size)

        assert _page_residency(mmap_obj, size) == [True, True, True]
        assert mmap_obj[0] == 0xAB
    finally:
        mmap_obj.close()


def test_madvise_unexpected_oserror_propagates(monkeypatch):
    """Only EINVAL triggers the fallback.  Other OSErrors (e.g. EIO) must
    propagate out of __init__, not be silently masked by the fallback branch.
    """
    from vllm.distributed.ec_transfer.ec_connector.cpu import ec_shared_region as esr

    monkeypatch.setattr(
        esr,
        "_madvise_populate_write",
        lambda mm, off, ln: (_ for _ in ()).throw(
            OSError(errno.EIO, "simulated I/O failure")
        ),
    )
    engine_id = str(uuid.uuid4())
    try:
        with pytest.raises(OSError) as exc_info:
            ECSharedRegion(
                engine_id=engine_id, num_blocks=4, block_size_bytes=mmap.PAGESIZE
            )
        assert exc_info.value.errno == errno.EIO
    finally:
        # __init__ raised past the unlink-owning cleanup(), so drop the
        # backing file here rather than leaking it into /dev/shm.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(f"/dev/shm/vllm_ec_{engine_id}.mmap")


# ── pin_memory ───────────────────────────────────────────────────────────────


def test_pin_memory_success_sets_flag(region):
    """When cudaHostRegister returns 0, _is_pinned flips to True
    and cleanup will correspondingly call cudaHostUnregister."""
    fake_cudart = MagicMock()
    success = MagicMock()
    success.value = 0
    fake_cudart.cudaHostRegister.return_value = success
    fake_cudart.cudaHostUnregister.return_value = success

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.cudart", return_value=fake_cudart),
    ):
        region.pin_memory()
        assert region._is_pinned is True
        # cleanup must pair with cudaHostUnregister exactly once.
        region.cleanup()
        fake_cudart.cudaHostUnregister.assert_called_once()


def test_pin_memory_failure_leaves_flag_false():
    """If cudaHostRegister fails (non-zero), don't pretend it succeeded —
    cleanup must NOT call cudaHostUnregister on memory we never registered."""
    r = _make_region()
    try:
        fake_cudart = MagicMock()
        fail = MagicMock()
        fail.value = 1  # non-zero == error
        fake_cudart.cudaHostRegister.return_value = fail

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.cudart", return_value=fake_cudart),
        ):
            r.pin_memory()
            assert r._is_pinned is False
            # Now run cleanup and verify cudaHostUnregister was NOT called.
            r.cleanup()
            fake_cudart.cudaHostUnregister.assert_not_called()
    finally:
        r.cleanup()


def test_pin_memory_noop_without_cuda(region):
    """pin_memory is a no-op when CUDA is not available."""
    with patch("torch.cuda.is_available", return_value=False):
        region.pin_memory()
        assert region._is_pinned is False


# ── cleanup idempotency ───────────────────────────────────────────────────────


def test_cleanup_is_idempotent(region):
    region.cleanup()
    region.cleanup()  # fixture calls a third time — must not raise
