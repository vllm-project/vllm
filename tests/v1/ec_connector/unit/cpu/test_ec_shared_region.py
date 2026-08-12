# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECSharedRegion (the mmap substrate)."""

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
